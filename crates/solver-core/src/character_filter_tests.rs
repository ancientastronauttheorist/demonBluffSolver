use super::*;
use serde_json::Value;
fn report() -> Value {
    serde_json::from_str(include_str!("../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_characters_filter_tail.json")).unwrap()
}
fn context(case: &Value) -> CharacterFilterContext {
    let o = &case["options"];
    let method = serde_json::from_value(case["method"].clone()).unwrap();
    let input: Option<Vec<Option<u16>>> = serde_json::from_value(case["input"].clone()).unwrap();
    let board: Option<Vec<Option<u16>>> = if o["null_board"] == true {
        None
    } else {
        Some(
            o["board"]
                .as_array()
                .map(|_| serde_json::from_value(o["board"].clone()).unwrap())
                .unwrap_or(vec![Some(0)]),
        )
    };
    let states: Vec<i32> = o["states"]
        .as_array()
        .map(|_| serde_json::from_value(o["states"].clone()).unwrap())
        .unwrap_or(vec![5, 0, 10, 20]);
    let characters = (0..4u16)
        .map(|i| {
            (
                i,
                FilterCharacter {
                    state: states[i as usize],
                    statuses: if o["missing"].is_string() {
                        None
                    } else {
                        Some(vec![10, i as i32])
                    },
                    data: (o["null_data"] != true).then_some(i),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut managed_contains = vec![];
    let mut unity_equals = vec![];
    let mut remaining = input.clone().unwrap_or_default();
    let canonical = |id: Option<u16>| {
        if o["unity_null_alias"] == true && id == Some(1) {
            None
        } else {
            id
        }
    };
    if method == CharacterFilterMethod::Unique {
        for ch in board.iter().flatten() {
            let Some(ch) = ch else {
                managed_contains.push(false);
                continue;
            };
            let target = characters[ch].data;
            let answer = o["contains_override"]
                .as_bool()
                .unwrap_or(remaining.contains(&target));
            managed_contains.push(answer);
            if answer {
                let mut kept = vec![];
                for value in remaining {
                    let equal = canonical(value) == canonical(target);
                    unity_equals.push(equal);
                    if !equal {
                        kept.push(value);
                    }
                }
                remaining = kept;
            }
        }
    }
    let fail_at = o["fail"].as_array().map(|f| FilterServiceFailure {
        gateway: serde_json::from_value(f[0].clone()).unwrap(),
        occurrence: f[1].as_u64().unwrap() as u16,
    });
    CharacterFilterContext {
        rule_version: CHARACTER_FILTER_NATIVE_V1.into(),
        metadata_initialized: true,
        services_preserve_input_contents: true,
        remove_all_commits_after_predicates: true,
        method,
        input,
        characters,
        board,
        requested_status: o["status"].as_i64().unwrap_or(10) as i32,
        gameplay_initialized: o["cold"] != true,
        unity_object_initialized: o["cold"] != true,
        input_version: 0,
        advance_input_version_after_add: o["mutate_source"] == true,
        managed_contains,
        unity_equals,
        fail_at,
    }
}
#[test]
fn matches_all_83_native_cases_and_partial_outputs() {
    let report = report();
    let mut count = 0;
    for case in report["cases"].as_array().unwrap() {
        let c = context(case);
        let before = c.clone();
        let r = replay_character_filter(&c).unwrap();
        assert_eq!(c, before);
        assert_eq!(
            serde_json::to_value(&r.output_prefix).unwrap(),
            case["output_prefix"],
            "{case}"
        );
        let queries: Vec<Value> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["event"] == "managed_contains")
            .map(|e| e["value"].clone())
            .collect();
        let pairs: Vec<Value> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["event"] == "unity_equal")
            .map(|e| serde_json::json!([e["left"], e["right"]]))
            .collect();
        assert_eq!(
            serde_json::to_value(&r.managed_contains_queries).unwrap(),
            serde_json::to_value(queries).unwrap(),
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&r.unity_equal_queries).unwrap(),
            serde_json::to_value(pairs).unwrap(),
            "{case}"
        );
        let native: Vec<_> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["event"].clone())
            .collect();
        assert_eq!(
            serde_json::to_value(&r.events).unwrap(),
            serde_json::to_value(native).unwrap(),
            "{case}"
        );
        let error = match r.failure {
            None => Value::Null,
            Some(CharacterFilterFailure::Null) => "null".into(),
            Some(CharacterFilterFailure::CollectionNull) => "collection_null".into(),
            Some(CharacterFilterFailure::Version) => "version".into(),
            Some(CharacterFilterFailure::Gateway(g)) => serde_json::to_value(g).unwrap(),
        };
        assert_eq!(error, case["error"], "{case}");
        assert_eq!(r.returned, error.is_null());
        count += 1;
    }
    assert_eq!(count, 83);
}
#[test]
fn distinct_equality_contract_and_missing_provenance_rejection() {
    let data = report();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| {
            v["method"] == "unique"
                && v["input"] == serde_json::json!([0, 1, 0, 2])
                && v["options"]["board"] == serde_json::json!([0])
        })
        .unwrap();
    let mut c = context(case);
    c.managed_contains = vec![false];
    c.unity_equals.clear();
    assert_eq!(
        replay_character_filter(&c).unwrap().output_prefix,
        vec![Some(0), Some(1), Some(0), Some(2)]
    );
    c.managed_contains = vec![true];
    c.unity_equals = vec![false; 4];
    assert_eq!(
        replay_character_filter(&c).unwrap().output_prefix,
        vec![Some(0), Some(1), Some(0), Some(2)]
    );
    c.unity_equals.clear();
    assert_eq!(
        replay_character_filter(&c),
        Err(FilterContextError::InvalidContext)
    );
    c = context(case);
    c.remove_all_commits_after_predicates = false;
    assert_eq!(
        replay_character_filter(&c),
        Err(FilterContextError::InvalidContext)
    );
    let mut value = serde_json::to_value(context(case)).unwrap();
    value["destroyed_object_truth"] = true.into();
    assert!(serde_json::from_value::<CharacterFilterContext>(value).is_err());
}

#[test]
fn supplied_equality_answers_still_record_exact_candidate_and_capture() {
    let data = report();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| {
            v["method"] == "unique"
                && v["input"] == serde_json::json!([0, 1, 0, 2])
                && v["options"]["board"] == serde_json::json!([0])
        })
        .unwrap();
    let mut c = context(case);
    c.characters.get_mut(&0).unwrap().data = None;
    c.managed_contains = vec![true];
    c.unity_equals = vec![false; 4];
    let r = replay_character_filter(&c).unwrap();
    assert_eq!(r.managed_contains_queries, vec![None]);
    assert_eq!(
        r.unity_equal_queries,
        vec![
            (Some(0), None),
            (Some(1), None),
            (Some(0), None),
            (Some(2), None)
        ]
    );
    assert_eq!(r.output_prefix, c.input.unwrap());
}
