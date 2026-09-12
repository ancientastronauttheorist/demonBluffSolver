use super::super::ascension_script::CustomScriptRecord;
use super::*;
use serde_json::Value;

fn report() -> Value {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_ascension_starting_sequence_audit.json")).unwrap()
}

fn context(input: &Value) -> AscensionStartingContext {
    let inline_scripts = serde_json::from_value(input["inline"].clone()).unwrap();
    let custom_scripts = input["custom"].as_array().map(|records| {
        records
            .iter()
            .map(|record| {
                if record.is_null() {
                    None
                } else {
                    Some(CustomScriptRecord {
                        script_id: record["script_info"].as_u64().map(|v| v as u16),
                    })
                }
            })
            .collect()
    });
    let method = match input["method"].as_str().unwrap() {
        "AllLazy" | "AllLazyGameData" => StartingMethod::AllLazy,
        "AllStored" => StartingMethod::AllStored,
        "Typed" => StartingMethod::Typed {
            character_type: input["type"].as_i64().unwrap() as i32,
        },
        other => panic!("unknown native method {other}"),
    };
    AscensionStartingContext {
        rule_version: ASCENSION_STARTING_NATIVE_V1.into(),
        services_preserve_inputs: true,
        method,
        selection: AscensionScriptContext {
            rule_version: ASCENSION_SCRIPT_NATIVE_V1.into(),
            script_ids: vec![1, 2],
            cached_script: input["initial_cache"].as_u64().map(|v| v as u16),
            inline_scripts,
            custom_scripts,
        },
        script_lists: serde_json::from_value(input["script_lists"].clone()).unwrap(),
        starting: serde_json::from_value(input["starting"].clone()).unwrap(),
        service_failures: StartingServiceFailures {
            to_array: input["service_options"]["to_array_fail"]
                .as_u64()
                .map(|v| v as u8),
            append: input["service_options"]["append_fail"]
                .as_u64()
                .map(|v| v as u8),
        },
    }
}

fn branching_context(report: &Value) -> AscensionStartingContext {
    let case = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| {
            let i = &case["input"];
            i["method"] == "AllLazy"
                && i["initial_cache"].is_null()
                && i["inline"] == serde_json::json!([2, 2])
                && i["custom"] == serde_json::json!([{"script_info":null},{"script_info":1}])
                && i["service_options"] == serde_json::json!({})
        })
        .unwrap();
    context(&case["input"])
}

#[test]
fn matches_every_in_scope_native_case_including_failure_prefixes() {
    let report = report();
    let mut checked = 0;
    for case in report["cases"].as_array().unwrap() {
        let input = &case["input"];
        let expected = &case["expected"];
        // This kernel receives an existing profile context; wrapper allocation
        // before null-profile rejection is independently covered by native code.
        if input["service_options"]["null_profile"] == true {
            continue;
        }
        let ctx = context(input);
        let before = ctx.clone();
        let paths = replay_ascension_starting(&ctx).unwrap();
        assert_eq!(ctx, before);
        let draws = expected["draws"].as_array().unwrap();
        let path = paths
            .iter()
            .find(|p| {
                p.draws.len() == draws.len()
                    && p.draws.iter().zip(draws).all(|(a, b)| {
                        a.occurrence_index as u64 == b["index"].as_u64().unwrap()
                            && a.width as u64 == b["width"].as_u64().unwrap()
                            && (if a.source == ScriptDrawSource::Inline {
                                "inline"
                            } else {
                                "custom"
                            }) == b["source"].as_str().unwrap()
                    })
            })
            .unwrap_or_else(|| panic!("native path missing: {case}"));
        assert_eq!(
            path.cached_script,
            expected["cache"].as_u64().map(|v| v as u16),
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&path.cache_writes).unwrap(),
            expected["cache_writes"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&path.faction_read_order).unwrap(),
            expected["faction_read_order"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&path.output_prefix).unwrap(),
            expected["output_prefix"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&path.typed_result).unwrap(),
            expected["typed_result"],
            "{case}"
        );
        assert_eq!(
            path.append_attempts as u64,
            expected["append_count"].as_u64().unwrap(),
            "{case}"
        );
        let native_error = match &path.failure {
            None => Value::Null,
            Some(StartingFailure::Selection { .. } | StartingFailure::NullSelectedList { .. }) => {
                Value::String("null".into())
            }
            Some(StartingFailure::ToArrayService { .. }) => Value::String("to_array".into()),
            Some(
                StartingFailure::NullAppendCollection { .. }
                | StartingFailure::AppendService { .. },
            ) => Value::String("collection".into()),
        };
        assert_eq!(native_error, expected["error"], "{case}");
        let denominator = draws
            .iter()
            .map(|d| d["width"].as_u64().unwrap())
            .product::<u64>();
        assert_eq!(
            path.probability,
            Probability {
                numerator: 1,
                denominator
            }
        );
        checked += 1;
    }
    assert_eq!(checked, report["native_cases"].as_u64().unwrap() - 1);
    assert_eq!(checked, 964);
}

#[test]
fn all_46_weighted_occurrence_paths_match_native_without_merging_discarded_draws() {
    let report = report();
    let paths = replay_ascension_starting(&branching_context(&report)).unwrap();
    assert_eq!(paths.len(), 46);
    let mut common_mass = 0;
    for case in report["weighted_native_cases"].as_array().unwrap() {
        let choices: Vec<u16> = serde_json::from_value(case["choices"].clone()).unwrap();
        let path = paths
            .iter()
            .find(|p| {
                p.draws
                    .iter()
                    .map(|d| d.occurrence_index)
                    .eq(choices.iter().copied())
            })
            .unwrap();
        let probability = case["probability"].as_str().unwrap();
        let (num, den) = probability.split_once('/').unwrap();
        assert_eq!(
            path.probability,
            Probability {
                numerator: num.parse().unwrap(),
                denominator: den.parse().unwrap()
            }
        );
        common_mass += 256 / path.probability.denominator;
    }
    assert_eq!(common_mass, 256);
    let mixed = paths
        .iter()
        .find(|p| p.draws.iter().map(|d| d.occurrence_index).eq([0, 0, 1, 1]))
        .unwrap();
    assert_eq!(mixed.cache_writes, vec![Some(2), None, Some(2), Some(1)]);
    assert_eq!(
        mixed.output_prefix,
        vec![
            Some(1001),
            Some(1001),
            None,
            Some(2101),
            Some(2101),
            Some(3101),
            Some(4101)
        ]
    );
}

#[test]
fn stored_ignores_cache_and_sources_typed_null_is_successful() {
    let mut ctx = branching_context(&report());
    ctx.method = StartingMethod::AllStored;
    ctx.selection.inline_scripts = None;
    ctx.selection.custom_scripts = None;
    ctx.selection.cached_script = Some(1);
    let path = replay_ascension_starting(&ctx).unwrap().remove(0);
    assert!(path.draws.is_empty() && path.cache_writes.is_empty());
    assert_eq!(path.cached_script, Some(1));
    assert_eq!(
        path.output_prefix,
        ctx.starting
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
    );
    ctx.selection.cached_script = None;
    ctx.selection.inline_scripts = Some(vec![]);
    ctx.selection.custom_scripts = Some(vec![]);
    ctx.starting[0] = None;
    ctx.method = StartingMethod::Typed { character_type: 10 };
    let path = replay_ascension_starting(&ctx).unwrap().remove(0);
    assert_eq!(path.failure, None);
    assert_eq!(path.typed_result, None);
    assert_eq!(path.cache_writes, vec![None]);
    assert_eq!(path.append_attempts, 0);
    ctx.method = StartingMethod::AllLazy;
    let path = replay_ascension_starting(&ctx).unwrap().remove(0);
    assert_eq!(
        path.failure,
        Some(StartingFailure::NullAppendCollection { character_type: 10 })
    );
}

#[test]
fn unsupported_type_still_selects_and_custom_record_failure_preserves_inline_write() {
    let mut ctx = branching_context(&report());
    ctx.method = StartingMethod::Typed {
        character_type: i32::MIN,
    };
    ctx.selection.inline_scripts = Some(vec![Some(2)]);
    ctx.selection.custom_scripts = Some(vec![None]);
    let p = replay_ascension_starting(&ctx).unwrap().remove(0);
    assert_eq!(p.cached_script, Some(2));
    assert_eq!(p.cache_writes, vec![Some(2)]);
    assert_eq!(p.draws.len(), 2);
    assert_eq!(
        p.failure,
        Some(StartingFailure::Selection {
            failure: ScriptSelectionFailure::NullCustomRecord
        })
    );
    ctx.selection.custom_scripts = Some(vec![Some(CustomScriptRecord { script_id: Some(1) })]);
    let p = replay_ascension_starting(&ctx).unwrap().remove(0);
    assert_eq!(p.failure, None);
    assert_eq!(p.typed_result, None);
    assert_eq!(p.to_array_attempts, 0);
    assert_eq!(p.draws.len(), 2);
}

#[test]
fn rejects_invalid_provenance_and_unknown_fields() {
    let original = branching_context(&report());
    for change in 0..6 {
        let mut ctx = original.clone();
        match change {
            0 => ctx.services_preserve_inputs = false,
            1 => ctx.rule_version = "future".into(),
            2 => ctx.selection.script_ids.push(1),
            3 => {
                ctx.script_lists.remove(&1);
            }
            4 => ctx.selection.cached_script = Some(99),
            _ => ctx.service_failures.append = Some(0),
        }
        assert_eq!(
            replay_ascension_starting(&ctx),
            Err(LedgerError::InvalidContext)
        );
    }
    let mut json = serde_json::to_value(original).unwrap();
    json["ignored_field"] = Value::Bool(true);
    assert!(serde_json::from_value::<AscensionStartingContext>(json).is_err());
}

#[test]
fn rejects_exponential_support_and_retained_memory_without_partial_results() {
    let mut ctx = branching_context(&report());
    ctx.selection.inline_scripts = Some(vec![None; 64]);
    ctx.selection.custom_scripts = Some(vec![Some(CustomScriptRecord { script_id: None }); 64]);
    assert_eq!(replay_ascension_starting(&ctx), Err(LedgerError::Capacity));
    ctx.selection.inline_scripts = Some(vec![Some(1); 512]);
    ctx.selection.custom_scripts = Some(vec![]);
    ctx.script_lists.get_mut(&1).unwrap()[0] = Some(vec![Some(u16::MAX); 4096]);
    assert_eq!(replay_ascension_starting(&ctx), Err(LedgerError::Capacity));
    ctx.method = StartingMethod::AllStored;
    // Unused source breadth does not trigger hypothetical selection expansion.
    assert!(replay_ascension_starting(&ctx).is_ok());
    ctx.starting[0] = Some(vec![None; 4097]);
    assert_eq!(replay_ascension_starting(&ctx), Err(LedgerError::Capacity));
}
