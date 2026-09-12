use super::super::ascension_script::{
    AscensionScriptContext, CustomScriptRecord, ASCENSION_SCRIPT_NATIVE_V1,
};
use super::super::ascension_starting::{StartingServiceFailures, ASCENSION_STARTING_NATIVE_V1};
use super::super::roster::{RosterAsset, RosterCharacter, RosterServiceFailures, ROSTER_NATIVE_V1};
use super::*;
use serde_json::Value;
fn report() -> Value {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_roster_starting_bridge_audit.json")).unwrap()
}
fn context(input: &Value) -> RosterStartingBridgeContext {
    let o = &input["options"];
    let custom_scripts = input["custom"].as_array().map(|v| {
        v.iter()
            .map(|r| {
                if r.is_null() {
                    None
                } else {
                    Some(CustomScriptRecord {
                        script_id: r["script_info"].as_u64().map(|v| v as u16),
                    })
                }
            })
            .collect()
    });
    let starting = AscensionStartingContext {
        rule_version: ASCENSION_STARTING_NATIVE_V1.into(),
        services_preserve_inputs: true,
        method: StartingMethod::AllLazy,
        selection: AscensionScriptContext {
            rule_version: ASCENSION_SCRIPT_NATIVE_V1.into(),
            script_ids: vec![1, 2],
            cached_script: input["initial_cache"].as_u64().map(|v| v as u16),
            inline_scripts: serde_json::from_value(input["inline"].clone()).unwrap(),
            custom_scripts,
        },
        script_lists: serde_json::from_value(input["script_lists"].clone()).unwrap(),
        starting: serde_json::from_value(input["starting"].clone()).unwrap(),
        service_failures: StartingServiceFailures {
            to_array: o["to_array_fail"].as_u64().map(|v| v as u8),
            append: None,
        },
    };
    let assets = [
        1001, 2001, 4001, 1101, 2101, 3101, 4101, 1201, 2201, 3201, 4201,
    ]
    .into_iter()
    .map(|id| {
        (
            id,
            RosterAsset {
                character_type: 0,
                starting_alignment: 0,
                can_appear_if: None,
            },
        )
    })
    .collect();
    let current_characters = input["current"].as_array().map(|v| {
        v.iter()
            .map(|v| {
                if v == "null_character" {
                    None
                } else {
                    Some(RosterCharacter {
                        data: v.as_u64().map(|v| v as u16),
                    })
                }
            })
            .collect()
    });
    let roster = RosterContext {
        rule_version: ROSTER_NATIVE_V1.into(),
        reference_equality: true,
        services_preserve_inputs: true,
        list_add_capacity_sufficient: true,
        allocation_capacity: 256,
        method: RosterMethod::GetNotInDeckCharacters,
        assets,
        lists: BTreeMap::new(),
        rosters: [None; 4],
        input: None,
        output: None,
        current_characters,
        starting_pools: std::array::from_fn(|_| None),
        current_script: None,
        project_available: o["null_project"] != true,
        gameplay_class_initialized: o["cold"] != true,
        service_failures: RosterServiceFailures {
            append_range: o["append_fail"].as_u64().map(|v| v as u16),
            remove: o["remove_fail"].as_u64().map(|v| v as u16),
            add: None,
            clear: None,
            typed_pool: o["typed_fail"].as_u64().map(|v| v as u16),
            class_initializer: o["class_fail"] == true,
        },
    };
    RosterStartingBridgeContext {
        rule_version: ROSTER_STARTING_BRIDGE_NATIVE_V1.into(),
        temporary_profile_stable: true,
        starting,
        roster,
    }
}
fn sample(report: &Value) -> RosterStartingBridgeContext {
    let case = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| {
            let i = &c["input"];
            i["initial_cache"].is_null()
                && i["inline"] == serde_json::json!([2, 2])
                && i["custom"] == serde_json::json!([{"script_info":null},{"script_info":1}])
                && i["current"] == serde_json::json!([])
                && i["options"] == serde_json::json!({})
        })
        .unwrap();
    context(&case["input"])
}
fn error(path: &RosterStartingBridgePath) -> Value {
    let text = match &path.failure {
        None => return Value::Null,
        Some(RosterStartingBridgeFailure::Getter { failure }) => match failure {
            StartingFailure::Selection { .. } | StartingFailure::NullSelectedList { .. } => "null",
            StartingFailure::ToArrayService { .. } => "to_array",
            other => panic!("unexpected Typed failure {other:?}"),
        },
        Some(RosterStartingBridgeFailure::Roster { failure }) => match failure {
            RosterFailure::Null => "null",
            RosterFailure::Collection => "collection",
            RosterFailure::RemoveService => "remove",
            RosterFailure::TypedPoolService => "typed",
            RosterFailure::ClassInitializer => "cctor",
            other => panic!("unexpected roster failure {other:?}"),
        },
    };
    Value::String(text.into())
}
#[test]
fn all_2122_actual_native_bridge_paths_match_cache_draws_output_versions_and_failures() {
    let report = report();
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2122);
    for case in cases {
        let ctx = context(&case["input"]);
        let before = ctx.clone();
        let paths = replay_roster_starting_bridge(&ctx).unwrap_or_else(|e| panic!("{e:?}: {case}"));
        assert_eq!(ctx, before);
        let e = &case["expected"];
        let draws = e["draws"].as_array().unwrap();
        let path = paths
            .iter()
            .find(|p| {
                p.draws.len() == draws.len()
                    && p.draws.iter().zip(draws).all(|(a, b)| {
                        a.occurrence_index as u64 == b["index"].as_u64().unwrap()
                            && a.width as u64 == b["width"].as_u64().unwrap()
                            && serde_json::to_value(a.source).unwrap() == b["source"]
                    })
            })
            .unwrap_or_else(|| panic!("missing path {case}"));
        assert_eq!(error(path), e["error"], "{case}");
        assert_eq!(
            serde_json::to_value(path.cached_script).unwrap(),
            e["cache"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&path.cache_writes).unwrap(),
            e["cache_writes"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&path.typed_requests).unwrap(),
            e["typed_order"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&path.lists[&path.local_output].items).unwrap(),
            e["output_prefix"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(path.lists[&path.local_output].version).unwrap(),
            e["version"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(path.removes.iter().map(|(_, id)| id).collect::<Vec<_>>())
                .unwrap(),
            e["removes"],
            "{case}"
        );
        assert_eq!(
            path.append_attempts as u64,
            e["append_count"].as_u64().unwrap(),
            "{case}"
        );
        assert_eq!(
            path.to_array_attempts as u64,
            e["to_array_count"].as_u64().unwrap(),
            "{case}"
        );
        assert_eq!(path.returned_list.is_some(), path.failure.is_none());
        assert_eq!(
            path.probability,
            Probability {
                numerator: 1,
                denominator: draws.iter().map(|d| d["width"].as_u64().unwrap()).product()
            }
        );
    }
}
#[test]
fn mixed_faction_sources_preserve_46_weighted_paths_then_remove_board_occurrences() {
    let report = report();
    let mut ctx = sample(&report);
    let paths = replay_roster_starting_bridge(&ctx).unwrap();
    assert_eq!(paths.len(), 46);
    assert_eq!(
        paths
            .iter()
            .map(|p| 256 / p.probability.denominator)
            .sum::<u64>(),
        256
    );
    ctx.roster.current_characters = Some(vec![
        Some(RosterCharacter { data: Some(4001) }),
        Some(RosterCharacter { data: Some(2101) }),
        Some(RosterCharacter { data: None }),
    ]);
    let paths = replay_roster_starting_bridge(&ctx).unwrap();
    let p = paths
        .iter()
        .find(|p| p.draws.iter().map(|d| d.occurrence_index).eq([0, 0, 1, 1]))
        .unwrap();
    assert_eq!(p.cache_writes, vec![Some(2), None, Some(2), Some(1)]);
    assert_eq!(p.cached_script, Some(1));
    assert_eq!(p.typed_requests, vec![100, 20, 30, 10]);
    assert_eq!(
        p.lists[&p.local_output].items,
        vec![Some(4001), Some(2101), Some(3101), Some(1101)]
    );
    assert_eq!(p.lists[&p.local_output].version, 7);
}
#[test]
fn earlier_append_failure_prevents_later_selection_and_later_null_payload_preserves_prefix() {
    let mut ctx = sample(&report());
    ctx.roster.service_failures.append_range = Some(1);
    let paths = replay_roster_starting_bridge(&ctx).unwrap();
    assert_eq!(paths.len(), 4);
    assert!(paths.iter().all(|p| p.draws.len() == 2
        && p.typed_requests == vec![100]
        && p.lists[&p.local_output].items.is_empty()));
    ctx.roster.service_failures.append_range = None;
    ctx.starting.script_lists.get_mut(&1).unwrap()[1] = None;
    let paths = replay_roster_starting_bridge(&ctx).unwrap();
    let p = paths
        .iter()
        .find(|p| p.draws.iter().map(|d| d.occurrence_index).eq([0, 0, 1, 1]))
        .unwrap();
    assert_eq!(p.cached_script, Some(1));
    assert_eq!(
        p.lists[&p.local_output].items,
        vec![Some(4001), Some(4001), None]
    );
    assert_eq!(p.append_attempts, 1);
    assert_eq!(
        p.failure,
        Some(RosterStartingBridgeFailure::Getter {
            failure: StartingFailure::NullSelectedList {
                script_id: 1,
                character_type: 20
            }
        })
    );
}
#[test]
fn strict_bridge_provenance_and_support_limits_reject_atomically() {
    let original = sample(&report());
    for n in 0..7 {
        let mut ctx = original.clone();
        match n {
            0 => ctx.rule_version = "future".into(),
            1 => ctx.temporary_profile_stable = false,
            2 => ctx.starting.service_failures.append = Some(1),
            3 => ctx.roster.starting_pools[0] = Some(vec![]),
            4 => ctx.roster.reference_equality = false,
            5 => ctx.starting.starting[0] = Some(vec![Some(999)]),
            _ => ctx.starting.method = StartingMethod::AllStored,
        }
        assert_eq!(
            replay_roster_starting_bridge(&ctx),
            Err(LedgerError::InvalidContext)
        );
    }
    let mut json = serde_json::to_value(original.clone()).unwrap();
    json["unknown"] = Value::Bool(true);
    assert!(serde_json::from_value::<RosterStartingBridgeContext>(json).is_err());
    let mut ctx = original;
    ctx.starting.selection.inline_scripts = Some(vec![Some(2); 32]);
    ctx.starting.selection.custom_scripts =
        Some(vec![Some(CustomScriptRecord { script_id: None }); 32]);
    assert_eq!(
        replay_roster_starting_bridge(&ctx),
        Err(LedgerError::Capacity)
    );
}

#[test]
fn finished_path_budget_includes_already_retained_active_paths() {
    let mut ctx = sample(&report());
    ctx.starting.selection.custom_scripts = Some(vec![None]);
    let path = replay_roster_starting_bridge(&ctx).unwrap().remove(0);
    assert!(matches!(
        path.failure,
        Some(RosterStartingBridgeFailure::Getter { .. })
    ));
    let size = path
        .lists
        .values()
        .map(|list| list.items.len() + 1)
        .sum::<usize>()
        + path.draws.len()
        + path.cache_writes.len()
        + path.removes.len()
        + path.typed_requests.len();
    let mut done = Vec::new();
    let mut retained_done = 0;
    assert_eq!(
        add_finished(
            &mut done,
            path.clone(),
            &mut retained_done,
            MAX_RETAINED - size + 1
        ),
        Err(LedgerError::Capacity)
    );
    assert!(done.is_empty());
    assert_eq!(retained_done, 0);
    add_finished(&mut done, path, &mut retained_done, MAX_RETAINED - size).unwrap();
    assert_eq!(done.len(), 1);
    assert_eq!(retained_done, size);
}

#[test]
fn mixed_active_and_getter_failure_paths_reject_retained_capacity_atomically() {
    let mut ctx = sample(&report());
    // Each inline occurrence branches first to a still-active null payload,
    // then to a missing custom record. Completed failures retain input lists;
    // the earlier active paths simultaneously retain their copied Demon pool.
    ctx.starting.selection.inline_scripts = Some(vec![Some(2); 32]);
    ctx.starting.selection.custom_scripts =
        Some(vec![Some(CustomScriptRecord { script_id: None }), None]);
    for id in 1..=8 {
        ctx.roster.lists.insert(
            id,
            RosterList {
                items: vec![Some(4001); 4094],
                version: 0,
                capacity: 4094,
            },
        );
    }
    let before = ctx.clone();
    assert_eq!(
        replay_roster_starting_bridge(&ctx),
        Err(LedgerError::Capacity)
    );
    assert_eq!(ctx, before);
}
