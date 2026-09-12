use super::*;
use serde_json::Value;

fn report() -> Value {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_gameplay_roster_composition_audit.json")).unwrap()
}
fn items(value: &Value) -> Occurrences {
    if value.is_null() {
        Vec::new()
    } else {
        serde_json::from_value(value.clone()).unwrap()
    }
}
fn context(case: &Value) -> RosterContext {
    let i = &case["input"];
    let o = &i["options"];
    let mut lists = BTreeMap::new();
    let rosters = std::array::from_fn(|n| {
        lists.insert(
            10 + n as u16,
            RosterList {
                items: items(&i["rosters"][n]),
                version: 3,
                capacity: 256,
            },
        );
        (!i["rosters"][n].is_null()).then_some(10 + n as u16)
    });
    lists.insert(
        30,
        RosterList {
            items: items(&i["in_data"]),
            version: 3,
            capacity: 256,
        },
    );
    lists.insert(
        31,
        RosterList {
            items: items(&i["out_data"]),
            version: 3,
            capacity: 256,
        },
    );
    let kinds = [10, 20, 30, 100, 20, 20, 123, 10, 100];
    let aligns = [10, 10, 20, 20, 10, 10, 77, 20, 10];
    let mut assets = BTreeMap::new();
    for n in 1..=9u16 {
        let dep = i["deps"].get(n.to_string());
        let present = dep.is_none_or(|v| !v.is_null());
        lists.insert(
            100 + n,
            RosterList {
                items: dep.map(items).unwrap_or_default(),
                version: 3,
                capacity: 256,
            },
        );
        assets.insert(
            n,
            RosterAsset {
                character_type: kinds[n as usize - 1],
                starting_alignment: aligns[n as usize - 1],
                can_appear_if: present.then_some(100 + n),
            },
        );
    }
    let mut input = Some(30);
    if let Some(n) = o["input_roster"].as_u64() {
        input = rosters[n as usize];
    }
    let mut output = if o["output_is_input"] == true {
        input
    } else {
        Some(31)
    };
    if o["null_input"] == true {
        input = None;
    }
    if o["null_output"] == true {
        output = None;
    }
    let current = if o["null_current"] == true {
        None
    } else {
        Some(
            i["current"]
                .as_array()
                .map(|v| {
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
                })
                .unwrap_or_default(),
        )
    };
    let method = match case["method"].as_str().unwrap() {
        "GetAllCurrentCharacters" => RosterMethod::GetAllCurrentCharacters,
        "CleanupCharactersList" => RosterMethod::CleanupCharactersList,
        "FilterIfCanAppearCharacters" => RosterMethod::FilterIfCanAppearCharacters {
            character_type: i["kind"].as_i64().unwrap() as i32,
        },
        "GetNotInPlayCharacters" => RosterMethod::GetNotInPlayCharacters,
        "GetNotInDeckCharacters" => RosterMethod::GetNotInDeckCharacters,
        "GetScriptCharactersOfAlignment" => RosterMethod::GetScriptCharactersOfAlignment {
            alignment: i["alignment"].as_i64().unwrap() as i32,
        },
        "UpdateCurrentCharacters" => RosterMethod::UpdateCurrentCharacters,
        x => panic!("unknown method {x}"),
    };
    RosterContext {
        rule_version: ROSTER_NATIVE_V1.into(),
        reference_equality: true,
        services_preserve_inputs: true,
        list_add_capacity_sufficient: true,
        allocation_capacity: 256,
        method,
        assets,
        lists,
        rosters,
        input,
        output,
        current_characters: current,
        starting_pools: std::array::from_fn(|n| {
            let v = &i["pools"][TYPES[n].to_string()];
            if v.is_null() {
                None
            } else {
                Some(items(v))
            }
        }),
        current_script: if o["null_counts"] == true {
            None
        } else {
            Some(serde_json::from_value(i["counts"].clone()).unwrap())
        },
        project_available: o["null_project"] != true,
        gameplay_class_initialized: o["cold"] != true,
        service_failures: RosterServiceFailures {
            append_range: o["append_fail"].as_u64().map(|v| v as u16),
            remove: o["remove_fail"].as_u64().map(|v| v as u16),
            add: o["add_fail"].as_u64().map(|v| v as u16),
            clear: o["clear_fail"].as_u64().map(|v| v as u16),
            typed_pool: o["typed_fail"].as_u64().map(|v| v as u16),
            class_initializer: o["class_fail"] == true,
        },
    }
}
fn name(path: &RosterPath, id: ListId) -> String {
    if let Some(n) = path.allocations.iter().position(|v| *v == id) {
        format!("allocated{}", n + 1)
    } else {
        match id {
            10..=13 => format!("roster{}", id - 10),
            30 => "input".into(),
            31 => "output".into(),
            101..=109 => format!("deps{}", id - 100),
            _ => panic!("unknown list {id}"),
        }
    }
}
fn error(path: &RosterPath) -> Value {
    match &path.failure {
        None => Value::Null,
        Some(f) => Value::String(
            match f {
                RosterFailure::Null => "null",
                RosterFailure::Collection => "collection",
                RosterFailure::RemoveService => "remove",
                RosterFailure::AddService => "add",
                RosterFailure::ClearService => "clear",
                RosterFailure::TypedPoolService => "typed",
                RosterFailure::ClassInitializer => "cctor",
                RosterFailure::EnumeratorVersion => "enumerator_version",
            }
            .into(),
        ),
    }
}

#[test]
fn matches_all_113_native_cases_with_versions_aliases_and_partial_lists() {
    let report = report();
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 113);
    for case in cases {
        let ctx = context(case);
        let original = ctx.clone();
        let paths = replay_roster(&ctx).unwrap_or_else(|e| panic!("{e:?} {case}"));
        assert_eq!(ctx, original);
        let draws = case["draws"].as_array().unwrap();
        let path = paths
            .iter()
            .find(|p| {
                p.draws.len() == draws.len()
                    && p.draws.iter().zip(draws).all(|(a, b)| {
                        a.width as u64 == b[0].as_u64().unwrap()
                            && a.occurrence_index as u64 == b[1].as_u64().unwrap()
                    })
            })
            .unwrap_or_else(|| panic!("missing native trace {case}"));
        assert_eq!(error(path), case["error"], "{case}");
        assert_eq!(
            serde_json::to_value(path.returned_list.map(|id| &path.lists[&id].items)).unwrap(),
            case["returned"],
            "{case}"
        );
        let rosters: Vec<_> = ctx
            .rosters
            .iter()
            .map(|id| id.map(|id| &path.lists[&id].items))
            .collect();
        assert_eq!(
            serde_json::to_value(rosters).unwrap(),
            case["rosters"],
            "{case}"
        );
        let versions: Vec<_> = ctx
            .rosters
            .iter()
            .map(|id| id.map(|id| path.lists[&id].version))
            .collect();
        assert_eq!(
            serde_json::to_value(versions).unwrap(),
            case["versions"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&path.typed_pool_requests).unwrap(),
            case["typed_order"],
            "{case}"
        );
        for (field, trace) in [("adds", &path.adds), ("removes", &path.removes)] {
            let trace: Vec<_> = trace.iter().map(|(id, a)| (name(path, *id), a)).collect();
            assert_eq!(serde_json::to_value(trace).unwrap(), case[field], "{case}");
        }
        assert_eq!(
            serde_json::to_value(
                path.clears
                    .iter()
                    .map(|id| name(path, *id))
                    .collect::<Vec<_>>()
            )
            .unwrap(),
            case["clears"],
            "{case}"
        );
        let mut partial = BTreeMap::new();
        partial.insert("output".to_string(), &path.lists[&31].items);
        for id in &path.allocations {
            partial.insert(name(path, *id), &path.lists[id].items);
        }
        assert_eq!(
            serde_json::to_value(partial).unwrap(),
            case["partial_lists"],
            "{case}"
        );
        assert_eq!(
            path.probability,
            Probability {
                numerator: 1,
                denominator: draws.iter().map(|d| d[0].as_u64().unwrap()).product()
            }
        );
    }
}

fn basic(method: RosterMethod) -> RosterContext {
    let report = report();
    let mut ctx = context(&report["cases"][0]);
    ctx.method = method;
    ctx
}
#[test]
fn cleanup_preserves_six_occurrence_paths_and_failed_mass() {
    let mut ctx = basic(RosterMethod::CleanupCharactersList);
    for id in ctx.rosters.iter().flatten() {
        ctx.lists.get_mut(id).unwrap().items.clear();
    }
    ctx.starting_pools = [
        Some(vec![]),
        Some(vec![Some(2), Some(5), Some(2)]),
        Some(vec![]),
        Some(vec![]),
    ];
    ctx.current_script = Some([0, 0, 3, 0, 0, 0, 0, 0]);
    let paths = replay_roster(&ctx).unwrap();
    assert_eq!(paths.len(), 6);
    for p in &paths {
        assert_eq!(
            p.probability,
            Probability {
                numerator: 1,
                denominator: 6
            }
        );
        assert_eq!(
            p.draws.iter().map(|d| d.width).collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
        assert!(p.draws.iter().all(|d| d.character_type == 20));
    }
    let p = paths
        .iter()
        .find(|p| p.draws.iter().map(|d| d.occurrence_index).eq([2, 1, 0]))
        .unwrap();
    assert_eq!(p.lists[&11].items, vec![Some(2), Some(2), Some(5)]);
    ctx.service_failures.add = Some(2);
    let paths = replay_roster(&ctx).unwrap();
    assert_eq!(paths.len(), 6);
    assert!(paths
        .iter()
        .all(|p| p.failure == Some(RosterFailure::AddService)
            && p.probability.denominator == 6
            && p.lists[&11].items.len() == 1));
}
#[test]
fn filter_alias_invalidates_enumerator_and_does_not_filter_unseen_candidates() {
    let mut ctx = basic(RosterMethod::FilterIfCanAppearCharacters { character_type: 20 });
    ctx.lists.get_mut(&30).unwrap().items = vec![Some(2), Some(2)];
    ctx.output = Some(30);
    ctx.lists.get_mut(&102).unwrap().items = vec![Some(8)];
    let p = replay_roster(&ctx).unwrap().remove(0);
    assert_eq!(p.failure, Some(RosterFailure::EnumeratorVersion));
    assert_eq!(p.lists[&30].items, vec![Some(2)]);
    assert_eq!(p.lists[&11].items, Vec::<Option<u16>>::new());
    ctx.lists.get_mut(&30).unwrap().items = vec![Some(1)];
    ctx.output = Some(31);
    ctx.lists.get_mut(&31).unwrap().items = vec![Some(6)];
    ctx.lists.get_mut(&106).unwrap().items = vec![Some(8)];
    let p = replay_roster(&ctx).unwrap().remove(0);
    assert_eq!(p.lists[&31].items, vec![Some(6)]);
    assert!(p.removes.is_empty());
}
#[test]
fn update_observes_destination_aliases_and_clear_failure_size_before_service() {
    let mut ctx = basic(RosterMethod::UpdateCurrentCharacters);
    ctx.input = ctx.rosters[0];
    let p = replay_roster(&ctx).unwrap().remove(0);
    assert!(p.adds.is_empty());
    assert!(ctx
        .rosters
        .iter()
        .flatten()
        .all(|id| p.lists[id].items.is_empty() && p.lists[id].version == 4));
    ctx.input = Some(30);
    ctx.service_failures.clear = Some(2);
    let p = replay_roster(&ctx).unwrap().remove(0);
    assert_eq!(p.failure, Some(RosterFailure::ClearService));
    assert!(p.lists[&13].items.is_empty() && p.lists[&12].items.is_empty());
    assert_eq!(p.lists[&11].items, vec![Some(2)]);
    assert_eq!(p.lists[&12].version, 4);
}
#[test]
fn deck_acquisition_order_is_separate_from_cleanup_order() {
    let mut ctx = basic(RosterMethod::GetNotInDeckCharacters);
    ctx.rosters = [None; 4];
    let p = replay_roster(&ctx).unwrap().remove(0);
    assert_eq!(p.typed_pool_requests, vec![100, 20, 30, 10]);
    assert!(p.draws.is_empty());
    assert_eq!(
        p.lists[&p.returned_list.unwrap()].items,
        vec![
            Some(4),
            Some(9),
            Some(2),
            Some(5),
            Some(6),
            Some(3),
            Some(1),
            Some(8)
        ]
    );
}
#[test]
fn strict_provenance_unknown_identity_and_capacity_are_rejected_atomically() {
    let original = basic(RosterMethod::GetAllCurrentCharacters);
    for n in 0..7 {
        let mut ctx = original.clone();
        match n {
            0 => ctx.rule_version = "future".into(),
            1 => ctx.reference_equality = false,
            2 => ctx.services_preserve_inputs = false,
            3 => ctx.list_add_capacity_sufficient = false,
            4 => ctx.rosters[0] = Some(999),
            5 => ctx.lists.get_mut(&10).unwrap().items.push(Some(999)),
            _ => ctx.service_failures.remove = Some(0),
        }
        assert_eq!(replay_roster(&ctx), Err(LedgerError::InvalidContext));
    }
    let mut json = serde_json::to_value(original.clone()).unwrap();
    json["unknown"] = Value::Bool(true);
    assert!(serde_json::from_value::<RosterContext>(json).is_err());
    let mut ctx = original.clone();
    ctx.allocation_capacity = 1;
    assert_eq!(replay_roster(&ctx), Err(LedgerError::Capacity));
    assert_eq!(original, basic(RosterMethod::GetAllCurrentCharacters));
    ctx = basic(RosterMethod::CleanupCharactersList);
    for id in ctx.rosters.iter().flatten() {
        ctx.lists.get_mut(id).unwrap().items.clear();
    }
    ctx.starting_pools[1] = Some(vec![Some(2); 12]);
    ctx.current_script = Some([0, 0, 12, 0, 0, 0, 0, 0]);
    assert_eq!(replay_roster(&ctx), Err(LedgerError::Capacity));
}

#[test]
fn input_budget_counts_each_list_once_before_native_null_failure() {
    let data = report();
    let mut c = context(&data["cases"][0]);
    c.method = RosterMethod::FilterIfCanAppearCharacters { character_type: 10 };
    c.input = None;
    c.output = None;
    c.rosters = [None; 4];
    for asset in c.assets.values_mut() {
        asset.can_appear_if = None;
    }
    c.lists = (0..257u16)
        .map(|id| {
            (
                id,
                RosterList {
                    items: vec![Some(1); 2048],
                    version: 0,
                    capacity: 2048,
                },
            )
        })
        .collect();
    c.starting_pools = [None, None, None, None];
    // 526,336 input entries fit the 1,048,576 retained-entry bound. Counting
    // each twice incorrectly masks the reached native null-input failure.
    let paths = replay_roster(&c).unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].failure, Some(RosterFailure::Null));
    assert_eq!(paths[0].lists.len(), 257);
}
