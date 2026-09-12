use super::*;
fn corpus() -> Value {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_round_candidate_composition.json")).unwrap()
}
fn context(report: &Value, case: &Value) -> Context {
    let options = &case["input"]["options"];
    let assets = report["asset_fields"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.parse().unwrap(),
                Asset {
                    real_type: v[0].as_i64().unwrap() as i32,
                    alignment: v[1].as_i64().unwrap() as i32,
                    bluffable: v[2].as_u64().unwrap() as u8,
                },
            )
        })
        .collect();
    let failure = options.get("fail").map(|v| ServiceFailure {
        gateway: serde_json::from_value(v[0].clone()).unwrap(),
        occurrence: v[1].as_u64().unwrap() as u16,
    });
    Context {
        version: ROUND_CANDIDATE_NATIVE_V1.into(),
        metadata_initialized: true,
        stable_services: true,
        distinct_lists_sufficient_capacity: true,
        reference_removal: true,
        uniform_occurrence_support: true,
        gameplay_initialized: !options["cold"].as_bool().unwrap_or(false),
        gameplay_present: !options["null_game"].as_bool().unwrap_or(false),
        pool_present: !options["null_pool"].as_bool().unwrap_or(false),
        rosters: serde_json::from_value(case["input"]["rosters"].clone()).unwrap(),
        assets,
        initial_pool: vec![Some(7)],
        initial_version: 3,
        failure,
    }
}
#[test]
fn all_124_native_full_traces_and_snapshots() {
    let report = corpus();
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 124);
    for (index, case) in cases.iter().enumerate() {
        let c = context(&report, case);
        let choices: Vec<usize> = case["input"]["options"]
            .get("choices")
            .map(|v| serde_json::from_value(v.clone()).unwrap())
            .unwrap_or_default();
        let actual = replay_trace(&c, &choices).unwrap();
        assert_eq!(
            json!(Run::new(&c).trace.lists),
            case["initial"],
            "initial case {index}"
        );
        assert_eq!(json!(actual.lists), case["final"], "final case {index}");
        assert_eq!(json!(actual.error), case["error"], "error case {index}");
        assert_eq!(
            json!(actual.entries),
            case["entries"],
            "entries case {index}"
        );
        assert_eq!(json!(actual.draws), case["draws"], "draws case {index}");
        let expected: Vec<Value> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                let mut e = e.clone();
                let slot = e
                    .as_object_mut()
                    .unwrap()
                    .remove("snapshot_index")
                    .unwrap()
                    .as_u64()
                    .unwrap() as usize;
                e["snapshot"] = report["snapshot_table"][slot].clone();
                e
            })
            .collect();
        assert_eq!(actual.events, expected, "events case {index}");
    }
}
#[test]
fn weighted_composition_matches_all_eighteen_native_paths() {
    let report = corpus();
    let c = context(&report, &report["cases"][0]);
    let paths = replay_weighted(&c).unwrap();
    assert_eq!(paths.len(), 18);
    for native in report["cases"].as_array().unwrap().iter().take(18) {
        let path = paths
            .iter()
            .find(|p| json!(p.trace.draws) == native["draws"])
            .unwrap();
        assert_eq!(json!(path.trace.lists), native["final"]);
        assert_eq!(json!(path.probability), native["probability"]);
    }
    assert!(paths
        .iter()
        .all(|p| p.probability.numerator == 1 && p.probability.denominator == 18));
}
#[test]
fn every_native_preparation_failure_retains_unit_mass() {
    let report = corpus();
    let mut checked = 0;
    for case in report["cases"].as_array().unwrap() {
        if case["error"].is_null() || !case["draws"].as_array().unwrap().is_empty() {
            continue;
        }
        let paths = replay_weighted(&context(&report, case)).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(
            paths[0].probability,
            Probability {
                numerator: 1,
                denominator: 1
            }
        );
        assert_eq!(json!(paths[0].trace.lists), case["final"]);
        assert_eq!(json!(paths[0].trace.error), case["error"]);
        checked += 1;
    }
    assert!(checked > 70);
}
#[test]
fn sampling_failure_mass_and_native_partial_write_order() {
    let report = corpus();
    let mut c = context(&report, &report["cases"][0]);
    c.failure = Some(ServiceFailure {
        gateway: Gateway::Rng,
        occurrence: 1,
    });
    let p = replay_weighted(&c).unwrap();
    assert_eq!(p.len(), 1);
    assert_eq!(
        p[0].probability,
        Probability {
            numerator: 1,
            denominator: 1
        }
    );
    assert_eq!(p[0].trace.error.as_deref(), Some("rng"));
    assert!(p[0].trace.lists["duplicates"].items.is_empty());
    c.failure = Some(ServiceFailure {
        gateway: Gateway::VillagerStore,
        occurrence: 1,
    });
    let p = replay_weighted(&c).unwrap();
    assert_eq!(p.len(), 3);
    for p in p {
        assert_eq!(
            p.probability,
            Probability {
                numerator: 1,
                denominator: 3
            }
        );
        assert_eq!(p.trace.lists["duplicates"].items.len(), 1);
        assert_eq!(p.trace.lists["villagers"].items.len(), 3);
        assert_eq!(p.trace.error.as_deref(), Some("villager_store"));
    }
    c.failure = Some(ServiceFailure {
        gateway: Gateway::OutcastAdd,
        occurrence: 1,
    });
    let p = replay_weighted(&c).unwrap();
    assert_eq!(p.len(), 18);
    assert!(p
        .iter()
        .all(|p| p.trace.lists["duplicates"].items.len() == 3
            && p.trace.lists["outcasts"].items.len() == 3
            && p.probability.denominator == 18));
}
#[test]
fn invalid_context_is_atomic_and_versions_wrap() {
    let report = corpus();
    let c = context(&report, &report["cases"][0]);
    for which in 0..7 {
        let mut bad = c.clone();
        match which {
            0 => bad.metadata_initialized = false,
            1 => bad.stable_services = false,
            2 => bad.distinct_lists_sufficient_capacity = false,
            3 => bad.reference_removal = false,
            4 => bad.uniform_occurrence_support = false,
            5 => bad.version = "unknown".into(),
            _ => {
                bad.failure = Some(ServiceFailure {
                    gateway: Gateway::Allocate,
                    occurrence: 0,
                })
            }
        };
        let before = bad.clone();
        assert_eq!(replay_trace(&bad, &[]), Err(LedgerError::InvalidContext));
        assert_eq!(replay_weighted(&bad), Err(LedgerError::InvalidContext));
        assert_eq!(bad, before);
    }
    let mut bad = c.clone();
    bad.rosters[0] = Some(vec![Some(999)]);
    assert_eq!(replay_trace(&bad, &[]), Err(LedgerError::InvalidContext));
    let mut big = c.clone();
    big.rosters[0] = Some(vec![Some(0); 33]);
    assert_eq!(replay_trace(&big, &[]), Err(LedgerError::Capacity));
    let mut wrap = c;
    wrap.initial_version = u32::MAX;
    let p = replay_trace(&wrap, &[]).unwrap();
    assert_eq!(p.lists["duplicates"].version, 4);
}

#[test]
fn weighted_bounds_reject_without_collapsing_duplicate_occurrences() {
    let report = corpus();
    let mut c = context(&report, &report["cases"][0]);
    c.rosters = [
        Some(vec![Some(0); 7]),
        Some(vec![Some(2); 3]),
        Some(vec![]),
        Some(vec![]),
    ];
    let before = c.clone();
    assert_eq!(replay_weighted(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    c.rosters = [
        Some(vec![Some(0); 7]),
        Some(vec![]),
        Some(vec![]),
        Some(vec![]),
    ];
    assert_eq!(replay_weighted(&c), Err(LedgerError::Capacity));
}
