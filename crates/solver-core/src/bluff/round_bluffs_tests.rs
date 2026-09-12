use super::*;
fn corpus() -> Value {
    serde_json::from_str(include_str!(
        "../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_round_bluffs.json"
    ))
    .unwrap()
}
fn context(report: &Value, case: &Value) -> Context {
    let i = &case["input"];
    let o = &i["options"];
    let list = |name: &str| serde_json::from_value(i[name].clone()).unwrap();
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
    let failure = o.get("fail").map(|v| ServiceFailure {
        gateway: serde_json::from_value(v[0].clone()).unwrap(),
        occurrence: v[1].as_u64().unwrap() as u16,
    });
    let string = |name: &str| o[name].as_str().map(str::to_owned);
    let predicate = (case["method"] == "predicate").then(|| Predicate {
        captured: !o["null_capture"].as_bool().unwrap_or(false),
        candidate: o["candidate"].as_u64().map(|v| v as u16),
        contains_return: o["contains_return"].as_u64().unwrap() as u32,
    });
    Context {
        version: ROUND_BLUFFS_NATIVE_V1.into(),
        metadata_initialized: true,
        stable_services: true,
        reference_membership_and_removal: true,
        distinct_lists_sufficient_capacity: true,
        uniform_occurrence_support: true,
        gameplay_initialized: !o["cold"].as_bool().unwrap_or(false),
        gameplay_present: !o["null_game"].as_bool().unwrap_or(false),
        pool_present: !o["null_pool"].as_bool().unwrap_or(false),
        all: list("all"),
        script: list("script"),
        fallback: list("fallback"),
        initial_pool: vec![Some(7)],
        initial_version: 3,
        assets,
        null_getter: string("null_getter"),
        null_filter: string("null_filter"),
        replace_after_getter: string("replace_after_getter"),
        failure,
        predicate,
    }
}
#[test]
fn all_108_native_caller_and_predicate_prefixes() {
    let report = corpus();
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 108);
    for (n, case) in cases.iter().enumerate() {
        let c = context(&report, case);
        let choices: Vec<usize> = case["input"]["options"]
            .get("choices")
            .map(|v| serde_json::from_value(v.clone()).unwrap())
            .unwrap_or_default();
        let trace = replay_trace(&c, &choices).unwrap();
        assert_eq!(
            json!(Run::new(&c).trace.state),
            case["initial"],
            "initial {n}"
        );
        assert_eq!(json!(trace.state), case["final"], "final {n}");
        assert_eq!(json!(trace.draws), case["draws"], "draws {n}");
        assert_eq!(json!(trace.error), case["error"], "error {n}");
        assert_eq!(json!(trace.return_al), case["return_al"], "return {n}");
        let expected: Vec<Value> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                let mut e = e.clone();
                let idx = e
                    .as_object_mut()
                    .unwrap()
                    .remove("snapshot_index")
                    .unwrap()
                    .as_u64()
                    .unwrap() as usize;
                e["snapshot"] = report["snapshot_table"][idx].clone();
                e
            })
            .collect();
        assert_eq!(trace.events.len(), expected.len(), "event count {n}");
        for (index, (actual, expected)) in trace.events.iter().zip(&expected).enumerate() {
            assert_eq!(actual, expected, "case {n}, first differing event {index}");
        }
    }
}
#[test]
fn eighteen_initial_and_nine_fallback_paths_match_native() {
    let report = corpus();
    for (start, count) in [(0, 18), (18, 3), (21, 3), (24, 3)] {
        let c = context(&report, &report["cases"][start]);
        let paths = replay_weighted(&c).unwrap();
        assert_eq!(paths.len(), count);
        for n in start..start + count {
            let case = &report["cases"][n];
            let p = paths
                .iter()
                .find(|p| json!(p.trace.draws) == case["draws"])
                .unwrap();
            assert_eq!(json!(p.trace.state), case["final"]);
            assert_eq!(json!(p.probability), case["probability"]);
        }
    }
}
#[test]
fn pre_draw_failures_have_unit_mass_and_fallback_is_single() {
    let report = corpus();
    let mut n = 0;
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
        assert_eq!(json!(paths[0].trace.state), case["final"]);
        assert_eq!(json!(paths[0].trace.error), case["error"]);
        n += 1;
    }
    assert!(n > 30);
    let c = context(&report, &report["cases"][18]);
    let paths = replay_weighted(&c).unwrap();
    assert!(paths
        .iter()
        .all(|p| p.trace.state.lists["unique"].items.len() == 1
            && p.trace.draws.len() == 1
            && p.trace.state.lists["fallback_villagers"].items == vec![Some(1), Some(1), Some(7)]));
}
#[test]
fn sampling_failure_mass_add_order_and_version_wrap() {
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
    assert!(p[0].trace.state.lists["unique"].items.is_empty());
    c.failure = Some(ServiceFailure {
        gateway: Gateway::Add,
        occurrence: 1,
    });
    let p = replay_weighted(&c).unwrap();
    assert_eq!(p.len(), 3);
    assert!(p
        .iter()
        .all(|p| p.probability.denominator == 3 && p.trace.state.lists["unique"].items.is_empty()));
    c.failure = Some(ServiceFailure {
        gateway: Gateway::Remove,
        occurrence: 1,
    });
    let p = replay_weighted(&c).unwrap();
    assert_eq!(p.len(), 3);
    assert!(p
        .iter()
        .all(|p| p.trace.state.lists["unique"].items.len() == 1
            && p.trace.state.lists["villagers"].items.len() == 3));
    c.failure = None;
    c.initial_version = u32::MAX;
    assert_eq!(
        replay_trace(&c, &[]).unwrap().state.lists["unique"].version,
        4
    );
}
#[test]
fn invalid_context_and_capacity_are_atomic() {
    let report = corpus();
    let c = context(&report, &report["cases"][0]);
    for which in 0..7 {
        let mut bad = c.clone();
        match which {
            0 => bad.metadata_initialized = false,
            1 => bad.stable_services = false,
            2 => bad.reference_membership_and_removal = false,
            3 => bad.distinct_lists_sufficient_capacity = false,
            4 => bad.uniform_occurrence_support = false,
            5 => bad.null_filter = Some("unknown".into()),
            _ => bad.version = "unknown".into(),
        };
        let before = bad.clone();
        assert_eq!(replay_trace(&bad, &[]), Err(LedgerError::InvalidContext));
        assert_eq!(replay_weighted(&bad), Err(LedgerError::InvalidContext));
        assert_eq!(bad, before);
    }
    let mut big = c.clone();
    big.all = vec![Some(0); 33];
    assert_eq!(replay_trace(&big, &[]), Err(LedgerError::Capacity));
    let mut branches = c;
    branches.all = vec![Some(0); 8];
    let before = branches.clone();
    assert_eq!(replay_weighted(&branches), Err(LedgerError::Capacity));
    assert_eq!(branches, before);
}

#[test]
fn null_fallback_source_preserves_native_gateway_diagnostic_names() {
    let report = corpus();
    let mut c = context(&report, &report["cases"][18]);
    for (getter, filter, destination) in [
        (Some("fallback"), None, "bluffable"),
        (None, Some("fallback_good"), "villagers"),
    ] {
        c.null_getter = getter.map(str::to_owned);
        c.null_filter = filter.map(str::to_owned);
        let trace = replay_trace(&c, &[]).unwrap();
        assert_eq!(trace.error.as_deref(), Some("null"));
        let last = trace.events.last().unwrap();
        assert_eq!(last["kind"], "filter");
        assert_eq!(last["source"], Value::Null);
        assert_eq!(last["destination"], destination);
        assert!(trace.state.lists["unique"].items.is_empty());
    }
}
