use super::*;
fn corpus() -> Value {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_unique_source_composition.json")).unwrap()
}
fn context(report: &Value, case: &Value) -> Context {
    Context {
        version: UNIQUE_SOURCE_NATIVE_V1.into(),
        metadata_initialized: true,
        distinct_collections_sufficient_capacity: true,
        stable_services_reference_equality: true,
        deferred_remove_all_commit: true,
        uniform_occurrence_support: true,
        initial_pool: vec![Some(7)],
        initial_version: 3,
        assets: report["asset_fields"]
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
            .collect(),
        input: serde_json::from_value(case["input"].clone()).unwrap(),
    }
}
#[test]
fn every_native_source_filter_and_predicate_trace_matches() {
    let report = corpus();
    let cases = report["cases"].as_array().unwrap();
    assert!(cases.len() >= 468);
    assert_eq!(
        cases.len(),
        report["cases_passed"].as_u64().unwrap() as usize
    );
    for (index, case) in cases.iter().enumerate() {
        let c = context(&report, case);
        let actual = replay_trace(&c, &c.input.options.choices).unwrap();
        assert_eq!(
            json!(Run::new(&c).trace.state),
            case["initial"],
            "initial {index}"
        );
        assert_eq!(json!(actual.error), case["error"], "error {index}");
        assert_eq!(json!(actual.state), case["final"], "final {index}");
        assert_eq!(json!(actual.entries), case["entries"], "entries {index}");
        assert_eq!(json!(actual.typed), case["typed"], "typed {index}");
        assert_eq!(json!(actual.draws), case["draws"], "draws {index}");
        assert_eq!(
            json!(actual.callbacks),
            case["callbacks"],
            "callbacks {index}"
        );
        assert_eq!(
            json!(actual.predicate_results),
            case["predicate_results"],
            "predicates {index}"
        );
        let events = case["events"].as_array().unwrap();
        assert_eq!(actual.events.len(), events.len(), "events count {index}");
        for (ordinal, (actual, expected)) in actual.events.iter().zip(events).enumerate() {
            let mut expected = expected.clone();
            let slot = expected
                .as_object_mut()
                .unwrap()
                .remove("snapshot_index")
                .unwrap()
                .as_u64()
                .unwrap() as usize;
            expected["snapshot"] = report["snapshot_table"][slot].clone();
            assert_eq!(*actual, expected, "event {index}:{ordinal}");
        }
    }
}
#[test]
fn weighted_repeated_sources_and_lazy_cache_share_one_chronology() {
    let report = corpus();
    let cases = report["cases"].as_array().unwrap();
    let mut families = BTreeMap::<String, Vec<&Value>>::new();
    families.insert("direct".into(), cases.iter().take(18).collect());
    families.insert("fallback".into(), cases.iter().skip(18).take(4).collect());
    for case in cases {
        if let Some(family) = case["support_family"].as_str() {
            families.entry(family.into()).or_default().push(case);
        }
    }
    assert_eq!(families.len(), 4);
    for (family, native) in families {
        let c = context(&report, native[0]);
        let paths = replay_weighted(&c).unwrap();
        assert_eq!(paths.len(), native.len(), "{family}");
        let mut mass = 0.0;
        for path in paths {
            let case = native
                .iter()
                .find(|v| v["draws"] == json!(path.trace.draws))
                .unwrap();
            assert_eq!(json!(path.trace.state), case["final"]);
            let parts = case["probability"]
                .as_str()
                .unwrap()
                .split('/')
                .map(|p| p.parse::<u64>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                path.probability,
                Probability {
                    numerator: parts[0],
                    denominator: *parts.get(1).unwrap_or(&1)
                }
            );
            mass += path.probability.numerator as f64 / path.probability.denominator as f64;
        }
        assert!((mass - 1.0).abs() < 1e-12);
    }
}
#[test]
fn preparation_failures_retain_unconditional_mass() {
    let report = corpus();
    let mut count = 0;
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
        count += 1;
    }
    assert!(count > 200);
}
#[test]
fn consumed_null_inline_draws_are_retained_before_reselection() {
    let report = corpus();
    let native = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| {
            c["support_family"] == "inline_[None, 1]"
                && c["draws"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|d| d["source"] == "inline")
                    .count()
                    == 4
        })
        .unwrap();
    let c = context(&report, native);
    let trace = replay_trace(&c, &c.input.options.choices).unwrap();
    assert_eq!(trace.typed.len(), 4);
    assert_eq!(
        trace
            .draws
            .iter()
            .filter(|d| d["source"] == "inline")
            .count(),
        4
    );
    assert_eq!(trace.error, None);
}
#[test]
fn unsupported_contracts_and_expansion_are_rejected() {
    let report = corpus();
    let c = context(&report, &report["cases"][0]);
    let mut bad = c.clone();
    bad.deferred_remove_all_commit = false;
    assert_eq!(replay_trace(&bad, &[]), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.input.options.inline = vec![Some(2)];
    assert_eq!(replay_trace(&bad, &[]), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.input.options.fail = Some((Gateway::Contains, 0));
    assert_eq!(replay_trace(&bad, &[]), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.input.starting[0] = Some(vec![Some(0); MAX_ITEMS + 1]);
    assert_eq!(replay_trace(&bad, &[]), Err(LedgerError::Capacity));
    let mut bad = c;
    bad.input.starting = [
        Some((0..12).map(|_| Some(0)).collect()),
        Some(vec![]),
        Some(vec![]),
        Some(vec![]),
    ];
    assert_eq!(replay_weighted(&bad), Err(LedgerError::Capacity));
}
