use super::*;
use serde_json::Value;
fn report() -> Value {
    serde_json::from_str(include_str!(
        "../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_round_duplicates.json"
    ))
    .unwrap()
}
fn context(v: &Value) -> RoundDuplicateContext {
    let o = &v["options"];
    let capacity = o["capacity"].as_u64().unwrap_or(16) as u16;
    RoundDuplicateContext {
        rule_version: ROUND_DUPLICATES_NATIVE_V1.into(),
        stable_services: true,
        reference_removal: true,
        uniform_occurrence_support: true,
        gameplay_initialized: !o["cold"].as_bool().unwrap_or(false),
        gameplay_present: !o["null_game"].as_bool().unwrap_or(false),
        pool_identity: (!o["null_pool"].as_bool().unwrap_or(false)).then_some(1),
        initial_pool: if capacity == 0 { vec![] } else { vec![Some(7)] },
        initial_version: 3,
        capacity,
        growth_slots: 16,
        villagers: (o["null_type"] != 10).then(|| CandidateList {
            identity: 2,
            items: serde_json::from_value(v["villagers"].clone()).unwrap(),
        }),
        outcasts: (o["null_type"] != 20).then(|| CandidateList {
            identity: 3,
            items: serde_json::from_value(v["outcasts"].clone()).unwrap(),
        }),
        failure: o["fail"].as_array().map(|f| ServiceFailure {
            gateway: serde_json::from_value(f[0].clone()).unwrap(),
            occurrence: f[1].as_u64().unwrap() as u16,
        }),
    }
}
#[test]
fn matches_all_47_native_duplicate_cases() {
    let r = report();
    let mut count = 0;
    for v in r["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["entry"] == "0x36d720")
    {
        let c = context(v);
        let before = c.clone();
        let paths = replay(&c).unwrap();
        assert_eq!(c, before);
        let draws = v["draws"].as_array().unwrap();
        let p = paths
            .iter()
            .find(|p| {
                p.draws.len() == draws.len()
                    && p.draws.iter().zip(draws).all(|(a, b)| {
                        a.width == b["width"].as_u64().unwrap() as usize
                            && a.index
                                .is_none_or(|i| i == b["index"].as_u64().unwrap() as usize)
                    })
            })
            .unwrap();
        assert_eq!(serde_json::to_value(&p.output).unwrap(), v["output"]);
        // A null filter result leaves the fixture's inaccessible candidate
        // object intact; this replay exposes only returned candidate lists.
        if c.villagers.is_some() {
            assert_eq!(
                serde_json::to_value(&p.remaining_villagers).unwrap(),
                v["remaining_villagers"]
            );
        }
        if c.outcasts.is_some() {
            assert_eq!(
                serde_json::to_value(&p.remaining_outcasts).unwrap(),
                v["remaining_outcasts"]
            );
        }
        assert_eq!(p.version as u64, v["version"].as_u64().unwrap());
        let error = match &p.error {
            None => Value::Null,
            Some(Failure::Null) => "null".into(),
            Some(Failure::Bounds) => "bounds".into(),
            Some(Failure::Service(g)) => serde_json::to_value(g).unwrap(),
        };
        assert_eq!(error, v["error"]);
        let mut events = v["events"].clone();
        // Native fixtures supply index zero even for a failed/empty range.
        // It is not a successfully selected occurrence in the replay.
        for (actual, expected) in p.events.iter().zip(events.as_array_mut().unwrap()) {
            if actual.event == Gateway::Rng && actual.index.is_none() {
                expected.as_object_mut().unwrap().remove("index");
            }
        }
        assert_eq!(serde_json::to_value(&p.events).unwrap(), events, "{v}");
        count += 1;
    }
    assert_eq!(count, 47);
}
#[test]
fn weighted_occurrences_preserve_first_equal_removal() {
    let r = report();
    let v = r["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["options"]["choices"].is_array())
        .unwrap();
    let c = context(v);
    let paths = replay(&c).unwrap();
    assert_eq!(paths.len(), 18);
    for p in &paths {
        assert_eq!(
            p.probability,
            Probability {
                numerator: 1,
                denominator: 18
            }
        );
        assert!(p.error.is_none());
        let choices: Vec<_> = p.draws.iter().map(|d| d.index.unwrap()).collect();
        let expected = r["weighted_paths"]
            .as_array()
            .unwrap()
            .iter()
            .find(|w| w["choices"] == serde_json::to_value(&choices).unwrap())
            .unwrap();
        assert_eq!(serde_json::to_value(&p.output).unwrap(), expected["output"]);
    }
}
#[test]
fn empty_and_failed_ranges_keep_incoming_mass_without_a_selected_occurrence() {
    let r = report();
    let mut c = context(&r["cases"][0]);
    let p = replay(&c).unwrap();
    assert_eq!(p.len(), 1);
    assert_eq!(p[0].error, Some(Failure::Bounds));
    assert_eq!(
        p[0].probability,
        Probability {
            numerator: 1,
            denominator: 1
        }
    );
    assert_eq!(
        p[0].draws,
        vec![Draw {
            width: 0,
            index: None
        }]
    );
    c.villagers.as_mut().unwrap().items = vec![Some(0), Some(1), Some(2)];
    c.failure = Some(ServiceFailure {
        gateway: Gateway::Rng,
        occurrence: 1,
    });
    let p = replay(&c).unwrap();
    assert_eq!(p.len(), 1);
    assert_eq!(
        p[0].probability,
        Probability {
            numerator: 1,
            denominator: 1
        }
    );
    assert_eq!(
        p[0].draws,
        vec![Draw {
            width: 3,
            index: None
        }]
    );
    assert!(p[0].output.is_empty());
}
#[test]
fn provenance_alias_and_capacity_guards_and_wrapping_versions() {
    let r = report();
    let base = context(&r["cases"][3]);
    for n in 0..7 {
        let mut c = base.clone();
        match n {
            0 => c.rule_version = "unknown".into(),
            1 => c.stable_services = false,
            2 => c.reference_removal = false,
            3 => c.uniform_occurrence_support = false,
            4 => c.villagers.as_mut().unwrap().identity = 1,
            5 => c.capacity = 0,
            _ => {
                c.failure = Some(ServiceFailure {
                    gateway: Gateway::Script,
                    occurrence: 0,
                })
            }
        };
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    }
    let mut c = base.clone();
    c.villagers.as_mut().unwrap().items = vec![Some(0); 100];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    c = base;
    c.initial_version = u32::MAX;
    let p = replay(&c).unwrap();
    assert_eq!(p[0].version, 1);
}
