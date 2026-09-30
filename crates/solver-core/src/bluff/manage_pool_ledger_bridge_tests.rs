use super::super::{
    ledger::Selector,
    manage_pool_composition::{Callback, CallbackKind},
    round_candidate_composition::Asset,
};
use super::*;
use serde_json::{json, Value};
fn corpus() -> Value {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_manage_pool_composition.json")).unwrap()
}
fn context(family: &str) -> Context {
    let report = corpus();
    let case = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["support_family"] == family)
        .unwrap();
    let mut input = case["input"].clone();
    let options = input["options"].as_object_mut().unwrap();
    options
        .entry("board")
        .or_insert(json!(["character", "other_character"]));
    options.entry("alternate_board").or_insert(json!([
        "other_character",
        "other_character",
        "other_character"
    ]));
    options.entry("manage_roster").or_insert(json!([7, 1, 2]));
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
    Context {
        version: MANAGE_POOL_LEDGER_NATIVE_V1.into(),
        construction: ConstructionContext {
            version: construction::MANAGE_POOL_NATIVE_V1.into(),
            metadata_initialized: true,
            distinct_pools_and_intermediates_sufficient_capacity: true,
            stable_services_reference_equality: true,
            deferred_remove_all_commit: true,
            uniform_occurrence_support: true,
            assets,
            initial_pool: vec![Some(7)],
            initial_version: 3,
            initial_duplicate_pool: vec![Some(6)],
            initial_duplicate_version: 9,
            input: serde_json::from_value(input).unwrap(),
        },
        asset_names: [
            "Baker",
            "Knight",
            "Rambler",
            "Minion",
            "Druid",
            "Bombardier",
            "Lilis",
            "Poet",
        ]
        .into_iter()
        .enumerate()
        .map(|(id, name)| (id as u16, name.into()))
        .collect(),
        must_include: vec![],
        live_current_build_asset_mapping: true,
        completed_setup_without_pool_script_writers: true,
        actual_selector_dispatch_and_order_verified: true,
        independent_uniform_selector_draws: true,
        events: vec![
            SelectorEvent {
                position: 1,
                acquisition_ordinal: 2,
                selector: Selector::Minion,
            },
            SelectorEvent {
                position: 2,
                acquisition_ordinal: 5,
                selector: Selector::Demon,
            },
        ],
    }
}
#[test]
fn native_shared_construction_history_and_exact_joint_probability() {
    let c = context("mixed");
    let before = c.clone();
    let replay = replay(&c).unwrap();
    assert_eq!(c, before);
    assert_eq!(replay.construction.len(), 8);
    assert_eq!(replay.outcomes.len(), 96);
    let native = corpus();
    for (index, construction) in replay.construction.iter().enumerate() {
        let case = native["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["support_family"] == "mixed")
            .find(|c| c["draws"] == json!(construction.trace.draws))
            .unwrap();
        assert_eq!(json!(construction.trace.state), case["final"]);
        assert_eq!(
            construction.probability,
            Probability {
                numerator: 1,
                denominator: 8
            }
        );
        let rows: Vec<_> = replay
            .outcomes
            .iter()
            .filter(|r| r.construction_path == index)
            .collect();
        assert_eq!(rows.len(), 12);
        for row in rows {
            let ResultKind::Selected { path } = &row.result else {
                panic!()
            };
            assert_eq!(path.trace[1].bluff_role, "Baker");
            assert_eq!(path.trace[1].event.acquisition_ordinal, 5);
            assert_eq!(path.pools.script.villagers, vec!["Knight", "Poet", "Baker"]);
            if matches!(
                path.trace[0].source,
                crate::types::BluffAcquisitionSource::DuplicatePool { .. }
            ) {
                assert_eq!(
                    row.probability,
                    Probability {
                        numerator: 1,
                        denominator: 120
                    }
                );
                assert_eq!(
                    path.probability,
                    Probability {
                        numerator: 1,
                        denominator: 15
                    }
                );
            } else {
                assert_eq!(
                    row.probability,
                    Probability {
                        numerator: 1,
                        denominator: 80
                    }
                );
                assert_eq!(
                    path.probability,
                    Probability {
                        numerator: 1,
                        denominator: 10
                    }
                );
            }
        }
    }
}
#[test]
fn lazy_source_choices_are_not_reselected_or_weighted_twice() {
    for (family, count) in [
        ("fallback", 4),
        ("inline_[0, 1]", 16),
        ("inline_[None, 1]", 56),
    ] {
        let mut c = context(family);
        c.events.clear();
        let r = replay(&c).unwrap();
        assert_eq!(r.construction.len(), count);
        assert_eq!(r.outcomes.len(), count);
        let mut mass = 0.0;
        for (outcome, construction) in r.outcomes.iter().zip(&r.construction) {
            assert_eq!(outcome.probability, construction.probability);
            let ResultKind::Selected { path } = &outcome.result else {
                panic!()
            };
            assert_eq!(
                path.probability,
                Probability {
                    numerator: 1,
                    denominator: 1
                }
            );
            assert!(path.trace.is_empty());
            mass += outcome.probability.numerator as f64 / outcome.probability.denominator as f64;
        }
        assert!((mass - 1.0).abs() < 1e-12);
        if family.contains("None") {
            assert!(r.construction.iter().any(|p| p
                .trace
                .draws
                .iter()
                .filter(|d| d["source"] == "inline")
                .count()
                == 4));
        }
    }
}
#[test]
fn final_roster_fields_override_earlier_script_snapshots() {
    let mut c = context("mixed");
    c.events.clear();
    c.construction.input.options.callbacks = vec![Callback {
        at: ("duplicate_script".into(), 9),
        kind: CallbackKind::Roster,
        value: Some(0),
        items: Some(vec![Some(7)]),
    }];
    let r = replay(&c).unwrap();
    for outcome in r.outcomes {
        let ResultKind::Selected { path } = outcome.result else {
            panic!()
        };
        assert_eq!(path.pools.script.villagers, vec!["Poet"]);
        assert!(path.pools.duplicate.iter().any(|role| role == "Knight"));
    }
    c.construction.input.options.callbacks[0].items = None;
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    assert_eq!(c, before);
}
#[test]
fn unsupported_field_aliases_and_dead_singleton_are_atomic() {
    let aliased = context("aliased_rosters");
    assert_eq!(replay(&aliased), Err(LedgerError::InvalidContext));
    let mut c = context("mixed");
    c.construction.input.options.callbacks = vec![Callback {
        at: ("duplicate_script".into(), 12),
        kind: CallbackKind::Game,
        value: None,
        items: None,
    }];
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    c.construction.input.options.callbacks.clear();
    c.completed_setup_without_pool_script_writers = false;
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
}
#[test]
fn prefix_failures_keep_unconditional_mass_and_skip_selectors() {
    let mut c = context("mixed");
    c.construction.input.options.null_game = true;
    let r = replay(&c).unwrap();
    assert_eq!(r.outcomes.len(), 1);
    assert_eq!(r.outcomes[0].result, ResultKind::PrefixFailure);
    assert_eq!(
        r.outcomes[0].probability,
        Probability {
            numerator: 1,
            denominator: 1
        }
    );
    c.construction.input.options.null_game = false;
    c.construction.input.options.fail = Some((construction::Gateway::Rng, 4));
    let r = replay(&c).unwrap();
    assert!(r
        .outcomes
        .iter()
        .all(|o| o.result == ResultKind::PrefixFailure));
    assert!(
        (r.outcomes
            .iter()
            .map(|o| o.probability.numerator as f64 / o.probability.denominator as f64)
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-12
    );
}
