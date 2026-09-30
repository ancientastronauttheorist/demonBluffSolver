use super::super::ledger::{CorruptionAttempt, Selector};
use super::*;
use serde_json::json;

fn names() -> BTreeMap<u16, String> {
    [
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
    .collect()
}
fn assets() -> BTreeMap<u16, Asset> {
    [
        (10, 20, 1),
        (10, 10, 1),
        (20, 20, 1),
        (30, 10, 1),
        (10, 10, 0),
        (20, 10, 1),
        (100, 20, 255),
        (10, 10, 1),
    ]
    .into_iter()
    .enumerate()
    .map(|(id, (real_type, alignment, bluffable))| {
        (
            id as u16,
            Asset {
                real_type,
                alignment,
                bluffable,
            },
        )
    })
    .collect()
}
fn context() -> Context {
    // Native corpus identities are synthetic. Canonical names here test the
    // explicit supplied bijection, rather than claiming these are shipped assets.
    let rosters = [
        Some(vec![Some(0), Some(0), Some(1), Some(4)]),
        Some(vec![Some(2), Some(2), Some(5)]),
        Some(vec![Some(3)]),
        Some(vec![Some(6)]),
    ];
    Context {
        version: POOL_LEDGER_BRIDGE_NATIVE_V1.into(),
        stable_script_and_assets: true,
        distinct_pool_identities: true,
        independent_uniform_draws: true,
        selector_order_and_no_intervening_writers: true,
        live_current_build_asset_mapping: true,
        unique: UniqueContext {
            version: round_bluffs::ROUND_BLUFFS_NATIVE_V1.into(),
            metadata_initialized: true,
            stable_services: true,
            reference_membership_and_removal: true,
            distinct_lists_sufficient_capacity: true,
            uniform_occurrence_support: true,
            gameplay_initialized: false,
            gameplay_present: true,
            pool_present: true,
            all: vec![Some(0), Some(0), Some(1), Some(2), Some(2), Some(5)],
            script: rosters.iter().flatten().flatten().copied().collect(),
            fallback: vec![Some(7)],
            initial_pool: vec![Some(7)],
            initial_version: 3,
            assets: assets(),
            null_getter: None,
            null_filter: None,
            replace_after_getter: None,
            failure: None,
            predicate: None,
        },
        duplicate: DuplicateContext {
            version: duplicate::ROUND_CANDIDATE_NATIVE_V1.into(),
            metadata_initialized: true,
            stable_services: true,
            distinct_lists_sufficient_capacity: true,
            reference_removal: true,
            uniform_occurrence_support: true,
            gameplay_initialized: false,
            gameplay_present: true,
            pool_present: true,
            rosters,
            assets: assets(),
            initial_pool: vec![Some(7)],
            initial_version: 3,
            failure: None,
        },
        asset_names: names(),
        must_include: vec![],
        events: vec![
            SelectorEvent {
                position: 2,
                acquisition_ordinal: 3,
                selector: Selector::Minion,
            },
            SelectorEvent {
                position: 4,
                acquisition_ordinal: 8,
                selector: Selector::Demon,
            },
            SelectorEvent {
                position: 6,
                acquisition_ordinal: 10,
                selector: Selector::Drunk {
                    corruption_resistant: false,
                },
            },
        ],
    }
}

#[test]
fn native_duplicate_construction_support_then_three_selector_kinds() {
    let c = context();
    let before = c.clone();
    let replay = replay(&c).unwrap();
    assert_eq!(c, before);
    assert_eq!(replay.unique.len(), 1);
    assert_eq!(replay.duplicate.len(), 18);
    assert_eq!(replay.outcomes.len(), 90);
    assert!(replay.unique[0].trace.gameplay_initialized);
    assert!(replay.duplicate.iter().all(|p| !p
        .trace
        .events
        .iter()
        .any(|e| e["kind"] == "class_init")));
    let native: Value = serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_round_candidate_composition.json")).unwrap();
    for (index, construction) in replay.duplicate.iter().enumerate() {
        let case = native["cases"]
            .as_array()
            .unwrap()
            .iter()
            .take(18)
            .find(|case| case["draws"] == json!(construction.trace.draws))
            .unwrap();
        assert_eq!(json!(construction.trace.lists), case["final"]);
        assert_eq!(
            construction.probability,
            Probability {
                numerator: 1,
                denominator: 18
            }
        );
        let outcomes: Vec<_> = replay
            .outcomes
            .iter()
            .filter(|p| p.duplicate_path == Some(index))
            .collect();
        assert_eq!(outcomes.len(), 5);
        let mut duplicate_picks = 0;
        let mut unique_picks = 0;
        for outcome in outcomes {
            let ResultKind::Selected { path } = &outcome.result else {
                panic!("successful construction")
            };
            assert_eq!(path.trace.len(), 3);
            assert_eq!(path.trace[1].bluff_role, "Poet");
            assert_eq!(path.trace[2].bluff_role, "Poet");
            assert_eq!(
                path.trace[2].corruption_attempt,
                Some(CorruptionAttempt::AcceptedSelf)
            );
            match path.trace[0].source {
                crate::types::BluffAcquisitionSource::DuplicatePool { .. } => {
                    duplicate_picks += 1;
                    assert_eq!(
                        outcome.probability,
                        Probability {
                            numerator: 1,
                            denominator: 180
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
                crate::types::BluffAcquisitionSource::UniquePool { .. } => {
                    unique_picks += 1;
                    assert_eq!(
                        outcome.probability,
                        Probability {
                            numerator: 1,
                            denominator: 30
                        }
                    );
                    assert_eq!(
                        path.probability,
                        Probability {
                            numerator: 3,
                            denominator: 5
                        }
                    );
                }
                _ => panic!("no must-include"),
            }
            assert_eq!(
                path.trace
                    .iter()
                    .map(|t| t.event.acquisition_ordinal)
                    .collect::<Vec<_>>(),
                vec![3, 8, 10]
            );
        }
        assert_eq!((duplicate_picks, unique_picks), (4, 1));
    }
}

#[test]
fn failures_stop_the_next_stage_without_conditioning_away_mass() {
    let mut c = context();
    c.unique.failure = Some(round_bluffs::ServiceFailure {
        gateway: round_bluffs::Gateway::Rng,
        occurrence: 1,
    });
    let r = replay(&c).unwrap();
    assert!(r.duplicate.is_empty());
    assert_eq!(r.outcomes.len(), 1);
    assert_eq!(r.outcomes[0].result, ResultKind::UniqueFailure);
    assert_eq!(
        r.outcomes[0].probability,
        Probability {
            numerator: 1,
            denominator: 1
        }
    );
    c.unique.failure = None;
    c.duplicate.failure = Some(duplicate::ServiceFailure {
        gateway: duplicate::Gateway::Remove,
        occurrence: 2,
    });
    let r = replay(&c).unwrap();
    assert_eq!(r.outcomes.len(), 6); // width3, then width2, all fail after second append
    assert!(r
        .outcomes
        .iter()
        .all(|p| p.result == ResultKind::DuplicateFailure
            && p.probability
                == Probability {
                    numerator: 1,
                    denominator: 6
                }));
    assert!(r
        .duplicate
        .iter()
        .all(|p| p.trace.lists["duplicates"].items.len() == 2));
}

#[test]
fn repeated_must_include_occurrences_keep_indices_and_first_equal_removal() {
    let mut c = context();
    c.events = vec![SelectorEvent {
        position: 1,
        acquisition_ordinal: 1,
        selector: Selector::Drunk {
            corruption_resistant: true,
        },
    }];
    c.must_include = vec![Some(1), Some(1)];
    let r = replay(&c).unwrap();
    assert_eq!(r.outcomes.len(), 36);
    for outcome in r.outcomes {
        let ResultKind::Selected { path } = outcome.result else {
            panic!()
        };
        assert_eq!(
            outcome.probability,
            Probability {
                numerator: 1,
                denominator: 36
            }
        );
        assert_eq!(path.pools.must_include, vec!["Knight"]);
        assert_eq!(path.trace[0].rng_draw_count, 2);
        assert_eq!(
            path.trace[0].corruption_attempt,
            Some(CorruptionAttempt::Resisted)
        );
        assert!(matches!(
            path.trace[0].source,
            crate::types::BluffAcquisitionSource::BluffMustInclude {
                occurrence_index: 0 | 1
            }
        ));
        assert_eq!(path.pools.unique, vec!["Poet"]);
    }
}

#[test]
fn unsupported_identity_and_provenance_are_atomic() {
    let original = context();
    for change in 0..9 {
        let mut c = original.clone();
        match change {
            0 => {
                c.asset_names.insert(7, "Baker".into());
            }
            1 => {
                c.asset_names.insert(7, "poet".into());
            }
            2 => {
                c.asset_names.insert(7, "Drunk".into());
            }
            3 => c.must_include.push(None),
            4 => c.unique.script.reverse(),
            5 => c.unique.replace_after_getter = Some("script".into()),
            6 => c.events[1].acquisition_ordinal = 3,
            7 => c.events[1].position = 2,
            8 => c.independent_uniform_draws = false,
            _ => unreachable!(),
        }
        let before = c.clone();
        assert_eq!(
            replay(&c),
            Err(LedgerError::InvalidContext),
            "change{change}"
        );
        assert_eq!(c, before);
    }
    let mut c = original;
    c.must_include = vec![Some(7); 33];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}

#[test]
fn empty_fallback_failure_remains_a_construction_outcome() {
    let mut c = context();
    c.unique.fallback = vec![];
    // This is a supported native construction failure, not a selector failure.
    let r = replay(&c).unwrap();
    assert!(r
        .outcomes
        .iter()
        .all(|p| p.result == ResultKind::UniqueFailure));
    // A pool can complete with only Outcasts. Demon cannot draw from it and the
    // successful-selector kernel must reject instead of selecting another role.
    c.unique.all = vec![Some(2), Some(5)];
    c.unique.script = vec![];
    c.duplicate.rosters = [Some(vec![]), Some(vec![]), Some(vec![]), Some(vec![])];
    c.unique.fallback = vec![Some(2), Some(5)];
    // Native fallback filters Good then Villager, so this still fails construction.
    assert!(replay(&c)
        .unwrap()
        .outcomes
        .iter()
        .all(|p| p.result == ResultKind::UniqueFailure));
}

#[test]
fn wide_selector_support_rejects_before_publishing_a_partial_joint_distribution() {
    let mut c = context();
    c.events = (1..=8)
        .map(|position| SelectorEvent {
            position,
            acquisition_ordinal: position as u16,
            selector: Selector::Minion,
        })
        .collect();
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
}
