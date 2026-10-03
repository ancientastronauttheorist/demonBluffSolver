use super::*;
use serde_json::{json, Value};

fn event(ordinal: u64, phase: Phase, action: Option<u64>, observation: Observation) -> PlayerEvent {
    PlayerEvent {
        ordinal,
        phase,
        action_ordinal: action,
        evidence_id: format!("capture_{ordinal}"),
        captured_at_ms: None,
        observation,
    }
}

fn fixture() -> PlayerHistory {
    PlayerHistory {
        schema_version: SCHEMA_VERSION.into(),
        build_id: BUILD_ID.into(),
        solver_commit: "92363e756f4faf6fdefc0f59a7a8cba527c5936a".into(),
        parser_version: "reviewed_public_text_v1".into(),
        corpus_version: "development_v1".into(),
        information_mode: InformationMode::Player,
        domain_id: PROJECTION_DOMAIN.into(),
        events: vec![
            event(
                1,
                Phase::Setup,
                None,
                Observation::DeckObserved(DeckObserved {
                    n_cards: 3,
                    n_evil: 1,
                    slots: vec![
                        DeckSlot::Exposed {
                            role: "Judge".into(),
                            faction: PublicFaction::Villager,
                        },
                        DeckSlot::Exposed {
                            role: "Judge".into(),
                            faction: PublicFaction::Villager,
                        },
                        DeckSlot::Exposed {
                            role: "Baa".into(),
                            faction: PublicFaction::Demon,
                        },
                    ],
                    header_counts: HeaderCounts {
                        villagers: Some(2),
                        outcasts: Some(0),
                        minions: Some(0),
                        demons: Some(1),
                        source: HeaderSource::VisibleHud,
                    },
                }),
            ),
            event(
                2,
                Phase::Day,
                None,
                Observation::PhaseObserved(PhaseObserved {
                    hp: Some(10),
                    remaining_evil: Some(1),
                    wrong_execution_cost: Some(5),
                    ability_resets: vec![],
                    reset_rule_version: None,
                }),
            ),
            event(
                3,
                Phase::Day,
                None,
                Observation::CardRevealed(CardRevealed {
                    position: 1,
                    apparent_role: "Judge".into(),
                    speech: Some(String::new()),
                    targets: vec![],
                    parser_version: "reviewed_public_text_v1".into(),
                    rule_version: "public_current".into(),
                }),
            ),
            event(
                4,
                Phase::Day,
                Some(4),
                Observation::ActionRequested(ActionRequested {
                    action: ActionKind::Ability,
                    actor: Some(1),
                    targets: vec![3],
                    public_cost: None,
                }),
            ),
            event(
                5,
                Phase::Day,
                Some(4),
                Observation::AbilityObserved(AbilityObserved {
                    actor: 1,
                    speech: Some("#3 is\nLying".into()),
                    targets: vec![3],
                    parser_version: "reviewed_public_text_v1".into(),
                    rule_version: "public_current".into(),
                }),
            ),
        ],
    }
}

fn review(history: &PlayerHistory) -> ReviewedEvidenceRegistry {
    let mut registry = ReviewedEvidenceRegistry::default();
    for event in &history.events {
        registry
            .record_trusted_ui_review(
                history,
                event.ordinal,
                &format!("reviewed_ui/{}", event.ordinal),
            )
            .unwrap();
    }
    registry
}

fn admitted(history: &PlayerHistory) -> AdmittedPlayerHistory {
    admit_history(history, &review(history)).unwrap()
}

#[test]
fn strict_roundtrip_preserves_exact_text_order_and_multiplicity() {
    let history = fixture();
    let value = serde_json::to_value(&history).unwrap();
    assert_eq!(value["events"][4]["kind"], "ability_observed");
    assert_eq!(value["events"][4]["payload"]["speech"], "#3 is\nLying");
    let parsed: PlayerHistory = serde_json::from_value(value).unwrap();
    assert_eq!(parsed, history);
    let projected = admitted(&history).project_legacy_snapshot().unwrap();
    assert_eq!(projected.deck.villagers, vec!["Judge", "Judge"]);
    assert_eq!(projected.reveal_order, vec![1]);
    assert_eq!(
        projected.card_at(1).unwrap().info_parsed["observations"],
        json!([{"target": 3, "is_lying": true}])
    );
    assert_eq!(
        projected.board_count_provenance,
        crate::types::BoardCountProvenance::LegacyUnknown
    );
    assert_eq!(projected.hp, 10);
    assert_eq!(projected.wrong_exec_cost, 5);
}

#[test]
fn all_envelope_and_payload_levels_reject_unknown_oracle_fields() {
    let base = serde_json::to_value(fixture()).unwrap();
    for field in [
        "pd_corruption_target",
        "twin_recipient_bluff_context",
        "twin_recipient_bluff_prefix_context",
        "evil_positions",
        "true_roles",
        "game_rng",
        "queue",
        "visible",
        "reviewed_evidence",
    ] {
        let mut value = base.clone();
        value[field] = json!(true);
        assert!(
            serde_json::from_value::<PlayerHistory>(value).is_err(),
            "top level {field}"
        );
        let mut value = base.clone();
        value["events"][4][field] = json!(true);
        assert!(
            serde_json::from_value::<PlayerHistory>(value).is_err(),
            "event {field}"
        );
        let mut value = base.clone();
        value["events"][4]["payload"][field] = json!(true);
        assert!(
            serde_json::from_value::<PlayerHistory>(value).is_err(),
            "payload {field}"
        );
    }
    let mut value = base.clone();
    value["events"][0]["payload"]["header_counts"]["trusted_pre_start"] = json!(true);
    assert!(serde_json::from_value::<PlayerHistory>(value).is_err());
    let mut value = base;
    value["events"][0]["payload"]["slots"][0]["true_role"] = json!("Baa");
    assert!(serde_json::from_value::<PlayerHistory>(value).is_err());
}

#[test]
fn no_evidence_id_or_self_reported_visibility_certifies_raw_history() {
    let history = fixture();
    assert!(matches!(
        admit_history(&history, &ReviewedEvidenceRegistry::default()),
        Err(HistoryError::EvidenceUnadmitted { ordinal: 1, .. })
    ));
    let mut value = serde_json::to_value(history).unwrap();
    value["events"][2]["payload"]["visible"] = json!(true);
    assert!(serde_json::from_value::<PlayerHistory>(value).is_err());
}

#[test]
fn trusted_review_binds_the_exact_event_and_complete_prior_prefix() {
    let history = fixture();
    let mut registry = review(&history);
    assert!(registry
        .record_trusted_ui_review(&history, 1, "second_capture")
        .is_err());
    let mut changed = history.clone();
    if let Observation::AbilityObserved(result) = &mut changed.events[4].observation {
        result.speech = Some("#3 is\nsaying Truth".into());
    }
    assert!(matches!(
        admit_history(&changed, &registry),
        Err(HistoryError::EvidenceUnadmitted { ordinal: 5, .. })
    ));
    // A differently reviewed earlier event cannot inherit the later review.
    let mut changed = history.clone();
    changed.events[1].evidence_id = "new_phase_capture".into();
    if let Observation::PhaseObserved(phase) = &mut changed.events[1].observation {
        phase.hp = Some(9);
    }
    registry
        .record_trusted_ui_review(&changed, 2, "new_ui_phase")
        .unwrap();
    assert!(matches!(
        admit_history(&changed, &registry),
        Err(HistoryError::EvidenceUnadmitted { ordinal: 3, .. })
    ));
    let mut prefix = history;
    prefix.events.truncate(4);
    // Truncation remains admissible: review never requires future observations.
    assert!(admit_history(&prefix, &registry).is_ok());
}

#[test]
fn memory_transcription_requires_ui_cross_check_and_prior_actor_reveal() {
    let history = fixture();
    let mut empty = ReviewedEvidenceRegistry::default();
    assert!(empty
        .bind_reviewed_memory_transcription("capture_5", "capture_3")
        .is_err());
    let mut registry = review(&history);
    registry
        .bind_reviewed_memory_transcription("capture_5", "capture_3")
        .unwrap();
    assert!(admit_history(&history, &registry).is_ok());
    assert!(registry
        .bind_reviewed_memory_transcription("capture_5", "capture_1")
        .is_err());
    assert!(registry
        .bind_reviewed_memory_transcription("capture_3", "capture_3")
        .is_err());
    let mut missing_gate = history;
    missing_gate.events.remove(2);
    assert!(matches!(
        admit_history(&missing_gate, &registry),
        Err(HistoryError::EvidenceUnadmitted { .. })
    ));
}

#[test]
fn paired_oracles_cannot_change_planner_inputs_or_public_projection() {
    // Different hidden worlds/Start targets are deliberately kept in the test
    // oracle lane. There is no production API accepting either oracle object.
    struct OracleFixture {
        public: PlayerHistory,
        hidden: Value,
    }
    let a = OracleFixture {
        public: fixture(),
        hidden: json!({"true_roles": ["Judge", "Baa", "Judge"], "pd_corruption_target": 1, "game_rng": 17}),
    };
    let b = OracleFixture {
        public: fixture(),
        hidden: json!({"true_roles": ["Judge", "Judge", "Baa"], "pd_corruption_target": 2, "game_rng": 900}),
    };
    assert_ne!(a.hidden, b.hidden);
    let a_input = admitted(&a.public);
    let b_input = admitted(&b.public);
    assert_eq!(a_input.planner_history(), b_input.planner_history());
    let a_snapshot = serde_json::to_value(a_input.project_legacy_snapshot().unwrap()).unwrap();
    let b_snapshot = serde_json::to_value(b_input.project_legacy_snapshot().unwrap()).unwrap();
    assert_eq!(a_snapshot, b_snapshot);
    assert_eq!(a_snapshot["pd_corruption_target"], Value::Null);
    assert!(a_snapshot.get("twin_recipient_bluff_context").is_none());
    assert!(a_snapshot
        .get("twin_recipient_bluff_prefix_context")
        .is_none());
}

#[test]
fn planner_input_ignores_capture_identity_and_wall_clock() {
    let a = fixture();
    let mut b = a.clone();
    b.solver_commit = "0".repeat(40);
    b.corpus_version = "other_reviewed_corpus".into();
    for event in &mut b.events {
        event.evidence_id.push_str("_other");
        event.captured_at_ms = Some(123456 + event.ordinal);
    }
    assert_eq!(
        admitted(&a).planner_history(),
        admitted(&b).planner_history()
    );
}

#[test]
fn chronology_rejects_reorder_future_actions_wrong_targets_and_duplicate_completions() {
    let base = fixture();
    let mut changed = base.clone();
    changed.events.swap(3, 4);
    assert!(matches!(
        validate_history_shape(&changed),
        Err(HistoryError::InvalidChronology { .. })
    ));
    let mut changed = base.clone();
    changed.events[2].action_ordinal = Some(4);
    assert!(validate_history_shape(&changed).is_err());
    let mut changed = base.clone();
    if let Observation::AbilityObserved(result) = &mut changed.events[4].observation {
        result.targets = vec![2];
    }
    assert!(validate_history_shape(&changed).is_err());
    let mut changed = base.clone();
    let mut duplicate = changed.events[4].clone();
    duplicate.ordinal = 6;
    duplicate.evidence_id = "duplicate".into();
    changed.events.push(duplicate);
    assert!(validate_history_shape(&changed).is_err());
    let mut changed = base;
    changed.events[2].phase = Phase::Night;
    assert!(validate_history_shape(&changed).is_err());
}

#[test]
fn out_of_board_positions_and_versionless_parser_are_rejected() {
    let mut history = fixture();
    if let Observation::CardRevealed(card) = &mut history.events[2].observation {
        card.position = 0;
    }
    assert!(matches!(
        validate_history_shape(&history),
        Err(HistoryError::InvalidPayload { .. })
    ));
    let mut history = fixture();
    if let Observation::CardRevealed(card) = &mut history.events[2].observation {
        card.parser_version.clear();
    }
    assert!(validate_history_shape(&history).is_err());
}

#[test]
fn repeated_judge_results_remain_ordered_in_history_but_temporal_flattening_is_unsupported() {
    let mut history = fixture();
    history.events.extend([
        event(
            6,
            Phase::Night,
            Some(4),
            Observation::PhaseObserved(PhaseObserved {
                hp: Some(10),
                remaining_evil: Some(1),
                wrong_execution_cost: None,
                ability_resets: vec![],
                reset_rule_version: None,
            }),
        ),
        event(
            7,
            Phase::Day,
            Some(4),
            Observation::PhaseObserved(PhaseObserved {
                hp: Some(10),
                remaining_evil: Some(1),
                wrong_execution_cost: None,
                ability_resets: vec![1],
                reset_rule_version: Some("public_current".into()),
            }),
        ),
        event(
            8,
            Phase::Day,
            Some(8),
            Observation::ActionRequested(ActionRequested {
                action: ActionKind::Ability,
                actor: Some(1),
                targets: vec![2],
                public_cost: None,
            }),
        ),
        event(
            9,
            Phase::Day,
            Some(8),
            Observation::AbilityObserved(AbilityObserved {
                actor: 1,
                speech: Some("#2 is\nsaying Truth".into()),
                targets: vec![2],
                parser_version: history.parser_version.clone(),
                rule_version: "public_current".into(),
            }),
        ),
    ]);
    let admitted = admitted(&history);
    let results: Vec<_> = admitted
        .planner_history()
        .events
        .into_iter()
        .filter_map(|event| match event.observation {
            Observation::AbilityObserved(result) => {
                Some((event.ordinal, result.targets, result.speech))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        results,
        vec![
            (5, vec![3], Some("#3 is\nLying".into())),
            (9, vec![2], Some("#2 is\nsaying Truth".into()))
        ]
    );
    assert!(matches!(
        admitted.project_legacy_snapshot(),
        Err(ProjectionError::Unsupported {
            ordinal: Some(6),
            ..
        })
    ));
}

#[test]
fn incomplete_and_unsupported_are_not_world_contradiction_claims() {
    let mut missing = fixture();
    missing.events.pop();
    assert!(matches!(
        admitted(&missing).project_legacy_snapshot(),
        Err(ProjectionError::Incomplete {
            ordinal: Some(4),
            ..
        })
    ));
    let mut missing = fixture();
    if let Observation::AbilityObserved(result) = &mut missing.events[4].observation {
        result.speech = None;
    }
    assert!(matches!(
        admitted(&missing).project_legacy_snapshot(),
        Err(ProjectionError::Incomplete {
            ordinal: Some(5),
            ..
        })
    ));
    let mut unsupported = fixture();
    if let Observation::CardRevealed(card) = &mut unsupported.events[2].observation {
        card.apparent_role = "Poet".into();
        card.speech = Some("#3 is Evil".into());
    }
    assert!(matches!(
        admitted(&unsupported).project_legacy_snapshot(),
        Err(ProjectionError::Unsupported {
            ordinal: Some(3),
            ..
        })
    ));
    let mut unsupported = fixture();
    if let Observation::DeckObserved(deck) = &mut unsupported.events[0].observation {
        deck.slots[2] = DeckSlot::Obscured {};
    }
    assert!(matches!(
        admitted(&unsupported).project_legacy_snapshot(),
        Err(ProjectionError::Unsupported {
            ordinal: Some(1),
            ..
        })
    ));
}

#[test]
fn memory_gate_cannot_substitute_another_history_at_the_same_ordinal() {
    let a = fixture();
    let mut b = fixture();
    for event in &mut b.events {
        event.evidence_id.push_str("_b");
    }
    if let Observation::CardRevealed(card) = &mut b.events[2].observation {
        card.position = 2;
    }
    let mut registry = review(&a);
    for event in &b.events {
        registry
            .record_trusted_ui_review(&b, event.ordinal, "separately_reviewed_b")
            .unwrap();
    }
    // The result has an actor-1 UI cross-check, and its history has an event at
    // ordinal 3, but that event reveals actor 2. Actor 1's gate belongs to A.
    assert!(registry
        .bind_reviewed_memory_transcription("capture_5_b", "capture_3")
        .is_err());
    assert!(registry
        .bind_reviewed_memory_transcription("capture_5_b", "capture_3_b")
        .is_err());
    registry
        .bind_reviewed_memory_transcription("capture_5", "capture_3")
        .unwrap();
    assert!(admit_history(&a, &registry).is_ok());
}

#[test]
fn single_day_projection_rejects_phase_reentry_and_unmodeled_public_budget_changes() {
    for next_phase in [Phase::Setup, Phase::Day] {
        let mut history = fixture();
        history.events.push(event(
            6,
            next_phase,
            Some(4),
            Observation::PhaseObserved(PhaseObserved {
                hp: Some(9),
                remaining_evil: Some(1),
                wrong_execution_cost: Some(5),
                ability_resets: vec![],
                reset_rule_version: None,
            }),
        ));
        assert!(matches!(
            admitted(&history).project_legacy_snapshot(),
            Err(ProjectionError::Unsupported {
                ordinal: Some(6),
                ..
            })
        ));
    }
    for budget in ["hp", "cost"] {
        let mut history = fixture();
        let mut initial = history.events[1].clone();
        initial.ordinal = 0;
        initial.phase = Phase::Setup;
        initial.evidence_id = "initial_setup".into();
        if let Observation::PhaseObserved(phase) = &mut initial.observation {
            phase.remaining_evil = None;
            if budget == "hp" {
                phase.hp = Some(11);
            } else {
                phase.wrong_execution_cost = Some(2);
            }
        }
        history.events.insert(0, initial);
        assert!(matches!(
            admitted(&history).project_legacy_snapshot(),
            Err(ProjectionError::Unsupported {
                ordinal: Some(2),
                ..
            })
        ));
    }
    let mut history = fixture();
    if let Observation::ActionRequested(action) = &mut history.events[3].observation {
        action.public_cost = Some(1);
    }
    assert!(matches!(
        admitted(&history).project_legacy_snapshot(),
        Err(ProjectionError::Unsupported {
            ordinal: Some(4),
            ..
        })
    ));
}

#[test]
fn conditional_hunter_baa_projection_preserves_public_pool_and_exact_speech() {
    for n in [4, 5] {
        let mut history = hunter_fixture(n);
        for position in 2..=n {
            history.events.push(event(
                u64::from(position) + 2,
                Phase::Day,
                None,
                Observation::CardRevealed(CardRevealed {
                    position,
                    apparent_role: "Hunter".into(),
                    speech: Some(format!("I am {} cards away from closest Evil", n - 1)),
                    targets: vec![],
                    parser_version: history.parser_version.clone(),
                    rule_version: "public_current".into(),
                }),
            ));
        }
        let state = admitted(&history).project_legacy_snapshot().unwrap();
        assert_eq!(state.deck.villagers, vec!["Hunter"; usize::from(n - 1)]);
        assert_eq!(state.deck.demons, vec!["Baa"]);
        assert_eq!(state.reveal_order, (1..=n).collect::<Vec<_>>());
        assert_eq!(
            state.card_at(1).unwrap().info_parsed,
            json!({"distance": 1, "hunter_variant": "public_current"})
                .as_object()
                .unwrap()
                .clone()
        );
        assert_eq!(
            state.card_at(n).unwrap().info_parsed["distance"],
            json!(n - 1)
        );
        assert_eq!(
            state.board_count_provenance,
            crate::types::BoardCountProvenance::LegacyUnknown
        );
        assert!(state.pd_corruption_target.is_none());
        assert!(state.twin_recipient_bluff_context.is_none());
        assert!(state.twin_recipient_bluff_prefix_context.is_none());
    }
}

fn hunter_fixture(n: u8) -> PlayerHistory {
    let mut history = fixture();
    history.domain_id = HUNTER_BAA_PROJECTION_DOMAIN.into();
    history.events.truncate(3);
    if let Observation::DeckObserved(deck) = &mut history.events[0].observation {
        deck.n_cards = n;
        deck.slots = (1..n)
            .map(|_| DeckSlot::Exposed {
                role: "Hunter".into(),
                faction: PublicFaction::Villager,
            })
            .collect();
        deck.slots.push(DeckSlot::Exposed {
            role: "Baa".into(),
            faction: PublicFaction::Demon,
        });
        deck.header_counts.villagers = Some(n - 1);
    }
    if let Observation::CardRevealed(card) = &mut history.events[2].observation {
        card.apparent_role = "Hunter".into();
        card.speech = Some("I am 1 card away from closest Evil".into());
    }
    history
}

#[test]
fn conditional_hunter_baa_rejects_nonempty_native_memory_references() {
    let mut history = hunter_fixture(4);
    if let Observation::CardRevealed(card) = &mut history.events[2].observation {
        card.targets = vec![2, 4];
    }
    assert!(matches!(
        admitted(&history).project_legacy_snapshot(),
        Err(ProjectionError::Unsupported {
            ordinal: Some(3),
            ..
        })
    ));
    // Even the native opposite-seat duplicate is not explicit public speech.
    if let Observation::CardRevealed(card) = &mut history.events[2].observation {
        card.speech = Some("I am 2 cards away from closest Evil".into());
        card.targets = vec![3, 3];
    }
    assert!(matches!(
        admitted(&history).project_legacy_snapshot(),
        Err(ProjectionError::Unsupported {
            ordinal: Some(3),
            ..
        })
    ));
}

#[test]
fn conditional_hunter_baa_missing_speech_and_header_are_incomplete_but_bad_text_is_unsupported() {
    let mut history = hunter_fixture(4);
    if let Observation::CardRevealed(card) = &mut history.events[2].observation {
        card.speech = None;
    }
    assert!(matches!(
        admitted(&history).project_legacy_snapshot(),
        Err(ProjectionError::Incomplete {
            ordinal: Some(3),
            ..
        })
    ));
    let mut history = hunter_fixture(4);
    if let Observation::DeckObserved(deck) = &mut history.events[0].observation {
        deck.header_counts.minions = None;
    }
    assert!(matches!(
        admitted(&history).project_legacy_snapshot(),
        Err(ProjectionError::Incomplete {
            ordinal: Some(1),
            ..
        })
    ));
    for text in [
        "I am 1 cards away from closest Evil",
        "I am 2 card away from closest Evil",
        "I am 02 cards away from closest Evil",
        "I am 0 cards away from closest Evil",
        "I am 4 cards away from closest Evil",
        "i am 1 card away from closest Evil",
    ] {
        let mut history = hunter_fixture(4);
        if let Observation::CardRevealed(card) = &mut history.events[2].observation {
            card.speech = Some(text.into());
        }
        assert!(
            matches!(
                admitted(&history).project_legacy_snapshot(),
                Err(ProjectionError::Unsupported {
                    ordinal: Some(3),
                    ..
                })
            ),
            "{text}"
        );
    }
}

#[test]
fn conditional_hunter_baa_rejects_other_pools_hud_counts_sizes_and_repeated_seats() {
    let mut impossible_native_output = hunter_fixture(5);
    if let Observation::CardRevealed(card) = &mut impossible_native_output.events[2].observation {
        card.speech = Some("I am 3 cards away from closest Evil".into());
    }
    assert!(matches!(
        admitted(&impossible_native_output).project_legacy_snapshot(),
        Err(ProjectionError::Unsupported {
            ordinal: Some(3),
            ..
        })
    ));
    for mutation in 0..6 {
        let mut history = hunter_fixture(4);
        if let Observation::DeckObserved(deck) = &mut history.events[0].observation {
            match mutation {
                0 => {
                    deck.slots.remove(0);
                }
                1 => {
                    deck.slots[0] = DeckSlot::Obscured {};
                }
                2 => {
                    deck.header_counts.outcasts = Some(1);
                }
                3 => {
                    deck.slots[0] = DeckSlot::Exposed {
                        role: "Hunter".into(),
                        faction: PublicFaction::Outcast,
                    };
                }
                4 => {
                    deck.n_cards = 6;
                }
                _ => {
                    deck.n_evil = 2;
                }
            }
        }
        assert!(matches!(
            admitted(&history).project_legacy_snapshot(),
            Err(ProjectionError::Unsupported {
                ordinal: Some(1),
                ..
            })
        ));
    }
    let mut history = hunter_fixture(4);
    let mut repeated = history.events[2].clone();
    repeated.ordinal = 4;
    repeated.evidence_id = "second_reveal".into();
    history.events.push(repeated);
    assert!(matches!(
        admitted(&history).project_legacy_snapshot(),
        Err(ProjectionError::Unsupported {
            ordinal: Some(4),
            ..
        })
    ));
}

#[test]
fn conditional_hunter_baa_projects_serial_reveal_completion_and_rejects_other_actions() {
    let mut history = hunter_fixture(4);
    history.events[2].ordinal = 4;
    history.events[2].evidence_id = "capture_4".into();
    history.events[2].action_ordinal = Some(3);
    history.events.insert(
        2,
        event(
            3,
            Phase::Day,
            Some(3),
            Observation::ActionRequested(ActionRequested {
                action: ActionKind::Reveal,
                actor: None,
                targets: vec![1],
                public_cost: Some(0),
            }),
        ),
    );
    assert_eq!(
        admitted(&history)
            .project_legacy_snapshot()
            .unwrap()
            .reveal_order,
        vec![1]
    );
    let mut pending = history.clone();
    pending.events.pop();
    assert!(matches!(
        admitted(&pending).project_legacy_snapshot(),
        Err(ProjectionError::Incomplete {
            ordinal: Some(3),
            ..
        })
    ));
    for action in [ActionKind::Ability, ActionKind::Execution] {
        let mut history = hunter_fixture(4);
        history.events.push(event(
            4,
            Phase::Day,
            Some(4),
            Observation::ActionRequested(ActionRequested {
                action,
                actor: Some(1),
                targets: vec![2],
                public_cost: None,
            }),
        ));
        assert!(matches!(
            admitted(&history).project_legacy_snapshot(),
            Err(ProjectionError::Unsupported {
                ordinal: Some(4),
                ..
            })
        ));
    }
    let mut history = hunter_fixture(4);
    history.events.push(event(
        4,
        Phase::Day,
        None,
        Observation::StatusObserved(StatusObserved {
            status: VisibleStatus::Blocked,
            positions: vec![2],
            speech: None,
        }),
    ));
    assert!(matches!(
        admitted(&history).project_legacy_snapshot(),
        Err(ProjectionError::Unsupported {
            ordinal: Some(4),
            ..
        })
    ));
}

#[test]
fn conditional_hunter_baa_pair_oracles_and_reveal_order_remain_separate() {
    let a_hidden_baa = 2;
    let b_hidden_baa = 4;
    assert_ne!(a_hidden_baa, b_hidden_baa);
    let public_a = hunter_fixture(4);
    let public_b = public_a.clone();
    assert_eq!(
        admitted(&public_a).planner_history(),
        admitted(&public_b).planner_history()
    );
    assert_eq!(
        serde_json::to_value(admitted(&public_a).project_legacy_snapshot().unwrap()).unwrap(),
        serde_json::to_value(admitted(&public_b).project_legacy_snapshot().unwrap()).unwrap()
    );
    let mut history = hunter_fixture(5);
    if let Observation::CardRevealed(card) = &mut history.events[2].observation {
        card.position = 5;
    }
    history.events.push(event(
        4,
        Phase::Day,
        None,
        Observation::CardRevealed(CardRevealed {
            position: 2,
            apparent_role: "Hunter".into(),
            speech: Some("I am 2 cards away from closest Evil".into()),
            targets: vec![],
            parser_version: history.parser_version.clone(),
            rule_version: "public_current".into(),
        }),
    ));
    assert_eq!(
        admitted(&history)
            .project_legacy_snapshot()
            .unwrap()
            .reveal_order,
        vec![5, 2]
    );
}

#[test]
fn obscured_slot_cannot_smuggle_hidden_identity_and_result_bool_cannot_smuggle_truth() {
    let mut value = serde_json::to_value(fixture()).unwrap();
    value["events"][0]["payload"]["slots"][2] = json!({"visibility": "obscured", "role": "Baa"});
    assert!(serde_json::from_value::<PlayerHistory>(value).is_err());
    let mut value = serde_json::to_value(fixture()).unwrap();
    value["events"][4]["payload"]["is_lying"] = json!(true);
    assert!(serde_json::from_value::<PlayerHistory>(value).is_err());
}
