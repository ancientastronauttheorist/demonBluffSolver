//! Conditional initial-Day Hunter/Baa deduction comparison, not generation or
//! likelihood certification. Reference rules never call production geometry,
//! validators or scenario generation. See the accompanying domain note.

use serde_json::{json, Value};
use solver_core::player_history as public;
use solver_core::solver::solve;
use solver_core::types::{BoardCountProvenance, CardInfo, DeckComposition, GameState, Scenario};
use std::collections::BTreeSet;

const FAMILY_SIZES: [u8; 2] = [4, 5];

#[derive(Clone, Debug)]
struct Domain {
    n: u8,
    public_hunter_occurrences: u8,
    distinct_fixed_circle: bool,
    one_baa_rest_hunter: bool,
    empty_outcast_minion_pools: bool,
    clean_initial_day: bool,
    finished_setup_and_hunter_bluff_supplied: bool,
}

impl Domain {
    fn supplied(n: u8) -> Self {
        Self {
            n,
            public_hunter_occurrences: n - 1,
            distinct_fixed_circle: true,
            one_baa_rest_hunter: true,
            empty_outcast_minion_pools: true,
            clean_initial_day: true,
            finished_setup_and_hunter_bluff_supplied: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Observation {
    actor: u8,
    distance: u8,
    text: String,
    targets: [u8; 2],
}

#[derive(Clone, Debug)]
enum Event {
    Reveal(Observation),
    Richer(&'static str),
}

#[derive(Clone, Debug)]
struct History {
    domain: Domain,
    events: Vec<Event>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct World {
    baa_seat: u8,
}

#[derive(Debug, PartialEq, Eq)]
enum ReferenceResult {
    Complete(BTreeSet<World>),
    Unsupported(&'static str),
}

// Walk the physical circle rather than importing the production distance helper.
fn step(seat: u8, n: u8, forward: bool) -> u8 {
    if forward {
        if seat == n {
            1
        } else {
            seat + 1
        }
    } else if seat == 1 {
        n
    } else {
        seat - 1
    }
}

fn native_distance(n: u8, actor: u8, baa: u8) -> u8 {
    let scan = |forward| {
        let mut seat = actor;
        for distance in 1..n {
            seat = step(seat, n, forward);
            if seat == baa {
                return distance;
            }
        }
        n - 1 // The actor is excluded, including when it is the sole Evil.
    };
    scan(true).min(scan(false))
}

fn native_targets(n: u8, actor: u8, distance: u8) -> [u8; 2] {
    let mut forward = actor;
    let mut reverse = actor;
    for _ in 0..distance {
        forward = step(forward, n, true);
        reverse = step(reverse, n, false);
    }
    [forward, reverse] // Retain ordering and an even-board opposite-seat duplicate.
}

fn native_text(distance: u8) -> String {
    if distance == 1 {
        "I am 1 card away from closest Evil".into()
    } else {
        format!("I am {distance} cards away from closest Evil")
    }
}

fn observation(n: u8, actor: u8, distance: u8) -> Observation {
    Observation {
        actor,
        distance,
        text: native_text(distance),
        targets: native_targets(n, actor, distance),
    }
}

fn admitted_observations(history: &History) -> Result<Vec<&Observation>, &'static str> {
    let d = &history.domain;
    if !FAMILY_SIZES.contains(&d.n) {
        return Err("outside the predeclared finite board-size family");
    }
    if d.public_hunter_occurrences != d.n - 1 {
        return Err("public Hunter multiplicity does not establish the conditional roster");
    }
    if !(d.distinct_fixed_circle
        && d.one_baa_rest_hunter
        && d.empty_outcast_minion_pools
        && d.clean_initial_day
        && d.finished_setup_and_hunter_bluff_supplied)
    {
        return Err("conditional finished-setup domain is not established");
    }
    let mut actors = BTreeSet::new();
    let mut observations = Vec::new();
    for event in &history.events {
        let o = match event {
            Event::Reveal(o) => o,
            Event::Richer(reason) => return Err(reason),
        };
        if o.actor == 0 || o.actor > d.n || !actors.insert(o.actor) {
            return Err("not one verified reveal per distinct physical seat");
        }
        // A coherent native-union result may contradict this conditional model.
        // Malformed capture data is Unsupported instead of an empty world set.
        if !(1..=d.n / 2).contains(&o.distance) && o.distance != d.n - 1 {
            return Err("distance is outside the current native output union");
        }
        if o.text != native_text(o.distance)
            || o.targets != native_targets(d.n, o.actor, o.distance)
        {
            return Err("speech and ordered target references are not coherent");
        }
        observations.push(o);
    }
    Ok(observations)
}

fn complete_worlds(history: &History) -> ReferenceResult {
    reference_worlds(history, false)
}

// The mutation is diagnostic reference code only; production is never patched.
fn reference_worlds(history: &History, linear_truth_mutation: bool) -> ReferenceResult {
    let observations = match admitted_observations(history) {
        Ok(observations) => observations,
        Err(reason) => return ReferenceResult::Unsupported(reason),
    };
    let n = history.domain.n;
    let worlds = (1..=n)
        .filter(|&baa| {
            observations.iter().all(|o| {
                let truth = if linear_truth_mutation && o.actor != baa {
                    o.actor.abs_diff(baa)
                } else {
                    native_distance(n, o.actor, baa)
                };
                if o.actor == baa {
                    // Native GetBluffInfo removes the truthful value from the
                    // distinct integer candidate list, then draws one index.
                    (1..=n / 2).any(|d| d != truth && d == o.distance)
                } else {
                    o.distance == truth
                }
            })
        })
        .map(|baa_seat| World { baa_seat })
        .collect();
    ReferenceResult::Complete(worlds)
}

fn project_legal_history(history: &History) -> Result<GameState, &'static str> {
    let observations = admitted_observations(history)?;
    let n = history.domain.n;
    Ok(GameState {
        n_cards: n,
        n_evil: 1,
        deck: DeckComposition {
            // Preserve the supplied public occurrence multiset. A single
            // Hunter record would not account for N-1 real Hunter actors.
            // This contains neither a privileged Baa seat nor an RNG draw.
            villagers: vec!["Hunter".into(); usize::from(history.domain.public_hunter_occurrences)],
            demons: vec!["Baa".into()],
            ..DeckComposition::default()
        },
        board_villager_count: Some(n - 1),
        board_outcast_count: Some(0),
        board_minion_count: Some(0),
        board_demon_count: Some(1),
        board_count_provenance: BoardCountProvenance::TrustedPreStart,
        cards: observations
            .iter()
            .map(|o| CardInfo {
                position: o.actor,
                apparent_role: "Hunter".into(),
                info_text: o.text.clone(),
                // Current production Hunter payload has no target field; the
                // reference gate above checks raw speech/references first.
                info_parsed: json!({"hunter_variant": "public_current", "distance": o.distance})
                    .as_object()
                    .unwrap()
                    .clone(),
            })
            .collect(),
        reveal_order: observations.iter().map(|o| o.actor).collect(),
        ..GameState::default()
    })
}

fn canonical_world(scenario: &Scenario, n: u8) -> World {
    assert_eq!(scenario.evil_positions.len(), 1);
    let (&baa_seat, role) = scenario.evil_positions.iter().next().unwrap();
    assert!((1..=n).contains(&baa_seat));
    assert_eq!(role, "Baa");
    // Reject residual corruption, identity traces, grouped alternatives and
    // future populated Scenario fields; do not silently discard correlations.
    let mut admitted = Scenario::default();
    admitted.evil_positions.insert(baa_seat, "Baa".into());
    assert_eq!(
        serde_json::to_value(scenario).unwrap(),
        serde_json::to_value(admitted).unwrap(),
        "unexpected state outside the complete conditional world"
    );
    World { baa_seat }
}

fn compare(history: &History) -> BTreeSet<World> {
    let state = project_legal_history(history).unwrap();
    compare_snapshot(history, &state)
}

fn compare_snapshot(history: &History, state: &GameState) -> BTreeSet<World> {
    let ReferenceResult::Complete(expected) = complete_worlds(history) else {
        panic!("comparison family must be admitted: {history:?}");
    };
    let result = solve(state);
    let mut actual = BTreeSet::new();
    for scenario in &result.surviving_scenarios {
        assert!(
            actual.insert(canonical_world(scenario, state.n_cards)),
            "duplicate canonical world"
        );
    }
    assert_eq!(actual, expected, "complete world mismatch at {history:?}");
    assert_eq!(result.n_surviving, expected.len());
    let expected_evil: Vec<_> = (1..=state.n_cards)
        .filter(|&p| !expected.is_empty() && expected.iter().all(|w| w.baa_seat == p))
        .collect();
    let expected_good: Vec<_> = (1..=state.n_cards)
        .filter(|&p| !expected.is_empty() && expected.iter().all(|w| w.baa_seat != p))
        .collect();
    assert_eq!(result.definite_evil, expected_evil);
    assert_eq!(result.definite_good, expected_good);
    expected
}

// Synthetic UI-review registrations exercise the production admission boundary;
// they do not certify pixel exposure or native generation. The original solver
// baseline is metadata, not an assertion about this run's source fingerprint.
fn public_history(history: &History) -> public::PlayerHistory {
    let n = history.domain.n;
    let make_event = |ordinal, phase, observation| public::PlayerEvent {
        ordinal,
        phase,
        action_ordinal: None,
        evidence_id: format!("synthetic_ui_{n}_{ordinal}"),
        captured_at_ms: None,
        observation,
    };
    let mut slots = vec![
        public::DeckSlot::Exposed {
            role: "Hunter".into(),
            faction: public::PublicFaction::Villager,
        };
        usize::from(history.domain.public_hunter_occurrences)
    ];
    slots.push(public::DeckSlot::Exposed {
        role: "Baa".into(),
        faction: public::PublicFaction::Demon,
    });
    let mut events = vec![
        make_event(
            1,
            public::Phase::Setup,
            public::Observation::DeckObserved(public::DeckObserved {
                n_cards: n,
                n_evil: 1,
                slots,
                header_counts: public::HeaderCounts {
                    villagers: Some(n - 1),
                    outcasts: Some(0),
                    minions: Some(0),
                    demons: Some(1),
                    source: public::HeaderSource::VisibleHud,
                },
            }),
        ),
        make_event(
            2,
            public::Phase::Day,
            public::Observation::PhaseObserved(public::PhaseObserved {
                hp: Some(10),
                remaining_evil: Some(1),
                wrong_execution_cost: Some(5),
                ability_resets: vec![],
                reset_rule_version: None,
            }),
        ),
    ];
    for (index, event) in history.events.iter().enumerate() {
        let Event::Reveal(observation) = event else {
            panic!("synthetic public family contains a richer event");
        };
        events.push(make_event(
            index as u64 + 3,
            public::Phase::Day,
            public::Observation::CardRevealed(public::CardRevealed {
                position: observation.actor,
                apparent_role: "Hunter".into(),
                speech: Some(observation.text.clone()),
                // Native-only reference ordering is never passed to this API.
                targets: vec![],
                parser_version: "reviewed_hunter_public_text_v1".into(),
                rule_version: "public_current".into(),
            }),
        ));
    }
    public::PlayerHistory {
        schema_version: public::SCHEMA_VERSION.into(),
        build_id: public::BUILD_ID.into(),
        solver_commit: "810f655f6fe9bccc13afb041b76b84fca37c7918".into(),
        parser_version: "reviewed_hunter_public_text_v1".into(),
        corpus_version: "conditional_hunter_baa_development_v2".into(),
        information_mode: public::InformationMode::Player,
        domain_id: public::HUNTER_BAA_PROJECTION_DOMAIN.into(),
        events,
    }
}

fn reviewed_public_history(history: &History) -> public::AdmittedPlayerHistory {
    let history = public_history(history);
    let mut registry = public::ReviewedEvidenceRegistry::default();
    for event in &history.events {
        registry
            .record_trusted_ui_review(
                &history,
                event.ordinal,
                &format!("synthetic_ui_review/{}", event.ordinal),
            )
            .unwrap();
    }
    public::admit_history(&history, &registry).unwrap()
}

fn compare_public(history: &History) -> BTreeSet<World> {
    let state = reviewed_public_history(history)
        .project_legacy_snapshot()
        .unwrap();
    assert_eq!(
        state.board_count_provenance,
        BoardCountProvenance::LegacyUnknown
    );
    assert!(state.pd_corruption_target.is_none());
    assert!(state.twin_recipient_bluff_context.is_none());
    assert!(state.twin_recipient_bluff_prefix_context.is_none());
    compare_snapshot(history, &state)
}

fn permutations(remaining: &mut Vec<u8>, prefix: &mut Vec<u8>, result: &mut Vec<Vec<u8>>) {
    if remaining.is_empty() {
        result.push(prefix.clone());
        return;
    }
    for i in 0..remaining.len() {
        let seat = remaining.remove(i);
        prefix.push(seat);
        permutations(remaining, prefix, result);
        prefix.pop();
        remaining.insert(i, seat);
    }
}

fn finite_histories() -> Vec<History> {
    let mut histories = Vec::new();
    for n in FAMILY_SIZES {
        let mut orders = Vec::new();
        permutations(&mut (1..=n).collect(), &mut Vec::new(), &mut orders);
        for baa in 1..=n {
            for bluff in 1..=n / 2 {
                assert_ne!(bluff, native_distance(n, baa, baa));
                for order in &orders {
                    histories.push(History {
                        domain: Domain::supplied(n),
                        events: order
                            .iter()
                            .map(|&actor| {
                                let d = if actor == baa {
                                    bluff
                                } else {
                                    native_distance(n, actor, baa)
                                };
                                Event::Reveal(observation(n, actor, d))
                            })
                            .collect(),
                    });
                }
            }
        }
    }
    histories
}

#[test]
fn complete_conditional_worlds_agree_at_every_finite_family_prefix() {
    let histories = finite_histories();
    let mut prefixes = 0;
    let mut ambiguous = 0;
    let mut unique = 0;
    for history in &histories {
        for length in 0..=history.events.len() {
            let prefix = History {
                domain: history.domain.clone(),
                events: history.events[..length].to_vec(),
            };
            let worlds = compare(&prefix);
            assert!(
                !worlds.is_empty(),
                "generated history lost all supporting worlds"
            );
            ambiguous += usize::from(worlds.len() > 1);
            unique += usize::from(worlds.len() == 1);
            prefixes += 1;
        }
    }
    assert_eq!(histories.len(), 1392);
    assert_eq!(prefixes, 8160);
    // Counts independently enumerated before the production comparison run.
    assert_eq!(ambiguous, 4376);
    assert_eq!(unique, 3784);
    eprintln!("conditional_hunter_baa_v1 histories={} prefixes={prefixes} ambiguous={ambiguous} unique={unique}", histories.len());
}

#[test]
fn reviewed_public_projection_preserves_every_complete_conditional_world() {
    let mut prefixes = 0;
    let mut ambiguous = 0;
    let mut unique = 0;
    for history in finite_histories() {
        for length in 0..=history.events.len() {
            let prefix = History {
                domain: history.domain.clone(),
                events: history.events[..length].to_vec(),
            };
            let worlds = compare_public(&prefix);
            assert!(!worlds.is_empty());
            ambiguous += usize::from(worlds.len() > 1);
            unique += usize::from(worlds.len() == 1);
            prefixes += 1;
        }
    }
    assert_eq!((prefixes, ambiguous, unique), (8160, 4376, 3784));
    eprintln!(
        "reviewed_public_hunter_baa prefixes={prefixes} ambiguous={ambiguous} unique={unique}"
    );
    for n in FAMILY_SIZES {
        for claimed in [1, n - 1] {
            let impossible = History {
                domain: Domain::supplied(n),
                events: (1..=n)
                    .map(|actor| Event::Reveal(observation(n, actor, claimed)))
                    .collect(),
            };
            assert!(compare_public(&impossible).is_empty());
        }
    }
}

#[test]
fn native_hunter_sentences_preserve_complete_conditional_worlds() {
    // These are native-generated sentences under supplied runtime/scheduling
    // services. The public history is a separate synthetic availability fixture,
    // not the complete native history: its retained `prior_info` stress record
    // is not an initial-Day public observation. Generation and pixels stay open.
    let report_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_hunter_role_publication.json",
    );
    let report: Value = serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["build_id"], public::BUILD_ID);
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["display_id_by_native_seat"], json!([4, 3, 2, 1]));
    let cases = report["cases"].as_array().unwrap();
    assert_eq!(report["case_count"], 20);
    assert_eq!(cases.len(), 20);
    let mut truths = 0;
    let mut bluffs = 0;
    for case in cases {
        assert_eq!(case["returned"], true);
        let native_actor = u8::try_from(case["options"]["actor_seat"].as_u64().unwrap()).unwrap();
        let native_baa = u8::try_from(case["options"]["baa_seat"].as_u64().unwrap()).unwrap();
        assert!(native_actor < 4 && native_baa < 4);
        // Native board order is the reverse of this reference circle. Reversal
        // preserves nearest circular distance; native ordered refs are not input.
        let actor = 4 - native_actor;
        let actual_world = World {
            baa_seat: 4 - native_baa,
        };
        let distance = u8::try_from(case["expected"]["distance"].as_u64().unwrap()).unwrap();
        let generated = case["final"]["generated"].as_array().unwrap();
        assert_eq!(generated.len(), 1);
        let text = generated[0]["description"].as_str().unwrap();
        assert_eq!(case["final"]["saved_speech"], text);
        assert_eq!(case["final"]["text"], text);
        assert_eq!(text, native_text(distance));
        let mut observed = observation(4, actor, distance);
        observed.text = text.to_owned();
        let history = History {
            domain: Domain::supplied(4),
            events: vec![Event::Reveal(observed)],
        };
        let public = public_history(&history);
        let public::Observation::CardRevealed(reveal) = &public.events[2].observation else {
            unreachable!()
        };
        assert_eq!(reveal.speech.as_deref(), Some(text));
        assert!(reveal.targets.is_empty());
        // Hidden seats grade inclusion only; production construction/admission
        // receives the public sentence/position and supplied public setup.
        assert!(compare_public(&history).contains(&actual_world));
        truths += usize::from(native_actor != native_baa);
        bluffs += usize::from(native_actor == native_baa);
    }
    assert_eq!((truths, bluffs), (12, 8));
}

#[test]
fn native_only_reference_mutation_does_not_change_public_input() {
    let a = History {
        domain: Domain::supplied(5),
        events: vec![Event::Reveal(observation(5, 1, 2))],
    };
    let mut b = a.clone();
    let Event::Reveal(observation) = &mut b.events[0] else {
        unreachable!()
    };
    observation.targets = [5, 5];
    assert!(matches!(
        complete_worlds(&b),
        ReferenceResult::Unsupported(_)
    ));
    assert_eq!(public_history(&a), public_history(&b));
    assert_eq!(
        reviewed_public_history(&a).planner_history(),
        reviewed_public_history(&b).planner_history()
    );
    let projected_a = reviewed_public_history(&a)
        .project_legacy_snapshot()
        .unwrap();
    let projected_b = reviewed_public_history(&b)
        .project_legacy_snapshot()
        .unwrap();
    assert_eq!(
        serde_json::to_value(&projected_a).unwrap(),
        serde_json::to_value(&projected_b).unwrap()
    );
    // The invalid private reference record is a validation failure, not an
    // observation the production public adapter can use to narrow these worlds.
    let expected = compare_snapshot(&a, &projected_a);
    assert_eq!(compare_snapshot(&a, &projected_b), expected);
}

#[test]
fn coherent_impossible_histories_have_complete_empty_world_sets() {
    for n in FAMILY_SIZES {
        for claimed in [1, n - 1] {
            let history = History {
                domain: Domain::supplied(n),
                events: (1..=n)
                    .map(|actor| Event::Reveal(observation(n, actor, claimed)))
                    .collect(),
            };
            // All-one claims contradict at least one truthful distant Hunter;
            // all-(N-1) claims also contradict Baa's bounded bluff domain.
            assert!(compare(&history).is_empty());
        }
    }
}

#[test]
fn ordered_references_preserve_duplicates_and_reject_incoherent_capture() {
    assert_eq!(native_targets(4, 1, 2), [3, 3]);
    assert_eq!(native_targets(5, 1, 2), [3, 4]);
    assert_eq!(native_targets(5, 5, 1), [1, 4]);
    let mut history = History {
        domain: Domain::supplied(5),
        events: vec![Event::Reveal(observation(5, 1, 2))],
    };
    compare(&history);
    let Event::Reveal(o) = &mut history.events[0] else {
        unreachable!()
    };
    o.targets.swap(0, 1);
    assert!(matches!(
        complete_worlds(&history),
        ReferenceResult::Unsupported(_)
    ));
    assert!(project_legal_history(&history).is_err());
    history.events = vec![Event::Reveal(observation(5, 1, 1))];
    let Event::Reveal(o) = &mut history.events[0] else {
        unreachable!()
    };
    o.text = "I am 1 cards away from closest Evil".into();
    assert!(matches!(
        complete_worlds(&history),
        ReferenceResult::Unsupported(_)
    ));
    assert!(project_legal_history(&history).is_err());
}

#[test]
fn richer_histories_and_unestablished_domain_are_explicitly_unsupported() {
    for reason in [
        "execution",
        "Night",
        "ability",
        "corruption",
        "identity writer",
    ] {
        let history = History {
            domain: Domain::supplied(4),
            events: vec![Event::Richer(reason)],
        };
        assert_eq!(
            complete_worlds(&history),
            ReferenceResult::Unsupported(reason)
        );
        assert!(project_legal_history(&history).is_err());
    }
    for field in 0..5 {
        let mut domain = Domain::supplied(4);
        match field {
            0 => domain.distinct_fixed_circle = false,
            1 => domain.one_baa_rest_hunter = false,
            2 => domain.empty_outcast_minion_pools = false,
            3 => domain.clean_initial_day = false,
            4 => domain.finished_setup_and_hunter_bluff_supplied = false,
            _ => unreachable!(),
        }
        let history = History {
            domain,
            events: Vec::new(),
        };
        assert!(matches!(
            complete_worlds(&history),
            ReferenceResult::Unsupported(_)
        ));
        assert!(project_legal_history(&history).is_err());
    }
    let mut history = History {
        domain: Domain::supplied(4),
        events: Vec::new(),
    };
    history.domain.public_hunter_occurrences = 1;
    assert!(matches!(
        complete_worlds(&history),
        ReferenceResult::Unsupported(_)
    ));
    assert!(project_legal_history(&history).is_err());
    history.domain = Domain::supplied(6);
    assert!(matches!(
        complete_worlds(&history),
        ReferenceResult::Unsupported(_)
    ));
    history.domain = Domain::supplied(4);
    history.events = vec![
        Event::Reveal(observation(4, 1, 1)),
        Event::Reveal(observation(4, 1, 1)),
    ];
    assert!(matches!(
        complete_worlds(&history),
        ReferenceResult::Unsupported(_)
    ));
}

#[test]
fn development_family_detects_linear_distance_reference_mutation() {
    let mut disagreements = 0;
    for history in finite_histories() {
        if complete_worlds(&history) != reference_worlds(&history, true) {
            disagreements += 1;
        }
    }
    assert_eq!(
        disagreements, 1104,
        "family must distinguish a line from the native circle"
    );
    eprintln!(
        "reference-only linear-distance mutation rejected on {disagreements} complete histories"
    );
}

#[test]
fn canonicalization_rejects_unexplained_residual_scenario_state() {
    let mut scenario = Scenario::default();
    scenario.evil_positions.insert(1, "Baa".into());
    scenario.corrupted.insert(2);
    assert!(std::panic::catch_unwind(|| canonical_world(&scenario, 4)).is_err());
    let encoded: Value = serde_json::to_value(&scenario).unwrap();
    assert_eq!(encoded["corrupted"], json!([2]));
    // skip_serializing_if(HashMap::is_empty) does not omit populated maps.
    scenario.corrupted.clear();
    scenario.pre_twin_current_roles.insert(2, "Hunter".into());
    assert!(std::panic::catch_unwind(|| canonical_world(&scenario, 4)).is_err());
    let encoded: Value = serde_json::to_value(&scenario).unwrap();
    assert_eq!(encoded["pre_twin_current_roles"], json!({"2": "Hunter"}));
}

#[test]
fn observationally_equivalent_hidden_worlds_have_identical_solver_input() {
    let make_history = |baa| History {
        domain: Domain::supplied(4),
        events: (1..=4)
            .map(|actor| {
                let d = if actor == baa {
                    2
                } else {
                    native_distance(4, actor, baa)
                };
                Event::Reveal(observation(4, actor, d))
            })
            .collect(),
    };
    // Opposite Baa seats with distance-two bluff generate the same legal clues.
    let a = make_history(1);
    let b = make_history(3);
    assert_eq!(
        serde_json::to_value(project_legal_history(&a).unwrap()).unwrap(),
        serde_json::to_value(project_legal_history(&b).unwrap()).unwrap()
    );
    let expected = BTreeSet::from([World { baa_seat: 1 }, World { baa_seat: 3 }]);
    assert_eq!(compare(&a), expected);
    assert_eq!(compare(&b), expected);
    assert_eq!(public_history(&a), public_history(&b));
    assert_eq!(
        reviewed_public_history(&a).planner_history(),
        reviewed_public_history(&b).planner_history()
    );
    assert_eq!(compare_public(&a), expected);
    assert_eq!(compare_public(&b), expected);
}
