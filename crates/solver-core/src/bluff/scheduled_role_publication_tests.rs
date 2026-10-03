use super::*;
use crate::bluff::character_role_publication::{
    Allocation, Event, ManagedString, ResultIterator, WaitObject,
};
use crate::bluff::wait_eligibility::WaitEligibility;
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    queue: WaitQueueState,
    callbacks: Vec<u64>,
    visits: Vec<u64>,
    history: Vec<Identity>,
    history_version: u32,
    uses_bits: u32,
    saved_text: String,
    shown: Vec<String>,
    result_states: Vec<i32>,
    speech_states: Vec<i32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    context: ScheduledPublicationContext,
    expected: Vec<Expected>,
}

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    native_report_sha256: String,
    cases: Vec<Case>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../../../../reverse_engineering/fixtures/synthetic/scheduled_role_publication_v1.json"
    ))
    .unwrap()
}

fn acquired_fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../../../../reverse_engineering/fixtures/synthetic/scheduled_role_publication_acquired_v1.json"
    ))
    .unwrap()
}

fn initial() -> ScheduledPublicationContext {
    fixture().cases.remove(2).context
}

fn text(replay: &Replay, id: Identity) -> String {
    String::from_utf16(
        &replay
            .context
            .strings
            .iter()
            .find(|s| s.identity == id)
            .unwrap()
            .units,
    )
    .unwrap()
}

fn callbacks(outcome: &PublicationDrainOutcome) -> Vec<u64> {
    outcome
        .queue_trace
        .iter()
        .filter_map(|event| match event {
            WaitQueueEvent::Callback { logical_id } => Some(*logical_id),
            _ => None,
        })
        .collect()
}

fn shown(replay: &Replay) -> Vec<String> {
    replay
        .events
        .iter()
        .filter_map(|event| match event {
            Event::Show { text: id, .. } => Some(text(replay, *id)),
            _ => None,
        })
        .collect()
}

#[test]
fn matches_all_29_native_cases_at_all_161_drain_checkpoints() {
    let fixture = fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.native_report_sha256,
        "e8182a5a51353941b4d6e07d6564dc2c78f7c9e30c7e8de7366eb37aca62efcf"
    );
    assert_eq!(fixture.cases.len(), 29);
    assert_eq!(check_native_fixture(fixture), 161);
}

#[test]
fn matches_all_six_acquired_native_cases_at_all_24_drain_checkpoints() {
    let fixture = acquired_fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.native_report_sha256,
        "3603e404ea1a55ff481fe075bfef36bfe5d339b6c9738bba4632bfc7c4f5c9d8"
    );
    assert_eq!(fixture.cases.len(), 6);
    assert_eq!(check_native_fixture(fixture), 24);
}

fn check_native_fixture(fixture: Fixture) -> usize {
    let mut checkpoints = 0;
    for case in fixture.cases {
        let output = replay_scheduled_publication(&case.context)
            .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
        assert_eq!(output.drains.len(), case.expected.len(), "{}", case.name);
        for (index, (got, want)) in output.drains.iter().zip(case.expected).enumerate() {
            checkpoints += 1;
            let label = format!("{} drain {index}", case.name);
            assert_eq!(got.queue, want.queue, "{label}");
            assert_eq!(callbacks(got), want.callbacks, "{label}");
            let visits: Vec<_> = got
                .queue_trace
                .iter()
                .filter_map(|event| match event {
                    WaitQueueEvent::Visit { logical_id, .. } => Some(*logical_id),
                    _ => None,
                })
                .collect();
            assert_eq!(visits, want.visits, "{label}");
            let p = &got.publication;
            assert_eq!(
                p.context.actor.infos,
                want.history.into_iter().map(Some).collect::<Vec<_>>(),
                "{label}"
            );
            assert_eq!(
                p.context.actor.info_version, want.history_version,
                "{label}"
            );
            assert_eq!(p.context.actor.uses as u32, want.uses_bits, "{label}");
            assert_eq!(
                text(p, p.context.actor.saved_act.unwrap()),
                want.saved_text,
                "{label}"
            );
            assert_eq!(
                p.context.ui.blank_text_value, p.context.actor.saved_act,
                "{label}"
            );
            assert_eq!(shown(p), want.shown, "{label}");
            assert_eq!(
                p.context
                    .result_iterators
                    .iter()
                    .map(|r| r.state)
                    .collect::<Vec<_>>(),
                want.result_states,
                "{label}"
            );
            assert_eq!(
                p.speech_iterators
                    .iter()
                    .map(|r| r.state)
                    .collect::<Vec<_>>(),
                want.speech_states,
                "{label}"
            );
            assert_eq!(
                p.context.reference_lists, case.context.publication.reference_lists,
                "{label}"
            );
            assert_eq!(p.context.infos, case.context.publication.infos, "{label}");
            assert_eq!(
                (
                    p.context.actor.data,
                    p.context.actor.bluff,
                    p.context.actor.register_as,
                    p.context.actor.role
                ),
                (
                    case.context.publication.actor.data,
                    case.context.publication.actor.bluff,
                    case.context.publication.actor.register_as,
                    case.context.publication.actor.role
                ),
                "{label}"
            );
        }
    }
    checkpoints
}

#[test]
fn acquired_use_is_consumed_after_clear_and_hides_picker_before_return() {
    let context = acquired_fixture().cases.remove(0).context;
    assert_eq!(context.publication.actor.uses, 1);
    assert!(context.publication.actor.infos.is_empty());
    assert_eq!(context.publication.actor.info_version, 10);
    assert!(context.publication.actor.statuses.active.is_empty());
    assert_eq!(context.publication.actor.statuses.version, 24);
    assert_eq!(context.publication.actor.statuses.resistances, vec![50]);
    assert_eq!(context.publication.actor.statuses.target, Some(1));
    assert_eq!(context.queue.next_id, 2);
    assert_eq!(context.result_bindings, BTreeMap::from([(1, 500)]));
    let output = replay_scheduled_publication(&context).unwrap();
    let skipped = &output.drains[0].publication;
    assert_eq!(skipped.context.actor.uses, 1);
    assert!(skipped.context.actor.infos.is_empty());
    assert!(skipped.events.is_empty());
    let published = &output.drains[1].publication;
    assert_eq!(published.context.actor.uses, 0);
    assert_eq!(published.context.actor.infos, vec![Some(200)]);
    assert_eq!(published.context.actor.info_version, 11);
    let save = published
        .events
        .iter()
        .position(|e| matches!(e, Event::SaveSpeech { .. }))
        .unwrap();
    let hide = published
        .events
        .iter()
        .position(|e| {
            matches!(
                e,
                Event::SetActive {
                    object: 411,
                    active: false
                }
            )
        })
        .unwrap();
    let done = published
        .events
        .iter()
        .position(|e| {
            matches!(
                e,
                Event::Return {
                    iterator: 500,
                    value: false
                }
            )
        })
        .unwrap();
    assert!(save < hide && hide < done);
    assert!(shown(published).is_empty());
    assert!(shown(&output.drains[2].publication).is_empty());
    assert_eq!(
        shown(&output.drains[3].publication),
        vec!["I am 1 card away from closest Evil"]
    );
}

#[test]
fn speech_text_is_retained_before_result_return_and_show_waits_for_later_gate() {
    let context = initial();
    let output = replay_scheduled_publication(&context).unwrap();
    for p in &output.drains[..3] {
        assert_eq!(p.publication.context.actor.infos, vec![Some(201)]);
        assert_eq!(p.publication.context.actor.saved_act, Some(103));
        assert!(shown(&p.publication).is_empty());
    }
    let prefix = &output.drains[3].publication;
    let save = prefix
        .events
        .iter()
        .position(|event| matches!(event, Event::SaveSpeech { .. }))
        .unwrap();
    let done = prefix
        .events
        .iter()
        .position(|event| {
            matches!(
                event,
                Event::Return {
                    iterator: 500,
                    value: false
                }
            )
        })
        .unwrap();
    assert!(save < done);
    assert_eq!(prefix.context.actor.saved_act, Some(100));
    assert!(shown(prefix).is_empty());
    assert_eq!(
        output.drains[3].queue.entries[0].timing.deadline,
        1.0 + f64::from(0.4_f32)
    );
    assert_eq!(output.drains[3].queue.entries[0].timing.frame_threshold, 9);
    assert!(shown(&output.drains[4].publication).is_empty());
    assert!(shown(&output.drains[5].publication).is_empty());
    assert_eq!(shown(&output.drains[6].publication).len(), 1);
}

fn two_results() -> ScheduledPublicationContext {
    let mut context = initial();
    let p = &mut context.publication;
    p.strings.push(ManagedString {
        identity: 101,
        units: "second captured clue".encode_utf16().collect(),
    });
    p.infos
        .push(super::super::character_role_publication::ActedInfo {
            identity: 202,
            description: Some(101),
            references: Some(300),
        });
    p.result_iterators.push(ResultIterator {
        identity: 510,
        actor: 1,
        info: Some(202),
        trigger_bits: 30,
        delay_bits: 0,
        state: 1,
        current: 511,
    });
    p.result_waits.push(WaitObject {
        identity: 511,
        seconds_bits: 0,
    });
    p.allocations.push(Allocation {
        result_iterator: 510,
        speech_iterator: 610,
        speech_wait: Some(611),
    });
    let mut second = context.queue.entries[0].clone();
    second.logical_id = 1;
    context.queue.entries.push(second);
    context.queue.next_id = 2;
    context.result_bindings.insert(1, 510);
    let mut first = context.drains[3].clone();
    first.dispatch.sampled_time = 2.0;
    first.dispatch.sampled_frame_counter = 10;
    first.dispatch.generation_before = 0;
    first.owners.insert(
        0,
        PublicationOwner::Matched {
            producer_time: 1.0,
            producer_frame_counter: 6,
        },
    );
    first.owners.insert(
        1,
        PublicationOwner::Matched {
            producer_time: 1.0,
            producer_frame_counter: 6,
        },
    );
    let mut later = first.clone();
    later.dispatch.generation_before = 1;
    later.owners = BTreeMap::from([
        (
            2,
            PublicationOwner::Matched {
                producer_time: 2.0,
                producer_frame_counter: 10,
            },
        ),
        (
            3,
            PublicationOwner::Matched {
                producer_time: 2.0,
                producer_frame_counter: 10,
            },
        ),
    ]);
    context.drains = vec![first, later];
    context
}

#[test]
fn overdue_new_speeches_are_visited_but_generation_suppresses_the_same_drain() {
    let output = replay_scheduled_publication(&two_results()).unwrap();
    let first = &output.drains[0];
    assert_eq!(callbacks(first), vec![0, 1]);
    assert!(shown(&first.publication).is_empty());
    for id in [2, 3] {
        assert!(first.queue_trace.contains(&WaitQueueEvent::Visit {
            logical_id: id,
            eligibility: WaitEligibility::SkipCurrentGeneration,
        }));
    }
    let final_state = &output.drains[1];
    assert_eq!(callbacks(final_state), vec![2, 3]);
    assert_eq!(
        shown(&final_state.publication),
        vec![
            text(&final_state.publication, 100),
            "second captured clue".into()
        ]
    );
    assert_eq!(final_state.publication.context.actor.saved_act, Some(101));
    assert!(final_state.pending.is_empty());
}

#[test]
fn owner_rejection_discards_the_wait_without_managed_publication() {
    for owner in [PublicationOwner::Unavailable, PublicationOwner::Mismatched] {
        let mut context = initial();
        context.drains = vec![context.drains[3].clone()];
        context.drains[0].dispatch.generation_before = 0;
        context.drains[0].owners.insert(0, owner.clone());
        let output = replay_scheduled_publication(&context).unwrap();
        let got = &output.drains[0];
        assert_eq!(got.publication.context, context.publication);
        assert!(got.publication.events.is_empty());
        assert_eq!(got.discarded, vec![PendingPublication::Result(500)]);
        assert!(got.pending.is_empty());
        assert!(got.queue.entries.is_empty());
        assert_eq!(
            callbacks(got),
            if owner == PublicationOwner::Mismatched {
                vec![0]
            } else {
                vec![]
            }
        );
        assert!(got
            .queue_trace
            .contains(&WaitQueueEvent::Release { logical_id: 0 }));
    }
}

#[test]
fn unsupported_or_inconsistent_contexts_fail_without_returning_a_prefix() {
    let original = initial();
    for variant in 0..14 {
        let mut bad = original.clone();
        match variant {
            0 => bad.normal_lifetime_and_stable_services_verified = false,
            1 => bad.publication.services.supplied_resume_order_verified = true,
            2 => bad.publication.result_resume_order.push(500),
            3 => bad.result_bindings.insert(0, 999).map(|_| ()).unwrap(),
            4 => bad.result_bindings.clear(),
            5 => bad.queue.entries[0].timing.phase_mask = 2,
            6 => bad.queue.entries[0].release_present = false,
            7 => bad.queue.next_id = 0,
            8 => bad.drains[3].owners.clear(),
            9 => bad.drains[4].dispatch.generation_before = 99,
            10 => {
                bad.drains[0]
                    .owners
                    .insert(999, PublicationOwner::Unavailable);
            }
            11 => bad.drains[3]
                .owners
                .insert(
                    0,
                    PublicationOwner::Matched {
                        producer_time: f64::NAN,
                        producer_frame_counter: 8,
                    },
                )
                .map(|_| ())
                .unwrap(),
            12 => bad.publication.services.preappend_callbacks_inert = false,
            _ => bad.publication.result_waits[0].seconds_bits = 1,
        }
        assert_eq!(
            replay_scheduled_publication(&bad),
            Err(LedgerError::InvalidContext),
            "variant {variant}"
        );
    }
    let mut bad = original.clone();
    bad.drains.clear();
    assert_eq!(
        replay_scheduled_publication(&bad),
        Err(LedgerError::InvalidContext)
    );
    bad.drains = vec![original.drains[0].clone(); 17];
    assert_eq!(
        replay_scheduled_publication(&bad),
        Err(LedgerError::InvalidContext)
    );
    let value = serde_json::to_value(&original).unwrap();
    assert_eq!(
        serde_json::from_value::<ScheduledPublicationContext>(value).unwrap(),
        original
    );
}

#[test]
fn sixteen_skipped_drains_need_no_owner_evidence_and_the_next_exceeds_the_bound() {
    let mut context = initial();
    let mut future = context.drains[0].clone();
    future.owners.clear();
    context.drains = (0..16)
        .map(|generation| {
            future.dispatch.generation_before = generation;
            future.clone()
        })
        .collect();
    let output = replay_scheduled_publication(&context).unwrap();
    assert_eq!(output.drains.len(), 16);
    let last = output.drains.last().unwrap();
    assert_eq!(last.queue.generation, 16);
    assert_eq!(last.queue.entries, context.queue.entries);
    assert_eq!(last.publication.context, context.publication);
    assert!(last.publication.events.is_empty());
    future.dispatch.generation_before = 16;
    context.drains.push(future);
    assert_eq!(
        replay_scheduled_publication(&context),
        Err(LedgerError::InvalidContext)
    );
}
