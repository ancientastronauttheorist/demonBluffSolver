//! Bounded one-shot queue replay with explicit callback/owner provenance.
//!
//! Finite records retain native deadline order and saved-successor mutation
//! semantics. Callback effects are supplied, not inferred from hidden game state.
//! Repeating waits, reentrant drains and mutation from release bodies are outside
//! this contract. No continuation-registry or live-solver caller is added.

use super::ledger::LedgerError;
use super::wait_eligibility::{
    evaluate_wait, make_wait_for_seconds, WaitDispatchContext, WaitEligibility,
    WaitForSecondsContext, WaitTimingRecord, UNITY_WAIT_ELIGIBILITY_NATIVE_V1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const UNITY_WAIT_QUEUE_NATIVE_V1: &str = "unity_wait_queue_native_v1";
const MAX_RECORDS: usize = 4096;
const MAX_MUTATIONS: usize = 4096;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaitQueueEntry {
    /// History-local label, not a native pointer or coroutine handle.
    pub logical_id: u64,
    pub timing: WaitTimingRecord,
    pub release_present: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaitQueueState {
    pub rule_version: String,
    pub generation: u32,
    /// Fresh labels are allocated monotonically; canceled labels are not reused.
    pub next_id: u64,
    /// Complete in-order queue. Equal deadlines preserve supplied occurrence order.
    pub entries: Vec<WaitQueueEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum WaitQueueMutation {
    /// Model a newly produced WaitForSeconds record during this callback. Its
    /// generation is the current drain's, and its frame threshold is producer+1.
    Insert {
        duration: f32,
        producer_time: f64,
        producer_frame_counter: i64,
        release_present: bool,
    },
    /// The caller identifies an actually linked wait record. Canceling an
    /// already consumed/canceled label is rejected, not treated as a no-op.
    Cancel { logical_id: u64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "owner", rename_all = "snake_case", deny_unknown_fields)]
pub enum WaitQueueResponse {
    /// Covers absent table, missing entry or a null resolved owner pointer.
    Unavailable,
    Resolved {
        callback_result: i32,
        mutations: Vec<WaitQueueMutation>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaitQueueContext {
    pub rule_version: String,
    pub initial: WaitQueueState,
    pub dispatch: WaitDispatchContext,
    /// A response is required only when a record passes timing gates. Responses
    /// describe that visit's owner lookup and callback, not permanent ownership.
    pub responses: BTreeMap<u64, WaitQueueResponse>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum WaitQueueEvent {
    Visit {
        logical_id: u64,
        eligibility: WaitEligibility,
    },
    Erase {
        logical_id: u64,
    },
    Callback {
        logical_id: u64,
    },
    Insert {
        entry: WaitQueueEntry,
    },
    CallbackResult {
        logical_id: u64,
        value: i32,
    },
    Release {
        logical_id: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WaitQueueOutcome {
    pub state: WaitQueueState,
    pub trace: Vec<WaitQueueEvent>,
}

fn validate(context: &WaitQueueContext) -> Result<(), LedgerError> {
    let state = &context.initial;
    if context.rule_version != UNITY_WAIT_QUEUE_NATIVE_V1
        || state.rule_version != UNITY_WAIT_QUEUE_NATIVE_V1
        || context.dispatch.rule_version != UNITY_WAIT_ELIGIBILITY_NATIVE_V1
        || context.dispatch.generation_before != state.generation
        || !context.dispatch.sampled_time.is_finite()
        || state.entries.len() > MAX_RECORDS
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut ids = BTreeSet::new();
    for entry in &state.entries {
        if entry.logical_id >= state.next_id
            || !ids.insert(entry.logical_id)
            || !entry.timing.deadline.is_finite()
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if state
        .entries
        .windows(2)
        .any(|w| w[0].timing.deadline > w[1].timing.deadline)
        || context.responses.keys().any(|id| !ids.contains(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    let mutations = context
        .responses
        .values()
        .try_fold(0usize, |count, response| {
            let extra = match response {
                WaitQueueResponse::Unavailable => 0,
                WaitQueueResponse::Resolved { mutations, .. } => mutations.len(),
            };
            count.checked_add(extra).ok_or(LedgerError::Capacity)
        })?;
    if mutations > MAX_MUTATIONS {
        return Err(LedgerError::Capacity);
    }
    Ok(())
}

/// Replay one drain atomically. A future deadline terminates traversal; skips
/// preserve records. Callback insertions never replace the saved successor,
/// while cancellation advances it if the canceled record was that successor.
/// The entry-time clock/frame snapshot remains fixed throughout this call.
pub fn replay_wait_queue(context: &WaitQueueContext) -> Result<WaitQueueOutcome, LedgerError> {
    validate(context)?;
    let mut state = context.initial.clone();
    state.generation = state.generation.wrapping_add(1);
    let mut trace = Vec::new();
    let mut cursor = state.entries.first().map(|entry| entry.logical_id);
    let mut visits = 0usize;
    while let Some(id) = cursor {
        visits += 1;
        if visits > MAX_RECORDS + MAX_MUTATIONS {
            return Err(LedgerError::Capacity);
        }
        let index = state
            .entries
            .iter()
            .position(|entry| entry.logical_id == id)
            .ok_or(LedgerError::InvalidContext)?;
        let entry = state.entries[index].clone();
        let eligibility = evaluate_wait(&entry.timing, &context.dispatch)?;
        trace.push(WaitQueueEvent::Visit {
            logical_id: id,
            eligibility,
        });
        if eligibility == WaitEligibility::StopAtFutureDeadline {
            break;
        }
        cursor = state.entries.get(index + 1).map(|next| next.logical_id);
        if eligibility != WaitEligibility::Eligible {
            continue;
        }
        let response = context
            .responses
            .get(&id)
            .ok_or(LedgerError::InvalidContext)?;
        state.entries.remove(index);
        trace.push(WaitQueueEvent::Erase { logical_id: id });
        let should_release = match response {
            WaitQueueResponse::Unavailable => true,
            WaitQueueResponse::Resolved {
                callback_result,
                mutations,
            } => {
                trace.push(WaitQueueEvent::Callback { logical_id: id });
                for mutation in mutations {
                    match mutation {
                        WaitQueueMutation::Insert {
                            duration,
                            producer_time,
                            producer_frame_counter,
                            release_present,
                        } => {
                            if state.entries.len() >= MAX_RECORDS {
                                return Err(LedgerError::Capacity);
                            }
                            let timing = make_wait_for_seconds(&WaitForSecondsContext {
                                rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
                                duration: *duration,
                                producer_time: *producer_time,
                                producer_frame_counter: *producer_frame_counter,
                                insertion_generation: state.generation,
                            })?;
                            let created = WaitQueueEntry {
                                logical_id: state.next_id,
                                timing,
                                release_present: *release_present,
                            };
                            state.next_id =
                                state.next_id.checked_add(1).ok_or(LedgerError::Capacity)?;
                            let insertion = state.entries.partition_point(|existing| {
                                existing.timing.deadline <= created.timing.deadline
                            });
                            state.entries.insert(insertion, created.clone());
                            trace.push(WaitQueueEvent::Insert { entry: created });
                        }
                        WaitQueueMutation::Cancel { logical_id } => {
                            let canceled_index = state
                                .entries
                                .iter()
                                .position(|entry| entry.logical_id == *logical_id)
                                .ok_or(LedgerError::InvalidContext)?;
                            if cursor == Some(*logical_id) {
                                cursor = state
                                    .entries
                                    .get(canceled_index + 1)
                                    .map(|next| next.logical_id);
                            }
                            let canceled = state.entries.remove(canceled_index);
                            trace.push(WaitQueueEvent::Erase {
                                logical_id: *logical_id,
                            });
                            if canceled.release_present {
                                trace.push(WaitQueueEvent::Release {
                                    logical_id: *logical_id,
                                });
                            }
                        }
                    }
                }
                trace.push(WaitQueueEvent::CallbackResult {
                    logical_id: id,
                    value: *callback_result,
                });
                *callback_result == 1
            }
        };
        if should_release && entry.release_present {
            trace.push(WaitQueueEvent::Release { logical_id: id });
        }
    }
    Ok(WaitQueueOutcome { state, trace })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_23_isolated_native_consumer_cases() {
        #[derive(Deserialize)]
        struct Case {
            name: String,
            context: WaitQueueContext,
            expected_state: WaitQueueState,
            expected_events: Vec<(String, u64)>,
            expected_visits: Vec<u64>,
        }
        #[derive(Deserialize)]
        struct Fixture {
            schema_version: u32,
            native_input_sha256: String,
            cases: Vec<Case>,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../reverse_engineering/fixtures/synthetic/unity_wait_consumer_v1.json"
        ))
        .unwrap();
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.native_input_sha256,
            "B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2"
        );
        assert_eq!(fixture.cases.len(), 23);
        for case in fixture.cases {
            let output = replay_wait_queue(&case.context)
                .unwrap_or_else(|error| panic!("native case {}: {error:?}", case.name));
            assert_eq!(output.state, case.expected_state, "{}", case.name);
            assert_eq!(visits(&output), case.expected_visits, "{}", case.name);
            let events = output
                .trace
                .iter()
                .filter_map(|event| {
                    let (name, logical_id) = match event {
                        WaitQueueEvent::Erase { logical_id } => ("erase", logical_id),
                        WaitQueueEvent::Callback { logical_id } => ("callback", logical_id),
                        WaitQueueEvent::Release { logical_id } => ("release", logical_id),
                        _ => return None,
                    };
                    Some((name.to_owned(), *logical_id))
                })
                .collect::<Vec<_>>();
            assert_eq!(events, case.expected_events, "{}", case.name);
        }
    }

    fn context(ids: &[u64]) -> WaitQueueContext {
        WaitQueueContext {
            rule_version: UNITY_WAIT_QUEUE_NATIVE_V1.into(),
            initial: WaitQueueState {
                rule_version: UNITY_WAIT_QUEUE_NATIVE_V1.into(),
                generation: 0,
                next_id: ids.iter().max().copied().unwrap_or(0) + 1,
                entries: ids
                    .iter()
                    .map(|id| WaitQueueEntry {
                        logical_id: *id,
                        timing: WaitTimingRecord {
                            deadline: 0.0,
                            frame_threshold: 0,
                            phase_mask: 10,
                            insertion_generation: 0,
                        },
                        release_present: true,
                    })
                    .collect(),
            },
            dispatch: WaitDispatchContext {
                rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
                sampled_time: 3.0,
                sampled_frame_counter: 1,
                phase_mask: 2,
                generation_before: 0,
            },
            responses: ids
                .iter()
                .map(|id| {
                    (
                        *id,
                        WaitQueueResponse::Resolved {
                            callback_result: 1,
                            mutations: vec![],
                        },
                    )
                })
                .collect(),
        }
    }

    fn mutations(input: &mut WaitQueueContext, id: u64, actions: Vec<WaitQueueMutation>) {
        input.responses.insert(
            id,
            WaitQueueResponse::Resolved {
                callback_result: 1,
                mutations: actions,
            },
        );
    }

    fn insertion(time: f64) -> WaitQueueMutation {
        WaitQueueMutation::Insert {
            duration: 0.0,
            producer_time: time,
            producer_frame_counter: 1,
            release_present: true,
        }
    }

    fn callbacks(output: &WaitQueueOutcome) -> Vec<u64> {
        output
            .trace
            .iter()
            .filter_map(|event| match event {
                WaitQueueEvent::Callback { logical_id } => Some(*logical_id),
                _ => None,
            })
            .collect()
    }

    fn visits(output: &WaitQueueOutcome) -> Vec<u64> {
        output
            .trace
            .iter()
            .filter_map(|event| match event {
                WaitQueueEvent::Visit { logical_id, .. } => Some(*logical_id),
                _ => None,
            })
            .collect()
    }

    fn releases(output: &WaitQueueOutcome) -> Vec<u64> {
        output
            .trace
            .iter()
            .filter_map(|event| match event {
                WaitQueueEvent::Release { logical_id } => Some(*logical_id),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn equal_deadlines_keep_occurrences_not_id_order_and_erase_before_callback() {
        let mut input = context(&[7, 3, 9]);
        input.initial.entries[0].timing.deadline = -0.0;
        let output = replay_wait_queue(&input).unwrap();
        assert_eq!(callbacks(&output), vec![7, 3, 9]);
        assert_eq!(releases(&output), vec![7, 3, 9]);
        assert!(output.state.entries.is_empty());
        assert!(matches!(
            output.trace[1],
            WaitQueueEvent::Erase { logical_id: 7 }
        ));
        assert!(matches!(
            output.trace[2],
            WaitQueueEvent::Callback { logical_id: 7 }
        ));
        assert_eq!(input.initial.entries.len(), 3);
    }

    #[test]
    fn future_deadline_stops_but_other_gates_skip_without_needing_owner_response() {
        let mut input = context(&[1, 2, 3, 4, 5]);
        input.initial.entries[0].timing.phase_mask = 1;
        input.initial.entries[1].timing.insertion_generation = 1;
        input.initial.entries[2].timing.frame_threshold = (1_i64 << 32) + 1;
        input.initial.entries[4].timing.deadline = 4.0;
        input.responses.retain(|id, _| *id == 4);
        let output = replay_wait_queue(&input).unwrap();
        assert_eq!(callbacks(&output), vec![4]);
        assert_eq!(visits(&output), vec![1, 2, 3, 4, 5]);
        assert_eq!(
            output
                .state
                .entries
                .iter()
                .map(|e| e.logical_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 5]
        );
        assert!(matches!(
            output.trace.last(),
            Some(WaitQueueEvent::Visit {
                eligibility: WaitEligibility::StopAtFutureDeadline,
                ..
            })
        ));
    }

    #[test]
    fn unavailable_owner_and_exact_callback_result_control_release() {
        let mut input = context(&[1, 2, 3, 4, 5]);
        input.responses.insert(1, WaitQueueResponse::Unavailable);
        for (id, result) in [(2, 0), (3, 2)] {
            input.responses.insert(
                id,
                WaitQueueResponse::Resolved {
                    callback_result: result,
                    mutations: vec![],
                },
            );
        }
        input.initial.entries[4].release_present = false;
        let output = replay_wait_queue(&input).unwrap();
        assert_eq!(callbacks(&output), vec![2, 3, 4, 5]);
        assert_eq!(releases(&output), vec![1, 4]);
    }

    #[test]
    fn canceling_successors_advances_cursor_and_releases_in_native_order() {
        let mut input = context(&[1, 2, 3, 4]);
        mutations(
            &mut input,
            1,
            vec![
                WaitQueueMutation::Cancel { logical_id: 2 },
                WaitQueueMutation::Cancel { logical_id: 3 },
            ],
        );
        let output = replay_wait_queue(&input).unwrap();
        assert_eq!(visits(&output), vec![1, 4]);
        assert_eq!(callbacks(&output), vec![1, 4]);
        assert_eq!(releases(&output), vec![2, 3, 1, 4]);
    }

    #[test]
    fn canceling_later_node_preserves_cursor_and_cancel_to_end_does_not_restart() {
        let mut input = context(&[1, 2, 3]);
        mutations(
            &mut input,
            1,
            vec![WaitQueueMutation::Cancel { logical_id: 3 }],
        );
        assert_eq!(visits(&replay_wait_queue(&input).unwrap()), vec![1, 2]);
        mutations(
            &mut input,
            1,
            vec![
                WaitQueueMutation::Cancel { logical_id: 2 },
                WaitQueueMutation::Cancel { logical_id: 3 },
                insertion(0.0),
            ],
        );
        let output = replay_wait_queue(&input).unwrap();
        assert_eq!(visits(&output), vec![1]);
        assert_eq!(output.state.entries[0].logical_id, 4);
    }

    #[test]
    fn insertion_before_cursor_is_unvisited_but_after_cursor_is_generation_skipped() {
        let mut before = context(&[1, 2]);
        before.initial.entries[1].timing.deadline = 2.0;
        mutations(&mut before, 1, vec![insertion(1.0)]);
        let output = replay_wait_queue(&before).unwrap();
        assert_eq!(visits(&output), vec![1, 2]);
        assert_eq!(output.state.entries[0].logical_id, 3);
        assert_eq!(output.state.entries[0].timing.insertion_generation, 1);
        assert_eq!(output.state.entries[0].timing.frame_threshold, 2);
        mutations(&mut before, 1, vec![insertion(2.0)]);
        let output = replay_wait_queue(&before).unwrap();
        assert_eq!(visits(&output), vec![1, 2, 3]);
        assert_eq!(callbacks(&output), vec![1, 2]);
        assert!(matches!(
            output.trace.last(),
            Some(WaitQueueEvent::Visit {
                logical_id: 3,
                eligibility: WaitEligibility::SkipCurrentGeneration
            })
        ));
    }

    #[test]
    fn next_same_frame_drain_changes_generation_but_retains_frame_gate() {
        let mut input = context(&[1]);
        mutations(&mut input, 1, vec![insertion(0.0)]);
        let first = replay_wait_queue(&input).unwrap();
        input.initial = first.state;
        input.dispatch.generation_before = 1;
        input.responses.clear();
        let second = replay_wait_queue(&input).unwrap();
        assert!(callbacks(&second).is_empty());
        assert!(matches!(
            second.trace[0],
            WaitQueueEvent::Visit {
                logical_id: 2,
                eligibility: WaitEligibility::SkipFutureFrame
            }
        ));
        input.initial = second.state;
        input.dispatch.generation_before = 2;
        input.dispatch.sampled_frame_counter = 2;
        input.responses.insert(
            2,
            WaitQueueResponse::Resolved {
                callback_result: 1,
                mutations: vec![],
            },
        );
        assert_eq!(callbacks(&replay_wait_queue(&input).unwrap()), vec![2]);
    }

    #[test]
    fn generation_wrap_and_signed_counter_match_native_gates() {
        let mut input = context(&[1, 2, 3]);
        input.initial.generation = u32::MAX;
        input.dispatch.generation_before = u32::MAX;
        input.dispatch.sampled_frame_counter = i64::MIN;
        input.initial.entries[1].timing.insertion_generation = u32::MAX;
        input.initial.entries[1].timing.frame_threshold = i64::MIN;
        input.initial.entries[2].timing.insertion_generation = u32::MAX;
        input.initial.entries[2].timing.frame_threshold = i64::MAX;
        let output = replay_wait_queue(&input).unwrap();
        assert_eq!(callbacks(&output), vec![2]);
        assert_eq!(output.state.generation, 0);
    }

    #[test]
    fn malformed_queue_or_missing_response_fails_atomically() {
        let input = context(&[1, 2]);
        let mut bad = input.clone();
        bad.responses.remove(&2);
        assert_eq!(replay_wait_queue(&bad), Err(LedgerError::InvalidContext));
        assert_eq!(bad.initial, input.initial);
        bad = input.clone();
        mutations(
            &mut bad,
            1,
            vec![
                WaitQueueMutation::Cancel { logical_id: 2 },
                WaitQueueMutation::Cancel { logical_id: 2 },
            ],
        );
        assert_eq!(replay_wait_queue(&bad), Err(LedgerError::InvalidContext));
        assert_eq!(bad.initial, input.initial);
        for variant in 0..6 {
            let mut bad = input.clone();
            match variant {
                0 => bad.initial.entries[1].logical_id = 1,
                1 => bad.initial.entries[0].timing.deadline = 1.0,
                2 => bad.initial.entries[0].timing.deadline = f64::NAN,
                3 => bad.initial.next_id = 2,
                4 => bad.dispatch.generation_before = 1,
                _ => {
                    bad.responses.insert(99, WaitQueueResponse::Unavailable);
                }
            }
            assert_eq!(replay_wait_queue(&bad), Err(LedgerError::InvalidContext));
        }
    }

    #[test]
    fn allocation_overflow_nonfinite_production_and_capacity_fail_whole_drain() {
        let mut input = context(&[1]);
        mutations(&mut input, 1, vec![insertion(0.0)]);
        input.initial.next_id = u64::MAX;
        assert_eq!(replay_wait_queue(&input), Err(LedgerError::Capacity));
        input.initial.next_id = 2;
        mutations(&mut input, 1, vec![insertion(f64::INFINITY)]);
        assert_eq!(replay_wait_queue(&input), Err(LedgerError::InvalidContext));
        mutations(&mut input, 1, vec![insertion(0.0); MAX_MUTATIONS + 1]);
        assert_eq!(replay_wait_queue(&input), Err(LedgerError::Capacity));
    }

    #[test]
    fn versioned_serialization_requires_explicit_provenance_even_for_empty_queue() {
        let input = context(&[]);
        assert_eq!(replay_wait_queue(&input).unwrap().state.generation, 1);
        let mut value = serde_json::to_value(&input).unwrap();
        value.as_object_mut().unwrap().remove("responses");
        assert!(serde_json::from_value::<WaitQueueContext>(value).is_err());
        let mut bad = input;
        bad.dispatch.rule_version = "unknown".into();
        assert_eq!(replay_wait_queue(&bad), Err(LedgerError::InvalidContext));
    }
}
