//! Retained result/speech publication driven by the native-tested one-shot queue.
//!
//! This offline boundary starts after verified result first yields. It supplies
//! clocks, owner lookup and normal lifetime/services explicitly. It establishes
//! neither acquisition, legal Day invocation nor rendered observation readiness.

use super::character_initialization::Identity;
use super::character_role_publication::{Context, PublicationStepper, Replay};
use super::ledger::LedgerError;
use super::wait_eligibility::WaitDispatchContext;
use super::wait_queue::{
    WaitQueueDrain, WaitQueueEvent, WaitQueueMutation, WaitQueueResponse, WaitQueueState,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEDULED_ROLE_PUBLICATION_NATIVE_V1: &str = "scheduled_role_publication_native_v1";
const MAX_DRAINS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "iterator", rename_all = "snake_case")]
pub enum PendingPublication {
    Result(Identity),
    Speech(Identity),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "owner", rename_all = "snake_case", deny_unknown_fields)]
pub enum PublicationOwner {
    /// Missing table/key or a null owner: erase/release without callback entry.
    Unavailable,
    /// Callback enters but rejects its owner before any managed MoveNext.
    Mismatched,
    /// The payload still belongs to the resolved owner. Normal native callback
    /// result one and inert release are required by this bounded lifetime model.
    Matched {
        producer_time: f64,
        producer_frame_counter: i64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationDrain {
    pub dispatch: WaitDispatchContext,
    /// Required only for an entry that actually passes timing gates. Unknown
    /// labels are rejected, including labels created later in this drain.
    pub owners: BTreeMap<u64, PublicationOwner>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledPublicationContext {
    pub rule_version: String,
    /// Storage/first-yield service contract from the publication primitive.
    /// Legacy resume vectors must be empty and their verification flag false;
    /// timing comes exclusively from this queue and the explicit drains.
    pub publication: Context,
    /// Complete result-only queue at this boundary; unrelated waits unsupported.
    pub queue: WaitQueueState,
    /// Queue-local labels -> suspended result iterator identities, a bijection.
    pub result_bindings: BTreeMap<u64, Identity>,
    /// Covers normal reference accounting returning one, no reentrant drain,
    /// no owner/release mutation, and stable clocks through each callback.
    pub normal_lifetime_and_stable_services_verified: bool,
    pub drains: Vec<PublicationDrain>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PublicationDrainOutcome {
    pub queue: WaitQueueState,
    pub queue_trace: Vec<WaitQueueEvent>,
    /// Exact retained managed state/events after this drain. SaveSpeech and
    /// Show stay distinct; neither is a pixel/capture admission certificate.
    pub publication: Replay,
    pub pending: BTreeMap<u64, PendingPublication>,
    pub discarded: Vec<PendingPublication>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScheduledPublicationOutcome {
    pub drains: Vec<PublicationDrainOutcome>,
}

/// Replay at most sixteen explicit drains over one/two result instances.
/// Newly yielded speech starts synchronously inside the result callback, then
/// enters the queue with its exact float duration and current drain generation.
/// The saved-successor kernel alone decides later visits and readiness.
pub fn replay_scheduled_publication(
    context: &ScheduledPublicationContext,
) -> Result<ScheduledPublicationOutcome, LedgerError> {
    if context.rule_version != SCHEDULED_ROLE_PUBLICATION_NATIVE_V1
        || !context.normal_lifetime_and_stable_services_verified
        || context.drains.is_empty()
        || context.drains.len() > MAX_DRAINS
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut stepper = PublicationStepper::new_scheduled(&context.publication)?;
    let results: BTreeSet<_> = context
        .publication
        .result_iterators
        .iter()
        .map(|r| r.identity)
        .collect();
    let bound: BTreeSet<_> = context.result_bindings.values().copied().collect();
    let queued: BTreeSet<_> = context.queue.entries.iter().map(|e| e.logical_id).collect();
    if results.is_empty()
        || bound != results
        || bound.len() != context.result_bindings.len()
        || context
            .result_bindings
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            != queued
        || context.queue.entries.len() != queued.len()
        || context
            .queue
            .entries
            .iter()
            .any(|e| e.timing.phase_mask != 0xA || !e.release_present)
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut queue = context.queue.clone();
    let mut pending: BTreeMap<_, _> = context
        .result_bindings
        .iter()
        .map(|(id, result)| (*id, PendingPublication::Result(*result)))
        .collect();
    let mut discarded = Vec::new();
    let mut outcomes = Vec::new();
    for input in &context.drains {
        if input.owners.keys().any(|id| !pending.contains_key(id)) {
            return Err(LedgerError::InvalidContext);
        }
        let mut drain = WaitQueueDrain::begin(&queue, &input.dispatch)?;
        while let Some(id) = drain.next_callback()? {
            let owner = input.owners.get(&id).ok_or(LedgerError::InvalidContext)?;
            let continuation = pending.remove(&id).ok_or(LedgerError::InvalidContext)?;
            match owner {
                PublicationOwner::Unavailable => {
                    discarded.push(continuation);
                    drain.resolve(&WaitQueueResponse::Unavailable)?;
                }
                PublicationOwner::Mismatched => {
                    discarded.push(continuation);
                    drain.resolve(&WaitQueueResponse::Resolved {
                        callback_result: 1,
                        mutations: Vec::new(),
                    })?;
                }
                PublicationOwner::Matched {
                    producer_time,
                    producer_frame_counter,
                } => {
                    if !producer_time.is_finite() {
                        return Err(LedgerError::InvalidContext);
                    }
                    let wait = match continuation {
                        PendingPublication::Result(iterator) => {
                            stepper.resume_result(iterator, true)?
                        }
                        PendingPublication::Speech(iterator) => stepper.resume_speech(iterator)?,
                    };
                    let mut mutations = Vec::new();
                    if let Some(wait) = wait {
                        let speech = stepper
                            .snapshot()
                            .speech_iterators
                            .iter()
                            .find(|s| s.state == 1 && s.current == Some(wait.identity))
                            .ok_or(LedgerError::InvalidContext)?;
                        if pending
                            .insert(
                                drain.state.next_id,
                                PendingPublication::Speech(speech.identity),
                            )
                            .is_some()
                        {
                            return Err(LedgerError::InvalidContext);
                        }
                        mutations.push(WaitQueueMutation::Insert {
                            duration: f32::from_bits(wait.seconds_bits),
                            producer_time: *producer_time,
                            producer_frame_counter: *producer_frame_counter,
                            release_present: true,
                        });
                    }
                    drain.resolve(&WaitQueueResponse::Resolved {
                        callback_result: 1,
                        mutations,
                    })?;
                }
            }
        }
        let output = drain.finish()?;
        queue = output.state;
        if queue.entries.len() != pending.len()
            || queue
                .entries
                .iter()
                .any(|e| !pending.contains_key(&e.logical_id))
        {
            return Err(LedgerError::InvalidContext);
        }
        outcomes.push(PublicationDrainOutcome {
            queue: queue.clone(),
            queue_trace: output.trace,
            publication: stepper.snapshot().clone(),
            pending: pending.clone(),
            discarded: discarded.clone(),
        });
    }
    Ok(ScheduledPublicationOutcome { drains: outcomes })
}

#[cfg(test)]
#[path = "scheduled_role_publication_tests.rs"]
mod tests;
