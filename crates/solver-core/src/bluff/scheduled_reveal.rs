//! Weighted DelayReveal completions driven by an explicitly supplied native
//! one-shot queue. This offline boundary requires a complete DelayReveal-only
//! queue, matching live owners, inert release bodies and producer snapshots.

use super::continuation_registry::{
    advance_ready_batch, validate_registry, ContinuationPath, ContinuationState,
};
use super::ledger::{LedgerError, Probability};
use super::twin_writer::retained_entries;
use super::wait_eligibility::WaitDispatchContext;
use super::wait_queue::{
    WaitQueueDrain, WaitQueueEvent, WaitQueueMutation, WaitQueueResponse, WaitQueueState,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEDULED_REVEAL_NATIVE_V1: &str = "scheduled_delay_reveal_native_v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledRevealState {
    pub rule_version: String,
    /// Complete pending DelayReveal instances, using the same labels as queue.
    pub continuations: ContinuationState,
    /// Complete queue for this boundary; unrelated waits are unsupported.
    pub queue: WaitQueueState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealCallbackBoundary {
    /// Explicit provenance: lookup succeeded and the dispatch payload still
    /// belongs to that live owner. False is unsupported, never a guessed resume.
    pub same_live_owner: bool,
    /// Native dispatch result, including its lifetime accounting. Only exactly
    /// one causes the queue's release call; completion alone does not imply it.
    pub callback_result: i32,
    /// Retained engine frame clock (+0x60), stable through this synchronous
    /// callback and all of its immediate writer-created DelayReveal producers.
    pub producer_time: f64,
    pub producer_frame_counter: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledRevealContext {
    pub rule_version: String,
    pub initial: ScheduledRevealState,
    pub dispatch: WaitDispatchContext,
    /// Required only for initial records that pass the native timing gates.
    /// Newly produced records carry this drain's generation and cannot resume.
    pub callbacks: BTreeMap<u64, RevealCallbackBoundary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScheduledRevealCallback {
    pub logical_id: u64,
    /// One registry batch per actual callback. Its local resume ordinal is zero;
    /// vector order and logical_id identify callbacks within this queue drain.
    pub replay: ContinuationPath,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScheduledRevealPath {
    /// Conditional RNG weight for this drain, not a prior-history probability.
    pub probability: Probability,
    pub state: ScheduledRevealState,
    pub queue_trace: Vec<WaitQueueEvent>,
    pub callbacks: Vec<ScheduledRevealCallback>,
}

fn validate_join(state: &ScheduledRevealState) -> Result<(), LedgerError> {
    if state.rule_version != SCHEDULED_REVEAL_NATIVE_V1 {
        return Err(LedgerError::InvalidContext);
    }
    validate_registry(&state.continuations)?;
    if state.queue.next_id != state.continuations.next_id
        || state.queue.entries.len() != state.continuations.pending.len()
        || state.queue.entries.iter().any(|entry| {
            !state.continuations.pending.contains_key(&entry.logical_id)
                || entry.timing.phase_mask != 0xA
                || !entry.release_present
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

#[derive(Clone)]
struct Branch {
    drain: WaitQueueDrain,
    registry: ContinuationState,
    probability: Probability,
    callbacks: Vec<ScheduledRevealCallback>,
}

fn registry_entries(state: &ContinuationState) -> usize {
    retained_entries(&state.initial.board) + state.initial.ui.len() * 4 + state.pending.len() * 2
}

// Include historical callback states, not just the final branch, in the bound.
fn callback_entries(callbacks: &[ScheduledRevealCallback]) -> usize {
    callbacks
        .iter()
        .map(|c| {
            registry_entries(&c.replay.state)
                + c.replay.created.len() * 5
                + c.replay
                    .trace
                    .iter()
                    .map(|t| {
                        8 + t.callbacks.len()
                            + t.replacement_views.len() * 3
                            + t.view.as_ref().map_or(0, |v| 2 + v.writes.len() * 4)
                            + t.start.as_ref().map_or(0, |s| {
                                1 + s
                                    .callbacks
                                    .iter()
                                    .map(|c| {
                                        2 + c.twin.as_ref().map_or(0, |w| 1 + w.replacements.len())
                                    })
                                    .sum::<usize>()
                            })
                    })
                    .sum::<usize>()
        })
        .sum()
}

/// Advance one native queue drain. Branch only on the existing exact Reveal
/// RNG model; queue order is fixed by supplied native records, not permuted or
/// assigned probabilities. Any unsupported branch invalidates the whole call.
pub fn replay_scheduled_reveal(
    context: &ScheduledRevealContext,
) -> Result<Vec<ScheduledRevealPath>, LedgerError> {
    if context.rule_version != SCHEDULED_REVEAL_NATIVE_V1 {
        return Err(LedgerError::InvalidContext);
    }
    validate_join(&context.initial)?;
    if context
        .callbacks
        .keys()
        .any(|id| !context.initial.continuations.pending.contains_key(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    let drain = WaitQueueDrain::begin(&context.initial.queue, &context.dispatch)?;
    let mut pending = vec![Branch {
        drain,
        registry: context.initial.continuations.clone(),
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        callbacks: vec![],
    }];
    let mut completed = Vec::new();
    let mut completed_entries = 0usize;
    while !pending.is_empty() {
        let mut next = Vec::new();
        let mut stage_entries = completed_entries;
        for mut branch in pending {
            let Some(id) = branch.drain.next_callback()? else {
                let outcome = branch.drain.finish()?;
                let state = ScheduledRevealState {
                    rule_version: SCHEDULED_REVEAL_NATIVE_V1.into(),
                    continuations: branch.registry,
                    queue: outcome.state,
                };
                validate_join(&state)?;
                let size = registry_entries(&state.continuations)
                    + state.queue.entries.len() * 7
                    + outcome.trace.len() * 8
                    + callback_entries(&branch.callbacks);
                completed_entries = completed_entries
                    .checked_add(size)
                    .ok_or(LedgerError::Capacity)?;
                stage_entries = stage_entries
                    .checked_add(size)
                    .ok_or(LedgerError::Capacity)?;
                if stage_entries > 1_048_576 || completed.len() + next.len() >= 65_536 {
                    return Err(LedgerError::Capacity);
                }
                completed.push(ScheduledRevealPath {
                    probability: branch.probability,
                    state,
                    queue_trace: outcome.trace,
                    callbacks: branch.callbacks,
                });
                continue;
            };
            if branch.callbacks.len() >= 16 {
                return Err(LedgerError::Capacity);
            }
            let boundary = context
                .callbacks
                .get(&id)
                .ok_or(LedgerError::InvalidContext)?;
            if !boundary.same_live_owner || !boundary.producer_time.is_finite() {
                return Err(LedgerError::InvalidContext);
            }
            // A singleton is a sealed batch: its callback runs synchronously,
            // and its newly produced waits cannot pass this drain's generation.
            let schedules = advance_ready_batch(&branch.registry, &[id])?;
            for replay in schedules.into_iter().flat_map(|s| s.paths) {
                let mut fork = branch.clone();
                fork.probability = fork
                    .probability
                    .multiply(replay.probability.numerator, replay.probability.denominator)?;
                fork.drain.resolve(&WaitQueueResponse::Resolved {
                    callback_result: boundary.callback_result,
                    mutations: replay
                        .created
                        .iter()
                        .map(|_| WaitQueueMutation::Insert {
                            duration: 0.3_f32,
                            producer_time: boundary.producer_time,
                            producer_frame_counter: boundary.producer_frame_counter,
                            release_present: true,
                        })
                        .collect(),
                })?;
                fork.registry = replay.state.clone();
                fork.callbacks.push(ScheduledRevealCallback {
                    logical_id: id,
                    replay,
                });
                stage_entries = stage_entries
                    .checked_add(
                        registry_entries(&fork.registry)
                            + fork.drain.state.entries.len() * 7
                            + fork.drain.trace.len() * 8
                            + callback_entries(&fork.callbacks),
                    )
                    .ok_or(LedgerError::Capacity)?;
                if stage_entries > 1_048_576 || completed.len() + next.len() >= 65_536 {
                    return Err(LedgerError::Capacity);
                }
                next.push(fork);
            }
        }
        pending = next;
    }
    Ok(completed)
}
