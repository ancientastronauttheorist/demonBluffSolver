//! Weighted DelayReveal completions driven by an explicitly supplied native
//! one-shot queue. V1 requires a complete DelayReveal-only queue. Both versions
//! require matching live owners, inert release bodies and producer snapshots.
//! V2 admits guarded original-role Hidden acquisition with typed Audio/Shuffle
//! waits preserved in the complete queue. Reaching either deferred callback is
//! unsupported and rejects the whole drain, including earlier callback effects.

use super::continuation_registry::{
    advance_ready_batch, validate_registry, ContinuationPath, ContinuationState,
};
use super::ledger::{LedgerError, Probability};
use super::reveal::SETUP_REVEAL_CALLBACKS_NATIVE_V5;
use super::twin_writer::retained_entries;
use super::wait_eligibility::WaitDispatchContext;
use super::wait_queue::{
    WaitQueueDrain, WaitQueueEvent, WaitQueueMutation, WaitQueueResponse, WaitQueueState,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEDULED_REVEAL_NATIVE_V1: &str = "scheduled_delay_reveal_native_v1";
pub const SCHEDULED_SETUP_REVEAL_NATIVE_V2: &str = "scheduled_setup_reveal_native_v2";

/// Caller-proven coroutine identity; no execution/effects are modeled here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeferredSetupWait {
    Audio,
    Shuffle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledRevealState {
    pub rule_version: String,
    /// Complete pending DelayReveal instances, using the same labels as queue.
    pub continuations: ContinuationState,
    /// Complete queue; V2's only non-Reveal entries are its typed deferred waits.
    pub queue: WaitQueueState,
    /// V2 preserves both distinct non-Reveal waits. V1 rejects any such entry.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub deferred_waits: BTreeMap<u64, DeferredSetupWait>,
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
    let setup = state.rule_version == SCHEDULED_SETUP_REVEAL_NATIVE_V2;
    if ![SCHEDULED_REVEAL_NATIVE_V1, SCHEDULED_SETUP_REVEAL_NATIVE_V2]
        .contains(&state.rule_version.as_str())
    {
        return Err(LedgerError::InvalidContext);
    }
    validate_registry(&state.continuations)?;
    if setup {
        if state.continuations.initial.board.reveal.rule_version != SETUP_REVEAL_CALLBACKS_NATIVE_V5
            || state
                .continuations
                .initial
                .board
                .bodies
                .values()
                .any(|body| body.state != 5 || body.revealed)
            || state.deferred_waits.len() != 2
            || state
                .deferred_waits
                .values()
                .copied()
                .collect::<BTreeSet<_>>()
                != BTreeSet::from([DeferredSetupWait::Audio, DeferredSetupWait::Shuffle])
        {
            return Err(LedgerError::InvalidContext);
        }
    } else if !state.deferred_waits.is_empty()
        || state.continuations.initial.board.reveal.rule_version == SETUP_REVEAL_CALLBACKS_NATIVE_V5
    {
        return Err(LedgerError::InvalidContext);
    }
    if state
        .deferred_waits
        .keys()
        .any(|id| state.continuations.pending.contains_key(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    if state.queue.next_id != state.continuations.next_id
        || state.queue.entries.len()
            != state.continuations.pending.len() + state.deferred_waits.len()
        || state.queue.entries.iter().any(|entry| {
            !(state.continuations.pending.contains_key(&entry.logical_id)
                || state.deferred_waits.contains_key(&entry.logical_id))
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
    if context.rule_version != context.initial.rule_version {
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
                    rule_version: context.initial.rule_version.clone(),
                    continuations: branch.registry,
                    queue: outcome.state,
                    deferred_waits: context.initial.deferred_waits.clone(),
                };
                validate_join(&state)?;
                let size = registry_entries(&state.continuations)
                    + state.queue.entries.len() * 7
                    + state.deferred_waits.len() * 2
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
            if context.initial.deferred_waits.contains_key(&id) {
                return Err(LedgerError::InvalidContext);
            }
            let boundary = context
                .callbacks
                .get(&id)
                .ok_or(LedgerError::InvalidContext)?;
            if !boundary.same_live_owner
                || !boundary.producer_time.is_finite()
                || (context.rule_version == SCHEDULED_SETUP_REVEAL_NATIVE_V2
                    && boundary.callback_result != 1)
            {
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
                            + context.initial.deferred_waits.len() * 2
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

#[cfg(test)]
#[path = "scheduled_setup_reveal_tests.rs"]
mod setup_tests;
