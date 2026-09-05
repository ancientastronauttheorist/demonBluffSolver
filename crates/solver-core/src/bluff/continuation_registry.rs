//! Logical continuation identities across caller-proven sealed ready batches.
//! IDs allocated here are simulation labels, never native handles or readiness.
use super::ledger::{LedgerError, Probability};
use super::ready_batch::{
    replay_ready_batch, ReadyBatchContext, ReadyContinuation, READY_BATCH_NATIVE_V1,
};
use super::reveal_writer::{
    replay_reveal_writers, RevealWriterContext, WriterResumeTrace, REVEAL_WRITER_VIEW_NATIVE_V2,
};
use super::twin_writer::retained_entries;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CONTINUATION_REGISTRY_NATIVE_V1: &str = "logical_continuation_registry_native_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationState {
    pub rule_version: String,
    pub initial: RevealWriterContext,
    /// Complete registry, including pending instances not ready for this batch.
    pub pending: BTreeMap<u64, u8>,
    /// All existing labels must be below this never-reused allocation cursor.
    pub next_id: u64,
    pub batch_ordinal: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CreatedContinuation {
    pub logical_id: u64,
    pub position: u8,
    pub resume_ordinal: u16,
    pub callback_index: usize,
    pub replacement_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContinuationPath {
    /// RNG probability for this batch/order only; not a prior-history weight.
    pub probability: Probability,
    pub state: ContinuationState,
    pub trace: Vec<WriterResumeTrace>,
    pub created: Vec<CreatedContinuation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContinuationSchedule {
    pub order: Vec<u64>,
    pub paths: Vec<ContinuationPath>,
}

fn validate_registry(state: &ContinuationState) -> Result<(), LedgerError> {
    if state.rule_version != CONTINUATION_REGISTRY_NATIVE_V1
        || state.initial.rule_version != REVEAL_WRITER_VIEW_NATIVE_V2
        || !state.initial.resumes.is_empty()
        || state.pending.len() > 4096
    {
        return Err(LedgerError::InvalidContext);
    }
    replay_reveal_writers(&state.initial)?;
    let mut counts = BTreeMap::<u8, usize>::new();
    for (id, position) in &state.pending {
        if *id >= state.next_id || !state.initial.board.bodies.contains_key(position) {
            return Err(LedgerError::InvalidContext);
        }
        *counts.entry(*position).or_default() += 1;
    }
    if state.initial.board.reveal.actors.iter().any(|a| {
        counts.get(&a.position).copied().unwrap_or(0) != usize::from(a.remaining_continuations)
    }) {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

/// Explore one sealed batch using logical IDs from the complete pending set.
/// The caller proves readiness and the no-new-eligibility boundary each time.
pub fn advance_ready_batch(
    state: &ContinuationState,
    ready_ids: &[u64],
) -> Result<Vec<ContinuationSchedule>, LedgerError> {
    validate_registry(state)?;
    if ready_ids.len() > 6
        || ready_ids.iter().copied().collect::<BTreeSet<_>>().len() != ready_ids.len()
    {
        return Err(LedgerError::InvalidContext);
    }
    let next_batch = state
        .batch_ordinal
        .checked_add(1)
        .ok_or(LedgerError::Capacity)?;
    let ready = ready_ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            Ok(ReadyContinuation {
                id: i as u16,
                position: *state.pending.get(id).ok_or(LedgerError::InvalidContext)?,
            })
        })
        .collect::<Result<Vec<_>, LedgerError>>()?;
    let batches = replay_ready_batch(&ReadyBatchContext {
        rule_version: READY_BATCH_NATIVE_V1.into(),
        initial: state.initial.clone(),
        ready,
    })?;
    let mut result = Vec::new();
    let mut retained = 0usize;
    for batch in batches {
        let order = batch
            .order
            .iter()
            .map(|id| ready_ids[usize::from(*id)])
            .collect();
        let mut paths = Vec::new();
        for branch in batch.paths {
            let mut pending = state.pending.clone();
            for id in ready_ids {
                pending.remove(id);
            }
            let mut next_id = state.next_id;
            let mut created = Vec::new();
            let mut created_counts = BTreeMap::<u8, u16>::new();
            for event in &branch.replay.trace {
                if let Some(start) = &event.start {
                    for (callback_index, callback) in start.callbacks.iter().enumerate() {
                        if let Some(writer) = &callback.twin {
                            for (replacement_index, replacement) in
                                writer.replacements.iter().enumerate()
                            {
                                let logical_id = next_id;
                                next_id = next_id.checked_add(1).ok_or(LedgerError::Capacity)?;
                                pending.insert(logical_id, replacement.position);
                                *created_counts.entry(replacement.position).or_default() += 1;
                                created.push(CreatedContinuation {
                                    logical_id,
                                    position: replacement.position,
                                    resume_ordinal: event.acquisition.event.resume_ordinal,
                                    callback_index,
                                    replacement_index,
                                });
                            }
                        }
                    }
                }
            }
            if created_counts != branch.new_continuations {
                return Err(LedgerError::InvalidContext);
            }
            if pending.len() > 4096 {
                return Err(LedgerError::Capacity);
            }
            let replay = branch.replay;
            // Include the registry and detailed trace in the retained-state cap.
            let trace_entries = replay
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
                .sum::<usize>();
            retained = retained
                .checked_add(
                    retained_entries(&replay.board)
                        + replay.ui.len() * 4
                        + pending.len() * 2
                        + created.len() * 5
                        + trace_entries,
                )
                .ok_or(LedgerError::Capacity)?;
            if retained > 1_048_576 {
                return Err(LedgerError::Capacity);
            }
            let next_state = ContinuationState {
                rule_version: CONTINUATION_REGISTRY_NATIVE_V1.into(),
                initial: RevealWriterContext {
                    rule_version: REVEAL_WRITER_VIEW_NATIVE_V2.into(),
                    board: replay.board,
                    ui: replay.ui,
                    resumes: vec![],
                },
                pending,
                next_id,
                batch_ordinal: next_batch,
            };
            validate_registry(&next_state)?;
            paths.push(ContinuationPath {
                probability: replay.probability,
                state: next_state,
                trace: replay.trace,
                created,
            });
        }
        result.push(ContinuationSchedule { order, paths });
    }
    Ok(result)
}
