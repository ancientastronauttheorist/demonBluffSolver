//! Explore all orders of an explicitly sealed, already-ready continuation set.
//! Each schedule has its own RNG distribution. Schedules have no probabilities.
//! Caller proves no other continuation becomes eligible before this batch drains.
use super::ledger::LedgerError;
use super::reveal::{BluffReference, ResumeEvent};
use super::reveal_writer::{
    replay_reveal_writers, RevealWriterContext, RevealWriterPath, REVEAL_WRITER_VIEW_NATIVE_V2,
};
use super::twin_writer::retained_entries;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const READY_BATCH_NATIVE_V1: &str = "sealed_ready_batch_native_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyContinuation {
    pub id: u16,
    pub position: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyBatchContext {
    pub rule_version: String,
    /// V2 full board/UI state with no supplied resumes.
    pub initial: RevealWriterContext,
    /// Explicit physical coroutine identities; list order is not scheduler order.
    pub ready: Vec<ReadyContinuation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReadyBatchPath {
    /// Probability conditional on this schedule, using the native RNG model.
    pub replay: RevealWriterPath,
    /// Newly created callbacks remain pending, never admitted to this batch.
    pub new_continuations: BTreeMap<u8, u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReadySchedule {
    pub order: Vec<u16>,
    pub paths: Vec<ReadyBatchPath>,
}

fn orders(
    remaining: &mut Vec<ReadyContinuation>,
    prefix: &mut Vec<ReadyContinuation>,
    out: &mut Vec<Vec<ReadyContinuation>>,
) {
    if remaining.is_empty() {
        out.push(prefix.clone());
        return;
    }
    for i in 0..remaining.len() {
        let item = remaining.remove(i);
        prefix.push(item.clone());
        orders(remaining, prefix, out);
        prefix.pop();
        remaining.insert(i, item);
    }
}

fn entries(path: &RevealWriterPath) -> usize {
    retained_entries(&path.board)
        + path.ui.len() * 4
        + path
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
                            .map(|c| 2 + c.twin.as_ref().map_or(0, |w| 1 + w.replacements.len()))
                            .sum::<usize>()
                    })
            })
            .sum::<usize>()
}

pub fn replay_ready_batch(context: &ReadyBatchContext) -> Result<Vec<ReadySchedule>, LedgerError> {
    if context.rule_version != READY_BATCH_NATIVE_V1
        || context.initial.rule_version != REVEAL_WRITER_VIEW_NATIVE_V2
        || !context.initial.resumes.is_empty()
        || context.ready.len() > 6
    {
        return Err(LedgerError::InvalidContext);
    }
    let initial = replay_reveal_writers(&context.initial)?
        .into_iter()
        .next()
        .ok_or(LedgerError::InvalidContext)?;
    let mut ids = BTreeSet::new();
    let mut counts = BTreeMap::<u8, u16>::new();
    for ready in &context.ready {
        if !ids.insert(ready.id) {
            return Err(LedgerError::InvalidContext);
        }
        let actor = initial
            .board
            .reveal
            .actors
            .iter()
            .find(|a| a.position == ready.position)
            .ok_or(LedgerError::InvalidContext)?;
        let count = counts.entry(ready.position).or_default();
        *count += 1;
        if *count > actor.remaining_continuations {
            return Err(LedgerError::InvalidContext);
        }
    }
    let mut permutations = Vec::new();
    orders(
        &mut context.ready.clone(),
        &mut Vec::new(),
        &mut permutations,
    );
    let mut result = Vec::new();
    let mut retained = 0usize;
    let mut total_paths = 0usize;
    for schedule in permutations {
        let mut paths = vec![initial.clone()];
        for (ordinal, ready) in schedule.iter().enumerate() {
            let mut next = Vec::new();
            let mut stage_entries = retained;
            for previous in paths {
                let actor = previous
                    .board
                    .reveal
                    .actors
                    .iter()
                    .find(|a| a.position == ready.position)
                    .ok_or(LedgerError::InvalidContext)?;
                // These are local simulation ordinals, not captured native event
                // provenance. Acquisition is derived independently on each path.
                let event = ResumeEvent {
                    position: ready.position,
                    resume_ordinal: ordinal as u16,
                    acquisition_ordinal: (!matches!(actor.bluff, BluffReference::Live { .. }))
                        .then_some(ordinal as u16),
                };
                let input = RevealWriterContext {
                    rule_version: REVEAL_WRITER_VIEW_NATIVE_V2.into(),
                    board: previous.board.clone(),
                    resumes: vec![event],
                    ui: previous.ui.clone(),
                };
                for mut path in replay_reveal_writers(&input)? {
                    path.probability = previous
                        .probability
                        .multiply(path.probability.numerator, path.probability.denominator)?;
                    let mut trace = previous.trace.clone();
                    trace.append(&mut path.trace);
                    path.trace = trace;
                    stage_entries = stage_entries
                        .checked_add(entries(&path))
                        .ok_or(LedgerError::Capacity)?;
                    if stage_entries > 1_048_576 || total_paths + next.len() >= 65_536 {
                        return Err(LedgerError::Capacity);
                    }
                    next.push(path);
                }
            }
            paths = next;
        }
        let mut completed = Vec::new();
        for path in paths {
            let mut new_continuations = BTreeMap::new();
            for actor in &path.board.reveal.actors {
                let old = initial
                    .board
                    .reveal
                    .actors
                    .iter()
                    .find(|a| a.position == actor.position)
                    .ok_or(LedgerError::InvalidContext)?;
                let deferred =
                    old.remaining_continuations - counts.get(&actor.position).copied().unwrap_or(0);
                let created = actor
                    .remaining_continuations
                    .checked_sub(deferred)
                    .ok_or(LedgerError::InvalidContext)?;
                if created > 0 {
                    new_continuations.insert(actor.position, created);
                }
            }
            retained = retained
                .checked_add(entries(&path) + new_continuations.len())
                .ok_or(LedgerError::Capacity)?;
            total_paths += 1;
            if retained > 1_048_576 || total_paths > 65_536 {
                return Err(LedgerError::Capacity);
            }
            completed.push(ReadyBatchPath {
                replay: path,
                new_continuations,
            });
        }
        result.push(ReadySchedule {
            order: schedule.iter().map(|r| r.id).collect(),
            paths: completed,
        });
    }
    Ok(result)
}
