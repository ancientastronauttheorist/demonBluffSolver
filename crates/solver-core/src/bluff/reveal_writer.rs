//! Ordered offline Reveal projection through acquisition, Start writers and
//! Init/AfterRoundStart. The caller proves resume order and absence of omitted
//! mutations (including UI epilogues). No scheduler timing is inferred.
use super::character_start::{
    replay_character_start, CharacterStartContext, StartCallTrace, CHARACTER_START_NATIVE_V1,
};
use super::ledger::{LedgerError, Probability};
use super::reveal::{callbacks, replay_acquisition, CallbackTrace, ResumeEvent, ResumeTrace};
use super::twin_writer::{retained_entries, validate_board, TwinWriterContext};
use serde::{Deserialize, Serialize};

pub const REVEAL_WRITER_NATIVE_V1: &str = "ordered_reveal_writer_native_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealWriterContext {
    pub rule_version: String,
    /// Full board with empty embedded resumes and copied_slot false.
    pub board: TwinWriterContext,
    /// Caller-proven chronological events; identical provenance must be valid
    /// on every positive-probability path, or the entire replay fails.
    pub resumes: Vec<ResumeEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RevealStartTrace {
    /// None means HealthyBluff requested Start but the latch suppressed it.
    pub initial_lying: Option<bool>,
    pub callbacks: Vec<StartCallTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WriterResumeTrace {
    /// Register-as, selector and GiveBluff effects; callbacks are empty here.
    pub acquisition: ResumeTrace,
    /// None means HealthyBluff was absent after acquisition.
    pub start: Option<RevealStartTrace>,
    pub callbacks: Vec<CallbackTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RevealWriterPath {
    pub probability: Probability,
    pub board: TwinWriterContext,
    pub trace: Vec<WriterResumeTrace>,
}

fn push(
    paths: &mut Vec<RevealWriterPath>,
    path: RevealWriterPath,
    entries: &mut usize,
) -> Result<(), LedgerError> {
    let count = retained_entries(&path.board)
        + path
            .trace
            .iter()
            .map(|t| {
                8 + t.callbacks.len()
                    + t.start.as_ref().map_or(0, |s| {
                        1 + s
                            .callbacks
                            .iter()
                            .map(|c| 2 + c.twin.as_ref().map_or(0, |t| 1 + t.replacements.len()))
                            .sum::<usize>()
                    })
            })
            .sum::<usize>();
    *entries = entries.checked_add(count).ok_or(LedgerError::Capacity)?;
    if paths.len() >= 65_536 || *entries > 1_048_576 {
        return Err(LedgerError::Capacity);
    }
    paths.push(path);
    Ok(())
}

fn finish(
    mut path: RevealWriterPath,
    acquisition: ResumeTrace,
    start: Option<RevealStartTrace>,
) -> Result<RevealWriterPath, LedgerError> {
    let actor = path
        .board
        .reveal
        .actors
        .iter_mut()
        .find(|a| a.position == acquisition.event.position)
        .ok_or(LedgerError::InvalidContext)?;
    // Read current real/copied identities after any reinitializer. Do not
    // repeat Start because its latch was reset by the just-completed writer.
    let callbacks = callbacks(actor, false)?;
    path.trace.push(WriterResumeTrace {
        acquisition,
        start,
        callbacks,
    });
    Ok(path)
}

pub fn replay_reveal_writers(
    context: &RevealWriterContext,
) -> Result<Vec<RevealWriterPath>, LedgerError> {
    if context.rule_version != REVEAL_WRITER_NATIVE_V1
        || context.board.copied_slot
        || context.resumes.len() > 16
    {
        return Err(LedgerError::InvalidContext);
    }
    validate_board(&context.board)?;
    let mut previous_resume = None;
    let mut previous_acquisition = None;
    for event in &context.resumes {
        if !context.board.bodies.contains_key(&event.position)
            || previous_resume.is_some_and(|n| n >= event.resume_ordinal)
            || event
                .acquisition_ordinal
                .is_some_and(|n| previous_acquisition.is_some_and(|p| p >= n))
        {
            return Err(LedgerError::InvalidContext);
        }
        previous_resume = Some(event.resume_ordinal);
        if event.acquisition_ordinal.is_some() {
            previous_acquisition = event.acquisition_ordinal;
        }
    }
    let mut paths = Vec::new();
    push(
        &mut paths,
        RevealWriterPath {
            probability: Probability {
                numerator: 1,
                denominator: 1,
            },
            board: context.board.clone(),
            trace: vec![],
        },
        &mut 0,
    )?;
    for event in &context.resumes {
        let mut next = Vec::new();
        let mut entries = 0;
        for path in paths {
            let mut input = path.board.reveal.clone();
            input.resumes = vec![event.clone()];
            for acquired in replay_acquisition(&input)? {
                let mut branch = path.clone();
                branch.probability = branch.probability.multiply(
                    acquired.probability.numerator,
                    acquired.probability.denominator,
                )?;
                branch.board.position = event.position;
                branch.board.reveal.actors = acquired.actors;
                branch.board.reveal.pools = acquired.pools;
                branch.board.reveal.spy_caches = acquired.spy_caches;
                let acquisition = acquired
                    .trace
                    .into_iter()
                    .next()
                    .ok_or(LedgerError::InvalidContext)?;
                let healthy = branch
                    .board
                    .reveal
                    .actors
                    .iter()
                    .find(|a| a.position == event.position)
                    .ok_or(LedgerError::InvalidContext)?
                    .statuses
                    .values
                    .contains(&30);
                if healthy {
                    for started in replay_character_start(&CharacterStartContext {
                        rule_version: CHARACTER_START_NATIVE_V1.into(),
                        board: branch.board.clone(),
                    })? {
                        let mut result = branch.clone();
                        result.probability = result.probability.multiply(
                            started.probability.numerator,
                            started.probability.denominator,
                        )?;
                        result.board = started.board;
                        let trace = RevealStartTrace {
                            initial_lying: started.initial_lying,
                            callbacks: started.callbacks,
                        };
                        push(
                            &mut next,
                            finish(result, acquisition.clone(), Some(trace))?,
                            &mut entries,
                        )?;
                    }
                } else {
                    push(&mut next, finish(branch, acquisition, None)?, &mut entries)?;
                }
            }
        }
        paths = next;
    }
    Ok(paths)
}
