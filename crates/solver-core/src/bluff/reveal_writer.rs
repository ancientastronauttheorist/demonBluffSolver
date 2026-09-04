//! Ordered offline Reveal projection through acquisition, Start writers and
//! Init/AfterRoundStart; v2 also carries bounded UI writes through replacements
//! and the Reveal tail. The caller proves resume order and absence of omitted
//! mutations. No scheduler timing or Unity lifecycle behavior is inferred.
use super::character_start::{
    replay_character_start, CharacterStartContext, StartCallTrace, CHARACTER_START_NATIVE_V1,
};
use super::ledger::{LedgerError, Probability};
use super::reveal::{callbacks, replay_acquisition, CallbackTrace, ResumeEvent, ResumeTrace};
use super::reveal_view::{
    replay_reveal_view, RevealViewContext, ViewWrite, VisualSource, REVEAL_VIEW_NATIVE_V1,
};
use super::twin_writer::{retained_entries, validate_board, TwinWriterContext};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REVEAL_WRITER_NATIVE_V1: &str = "ordered_reveal_writer_native_v1";
pub const REVEAL_WRITER_VIEW_NATIVE_V2: &str = "ordered_reveal_writer_view_native_v2";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewUiState {
    pub pickable_active: bool,
    pub rip_active: bool,
    /// Explicit null means absent/destroyed; omission is not provenance.
    #[serde(deserialize_with = "required_icon")]
    pub disguise_icon_active: Option<bool>,
}

fn required_icon<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<bool>, D::Error> {
    Option::<bool>::deserialize(d)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplacementViewTrace {
    pub position: u8,
    pub rip_write: Option<bool>,
    pub disguise_write: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RevealTailTrace {
    pub name_art_source: VisualSource,
    pub final_color_source: VisualSource,
    pub writes: Vec<ViewWrite>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealWriterContext {
    pub rule_version: String,
    /// Full board with empty embedded resumes and copied_slot false.
    pub board: TwinWriterContext,
    /// Caller-proven chronological events; identical provenance must be valid
    /// on every positive-probability path, or the entire replay fails.
    pub resumes: Vec<ResumeEvent>,
    /// V2 requires exactly one explicit UI snapshot per physical body.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ui: BTreeMap<u8, ViewUiState>,
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub replacement_views: Vec<ReplacementViewTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub view: Option<RevealTailTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RevealWriterPath {
    pub probability: Probability,
    pub board: TwinWriterContext,
    pub trace: Vec<WriterResumeTrace>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub ui: BTreeMap<u8, ViewUiState>,
}

fn push(
    paths: &mut Vec<RevealWriterPath>,
    path: RevealWriterPath,
    entries: &mut usize,
) -> Result<(), LedgerError> {
    let count = retained_entries(&path.board)
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
    replacement_views: Vec<ReplacementViewTrace>,
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
    let position = actor.position;
    let view = if let Some(ui) = path.ui.get_mut(&position) {
        let body = path
            .board
            .bodies
            .get_mut(&position)
            .ok_or(LedgerError::InvalidContext)?;
        let result = replay_reveal_view(&RevealViewContext {
            rule_version: REVEAL_VIEW_NATIVE_V1.into(),
            raw_bluff: actor.bluff,
            body: body.clone(),
            pickable_active: ui.pickable_active,
            rip_active: ui.rip_active,
            disguise_icon_active: ui.disguise_icon_active,
        })?;
        *body = result.context.body;
        ui.pickable_active = result.context.pickable_active;
        ui.rip_active = result.context.rip_active;
        ui.disguise_icon_active = result.context.disguise_icon_active;
        Some(RevealTailTrace {
            name_art_source: result.name_art_source,
            final_color_source: result.final_color_source,
            writes: result.writes,
        })
    } else {
        None
    };
    path.trace.push(WriterResumeTrace {
        acquisition,
        start,
        callbacks,
        replacement_views,
        view,
    });
    Ok(path)
}

pub fn replay_reveal_writers(
    context: &RevealWriterContext,
) -> Result<Vec<RevealWriterPath>, LedgerError> {
    if ![REVEAL_WRITER_NATIVE_V1, REVEAL_WRITER_VIEW_NATIVE_V2]
        .contains(&context.rule_version.as_str())
        || context.board.copied_slot
        || context.resumes.len() > 16
    {
        return Err(LedgerError::InvalidContext);
    }
    validate_board(&context.board)?;
    if (context.rule_version == REVEAL_WRITER_NATIVE_V1 && !context.ui.is_empty())
        || (context.rule_version == REVEAL_WRITER_VIEW_NATIVE_V2
            && context.ui.keys().copied().collect::<BTreeSet<_>>()
                != context.board.bodies.keys().copied().collect())
    {
        return Err(LedgerError::InvalidContext);
    }
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
            ui: context.ui.clone(),
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
                        let mut replacement_views = Vec::new();
                        let mut replaced = BTreeSet::new();
                        for replacement in started
                            .callbacks
                            .iter()
                            .filter_map(|c| c.twin.as_ref())
                            .flat_map(|t| &t.replacements)
                        {
                            if let Some(ui) = result.ui.get_mut(&replacement.position) {
                                // InitWithNoReset hides RIP only when destroying a
                                // live death presentation. Repeated replacements
                                // see the cleared pointer. Hidden-state refreshes
                                // then hide the optional disguise icon; uses=1
                                // leaves pickable activity unchanged.
                                let first = replaced.insert(replacement.position);
                                let rip_write = (first
                                    && branch.board.bodies[&replacement.position]
                                        .created_dead_presentation)
                                    .then_some(false);
                                if let Some(active) = rip_write {
                                    ui.rip_active = active;
                                }
                                let disguise_write = ui.disguise_icon_active.map(|_| false);
                                if let Some(active) = disguise_write {
                                    ui.disguise_icon_active = Some(active);
                                }
                                replacement_views.push(ReplacementViewTrace {
                                    position: replacement.position,
                                    rip_write,
                                    disguise_write,
                                });
                            }
                        }
                        result.board = started.board;
                        let trace = RevealStartTrace {
                            initial_lying: started.initial_lying,
                            callbacks: started.callbacks,
                        };
                        push(
                            &mut next,
                            finish(result, acquisition.clone(), Some(trace), replacement_views)?,
                            &mut entries,
                        )?;
                    }
                } else {
                    push(
                        &mut next,
                        finish(branch, acquisition, None, vec![])?,
                        &mut entries,
                    )?;
                }
            }
        }
        paths = next;
    }
    Ok(paths)
}
