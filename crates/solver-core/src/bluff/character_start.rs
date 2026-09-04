//! One explicit Character.Act(Start), through real then current copied role.
//! Offline only: no Reveal/resume scheduling or onTrigger/presentation writers.
use super::ledger::{LedgerError, Probability};
use super::reveal::{CallbackRole, Dispatch, RoleSlot, StatusApplication};
use super::twin_writer::{
    replay_twin_start, retained_entries, validate_board, ReplacementTrace, TwinWriterContext,
};
use serde::{Deserialize, Serialize};

pub const CHARACTER_START_NATIVE_V1: &str = "character_start_native_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CharacterStartContext {
    pub rule_version: String,
    /// Explicit full board; copied_slot must be false because Character.Act
    /// selects both slots itself. Pending continuations are not consumed.
    pub board: TwinWriterContext,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TwinCallTrace {
    pub demon_occurrence: Option<usize>,
    pub demon_position: Option<u8>,
    pub neighbor_occurrence: Option<u8>,
    pub rng_draw_count: u8,
    pub replacements: Vec<ReplacementTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StartCallTrace {
    pub slot: RoleSlot,
    pub role: CallbackRole,
    pub dispatch: Dispatch,
    pub status_application: Option<StatusApplication>,
    pub twin: Option<TwinCallTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CharacterStartPath {
    pub probability: Probability,
    pub board: TwinWriterContext,
    /// None means the one-shot guard returned before CheckLying.
    pub initial_lying: Option<bool>,
    pub callbacks: Vec<StartCallTrace>,
}

fn push(
    paths: &mut Vec<CharacterStartPath>,
    path: CharacterStartPath,
    entries: &mut usize,
) -> Result<(), LedgerError> {
    let count = retained_entries(&path.board)
        + path
            .callbacks
            .iter()
            .map(|t| 1 + t.twin.as_ref().map_or(0, |t| 1 + t.replacements.len()))
            .sum::<usize>();
    *entries = entries.checked_add(count).ok_or(LedgerError::Capacity)?;
    if paths.len() >= 65_536 || *entries > 1_048_576 {
        return Err(LedgerError::Capacity);
    }
    paths.push(path);
    Ok(())
}

fn dispatch(
    path: CharacterStartPath,
    slot: RoleSlot,
    role: CallbackRole,
    dispatch: Dispatch,
) -> Result<Vec<CharacterStartPath>, LedgerError> {
    let trace = StartCallTrace {
        slot,
        role,
        dispatch,
        status_application: None,
        twin: None,
    };
    if role == CallbackRole::Spy {
        return Err(LedgerError::InvalidContext);
    }
    if role == CallbackRole::TwinMinion {
        // Twin inherits Role.BluffAct -> concrete Act, so both routes swap.
        let mut input = path.board.clone();
        input.copied_slot = slot == RoleSlot::Bluff;
        let mut result = Vec::new();
        let mut entries = 0;
        for writer in replay_twin_start(&input)? {
            let mut branch = path.clone();
            branch.probability = branch
                .probability
                .multiply(writer.probability.numerator, writer.probability.denominator)?;
            branch.board = writer.context;
            branch.board.copied_slot = false;
            let mut callback = trace.clone();
            callback.twin = Some(TwinCallTrace {
                demon_occurrence: writer.demon_occurrence,
                demon_position: writer.demon_position,
                neighbor_occurrence: writer.neighbor_occurrence,
                rng_draw_count: writer.rng_draw_count,
                replacements: writer.replacements,
            });
            branch.callbacks.push(callback);
            push(&mut result, branch, &mut entries)?;
        }
        return Ok(result);
    }
    let mut path = path;
    let actor = path
        .board
        .reveal
        .actors
        .iter_mut()
        .find(|a| a.position == path.board.position)
        .ok_or(LedgerError::InvalidContext)?;
    let mut callback = trace;
    callback.status_application = match role {
        CallbackRole::Drunk => Some(actor.statuses.apply(10, Some(actor.position))),
        CallbackRole::Lilis => Some(actor.statuses.apply(60, None)),
        CallbackRole::Scout | CallbackRole::Witness | CallbackRole::Confessor => None,
        CallbackRole::TwinMinion | CallbackRole::Spy => unreachable!(),
    };
    path.callbacks.push(callback);
    Ok(vec![path])
}

pub fn replay_character_start(
    context: &CharacterStartContext,
) -> Result<Vec<CharacterStartPath>, LedgerError> {
    if context.rule_version != CHARACTER_START_NATIVE_V1 || context.board.copied_slot {
        return Err(LedgerError::InvalidContext);
    }
    validate_board(&context.board)?;
    let mut path = CharacterStartPath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        board: context.board.clone(),
        initial_lying: None,
        callbacks: vec![],
    };
    let actor = path
        .board
        .reveal
        .actors
        .iter_mut()
        .find(|a| a.position == context.board.position)
        .ok_or(LedgerError::InvalidContext)?;
    if actor.character_start_acted == Some(true) {
        let mut result = Vec::new();
        push(&mut result, path, &mut 0)?;
        return Ok(result);
    }
    // Set before dispatch. A Twin replacement can reset this; do not restore it
    // or recursively restart the newly cloned real role on method return.
    actor.character_start_acted = Some(true);
    let lying = actor.is_lying();
    let real_role = actor.action_role;
    let real_dispatch = if !lying || (actor.runtime_evil && actor.bluff_role.is_some()) {
        Dispatch::Act
    } else {
        Dispatch::BluffAct
    };
    path.initial_lying = Some(lying);
    let mut result = Vec::new();
    let mut entries = 0;
    for branch in dispatch(path, RoleSlot::Real, real_role, real_dispatch)? {
        // Read the current copied slot after real dispatch, even when raw bluff
        // was cleared. Reuse CheckLying from before either slot's mutations.
        let copied = branch
            .board
            .reveal
            .actors
            .iter()
            .find(|a| a.position == context.board.position)
            .ok_or(LedgerError::InvalidContext)?
            .bluff_role;
        if let Some(role) = copied {
            for completed in dispatch(
                branch,
                RoleSlot::Bluff,
                role,
                if lying {
                    Dispatch::BluffAct
                } else {
                    Dispatch::Act
                },
            )? {
                push(&mut result, completed, &mut entries)?;
            }
        } else {
            push(&mut result, branch, &mut entries)?;
        }
    }
    Ok(result)
}
