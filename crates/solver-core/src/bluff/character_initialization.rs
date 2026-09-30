//! Bounded Character.Init / InitWithNoReset through native first-yield publication.
//!
//! Offline only. Required objects/lists are valid; callbacks and presentation
//! services are inert. The clone result and synchronous first-yield handoff are
//! explicit inputs. This does not infer Unity scheduling or run role actions.

use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CHARACTER_INITIALIZATION_NATIVE_V1: &str = "character_initialization_native_v1";
pub type Identity = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Init,
    InitWithNoReset,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectReference {
    pub identity: Identity,
    /// A retained destroyed Unity object is distinct from an absent pointer.
    pub live: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statuses {
    pub active: Vec<i32>,
    pub version: u32,
    pub resistances: Vec<i32>,
    pub target: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Actor {
    pub identity: Identity,
    pub data: Option<Identity>,
    pub bluff: Option<Identity>,
    pub register_as: Option<Identity>,
    pub trailer: Option<Identity>,
    pub runtime: Option<Identity>,
    pub dead_prefab: Option<ObjectReference>,
    pub revealed: bool,
    pub uses: i32,
    pub previous: i32,
    pub state: i32,
    pub killed_hidden: bool,
    pub killed_demon: bool,
    pub alignment: i32,
    pub id: i32,
    pub started: bool,
    pub role: Option<Identity>,
    pub bluff_role: Option<Identity>,
    pub saved_act: Option<Identity>,
    pub infos: Vec<Option<Identity>>,
    pub info_version: u32,
    pub statuses: Statuses,
    pub state_callback: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Continuation {
    pub identity: Identity,
    pub actor: Identity,
    pub state: i32,
    pub current: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub method: Method,
    pub actor: Actor,
    pub data: Identity,
    pub starting_alignment: i32,
    /// Native -100 preserves the existing displayed and stored ID.
    pub id: i32,
    pub required_objects_and_lists_valid: bool,
    pub callbacks_and_ui_inert: bool,
    pub clone_result_verified: bool,
    pub synchronous_first_yield_verified: bool,
    pub clone_result: Option<Identity>,
    pub continuation_identity: Identity,
    pub wait_identity: Identity,
    /// Existing continuations are retained; initialization cancels none.
    pub continuations: Vec<Continuation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    ClearTrailer,
    HideActed,
    ClearInfos,
    ClearRuntime,
    ClearStartGuard,
    HideRip,
    DestroyDeathPresentation,
    ClearBluff,
    ClearRevealed,
    StoreData,
    DiagnosticLog,
    ClearRegisterAs,
    ResetKilledDemonAndUses,
    StoreStartingAlignment,
    SetNumberText,
    StoreId,
    SetHidden,
    StateCallback,
    ClearStatuses,
    RefreshCharacter,
    RefreshView,
    RegisterContinuation,
    PublishRole,
    FirstYield,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub actor: Actor,
    /// The inert callback's exact actor observation precedes active-status clear.
    pub callback_observation: Option<Actor>,
    pub continuations: Vec<Continuation>,
    pub events: Vec<Event>,
    /// Bits of the native float32 literal, avoiding JSON decimal rounding.
    pub wait_seconds_f32_bits: u32,
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    if c.version != CHARACTER_INITIALIZATION_NATIVE_V1
        || !c.required_objects_and_lists_valid
        || !c.callbacks_and_ui_inert
        || !c.clone_result_verified
        || !c.synchronous_first_yield_verified
        || [
            c.actor.identity,
            c.data,
            c.continuation_identity,
            c.wait_identity,
        ]
        .contains(&0)
        || c.clone_result == Some(0)
        || c.continuation_identity == c.wait_identity
        || [c.actor.identity, c.data].contains(&c.continuation_identity)
        || [c.actor.identity, c.data].contains(&c.wait_identity)
    {
        return Err(LedgerError::InvalidContext);
    }
    if [
        c.actor.data,
        c.actor.bluff,
        c.actor.register_as,
        c.actor.trailer,
        c.actor.runtime,
        c.actor.role,
        c.actor.bluff_role,
        c.actor.saved_act,
        c.actor.state_callback,
        c.actor.statuses.target,
    ]
    .contains(&Some(0))
        || c.actor.infos.contains(&Some(0))
        || c.actor
            .dead_prefab
            .as_ref()
            .is_some_and(|p| p.identity == 0)
    {
        return Err(LedgerError::InvalidContext);
    }
    if c.actor.infos.len() > 4096
        || c.actor.statuses.active.len() > 4096
        || c.actor.statuses.resistances.len() > 4096
        || c.continuations.len() >= 4096
    {
        return Err(LedgerError::Capacity);
    }
    let mut ids = BTreeSet::new();
    let mut retained = BTreeSet::from([c.actor.identity, c.data]);
    retained.extend(
        [
            c.actor.data,
            c.actor.bluff,
            c.actor.register_as,
            c.actor.trailer,
            c.actor.runtime,
            c.actor.role,
            c.actor.bluff_role,
            c.actor.saved_act,
            c.actor.state_callback,
            c.actor.statuses.target,
            c.clone_result,
        ]
        .into_iter()
        .flatten(),
    );
    retained.extend(c.actor.infos.iter().flatten().copied());
    retained.extend(c.actor.dead_prefab.as_ref().map(|p| p.identity));
    for pending in &c.continuations {
        if pending.identity == 0
            || pending.actor == 0
            || !ids.insert(pending.identity)
            || pending.identity == c.continuation_identity
            || pending.identity == c.wait_identity
            || pending.current == Some(0)
        {
            return Err(LedgerError::InvalidContext);
        }
        retained.extend([pending.identity, pending.actor]);
        retained.extend(pending.current);
    }
    if retained.contains(&c.continuation_identity) || retained.contains(&c.wait_identity) {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut actor = c.actor.clone();
    let mut events = Vec::new();
    let reset = c.method == Method::Init;
    if reset {
        actor.trailer = None;
        events.push(Event::ClearTrailer);
    }
    events.push(Event::HideActed);
    actor.infos.clear();
    actor.info_version = actor.info_version.wrapping_add(1);
    events.push(Event::ClearInfos);
    if reset {
        actor.runtime = None;
        events.push(Event::ClearRuntime);
    }
    actor.started = false;
    events.push(Event::ClearStartGuard);
    if actor.dead_prefab.as_ref().is_some_and(|p| p.live) {
        events.extend([Event::HideRip, Event::DestroyDeathPresentation]);
        actor.dead_prefab = None;
    }
    actor.bluff = None;
    events.push(Event::ClearBluff);
    if !reset {
        actor.revealed = false;
        events.push(Event::ClearRevealed);
    }
    actor.data = Some(c.data);
    events.extend([Event::StoreData, Event::DiagnosticLog]);
    if reset {
        actor.register_as = None;
        actor.revealed = false;
        events.extend([Event::ClearRegisterAs, Event::ClearRevealed]);
    }
    actor.killed_demon = false;
    actor.uses = 1;
    events.push(Event::ResetKilledDemonAndUses);
    if reset {
        actor.alignment = c.starting_alignment;
        events.push(Event::StoreStartingAlignment);
    }
    if c.id != -100 {
        events.push(Event::SetNumberText);
        actor.id = c.id;
        events.push(Event::StoreId);
    }
    actor.previous = actor.state;
    actor.state = 5;
    events.push(Event::SetHidden);
    let callback_observation = actor.state_callback.map(|_| {
        events.push(Event::StateCallback);
        actor.clone()
    });
    if reset {
        actor.statuses.active.clear();
        actor.statuses.version = actor.statuses.version.wrapping_add(1);
        events.push(Event::ClearStatuses);
    }
    // Hidden and positive uses bound both refreshes to no additional actor writes.
    // UI writes are outside Actor; required engine objects are explicitly valid.
    events.extend([
        Event::RefreshCharacter,
        Event::RefreshView,
        Event::RegisterContinuation,
    ]);
    let mut continuations = c.continuations.clone();
    actor.role = c.clone_result;
    events.push(Event::PublishRole);
    continuations.push(Continuation {
        identity: c.continuation_identity,
        actor: actor.identity,
        state: 1,
        current: Some(c.wait_identity),
    });
    events.push(Event::FirstYield);
    Ok(Replay {
        actor,
        callback_observation,
        continuations,
        events,
        wait_seconds_f32_bits: 0x3e99_999a,
    })
}

#[cfg(test)]
#[path = "character_initialization_tests.rs"]
mod tests;
