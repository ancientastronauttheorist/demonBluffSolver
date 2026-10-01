//! OnHover and HideDescription caller projections with explicit inert services.
//!
//! Metadata/class operations are excluded from the semantic trace, including
//! supported cold projections; runtime provenance is independently supplied.
//! Full logical Actor semantics are retained inputs. Native A5 fixture padding
//! does not establish valid typed values for its unused Actor reference fields.
//! No Acted.Act/DisableHighlightAll body, coroutine behavior or Unity effect is
//! inferred. Action header/identity and normal supplied effects are explicit.

use super::character_initialization::{Actor, Identity};
use super::character_refresh_view::UiObject;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHARACTER_DESCRIPTION_HIDE_NATIVE_V1: &str = "character_description_hide_native_v1";
const MAX_CALLS: usize = 32;
const MAX_RETAINED: usize = 16_384;
const MAX_WORK: usize = 262_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    OnHover,
    HideDescription,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Selected physical delegate, target and MethodInfo consumed by this caller.
/// The callable pointer and its implementation remain supplied provenance.
pub struct Action {
    pub identity: Identity,
    pub target: Option<Identity>,
    pub method_info: Identity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bindings {
    pub left_acted: Option<Identity>,
    pub left_version: Option<Identity>,
    pub left_game_object: Option<Identity>,
    pub acted: Option<Identity>,
    pub characters_static: Option<Identity>,
    pub characters: Option<Identity>,
    pub ui_events_static: Option<Identity>,
    pub hide_custom_hint: Option<Action>,
    pub hide_hint: Option<Action>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestoredSpeech {
    pub component: Identity,
    pub text: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub actor: Actor,
    /// Native byte gates preserve noncanonical values until an explicit store.
    pub hover_bits: u8,
    pub left_bits: u8,
    pub ui_objects: Vec<UiObject>,
    pub stopped_components: Vec<Identity>,
    pub restored_speech: Vec<RestoredSpeech>,
    pub highlight_clears: Vec<Identity>,
    pub action_calls: Vec<Action>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub retained_actor_projection_verified: bool,
    pub native_runtime_and_bindings_verified: bool,
    pub metadata_and_class_state_verified: bool,
    pub object_bindings_verified: bool,
    pub callbacks_and_services_inert: bool,
    pub normal_completion_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub bindings: Bindings,
    pub calls: Vec<Method>,
    pub services: Services,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    StoreHover {
        actor: Identity,
        bits: u8,
    },
    StopAllCoroutines {
        component: Identity,
    },
    GetGameObject {
        component: Identity,
        result: Identity,
    },
    SetActive {
        object: Identity,
        active: bool,
    },
    ActedAct {
        component: Identity,
        text: Option<Identity>,
    },
    DisableHighlightAll {
        characters: Identity,
    },
    HideCustomHint {
        action: Action,
    },
    HideHint {
        action: Action,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Step {
    pub call: usize,
    pub event: Event,
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallResult {
    pub method: Method,
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub state: State,
    pub steps: Vec<Step>,
    pub calls: Vec<CallResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Character,
    Data,
    Trailer,
    Runtime,
    Role,
    Action,
    String,
    ActedInfo,
    GameObject,
    Acted,
    ActedVersion,
    Characters,
    CharactersStatic,
    UiStatic,
    MethodInfo,
}

fn bind(
    types: &mut BTreeMap<Identity, Kind>,
    id: Option<Identity>,
    kind: Kind,
) -> Result<(), LedgerError> {
    if let Some(id) = id {
        if id == 0 || types.get(&id).is_some_and(|old| *old != kind) {
            return Err(LedgerError::InvalidContext);
        }
        types.insert(id, kind);
    }
    Ok(())
}

fn budget(c: &Context) -> Result<(), LedgerError> {
    // Reserve all future retained effects and every whole-state snapshot before
    // constructing maps or cloning any input. Fixed scalar/reference units and
    // worst-case eight steps per call deliberately over-reserve short branches.
    let mut units = 48usize;
    for (len, weight) in [
        (c.state.actor.infos.len(), 1),
        (c.state.actor.statuses.active.len(), 1),
        (c.state.actor.statuses.resistances.len(), 1),
        (c.state.ui_objects.len(), 2),
        (c.state.stopped_components.len(), 1),
        (c.state.restored_speech.len(), 2),
        (c.state.highlight_clears.len(), 1),
        (c.state.action_calls.len(), 3),
    ] {
        units = units
            .checked_add(len.checked_mul(weight).ok_or(LedgerError::Capacity)?)
            .ok_or(LedgerError::Capacity)?;
    }
    let future = c.calls.len().checked_mul(10).ok_or(LedgerError::Capacity)?;
    units = units.checked_add(future).ok_or(LedgerError::Capacity)?;
    let snapshots = c
        .calls
        .len()
        .checked_mul(9)
        .and_then(|n| n.checked_add(2))
        .ok_or(LedgerError::Capacity)?;
    if c.calls.len() > MAX_CALLS
        || units > MAX_RETAINED
        || units.checked_mul(snapshots).ok_or(LedgerError::Capacity)? > MAX_WORK
    {
        return Err(LedgerError::Capacity);
    }
    Ok(())
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    budget(c)?;
    let s = &c.services;
    let hides = c.calls.contains(&Method::HideDescription);
    if c.version != CHARACTER_DESCRIPTION_HIDE_NATIVE_V1
        || c.calls.is_empty()
        || !s.retained_actor_projection_verified
        || !s.normal_completion_verified
        || (hides
            && (!s.native_runtime_and_bindings_verified
                || !s.metadata_and_class_state_verified
                || !s.object_bindings_verified
                || !s.callbacks_and_services_inert))
    {
        return Err(LedgerError::InvalidContext);
    }
    let a = &c.state.actor;
    let b = &c.bindings;
    let mut types = BTreeMap::new();
    for (id, kind) in [
        (Some(a.identity), Kind::Character),
        (a.data, Kind::Data),
        (a.bluff, Kind::Data),
        (a.register_as, Kind::Data),
        (a.trailer, Kind::Trailer),
        (a.runtime, Kind::Runtime),
        (a.role, Kind::Role),
        (a.bluff_role, Kind::Role),
        (a.saved_act, Kind::String),
        (a.state_callback, Kind::Action),
        (a.statuses.target, Kind::Character),
        (a.dead_prefab.as_ref().map(|p| p.identity), Kind::GameObject),
        (b.left_acted, Kind::Acted),
        (b.acted, Kind::Acted),
        (b.left_version, Kind::ActedVersion),
        (b.left_game_object, Kind::GameObject),
        (b.characters, Kind::Characters),
        (b.characters_static, Kind::CharactersStatic),
        (b.ui_events_static, Kind::UiStatic),
    ] {
        bind(&mut types, id, kind)?;
    }
    for id in &a.infos {
        bind(&mut types, *id, Kind::ActedInfo)?;
    }
    let mut ui = BTreeMap::new();
    for object in &c.state.ui_objects {
        bind(&mut types, Some(object.identity), Kind::GameObject)?;
        if ui.insert(object.identity, object.active).is_some() {
            return Err(LedgerError::InvalidContext);
        }
    }
    for id in &c.state.stopped_components {
        bind(&mut types, Some(*id), Kind::Acted)?;
    }
    for speech in &c.state.restored_speech {
        bind(&mut types, Some(speech.component), Kind::Acted)?;
        bind(&mut types, speech.text, Kind::String)?;
    }
    for id in &c.state.highlight_clears {
        bind(&mut types, Some(*id), Kind::Characters)?;
    }
    let mut headers = BTreeMap::new();
    for action in [b.hide_custom_hint.as_ref(), b.hide_hint.as_ref()]
        .into_iter()
        .flatten()
    {
        bind(&mut types, Some(action.identity), Kind::Action)?;
        bind(&mut types, Some(action.method_info), Kind::MethodInfo)?;
        if headers
            .insert(action.identity, *action)
            .is_some_and(|old| old != *action)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    for action in &c.state.action_calls {
        bind(&mut types, Some(action.identity), Kind::Action)?;
        bind(&mut types, Some(action.method_info), Kind::MethodInfo)?;
        if headers
            .insert(action.identity, *action)
            .is_some_and(|old| old != *action)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    for action in headers.values() {
        if action.target == Some(0)
            || action.target.is_some_and(|p| {
                matches!(
                    types.get(&p),
                    Some(Kind::MethodInfo | Kind::CharactersStatic | Kind::UiStatic)
                )
            })
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if hides
        && (b.characters.is_none()
            || b.characters_static.is_none()
            || b.ui_events_static.is_none()
            || (c.state.left_bits != 0
                && (b.left_acted.is_none()
                    || b.left_version.is_none()
                    || b.acted.is_none()
                    || b.left_game_object.is_none()
                    || !ui.contains_key(&b.left_game_object.unwrap()))))
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

fn step(out: &mut Replay, call: usize, event: Event) {
    out.steps.push(Step {
        call,
        event,
        state: out.state.clone(),
    });
}

pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut out = Replay {
        state: c.state.clone(),
        steps: vec![],
        calls: vec![],
    };
    let b = &c.bindings;
    for (index, method) in c.calls.iter().enumerate() {
        match method {
            Method::OnHover => {
                out.state.hover_bits = 1;
                let actor = out.state.actor.identity;
                step(&mut out, index, Event::StoreHover { actor, bits: 1 });
            }
            Method::HideDescription => {
                if out.state.left_bits != 0 {
                    let left = b.left_acted.unwrap();
                    let version = b.left_version.unwrap();
                    let object = b.left_game_object.unwrap();
                    step(
                        &mut out,
                        index,
                        Event::StopAllCoroutines { component: left },
                    );
                    out.state.stopped_components.push(left);
                    step(
                        &mut out,
                        index,
                        Event::GetGameObject {
                            component: version,
                            result: object,
                        },
                    );
                    step(
                        &mut out,
                        index,
                        Event::SetActive {
                            object,
                            active: false,
                        },
                    );
                    out.state
                        .ui_objects
                        .iter_mut()
                        .find(|p| p.identity == object)
                        .unwrap()
                        .active = false;
                    let component = b.acted.unwrap();
                    let text = out.state.actor.saved_act;
                    step(&mut out, index, Event::ActedAct { component, text });
                    out.state
                        .restored_speech
                        .push(RestoredSpeech { component, text });
                }
                let characters = b.characters.unwrap();
                step(&mut out, index, Event::DisableHighlightAll { characters });
                out.state.highlight_clears.push(characters);
                if let Some(action) = b.hide_custom_hint {
                    step(&mut out, index, Event::HideCustomHint { action });
                    out.state.action_calls.push(action);
                }
                if let Some(action) = b.hide_hint {
                    step(&mut out, index, Event::HideHint { action });
                    out.state.action_calls.push(action);
                }
            }
        }
        out.calls.push(CallResult {
            method: *method,
            state: out.state.clone(),
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "character_description_hide_tests.rs"]
mod tests;
