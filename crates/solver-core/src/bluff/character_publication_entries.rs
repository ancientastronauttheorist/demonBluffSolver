//! Six normal native entry callers with independently supplied inert services.
//!
//! Events describe caller stores, barriers and UI/service order. Metadata and
//! class initialization are excluded, including on supported cold projections;
//! their validity is supplied provenance rather than a reconstructed trace.
//! No iterator is resumed. Registration results are verified fresh authored
//! objects; the void ShowActed caller does not publish their return value.
//! Known physical identities must have compatible nominal field types, including
//! untouched retained Actor fields. Compatible asset/GameObject/target aliases
//! and repeated layout occurrences remain legal; no runtime cast is inferred.

use super::character_initialization::{Actor, Identity};
use super::character_refresh_view::UiObject;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CHARACTER_PUBLICATION_ENTRIES_NATIVE_V1: &str = "character_publication_entries_native_v1";
const MAX_CALLS: usize = 32;
const MAX_RETAINED: usize = 16_384;
const MAX_WORK: usize = 262_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    DelayReveal,
    DelayedDemonKill,
    ShowActed,
    ShowInfoDelayed,
    ShowTrailerAct,
    HideActed,
}

impl Method {
    fn factory(self) -> bool {
        matches!(
            self,
            Self::DelayReveal | Self::DelayedDemonKill | Self::ShowActed | Self::ShowInfoDelayed
        )
    }
    fn capture_only(self) -> bool {
        matches!(
            self,
            Self::DelayReveal | Self::DelayedDemonKill | Self::ShowInfoDelayed
        )
    }
    fn argument_kind(self) -> Option<ArgumentKind> {
        match self {
            Self::DelayedDemonKill => Some(ArgumentKind::Character),
            Self::ShowActed => Some(ArgumentKind::ActedInfo),
            Self::ShowInfoDelayed | Self::ShowTrailerAct => Some(ArgumentKind::String),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Signature-based supplied argument declarations; no native cast or class
/// inspection is inferred from these capture-only fields or UI string inputs.
pub enum ArgumentKind {
    Character,
    ActedInfo,
    String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Argument {
    pub identity: Identity,
    pub kind: ArgumentKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Iterator {
    pub identity: Identity,
    pub method: Method,
    pub state: i32,
    pub current: Option<Identity>,
    pub owner: Option<Identity>,
    pub argument: Option<Identity>,
    pub delay_bits: u32,
    pub trigger_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registration {
    pub iterator: Identity,
    pub result: Identity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Show {
    pub version: Identity,
    pub text: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Components {
    pub acted_component: Option<Identity>,
    pub game_object: Option<Identity>,
    pub version: Option<Identity>,
    pub layout_array: Option<Identity>,
    /// Ordered occurrences; aliases cause repeated native rebuild requests.
    pub layouts: Vec<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub actor: Actor,
    pub ui_objects: Vec<UiObject>,
    pub iterators: Vec<Iterator>,
    pub registrations: Vec<Registration>,
    pub shown: Vec<Show>,
    pub layout_rebuilds: Vec<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub method: Method,
    pub owner: Option<Identity>,
    pub argument: Option<Identity>,
    pub trigger_bits: u32,
    pub delay_bits: u32,
    pub fresh_iterator: Option<Identity>,
    pub registration_result: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub native_runtime_and_bindings_verified: bool,
    pub metadata_and_class_state_verified: bool,
    pub zeroed_fresh_allocations_verified: bool,
    pub empty_base_body_verified: bool,
    pub arguments_and_ui_bindings_verified: bool,
    pub gc_and_other_services_inert: bool,
    pub registration_without_resume_verified: bool,
    pub fresh_registration_results_verified: bool,
    pub ui_services_and_callbacks_inert: bool,
    pub normal_completion_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub components: Components,
    pub arguments: Vec<Argument>,
    pub calls: Vec<Call>,
    pub services: Services,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Allocate {
        iterator: Identity,
        method: Method,
    },
    EmptyBase {
        iterator: Identity,
    },
    StoreOwner {
        iterator: Identity,
        owner: Option<Identity>,
    },
    StoreStateZero {
        iterator: Identity,
    },
    StoreArgument {
        iterator: Identity,
        argument: Option<Identity>,
    },
    StoreDelay {
        iterator: Identity,
        bits: u32,
    },
    StoreTrigger {
        iterator: Identity,
        bits: u32,
    },
    Barrier {
        iterator: Identity,
        offset: u32,
        value: Option<Identity>,
    },
    FactoryReturn {
        iterator: Identity,
    },
    Register {
        owner: Identity,
        iterator: Identity,
        result: Identity,
    },
    GetGameObject {
        component: Identity,
        result: Identity,
    },
    SetActive {
        object: Identity,
        active: bool,
    },
    ReloadActed {
        component: Identity,
    },
    ShowVersion {
        version: Identity,
        text: Option<Identity>,
    },
    RebuildLayout {
        object: Identity,
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
    pub returned_iterator: Option<Identity>,
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub state: State,
    pub steps: Vec<Step>,
    pub calls: Vec<CallResult>,
}

fn nonzero_optional(values: impl IntoIterator<Item = Option<Identity>>) -> bool {
    values.into_iter().all(|p| p != Some(0))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NominalType {
    Character,
    CharacterData,
    CharacterTrailerInfo,
    RuntimeCharacterData,
    Role,
    Action,
    String,
    ActedInfo,
    GameObject,
    Acted,
    ActedVersion,
    LayoutArray,
    RectTransform,
    Iterator(Method),
    Coroutine,
}

fn bind_type(
    types: &mut BTreeMap<Identity, NominalType>,
    identity: Option<Identity>,
    kind: NominalType,
) -> Result<(), LedgerError> {
    if let Some(id) = identity {
        if id == 0 || types.get(&id).is_some_and(|old| *old != kind) {
            return Err(LedgerError::InvalidContext);
        }
        types.insert(id, kind);
    }
    Ok(())
}

fn validate_retained_types(c: &Context) -> Result<(), LedgerError> {
    // Called only after the complete retained/snapshot budget was accepted.
    let a = &c.state.actor;
    let mut types = BTreeMap::new();
    for (id, kind) in [
        (Some(a.identity), NominalType::Character),
        (a.data, NominalType::CharacterData),
        (a.bluff, NominalType::CharacterData),
        (a.register_as, NominalType::CharacterData),
        (a.trailer, NominalType::CharacterTrailerInfo),
        (a.runtime, NominalType::RuntimeCharacterData),
        (a.role, NominalType::Role),
        (a.bluff_role, NominalType::Role),
        (a.state_callback, NominalType::Action),
        (a.saved_act, NominalType::String),
        (a.statuses.target, NominalType::Character),
        (
            a.dead_prefab.as_ref().map(|p| p.identity),
            NominalType::GameObject,
        ),
        (c.components.acted_component, NominalType::Acted),
        (c.components.version, NominalType::ActedVersion),
        (c.components.layout_array, NominalType::LayoutArray),
    ] {
        bind_type(&mut types, id, kind)?;
    }
    for id in &a.infos {
        bind_type(&mut types, *id, NominalType::ActedInfo)?;
    }
    for object in &c.state.ui_objects {
        bind_type(&mut types, Some(object.identity), NominalType::GameObject)?;
    }
    for id in &c.components.layouts {
        bind_type(&mut types, Some(*id), NominalType::RectTransform)?;
    }
    for arg in &c.arguments {
        bind_type(
            &mut types,
            Some(arg.identity),
            match arg.kind {
                ArgumentKind::Character => NominalType::Character,
                ArgumentKind::ActedInfo => NominalType::ActedInfo,
                ArgumentKind::String => NominalType::String,
            },
        )?;
    }
    for iterator in &c.state.iterators {
        bind_type(
            &mut types,
            Some(iterator.identity),
            NominalType::Iterator(iterator.method),
        )?;
    }
    for registration in &c.state.registrations {
        bind_type(
            &mut types,
            Some(registration.result),
            NominalType::Coroutine,
        )?;
    }
    Ok(())
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    // Reserve all future retained records and whole-state snapshots before maps/clones.
    let mut retained = 1usize;
    for count in [
        c.state.actor.infos.len(),
        c.state.actor.statuses.active.len(),
        c.state.actor.statuses.resistances.len(),
        c.state.ui_objects.len(),
        c.state.iterators.len(),
        c.state.registrations.len(),
        c.state.shown.len(),
        c.state.layout_rebuilds.len(),
        c.components.layouts.len(),
        c.arguments.len(),
    ] {
        retained = retained.checked_add(count).ok_or(LedgerError::Capacity)?;
    }
    let mut steps = 0usize;
    for call in &c.calls {
        retained = retained
            .checked_add(if call.method.factory() { 2 } else { 1 })
            .and_then(|n| {
                if call.method == Method::ShowTrailerAct {
                    n.checked_add(c.components.layouts.len())
                } else {
                    Some(n)
                }
            })
            .ok_or(LedgerError::Capacity)?;
        steps = steps
            .checked_add(12)
            .and_then(|n| {
                if call.method == Method::ShowTrailerAct {
                    n.checked_add(c.components.layouts.len())
                } else {
                    Some(n)
                }
            })
            .ok_or(LedgerError::Capacity)?;
    }
    let snapshots = steps
        .checked_add(c.calls.len())
        .and_then(|n| n.checked_add(2))
        .ok_or(LedgerError::Capacity)?;
    if c.calls.len() > MAX_CALLS
        || retained > MAX_RETAINED
        || retained
            .checked_mul(snapshots)
            .ok_or(LedgerError::Capacity)?
            > MAX_WORK
    {
        return Err(LedgerError::Capacity);
    }
    let s = &c.services;
    if c.version != CHARACTER_PUBLICATION_ENTRIES_NATIVE_V1
        || c.calls.is_empty()
        || !s.native_runtime_and_bindings_verified
        || !s.metadata_and_class_state_verified
        || !s.zeroed_fresh_allocations_verified
        || !s.empty_base_body_verified
        || !s.arguments_and_ui_bindings_verified
        || !s.gc_and_other_services_inert
        || !s.registration_without_resume_verified
        || !s.fresh_registration_results_verified
        || !s.ui_services_and_callbacks_inert
        || !s.normal_completion_verified
        || c.state.actor.identity == 0
    {
        return Err(LedgerError::InvalidContext);
    }
    let actor = &c.state.actor;
    let opaque = [
        actor.data,
        actor.bluff,
        actor.register_as,
        actor.trailer,
        actor.runtime,
        actor.role,
        actor.bluff_role,
        actor.state_callback,
    ];
    if !nonzero_optional(
        opaque
            .into_iter()
            .chain([actor.saved_act, actor.statuses.target])
            .chain(actor.infos.iter().copied())
            .chain(actor.dead_prefab.as_ref().map(|p| Some(p.identity))),
    ) {
        return Err(LedgerError::InvalidContext);
    }
    let ui: BTreeMap<_, _> = c.state.ui_objects.iter().map(|o| (o.identity, o)).collect();
    let mut typed = BTreeSet::from([actor.identity]);
    for id in c.state.ui_objects.iter().map(|o| o.identity).chain(
        [
            c.components.acted_component,
            c.components.version,
            c.components.layout_array,
        ]
        .into_iter()
        .flatten(),
    ) {
        if id == 0 || !typed.insert(id) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let layouts: BTreeSet<_> = c.components.layouts.iter().copied().collect();
    for id in &layouts {
        if *id == 0 || !typed.insert(*id) {
            return Err(LedgerError::InvalidContext);
        }
    }
    if !nonzero_optional([
        c.components.acted_component,
        c.components.game_object,
        c.components.version,
        c.components.layout_array,
    ]) || c
        .components
        .game_object
        .is_some_and(|id| !ui.contains_key(&id))
        || (c.components.layout_array.is_none() && !c.components.layouts.is_empty())
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut arguments = BTreeMap::new();
    for arg in &c.arguments {
        if arg.identity == 0
            || arguments.insert(arg.identity, arg.kind).is_some()
            || (typed.contains(&arg.identity)
                && !(arg.identity == actor.identity && arg.kind == ArgumentKind::Character))
            || opaque.contains(&Some(arg.identity))
            || (actor.saved_act == Some(arg.identity) && arg.kind != ArgumentKind::String)
            || (actor.statuses.target == Some(arg.identity) && arg.kind != ArgumentKind::Character)
            || (actor.infos.contains(&Some(arg.identity)) && arg.kind != ArgumentKind::ActedInfo)
            || actor
                .dead_prefab
                .as_ref()
                .is_some_and(|p| p.identity == arg.identity)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    arguments.insert(actor.identity, ArgumentKind::Character);
    validate_retained_types(c)?;
    let mut retained_ids = typed;
    retained_ids.extend(opaque.into_iter().flatten());
    retained_ids.extend(
        [actor.saved_act, actor.statuses.target]
            .into_iter()
            .flatten(),
    );
    retained_ids.extend(actor.infos.iter().flatten().copied());
    retained_ids.extend(actor.dead_prefab.as_ref().map(|p| p.identity));
    retained_ids.extend(arguments.keys().copied());
    for it in &c.state.iterators {
        if !it.method.factory()
            || !nonzero_optional([it.current, it.owner, it.argument])
            || (it.method != Method::ShowActed && (it.delay_bits != 0 || it.trigger_bits != 0))
            || (it.method == Method::DelayReveal && it.argument.is_some())
            || it
                .owner
                .is_some_and(|id| arguments.get(&id) != Some(&ArgumentKind::Character))
            || it
                .argument
                .is_some_and(|id| arguments.get(&id).copied() != it.method.argument_kind())
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    // Existing currents are opaque retained outputs, not fresh allocations here.
    retained_ids.extend(c.state.iterators.iter().filter_map(|it| it.current));
    let mut iterators = BTreeSet::new();
    for it in &c.state.iterators {
        if it.identity == 0 || retained_ids.contains(&it.identity) || !iterators.insert(it.identity)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    retained_ids.extend(iterators.iter().copied());
    for reg in &c.state.registrations {
        if c.state
            .iterators
            .iter()
            .find(|it| it.identity == reg.iterator)
            .is_none_or(|it| it.method != Method::ShowActed)
            || reg.result == 0
            || !retained_ids.insert(reg.result)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if c.state.shown.iter().any(|show| {
        Some(show.version) != c.components.version
            || show
                .text
                .is_some_and(|id| arguments.get(&id) != Some(&ArgumentKind::String))
    }) || c
        .state
        .layout_rebuilds
        .iter()
        .any(|id| !layouts.contains(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    for call in &c.calls {
        if !nonzero_optional([
            call.owner,
            call.argument,
            call.fresh_iterator,
            call.registration_result,
        ]) || call.owner.is_some_and(|id| id != actor.identity)
            || (!call.method.capture_only() && call.owner != Some(actor.identity))
            || call
                .argument
                .is_some_and(|id| arguments.get(&id).copied() != call.method.argument_kind())
            || (call.method.argument_kind().is_none() && call.argument.is_some())
            || (call.method != Method::ShowActed
                && (call.delay_bits != 0 || call.trigger_bits != 0))
            || call.method.factory() != call.fresh_iterator.is_some()
            || (call.method == Method::ShowActed) != call.registration_result.is_some()
            || (matches!(call.method, Method::ShowTrailerAct | Method::HideActed)
                && (c.components.acted_component.is_none() || c.components.game_object.is_none()))
            || (call.method == Method::ShowTrailerAct
                && (c.components.version.is_none() || c.components.layout_array.is_none()))
        {
            return Err(LedgerError::InvalidContext);
        }
        for id in [call.fresh_iterator, call.registration_result]
            .into_iter()
            .flatten()
        {
            if !retained_ids.insert(id) {
                return Err(LedgerError::InvalidContext);
            }
        }
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
    for (index, call) in c.calls.iter().enumerate() {
        let mut returned = None;
        if let Some(id) = call.fresh_iterator {
            step(
                &mut out,
                index,
                Event::Allocate {
                    iterator: id,
                    method: call.method,
                },
            );
            out.state.iterators.push(Iterator {
                identity: id,
                method: call.method,
                state: 0,
                current: None,
                owner: None,
                argument: None,
                delay_bits: 0,
                trigger_bits: 0,
            });
            step(&mut out, index, Event::EmptyBase { iterator: id });
            out.state.iterators.last_mut().unwrap().owner = call.owner;
            step(
                &mut out,
                index,
                Event::StoreOwner {
                    iterator: id,
                    owner: call.owner,
                },
            );
            step(&mut out, index, Event::StoreStateZero { iterator: id });
            let owner_offset = if call.method == Method::ShowActed {
                0x28
            } else {
                0x20
            };
            step(
                &mut out,
                index,
                Event::Barrier {
                    iterator: id,
                    offset: owner_offset,
                    value: call.owner,
                },
            );
            if call.method.argument_kind().is_some() {
                out.state.iterators.last_mut().unwrap().argument = call.argument;
                step(
                    &mut out,
                    index,
                    Event::StoreArgument {
                        iterator: id,
                        argument: call.argument,
                    },
                );
                if call.method == Method::ShowActed {
                    out.state.iterators.last_mut().unwrap().delay_bits = call.delay_bits;
                    step(
                        &mut out,
                        index,
                        Event::StoreDelay {
                            iterator: id,
                            bits: call.delay_bits,
                        },
                    );
                }
                step(
                    &mut out,
                    index,
                    Event::Barrier {
                        iterator: id,
                        offset: owner_offset + 8,
                        value: call.argument,
                    },
                );
            }
            if call.method == Method::ShowActed {
                out.state.iterators.last_mut().unwrap().trigger_bits = call.trigger_bits;
                step(
                    &mut out,
                    index,
                    Event::StoreTrigger {
                        iterator: id,
                        bits: call.trigger_bits,
                    },
                );
            }
            step(&mut out, index, Event::FactoryReturn { iterator: id });
            if let Some(result) = call.registration_result {
                step(
                    &mut out,
                    index,
                    Event::Register {
                        owner: call.owner.unwrap(),
                        iterator: id,
                        result,
                    },
                );
                out.state.registrations.push(Registration {
                    iterator: id,
                    result,
                });
            } else {
                returned = Some(id);
            }
        } else {
            let component = c.components.acted_component.unwrap();
            let object = c.components.game_object.unwrap();
            step(
                &mut out,
                index,
                Event::GetGameObject {
                    component,
                    result: object,
                },
            );
            let active = call.method == Method::ShowTrailerAct;
            step(&mut out, index, Event::SetActive { object, active });
            out.state
                .ui_objects
                .iter_mut()
                .find(|o| o.identity == object)
                .unwrap()
                .active = active;
            if active {
                step(&mut out, index, Event::ReloadActed { component });
                let version = c.components.version.unwrap();
                step(
                    &mut out,
                    index,
                    Event::ShowVersion {
                        version,
                        text: call.argument,
                    },
                );
                out.state.shown.push(Show {
                    version,
                    text: call.argument,
                });
                for object in &c.components.layouts {
                    step(&mut out, index, Event::RebuildLayout { object: *object });
                    out.state.layout_rebuilds.push(*object);
                }
            }
        }
        out.calls.push(CallResult {
            method: call.method,
            returned_iterator: returned,
            state: out.state.clone(),
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "character_publication_entries_tests.rs"]
mod tests;
