//! Offline tutorial publication callers. Runtime allocation, metadata, GC,
//! class initialization and registration are supplied stable services. No
//! generated routine executes and no engine scheduling is inferred.

use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const TUTORIAL_HANDLER_PUBLICATION_NATIVE_V1: &str = "tutorial_handler_publication_native_v1";
pub const METADATA_RVAS: [&str; 10] = [
    "0x288c30d",
    "0x288c309",
    "0x288c307",
    "0x288c30c",
    "0x288c30b",
    "0x288c308",
    "0x288c30a",
    "0x288c305",
    "0x288c306",
    "0x288c304",
];
const MAX_CALLS: usize = 32;
const MAX_RETAINED: usize = 16_384;
const MAX_WORK: usize = 262_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Start,
    Info,
    Killed,
    Level,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Reveal,
    Kill,
    HoverObjective,
    CancelKill,
    CharacterInfo,
    CharacterKill,
    PoisonKilled,
    SecondLevel,
    ThirdLevel,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Reveal => "TutorialsController.<RevealCardTutorial>d__19_TypeInfo",
            Self::Kill => "TutorialsController.<KillTutorial>d__20_TypeInfo",
            Self::HoverObjective => "TutorialsController.<HoverObjective>d__21_TypeInfo",
            Self::CancelKill => "TutorialsController.<CancelKillTutorial>d__22_TypeInfo",
            Self::CharacterInfo => "TutorialsController.<CharacterInfoTutorial>d__18_TypeInfo",
            Self::CharacterKill => "TutorialsController.<CharacterKillRoutine>d__11_TypeInfo",
            Self::PoisonKilled => "TutorialsController.<PoisonKilledRoutine>d__12_TypeInfo",
            Self::SecondLevel => "TutorialsController.<SecondLevelNoteCoroutine>d__8_TypeInfo",
            Self::ThirdLevel => "TutorialsController.<ThirdLevelNoteCoroutine>d__9_TypeInfo",
        }
    }
    fn metadata_index(self) -> usize {
        match self {
            Self::Reveal => 6,
            Self::Kill => 4,
            Self::HoverObjective => 3,
            Self::CancelKill => 0,
            Self::CharacterInfo => 1,
            Self::CharacterKill => 2,
            Self::PoisonKilled => 5,
            Self::SecondLevel => 7,
            Self::ThirdLevel => 8,
        }
    }
    pub fn controller_offset(self) -> u32 {
        if self == Self::PoisonKilled {
            0x28
        } else {
            0x20
        }
    }
    pub fn character_offset(self) -> Option<u32> {
        match self {
            Self::CharacterInfo | Self::CharacterKill => Some(0x28),
            Self::PoisonKilled => Some(0x20),
            _ => None,
        }
    }
    pub fn has_closure(self) -> bool {
        matches!(self, Self::Kill | Self::HoverObjective)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Routine {
    pub identity: u64,
    pub kind: Kind,
    pub state: i32,
    pub current: Option<u64>,
    pub controller: Option<u64>,
    pub character: Option<u64>,
    pub closure: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetainedStorage {
    pub identity: u64,
    pub offset: u32,
    /// Unconsumed storage is retained byte-for-byte, never decoded as a runtime object.
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub routines: Vec<Routine>,
    pub publications: Vec<u64>,
    pub retained: Vec<RetainedStorage>,
    /// Cancel, Info, CharacterKill, HoverObjective, Kill, Poison, Reveal,
    /// SecondLevel, ThirdLevel and Level-handler metadata flags, in that order.
    pub metadata_bytes: [u8; 10],
    pub class_initialized_word: u32,
    pub current_level: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub native_bindings_and_layouts_verified: bool,
    pub metadata_and_class_state_verified: bool,
    pub zeroed_fresh_allocations_and_base_verified: bool,
    pub gc_and_retained_storage_inert: bool,
    pub class_initializer_sets_one_and_preserves_level: bool,
    pub registration_acceptance_without_resume_verified: bool,
    pub callbacks_and_gameplay_inert: bool,
    pub normal_completion_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub method: Method,
    pub fresh_routines: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub controller: Option<u64>,
    pub character: Option<u64>,
    pub gameplay_class: u64,
    pub gameplay_static: u64,
    pub gameplay_instance: Option<u64>,
    pub state: State,
    pub services: Services,
    pub calls: Vec<Call>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Metadata {
        name: String,
    },
    ClassInitialize {
        class: u64,
    },
    Allocate {
        routine: u64,
        routine_kind: Kind,
    },
    Barrier {
        routine: u64,
        offset: u32,
        value: Option<u64>,
    },
    Register {
        controller: Option<u64>,
        routine: u64,
        routine_kind: Kind,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub event: Event,
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replay {
    pub state: State,
    pub steps: Vec<Step>,
}

fn kinds(method: Method, level: i32) -> &'static [Kind] {
    match method {
        Method::Start => &[
            Kind::Reveal,
            Kind::Kill,
            Kind::HoverObjective,
            Kind::CancelKill,
        ],
        Method::Info => &[Kind::CharacterInfo],
        Method::Killed => &[Kind::CharacterKill, Kind::PoisonKilled],
        Method::Level if level == 1 => &[Kind::SecondLevel],
        Method::Level if level == 2 => &[Kind::ThirdLevel],
        Method::Level => &[],
    }
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    // Reserve the complete future state and all projected snapshots before
    // creating maps/sets or cloning retained bytes.
    let mut retained = c
        .state
        .routines
        .len()
        .checked_mul(8)
        .and_then(|n| n.checked_add(c.state.publications.len()))
        .and_then(|n| n.checked_add(c.state.retained.len()))
        .and_then(|n| n.checked_add(16))
        .ok_or(LedgerError::Capacity)?;
    for object in &c.state.retained {
        retained = retained
            .checked_add(object.bytes.len())
            .ok_or(LedgerError::Capacity)?;
    }
    for call in &c.calls {
        retained = retained
            .checked_add(
                call.fresh_routines
                    .len()
                    .checked_mul(9)
                    .ok_or(LedgerError::Capacity)?,
            )
            .ok_or(LedgerError::Capacity)?;
    }
    let snapshots = c
        .calls
        .len()
        .checked_mul(22)
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
    if c.version != TUTORIAL_HANDLER_PUBLICATION_NATIVE_V1
        || c.calls.is_empty()
        || !s.native_bindings_and_layouts_verified
        || !s.metadata_and_class_state_verified
        || !s.zeroed_fresh_allocations_and_base_verified
        || !s.gc_and_retained_storage_inert
        || !s.class_initializer_sets_one_and_preserves_level
        || !s.registration_acceptance_without_resume_verified
        || !s.callbacks_and_gameplay_inert
        || !s.normal_completion_verified
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut objects = BTreeSet::new();
    for id in [
        Some(c.gameplay_class),
        Some(c.gameplay_static),
        c.gameplay_instance,
        c.controller,
        c.character,
    ]
    .into_iter()
    .flatten()
    {
        if id == 0 || !objects.insert(id) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let mut retained_objects = BTreeSet::new();
    for object in &c.state.retained {
        let end = u64::from(object.offset)
            .checked_add(object.bytes.len() as u64)
            .ok_or(LedgerError::Capacity)?;
        if object.identity == 0
            || !retained_objects.insert(object.identity)
            || end > u64::from(u32::MAX) + 1
            || object.identity == c.gameplay_class && object.offset < 0xE4 && end > 0xE0
            || object.identity == c.gameplay_class && object.offset < 0xC0 && end > 0xB8
            || object.identity == c.gameplay_static && object.offset < 0x18 && end > 0x10
            || Some(object.identity) == c.gameplay_instance && object.offset < 0x7C && end > 0x78
        {
            return Err(LedgerError::InvalidContext);
        }
        // A retained byte image can describe one of the declared physical roots.
        objects.insert(object.identity);
    }
    let mut existing = BTreeSet::new();
    for r in &c.state.routines {
        if r.identity == 0
            || !existing.insert(r.identity)
            || objects.contains(&r.identity)
            || r.character.is_some() && r.kind.character_offset().is_none()
            || r.closure.is_some() && !r.kind.has_closure()
            || [r.current, r.controller, r.character, r.closure].contains(&Some(0))
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if c.state.publications.iter().any(|id| !existing.contains(id)) {
        return Err(LedgerError::InvalidContext);
    }
    objects.extend(existing);
    let mut nominal = BTreeMap::new();
    let mut bind = |id: Option<u64>, ty: u8| -> Result<(), LedgerError> {
        if let Some(id) = id {
            if id == 0 || nominal.get(&id).is_some_and(|old| *old != ty) {
                return Err(LedgerError::InvalidContext);
            }
            nominal.insert(id, ty);
        }
        Ok(())
    };
    for (id, ty) in [
        (c.controller, 1),
        (c.character, 2),
        (Some(c.gameplay_class), 3),
        (Some(c.gameplay_static), 4),
        (c.gameplay_instance, 5),
    ] {
        bind(id, ty)?;
    }
    for r in &c.state.routines {
        bind(Some(r.identity), 6)?;
    }
    for r in &c.state.routines {
        bind(r.controller, 1)?;
        bind(r.character, 2)?;
        // Two different generated closure classes, both nominally distinct
        // from controllers, Characters, iterators and each other.
        bind(r.closure, if r.kind == Kind::Kill { 7 } else { 8 })?;
    }
    for r in &c.state.routines {
        objects.extend(
            [r.current, r.controller, r.character, r.closure]
                .into_iter()
                .flatten(),
        );
    }
    for call in &c.calls {
        if call.method == Method::Level && c.gameplay_instance.is_none()
            || call.fresh_routines.len() != kinds(call.method, c.state.current_level).len()
        {
            return Err(LedgerError::InvalidContext);
        }
        for id in &call.fresh_routines {
            if *id == 0 || !objects.insert(*id) {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    Ok(())
}

fn step(out: &mut Replay, event: Event) {
    out.steps.push(Step {
        event,
        state: out.state.clone(),
    });
}

fn metadata(out: &mut Replay, index: usize, name: &str) {
    if out.state.metadata_bytes[index] == 0 {
        step(
            out,
            Event::Metadata {
                name: name.to_owned(),
            },
        );
        out.state.metadata_bytes[index] = 1;
    }
}

fn publish(out: &mut Replay, c: &Context, kind: Kind, identity: u64) {
    metadata(out, kind.metadata_index(), kind.name());
    step(
        out,
        Event::Allocate {
            routine: identity,
            routine_kind: kind,
        },
    );
    out.state.routines.push(Routine {
        identity,
        kind,
        state: 0,
        current: None,
        controller: None,
        character: None,
        closure: None,
    });
    let index = out.state.routines.len() - 1;
    // Exact constructor state and native capture layout. Zeroed allocation
    // supplies current/closure, and GC observes each preceding reference store.
    if kind == Kind::PoisonKilled {
        out.state.routines[index].controller = c.controller;
        step(
            out,
            Event::Barrier {
                routine: identity,
                offset: 0x28,
                value: c.controller,
            },
        );
        out.state.routines[index].character = c.character;
        step(
            out,
            Event::Barrier {
                routine: identity,
                offset: 0x20,
                value: c.character,
            },
        );
    } else {
        out.state.routines[index].controller = c.controller;
        step(
            out,
            Event::Barrier {
                routine: identity,
                offset: 0x20,
                value: c.controller,
            },
        );
        if kind.character_offset().is_some() {
            out.state.routines[index].character = c.character;
            step(
                out,
                Event::Barrier {
                    routine: identity,
                    offset: 0x28,
                    value: c.character,
                },
            );
        }
    }
    step(
        out,
        Event::Register {
            controller: c.controller,
            routine: identity,
            routine_kind: kind,
        },
    );
    out.state.publications.push(identity);
}

pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut out = Replay {
        state: c.state.clone(),
        steps: Vec::new(),
    };
    for call in &c.calls {
        if call.method == Method::Level {
            metadata(&mut out, 9, "Gameplay_TypeInfo");
            if out.state.class_initialized_word == 0 {
                step(
                    &mut out,
                    Event::ClassInitialize {
                        class: c.gameplay_class,
                    },
                );
                out.state.class_initialized_word = 1;
            }
        }
        for (kind, id) in kinds(call.method, out.state.current_level)
            .iter()
            .zip(&call.fresh_routines)
        {
            publish(&mut out, c, *kind, *id);
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "tutorial_handler_publication_tests.rs"]
mod tests;
