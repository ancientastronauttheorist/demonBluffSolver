//! Guarded offline replay of the two death-tutorial MoveNext callers.
//! Resumes are explicit inputs. Class initialization, list membership, Unity
//! transform lookup and Show acceptance are independent supplied outcomes.
//! This does not implement tutorial UI, saving, scheduling or exception paths.
//! Actor bytes are an explicit 0x1B8 diagnostic record. Only icon +0x20 and
//! statuses +0xF0 are typed here; unconsumed sentinel bytes remain opaque.

use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const TUTORIAL_DEATH_GENERATORS_NATIVE_V1: &str = "tutorial_death_generators_native_v1";
const MAX_WORK: usize = 1_048_576;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Kill,
    Poison,
}
impl Kind {
    fn offsets(self) -> (usize, usize) {
        if self == Self::Kill {
            (0x20, 0x28)
        } else {
            (0x28, 0x20)
        }
    }
    fn index(self) -> usize {
        if self == Self::Kill {
            0
        } else {
            1
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageKind {
    Routine,
    Actor,
    Statuses,
    List,
    Array,
    Class,
    Static,
    Transform,
    Controller,
    Wait,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Storage {
    pub identity: u64,
    pub kind: StorageKind,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Routine {
    pub identity: u64,
    pub kind: Kind,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub storage: Vec<Storage>,
    pub routines: Vec<Routine>,
    /// Kill MoveNext, Poison MoveNext, CharacterStatuses.Contains byte flags.
    pub metadata: [u8; 3],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassOutcome {
    pub receiver: u64,
    pub initialized_word: u32,
    pub retained_gameplay_state: i32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContainsOutcome {
    pub receiver: u64,
    pub status: i32,
    pub method_info: u64,
    pub returned: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformOutcome {
    pub receiver: u64,
    pub method_info: u64,
    pub returned: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShowOutcome {
    pub controller: u64,
    pub tutorial_type: i32,
    pub pivot: u64,
    pub method_info: u64,
    /// Only the represented caller storage is promised unchanged; no general
    /// Show implementation or persistence behavior follows from acceptance.
    pub accepted_normally_with_caller_storage_retained: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resume {
    pub routine: u64,
    pub fresh_wait: Option<Storage>,
    pub class_initializer: Option<ClassOutcome>,
    pub contains: Option<ContainsOutcome>,
    pub transform: Option<TransformOutcome>,
    pub show: Option<ShowOutcome>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub native_layout_and_metadata_verified: bool,
    pub native_wait_and_base_constructor_verified: bool,
    pub metadata_and_gc_preserve_caller_storage: bool,
    pub no_external_mutation_failure_or_implicit_resume: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub gameplay_class: u64,
    pub wait_class: u64,
    pub status_contains_method: u64,
    pub state: State,
    pub services: Services,
    pub resumes: Vec<Resume>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub routine: u64,
    pub returned: bool,
    pub show: Option<ShowOutcome>,
    pub state: State,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replay {
    pub steps: Vec<Step>,
    pub final_state: State,
}

fn bad<T>() -> Result<T, LedgerError> {
    Err(LedgerError::InvalidContext)
}
fn word(o: &Storage, a: usize) -> u32 {
    u32::from_le_bytes(o.bytes[a..a + 4].try_into().unwrap())
}
fn pointer(o: &Storage, a: usize) -> u64 {
    u64::from_le_bytes(o.bytes[a..a + 8].try_into().unwrap())
}
fn put_word(o: &mut Storage, a: usize, v: u32) {
    o.bytes[a..a + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_pointer(o: &mut Storage, a: usize, v: u64) {
    o.bytes[a..a + 8].copy_from_slice(&v.to_le_bytes());
}
fn object(s: &State, id: u64, kind: StorageKind) -> Result<&Storage, LedgerError> {
    s.storage
        .iter()
        .find(|o| o.identity == id && o.kind == kind)
        .ok_or(LedgerError::InvalidContext)
}
fn object_mut(s: &mut State, id: u64) -> &mut Storage {
    s.storage.iter_mut().find(|o| o.identity == id).unwrap()
}
fn validate_storage(o: &Storage) -> Result<(), LedgerError> {
    let size = match o.kind {
        StorageKind::Actor => 0x1b8,
        StorageKind::Class => 0x180,
        StorageKind::Static => 0x100,
        StorageKind::Controller => 0x40,
        StorageKind::List => 0x28,
        StorageKind::Array => {
            if o.bytes.len() < 0x20 || (o.bytes.len() - 0x20) % 4 != 0 {
                return bad();
            }
            let slots = (o.bytes.len() - 0x20) / 4;
            if slots > 4096 || pointer(o, 0x18) != slots as u64 {
                return bad();
            }
            o.bytes.len()
        }
        _ => 0x80,
    };
    if o.identity == 0 || o.bytes.len() != size {
        return bad();
    }
    Ok(())
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let g = &c.services;
    if c.version != TUTORIAL_DEATH_GENERATORS_NATIVE_V1
        || c.status_contains_method == 0
        || c.wait_class == 0
        || c.wait_class == c.status_contains_method
        || !g.native_layout_and_metadata_verified
        || !g.native_wait_and_base_constructor_verified
        || !g.metadata_and_gc_preserve_caller_storage
        || !g.no_external_mutation_failure_or_implicit_resume
    {
        return bad();
    }
    if c.resumes.len() > 32 || c.state.routines.len() > 16 || c.state.storage.len() > 128 {
        return Err(LedgerError::Capacity);
    }
    let mut size = 0usize;
    let mut slots = 0usize;
    for o in c
        .state
        .storage
        .iter()
        .chain(c.resumes.iter().filter_map(|r| r.fresh_wait.as_ref()))
    {
        validate_storage(o)?;
        size = size
            .checked_add(o.bytes.len())
            .ok_or(LedgerError::Capacity)?;
        if o.kind == StorageKind::Array {
            slots = slots
                .checked_add((o.bytes.len() - 0x20) / 4)
                .ok_or(LedgerError::Capacity)?;
        }
    }
    // Reserve the full retained snapshot volume, including future allocations,
    // before cloning any physical bytes or constructing output snapshots.
    if slots > 4096
        || size
            .checked_mul(c.resumes.len() + 2)
            .ok_or(LedgerError::Capacity)?
            > MAX_WORK
    {
        return Err(LedgerError::Capacity);
    }
    // Allocation-free shape and snapshot budget pass above precedes identity
    // maps as well as every retained-byte clone.
    let mut ids = BTreeSet::new();
    for o in c
        .state
        .storage
        .iter()
        .chain(c.resumes.iter().filter_map(|r| r.fresh_wait.as_ref()))
    {
        if !ids.insert(o.identity) {
            return bad();
        }
        if o.kind == StorageKind::Wait && pointer(o, 0) != c.wait_class {
            return bad();
        }
    }
    if ids.contains(&c.status_contains_method) || ids.contains(&c.wait_class) {
        return bad();
    }
    let s = &c.state;
    let class = object(s, c.gameplay_class, StorageKind::Class)?;
    object(s, pointer(class, 0xb8), StorageKind::Static)?;
    let mut routines = BTreeSet::new();
    for r in &s.routines {
        if !routines.insert(r.identity) {
            return bad();
        }
        let o = object(s, r.identity, StorageKind::Routine)?;
        let (controller, character) = r.kind.offsets();
        object(s, pointer(o, controller), StorageKind::Controller)?;
        let actor = object(s, pointer(o, character), StorageKind::Actor)?;
        object(s, pointer(actor, 0x20), StorageKind::Transform)?;
        object(s, pointer(actor, 0xf0), StorageKind::Statuses)?;
        if pointer(o, 0x18) != 0 {
            object(s, pointer(o, 0x18), StorageKind::Wait)?;
        }
    }
    for o in &s.storage {
        if o.kind == StorageKind::Statuses {
            object(s, pointer(o, 0x10), StorageKind::List)?;
            object(s, pointer(o, 0x18), StorageKind::List)?;
            if pointer(o, 0x20) != 0 {
                object(s, pointer(o, 0x20), StorageKind::Actor)?;
            }
        }
        if o.kind == StorageKind::List {
            let a = object(s, pointer(o, 0x10), StorageKind::Array)?;
            if word(o, 0x18) as usize > (a.bytes.len() - 0x20) / 4 {
                return bad();
            }
        }
    }
    for r in &c.resumes {
        if !routines.contains(&r.routine) {
            return bad();
        }
    }
    Ok(())
}

pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut s = c.state.clone();
    let mut steps = Vec::with_capacity(c.resumes.len());
    for call in &c.resumes {
        let kind = s
            .routines
            .iter()
            .find(|r| r.identity == call.routine)
            .unwrap()
            .kind;
        if s.metadata[kind.index()] == 0 {
            s.metadata[kind.index()] = 1;
        }
        let initial = word(object(&s, call.routine, StorageKind::Routine)?, 0x10) as i32;
        let mut expected_wait = false;
        let mut expected_class = false;
        let mut expected_contains = false;
        let mut expected_transform = false;
        let mut expected_show = false;
        let mut returned = false;
        if initial == 0 {
            expected_wait = true;
            let wait = call
                .fresh_wait
                .as_ref()
                .ok_or(LedgerError::InvalidContext)?;
            if wait.kind != StorageKind::Wait
                || pointer(wait, 0) != c.wait_class
                || wait.bytes[8..].iter().any(|b| *b != 0)
            {
                return bad();
            }
            put_word(object_mut(&mut s, call.routine), 0x10, u32::MAX);
            let mut wait = wait.clone();
            put_word(&mut wait, 0x10, 0x3e4ccccd);
            s.storage.push(wait);
            put_pointer(
                object_mut(&mut s, call.routine),
                0x18,
                call.fresh_wait.as_ref().unwrap().identity,
            );
            put_word(object_mut(&mut s, call.routine), 0x10, 1);
            returned = true;
        } else if initial == 1 {
            put_word(object_mut(&mut s, call.routine), 0x10, u32::MAX);
            let class = object(&s, c.gameplay_class, StorageKind::Class)?;
            let static_id = pointer(class, 0xb8);
            let gameplay = word(object(&s, static_id, StorageKind::Static)?, 0x28) as i32;
            if word(class, 0xe0) == 0 {
                expected_class = true;
                let outcome = call
                    .class_initializer
                    .as_ref()
                    .ok_or(LedgerError::InvalidContext)?;
                if outcome.receiver != c.gameplay_class
                    || outcome.initialized_word != 1
                    || outcome.retained_gameplay_state != gameplay
                {
                    return bad();
                }
                put_word(object_mut(&mut s, c.gameplay_class), 0xe0, 1);
            }
            if gameplay != 50 {
                let (controller_offset, actor_offset) = kind.offsets();
                let actor_id = pointer(
                    object(&s, call.routine, StorageKind::Routine)?,
                    actor_offset,
                );
                let mut eligible = true;
                if kind == Kind::Poison {
                    expected_contains = true;
                    if s.metadata[2] == 0 {
                        s.metadata[2] = 1;
                    }
                    let statuses = object(
                        &s,
                        pointer(object(&s, actor_id, StorageKind::Actor)?, 0xf0),
                        StorageKind::Statuses,
                    )?;
                    let list_id = pointer(statuses, 0x10);
                    let list = object(&s, list_id, StorageKind::List)?;
                    let array = object(&s, pointer(list, 0x10), StorageKind::Array)?;
                    eligible = (0..word(list, 0x18) as usize)
                        .any(|i| word(array, 0x20 + i * 4) as i32 == 10);
                    let outcome = call.contains.as_ref().ok_or(LedgerError::InvalidContext)?;
                    if outcome.receiver != list_id
                        || outcome.status != 10
                        || outcome.method_info != c.status_contains_method
                        || outcome.returned != eligible
                    {
                        return bad();
                    }
                }
                if eligible {
                    expected_transform = true;
                    expected_show = true;
                    let icon = pointer(object(&s, actor_id, StorageKind::Actor)?, 0x20);
                    let transform = call.transform.as_ref().ok_or(LedgerError::InvalidContext)?;
                    if transform.receiver != icon || transform.method_info != 0 {
                        return bad();
                    }
                    if transform.returned != 0 {
                        object(&s, transform.returned, StorageKind::Transform)?;
                    }
                    let controller = pointer(
                        object(&s, call.routine, StorageKind::Routine)?,
                        controller_offset,
                    );
                    let show = call.show.as_ref().ok_or(LedgerError::InvalidContext)?;
                    if show.controller != controller
                        || show.pivot != transform.returned
                        || show.tutorial_type != if kind == Kind::Kill { 100 } else { 45 }
                        || show.method_info != 0
                        || !show.accepted_normally_with_caller_storage_retained
                    {
                        return bad();
                    }
                }
            }
        }
        if call.fresh_wait.is_some() != expected_wait
            || call.class_initializer.is_some() != expected_class
            || call.contains.is_some() != expected_contains
            || call.transform.is_some() != expected_transform
            || call.show.is_some() != expected_show
        {
            return bad();
        }
        steps.push(Step {
            routine: call.routine,
            returned,
            show: call.show.clone(),
            state: s.clone(),
        });
    }
    Ok(Replay {
        steps,
        final_state: s,
    })
}

#[cfg(test)]
#[path = "tutorial_death_generators_tests.rs"]
mod tests;
