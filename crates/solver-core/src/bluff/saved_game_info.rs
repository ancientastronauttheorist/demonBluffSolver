//! Guarded SavedGameInfo mutation callers over supplied managed List services.
//!
//! This retains physical List records and their private versions. String
//! equality, growth, constructors, barriers and clearing are explicit contracts;
//! no persistence or cross-emulator object identity is inferred.

use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SAVED_GAME_INFO_NATIVE_V1: &str = "saved_game_info_native_v1";
const MAX_LISTS: usize = 256;
const MAX_CAPACITY: usize = 256;
const MAX_TEXT_BYTES: usize = 4096;
const MAX_TOTAL_SLOTS: usize = 4096;
const MAX_TOTAL_TEXT_BYTES: usize = 1_048_576;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct List {
    pub identity: u64,
    pub count: usize,
    pub version: u32,
    /// None denotes a freshly allocated, not yet constructed List.
    /// Populated Lists require separately verified, unaliased backing storage.
    pub backing: Option<Vec<Option<String>>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub key: Option<String>,
    pub completed_tutorials: Option<u64>,
    pub unlocked_characters: Option<u64>,
    pub lists: Vec<List>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    AddTutorial,
    AddCharacter,
    ClearTutorials,
    ClearUnlockedCharacters,
    Construct,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub runtime_and_metadata_verified: bool,
    pub initialization_and_callbacks_inert: bool,
    pub nullable_text_contains_verified: bool,
    pub backing_storage_unaliased_verified: bool,
    pub array_clear_zeroes_requested_slots: bool,
    pub constructors_publish_empty_lists: bool,
    pub normal_completion_verified: bool,
    /// Supplied growth outcome, required only for an append at full capacity.
    /// This is not an inferred runtime growth algorithm.
    pub resized_capacity: Option<usize>,
    /// Ordered fresh allocation identities, required only for Construct.
    pub constructor_list_identities: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub operation: Operation,
    pub argument: Option<String>,
    pub services: Services,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Contains { list: u64, value: Option<String> },
    ResizeAppend { list: u64, value: Option<String> },
    WriteBarrier,
    ArrayClear { count: usize },
    ListAllocate,
    ListConstructor { list: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Step {
    pub event: Event,
    /// Caller state at service entry, before the supplied service runs.
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub state: State,
    pub steps: Vec<Step>,
}

fn valid_text(value: &Option<String>) -> bool {
    value.as_ref().is_none_or(|s| s.len() <= MAX_TEXT_BYTES)
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    let s = &c.services;
    if c.version != SAVED_GAME_INFO_NATIVE_V1
        || !s.runtime_and_metadata_verified
        || !s.initialization_and_callbacks_inert
        || !s.nullable_text_contains_verified
        || !s.backing_storage_unaliased_verified
        || !s.array_clear_zeroes_requested_slots
        || !s.constructors_publish_empty_lists
        || !s.normal_completion_verified
    {
        return Err(LedgerError::InvalidContext);
    }
    if c.state.lists.len() > MAX_LISTS
        || !valid_text(&c.state.key)
        || !valid_text(&c.argument)
        || s.resized_capacity.is_some_and(|n| n > MAX_CAPACITY)
    {
        return Err(LedgerError::Capacity);
    }
    let identities: BTreeSet<_> = c.state.lists.iter().map(|l| l.identity).collect();
    if identities.len() != c.state.lists.len() || identities.contains(&0) {
        return Err(LedgerError::InvalidContext);
    }
    let mut total_slots = 0usize;
    let old_key_bytes = c.state.key.as_ref().map_or(0, String::len);
    let key_bytes = if c.operation == Operation::Construct {
        old_key_bytes.max("Tutorials".len())
    } else {
        old_key_bytes
    };
    let mut total_text_bytes = key_bytes + c.argument.as_ref().map_or(0, String::len);
    for list in &c.state.lists {
        let Some(backing) = &list.backing else {
            return Err(LedgerError::InvalidContext);
        };
        if backing.len() > MAX_CAPACITY || backing.iter().any(|v| !valid_text(v)) {
            return Err(LedgerError::Capacity);
        }
        total_slots += backing.len();
        total_text_bytes += backing.iter().flatten().map(String::len).sum::<usize>();
        if total_slots > MAX_TOTAL_SLOTS || total_text_bytes > MAX_TOTAL_TEXT_BYTES {
            return Err(LedgerError::Capacity);
        }
        if list.count > backing.len() {
            return Err(LedgerError::InvalidContext);
        }
    }
    for identity in [c.state.completed_tutorials, c.state.unlocked_characters]
        .into_iter()
        .flatten()
    {
        if !identities.contains(&identity) {
            return Err(LedgerError::InvalidContext);
        }
    }
    if c.operation == Operation::Construct {
        let fresh = &s.constructor_list_identities;
        if fresh.len() != 2
            || fresh[0] == 0
            || fresh[1] == 0
            || fresh[0] == fresh[1]
            || fresh.iter().any(|id| identities.contains(id))
            || s.resized_capacity.is_some()
        {
            return Err(LedgerError::InvalidContext);
        }
        if c.state.lists.len() > MAX_LISTS - 2 {
            return Err(LedgerError::Capacity);
        }
    } else if !s.constructor_list_identities.is_empty() {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

fn step(result: &mut Replay, event: Event) {
    result.steps.push(Step {
        event,
        state: result.state.clone(),
    });
}

/// Reconstruct normal caller completion with explicit supplied service outcomes.
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut result = Replay {
        state: c.state.clone(),
        steps: Vec::new(),
    };
    if c.operation == Operation::Construct {
        result.state.key = Some("Tutorials".into());
        step(&mut result, Event::WriteBarrier);
        for (ordinal, &identity) in c.services.constructor_list_identities.iter().enumerate() {
            step(&mut result, Event::ListAllocate);
            result.state.lists.push(List {
                identity,
                count: 0,
                version: 0,
                backing: None,
            });
            step(&mut result, Event::ListConstructor { list: identity });
            result.state.lists.last_mut().unwrap().backing = Some(Vec::new());
            if ordinal == 0 {
                result.state.completed_tutorials = Some(identity);
            } else {
                result.state.unlocked_characters = Some(identity);
            }
            step(&mut result, Event::WriteBarrier);
        }
        return Ok(result);
    }
    let identity = match c.operation {
        Operation::AddTutorial | Operation::ClearTutorials => result.state.completed_tutorials,
        Operation::AddCharacter | Operation::ClearUnlockedCharacters => {
            result.state.unlocked_characters
        }
        Operation::Construct => unreachable!(),
    }
    .ok_or(LedgerError::InvalidContext)?;
    let index = result
        .state
        .lists
        .iter()
        .position(|l| l.identity == identity)
        .unwrap();
    if matches!(
        c.operation,
        Operation::AddTutorial | Operation::AddCharacter
    ) {
        step(
            &mut result,
            Event::Contains {
                list: identity,
                value: c.argument.clone(),
            },
        );
        let list = &result.state.lists[index];
        if list.backing.as_ref().unwrap()[..list.count].contains(&c.argument) {
            if c.services.resized_capacity.is_some() {
                return Err(LedgerError::InvalidContext);
            }
            return Ok(result);
        }
        let needs_resize = list.count == list.backing.as_ref().unwrap().len();
        let growth = if needs_resize {
            let capacity = c
                .services
                .resized_capacity
                .ok_or(LedgerError::InvalidContext)?;
            if capacity <= list.count {
                return Err(LedgerError::InvalidContext);
            }
            let slots: usize = result
                .state
                .lists
                .iter()
                .map(|l| l.backing.as_ref().unwrap().len())
                .sum();
            if slots + capacity - list.count > MAX_TOTAL_SLOTS {
                return Err(LedgerError::Capacity);
            }
            Some(capacity)
        } else {
            if c.services.resized_capacity.is_some() {
                return Err(LedgerError::InvalidContext);
            }
            None
        };
        result.state.lists[index].version = result.state.lists[index].version.wrapping_add(1);
        if let Some(capacity) = growth {
            step(
                &mut result,
                Event::ResizeAppend {
                    list: identity,
                    value: c.argument.clone(),
                },
            );
            // Only the live prefix is copied by the supplied resize service.
            let list = &mut result.state.lists[index];
            let backing = list.backing.as_mut().unwrap();
            backing.truncate(list.count);
            backing.resize(capacity, None);
            backing[list.count] = c.argument.clone();
            list.count += 1;
        } else {
            let list = &mut result.state.lists[index];
            list.backing.as_mut().unwrap()[list.count] = c.argument.clone();
            list.count += 1;
            step(&mut result, Event::WriteBarrier);
        }
    } else {
        if c.services.resized_capacity.is_some() {
            return Err(LedgerError::InvalidContext);
        }
        let list = &mut result.state.lists[index];
        let count = list.count;
        list.version = list.version.wrapping_add(1);
        list.count = 0;
        if count != 0 {
            step(&mut result, Event::ArrayClear { count });
            result.state.lists[index].backing.as_mut().unwrap()[..count].fill(None);
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "saved_game_info_tests.rs"]
mod tests;
