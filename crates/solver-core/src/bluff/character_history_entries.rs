//! Guarded offline Character history leaves and register-as type selection.
//! All runtime, GC and Unity liveness services are independently verified and
//! inert, and every call must complete normally. No allocation growth, exception
//! unwinding or service mutation is modeled. History and hover may name the same
//! physical List; distinct Lists must have distinct backing arrays. Slots retain
//! the entire supplied backing-memory view, including untouched diagnostic bytes
//! beyond declared native array capacity; those bytes are not valid array slots.
//! Signed-negative and oversized List counts are conservatively rejected even
//! when a particular native leaf skips them. Events describe exact caller service
//! order and supplied normal metadata/class effects, excluding raw runtime work.

use super::character_initialization::{Actor, Identity};
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CHARACTER_HISTORY_ENTRIES_NATIVE_V1: &str = "character_history_entries_native_v1";
pub const METADATA_FLAG_RVAS: [u32; 14] = [
    0x288c158, 0x288c159, 0x288c15f, 0x288c162, 0x288c164, 0x288c173, 0x288c17f, 0x288c185,
    0x288c18b, 0x288c18c, 0x288c18e, 0x288c18f, 0x288c194, 0x288c195,
];
const MAX_RETAINED: usize = 16_384;
const MAX_WORK: usize = 262_144;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceList {
    pub identity: Identity,
    pub backing_array: Identity,
    pub capacity: u32,
    pub count_bits: u32,
    pub version: u32,
    /// Retained supplied memory view; entries beyond capacity are diagnostics,
    /// never read or written as managed array elements by this replay.
    pub slots: Vec<Option<Identity>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataAsset {
    pub identity: Identity,
    pub type_bits: u32,
    pub live: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub actor: Actor,
    pub history: Identity,
    pub hover: Identity,
    pub lists: Vec<ReferenceList>,
    pub assets: Vec<DataAsset>,
    pub metadata_flags: [u8; 14],
    pub object_class_initialized: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum Call {
    AddOnHoverInfo { info: Option<Identity> },
    ClearRecentMemory,
    GetCurrentActedInfo,
    GetCharacterType { unity_null_return_bits: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub runtime_verified_inert: bool,
    pub gc_verified_inert: bool,
    pub unity_liveness_verified_inert: bool,
    pub storage_verified: bool,
    pub normal_completion_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub object_class: Identity,
    pub info_identities: Vec<Identity>,
    pub calls: Vec<Call>,
    pub services: Services,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    MetadataService {
        slot_rva: u32,
    },
    ClassInitializationService {
        class: Identity,
    },
    RegisterUnityNullService {
        object: Option<Identity>,
    },
    BarrierService {
        address: Identity,
        value: Option<Identity>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Step {
    pub event: Event,
    /// State at service entry, before its supplied inert effects.
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Return {
    Void,
    Info { identity: Option<Identity> },
    Type { bits: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallResult {
    pub result: Return,
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub state: State,
    pub steps: Vec<Step>,
    pub calls: Vec<CallResult>,
}

fn actor_references(a: &Actor) -> impl Iterator<Item = Identity> + '_ {
    [
        a.data,
        a.bluff,
        a.register_as,
        a.trailer,
        a.runtime,
        a.role,
        a.bluff_role,
        a.saved_act,
        a.state_callback,
        a.statuses.target,
        a.dead_prefab.as_ref().map(|v| v.identity),
    ]
    .into_iter()
    .flatten()
    .chain(a.infos.iter().flatten().copied())
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    // The aggregate includes all snapshots and projected Actor history copies.
    // Check before constructing identity maps or cloning any retained storage.
    let retained = c
        .state
        .lists
        .iter()
        .try_fold(32usize, |n, l| {
            n.checked_add(l.slots.len()).and_then(|n| n.checked_add(2))
        })
        .and_then(|n| n.checked_add(c.state.assets.len()))
        .and_then(|n| n.checked_add(c.info_identities.len()))
        .and_then(|n| n.checked_add(c.state.actor.infos.len()))
        .and_then(|n| n.checked_add(c.state.actor.statuses.active.len()))
        .and_then(|n| n.checked_add(c.state.actor.statuses.resistances.len()))
        .and_then(|n| c.calls.len().checked_mul(2).and_then(|m| n.checked_add(m)));
    let work = retained.and_then(|n| {
        c.calls
            .len()
            .checked_mul(5)
            .and_then(|m| m.checked_add(2))
            .and_then(|m| n.checked_mul(m))
    });
    if c.calls.len() > 32
        || retained.is_none_or(|n| n > MAX_RETAINED)
        || work.is_none_or(|n| n > MAX_WORK)
    {
        return Err(LedgerError::Capacity);
    }
    let s = &c.state;
    if c.version != CHARACTER_HISTORY_ENTRIES_NATIVE_V1
        || c.calls.is_empty()
        || !c.services.runtime_verified_inert
        || !c.services.gc_verified_inert
        || !c.services.unity_liveness_verified_inert
        || !c.services.storage_verified
        || !c.services.normal_completion_verified
        || s.actor.identity == 0
        || c.object_class == 0
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut ids = BTreeSet::from([s.actor.identity, c.object_class]);
    if ids.len() != 2 {
        return Err(LedgerError::InvalidContext);
    }
    let mut lists = BTreeMap::new();
    for l in &s.lists {
        if l.identity == 0
            || l.backing_array == 0
            || !ids.insert(l.identity)
            || !ids.insert(l.backing_array)
            || l.count_bits > i32::MAX as u32
            || l.count_bits > l.capacity
            || l.capacity as usize > l.slots.len()
            || l.slots.contains(&Some(0))
        {
            return Err(LedgerError::InvalidContext);
        }
        // Slot address arithmetic must never wrap.
        if l.backing_array
            .checked_add(0x20 + l.slots.len() as u64 * 8)
            .is_none()
        {
            return Err(LedgerError::InvalidContext);
        }
        lists.insert(l.identity, l);
    }
    let Some(history) = lists.get(&s.history) else {
        return Err(LedgerError::InvalidContext);
    };
    if !lists.contains_key(&s.hover)
        || s.actor.infos != history.slots[..history.count_bits as usize]
        || s.actor.info_version != history.version
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut assets = BTreeMap::new();
    for a in &s.assets {
        if a.identity == 0 || !ids.insert(a.identity) {
            return Err(LedgerError::InvalidContext);
        }
        assets.insert(a.identity, a);
    }
    for p in [s.actor.data, s.actor.register_as].into_iter().flatten() {
        if !assets.contains_key(&p) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let mut infos = BTreeSet::new();
    for id in &c.info_identities {
        if *id == 0 || !ids.insert(*id) || !infos.insert(*id) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let storage: BTreeSet<_> = s
        .lists
        .iter()
        .flat_map(|l| [l.identity, l.backing_array])
        .collect();
    let incompatible_actor_fields = [
        // Character.trailerInfo is CharacterTrailerInfo, not CharacterData.
        s.actor.trailer,
        s.actor.runtime,
        s.actor.role,
        s.actor.bluff_role,
        s.actor.saved_act,
        s.actor.state_callback,
        s.actor.statuses.target,
        s.actor.dead_prefab.as_ref().map(|p| p.identity),
    ];
    if incompatible_actor_fields
        .into_iter()
        .flatten()
        .any(|p| infos.contains(&p) || assets.contains_key(&p))
    {
        return Err(LedgerError::InvalidContext);
    }
    if [s.actor.bluff, s.actor.trailer]
        .into_iter()
        .flatten()
        .any(|p| infos.contains(&p))
        || [
            s.actor.bluff,
            s.actor.trailer,
            s.actor.runtime,
            s.actor.role,
            s.actor.bluff_role,
            s.actor.saved_act,
            s.actor.state_callback,
            s.actor.dead_prefab.as_ref().map(|p| p.identity),
        ]
        .into_iter()
        .flatten()
        .any(|p| p == s.actor.identity)
    {
        return Err(LedgerError::InvalidContext);
    }
    if actor_references(&s.actor).any(|id| id == 0 || id == c.object_class || storage.contains(&id))
        || s.lists
            .iter()
            .flat_map(|l| l.slots.iter().flatten())
            .any(|p| !infos.contains(p))
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut counts: BTreeMap<_, _> = s.lists.iter().map(|l| (l.identity, l.count_bits)).collect();
    for call in &c.calls {
        match call {
            Call::AddOnHoverInfo { info } => {
                if info.is_some_and(|p| !infos.contains(&p)) {
                    return Err(LedgerError::InvalidContext);
                }
                let count = counts.get_mut(&s.hover).unwrap();
                if *count >= lists[&s.hover].capacity {
                    return Err(LedgerError::InvalidContext);
                }
                *count += 1;
            }
            Call::ClearRecentMemory => {
                let count = counts.get_mut(&s.history).unwrap();
                if *count > 0 {
                    *count -= 1;
                }
            }
            Call::GetCurrentActedInfo => {
                if counts[&s.history] == 0 {
                    return Err(LedgerError::InvalidContext);
                }
            }
            Call::GetCharacterType {
                unity_null_return_bits,
            } => {
                let absent = s.actor.register_as.is_none_or(|p| !assets[&p].live);
                if (*unity_null_return_bits as u8 != 0) != absent
                    || (absent && s.actor.data.is_none())
                {
                    return Err(LedgerError::InvalidContext);
                }
            }
        }
    }
    Ok(())
}

fn sync_history(s: &mut State) {
    let history = s.lists.iter().find(|l| l.identity == s.history).unwrap();
    s.actor.infos = history.slots[..history.count_bits as usize].to_vec();
    s.actor.info_version = history.version;
}

fn step(s: &State, steps: &mut Vec<Step>, event: Event) {
    steps.push(Step {
        event,
        state: s.clone(),
    });
}

pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut state = c.state.clone();
    let mut steps = Vec::new();
    let mut calls = Vec::new();
    for call in &c.calls {
        let (flag, slots): (usize, &[u32]) = match call {
            Call::AddOnHoverInfo { .. } => (2, &[0x270ddb8]),
            Call::ClearRecentMemory => (9, &[0x270deb8, 0x270df38]),
            Call::GetCurrentActedInfo => (7, &[0x270df38, 0x270dfb8]),
            Call::GetCharacterType { .. } => (3, &[0x2718bf0]),
        };
        if state.metadata_flags[flag] == 0 {
            for slot in slots {
                step(
                    &state,
                    &mut steps,
                    Event::MetadataService { slot_rva: *slot },
                );
            }
            state.metadata_flags[flag] = 1;
        }
        let result = match call {
            Call::AddOnHoverInfo { info } => {
                let l = state
                    .lists
                    .iter_mut()
                    .find(|l| l.identity == state.hover)
                    .unwrap();
                let index = l.count_bits as usize;
                l.version = l.version.wrapping_add(1);
                l.count_bits += 1;
                l.slots[index] = *info;
                let address = l.backing_array + 0x20 + index as u64 * 8;
                sync_history(&mut state);
                step(
                    &state,
                    &mut steps,
                    Event::BarrierService {
                        address,
                        value: *info,
                    },
                );
                Return::Void
            }
            Call::ClearRecentMemory => {
                let l = state
                    .lists
                    .iter_mut()
                    .find(|l| l.identity == state.history)
                    .unwrap();
                if l.count_bits > 0 {
                    l.count_bits -= 1;
                    let index = l.count_bits as usize;
                    l.slots[index] = None;
                    let address = l.backing_array + 0x20 + index as u64 * 8;
                    sync_history(&mut state);
                    step(
                        &state,
                        &mut steps,
                        Event::BarrierService {
                            address,
                            value: None,
                        },
                    );
                    let l = state
                        .lists
                        .iter_mut()
                        .find(|l| l.identity == state.history)
                        .unwrap();
                    l.version = l.version.wrapping_add(1);
                    sync_history(&mut state);
                }
                Return::Void
            }
            Call::GetCurrentActedInfo => {
                let l = state
                    .lists
                    .iter()
                    .find(|l| l.identity == state.history)
                    .unwrap();
                Return::Info {
                    identity: l.slots[l.count_bits as usize - 1],
                }
            }
            Call::GetCharacterType {
                unity_null_return_bits,
            } => {
                if state.object_class_initialized == 0 {
                    step(
                        &state,
                        &mut steps,
                        Event::ClassInitializationService {
                            class: c.object_class,
                        },
                    );
                    state.object_class_initialized = 1;
                }
                step(
                    &state,
                    &mut steps,
                    Event::RegisterUnityNullService {
                        object: state.actor.register_as,
                    },
                );
                // Native tests AL; upper return bits are deliberately ignored.
                let p = if *unity_null_return_bits as u8 == 0 {
                    state.actor.register_as
                } else {
                    state.actor.data
                }
                .unwrap();
                let asset = state.assets.iter().find(|a| a.identity == p).unwrap();
                Return::Type {
                    bits: asset.type_bits,
                }
            }
        };
        calls.push(CallResult {
            result,
            state: state.clone(),
        });
    }
    Ok(Replay {
        state,
        steps,
        calls,
    })
}

#[cfg(test)]
#[path = "character_history_entries_tests.rs"]
mod tests;
