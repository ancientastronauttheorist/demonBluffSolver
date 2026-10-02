//! Complete normal InitReward storage chronology with inert supplied services.
//! The diagnostic graph is nominal, bounded and disjoint. Unity, Action, GC and
//! RevealReal implementations, callback mutations and fault/unwind paths remain
//! supplied boundaries; no scene or managed-object admission is inferred.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CHARACTER_REWARD_INITIALIZATION_NATIVE_V1: &str =
    "character_reward_initialization_native_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Character,
    Data,
    Acted,
    GameObject,
    Action,
    Class,
    Token,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub identity: Identity,
    pub kind: Kind,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameObject {
    pub identity: Identity,
    pub active: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativePhase {
    Entry { input_data: Identity },
    AlignmentLoad { input_data: Identity, bits: u32 },
    CurrentStateLoad { bits: u32 },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub games: Vec<GameObject>,
    pub activation_requests: Vec<Identity>,
    pub callbacks: Vec<[u64; 4]>,
    pub reveal_requests: Vec<[u64; 4]>,
    pub native_phases: Vec<NativePhase>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Service {
    GetGameObject,
    SetActive,
    Barrier,
    Callback,
    RevealReal,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Field {
    Bluff,
    DataRef,
    RegisterAs,
}
impl Field {
    fn offset(self) -> usize {
        match self {
            Self::Bluff => 0x58,
            Self::DataRef => 0x50,
            Self::RegisterAs => 0x60,
        }
    }
    fn caller(self) -> u32 {
        match self {
            Self::Bluff => 0x36568F,
            Self::DataRef => 0x36569E,
            Self::RegisterAs => 0x3656B0,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    GetGameObject { acted: Identity, result: Identity },
    SetActive { game: Identity },
    Barrier { field: Field, value: Identity },
    Callback,
    RevealReal,
}
impl Event {
    fn service(&self) -> Service {
        match self {
            Self::GetGameObject { .. } => Service::GetGameObject,
            Self::SetActive { .. } => Service::SetActive,
            Self::Barrier { .. } => Service::Barrier,
            Self::Callback => Service::Callback,
            Self::RevealReal => Service::RevealReal,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub input_data: Identity,
    /// Explicit whole get_gameObject output; no implicit Unity lookup.
    pub gameobject_result: Identity,
    pub entry_r8_bits: u64,
    pub entry_r9_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub owner: Identity,
    pub state: State,
    pub calls: Vec<Call>,
    pub service_counts: BTreeMap<Service, u64>,
    pub native_base: u64,
    pub caller_return_bits: u64,
    pub callback_entry_bits: u64,
    /// Full RCX/RDX/R8/R9 bits left by every inert supplied return.
    pub volatile_return_bits: [u64; 4],
    pub storage_verified: bool,
    pub inert_services_verified: bool,
    pub normal_completion_verified: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub call: usize,
    pub ordinal: u64,
    pub event: Event,
    pub raw_args: [u64; 4],
    pub caller_return_rva: Option<u32>,
    pub caller_return_bits: u64,
    pub state: State,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replay {
    pub steps: Vec<Step>,
    pub completed: Vec<State>,
    pub final_state: State,
    pub service_counts: BTreeMap<Service, u64>,
}
fn word(r: &Record, off: usize) -> u64 {
    u64::from_le_bytes(r.bytes[off..off + 8].try_into().expect("validated storage"))
}
fn dword(r: &Record, off: usize) -> u32 {
    u32::from_le_bytes(r.bytes[off..off + 4].try_into().expect("validated storage"))
}
fn record(s: &State, id: Identity) -> &Record {
    s.records
        .iter()
        .find(|r| r.identity == id)
        .expect("validated identity")
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let units = (|| {
        let mut n = 96usize.checked_add(c.version.len())?;
        for r in &c.state.records {
            n = n.checked_add(4)?.checked_add(r.bytes.len())?;
        }
        n = n
            .checked_add(c.state.games.len().checked_mul(2)?)?
            .checked_add(c.state.activation_requests.len())?
            .checked_add(c.state.callbacks.len().checked_mul(4)?)?
            .checked_add(c.state.reveal_requests.len().checked_mul(4)?)?
            .checked_add(c.state.native_phases.len().checked_mul(3)?)?
            .checked_add(c.service_counts.len().checked_mul(2)?)?;
        n.checked_add(c.calls.len().checked_mul(4)?)
    })();
    let work = units.and_then(|n| {
        // Seven entry snapshots, completed-call states and all future ledgers
        // are accounted before allocating maps or cloning the authored graph.
        n.checked_add(c.calls.len().checked_mul(18)?)?
            .checked_mul(c.calls.len().checked_mul(8)?.checked_add(2)?)
    });
    if c.state.records.len() > 32
        || c.calls.len() > 16
        || c.state.games.len() > 16
        || c.state.activation_requests.len() > 128
        || c.state.callbacks.len() > 128
        || c.state.reveal_requests.len() > 128
        || c.state.native_phases.len() > 128
        || units.is_none_or(|n| n > 8192)
        || work.is_none_or(|n| n > 262_144)
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != CHARACTER_REWARD_INITIALIZATION_NATIVE_V1
        || !c.storage_verified
        || !c.inert_services_verified
        || !c.normal_completion_verified
        || c.native_base == 0
        || c.native_base.checked_add(0x3656F8).is_none()
        || c.caller_return_bits == 0
        || c.callback_entry_bits == 0
        || c.service_counts
            .values()
            .any(|&v| v.checked_add((c.calls.len() * 3) as u64).is_none())
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        let valid_size = match r.kind {
            Kind::Character => r.bytes.len() == 0x200,
            // Unconsumed prior Data references can retain their smaller
            // diagnostic windows. A consumed input requires the full window.
            Kind::Data => matches!(r.bytes.len(), 0x80 | 0x200),
            Kind::Class => r.bytes.len() == 0x100,
            _ => r.bytes.len() == 0x80,
        };
        if r.identity == 0 || !valid_size || records.insert(r.identity, r).is_some() {
            return Err(LedgerError::InvalidContext);
        }
    }
    let mut previous_end = 0;
    for (&id, r) in &records {
        let Some(end) = id.checked_add(r.bytes.len() as u64) else {
            return Err(LedgerError::InvalidContext);
        };
        if id < previous_end {
            return Err(LedgerError::InvalidContext);
        }
        previous_end = end;
    }
    let typed = |id, kind, nullable| {
        id == 0 && nullable || records.get(&id).is_some_and(|r| r.kind == kind)
    };
    if !typed(c.owner, Kind::Character, false) {
        return Err(LedgerError::InvalidContext);
    }
    let owner = records[&c.owner];
    if [0x50, 0x58, 0x60]
        .into_iter()
        .any(|off| !typed(word(owner, off), Kind::Data, true))
        || !typed(word(owner, 0xA8), Kind::Acted, c.calls.is_empty())
        || !typed(word(owner, 0x180), Kind::Action, true)
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut games = BTreeSet::new();
    for g in &c.state.games {
        if !typed(g.identity, Kind::GameObject, false) || !games.insert(g.identity) {
            return Err(LedgerError::InvalidContext);
        }
    }
    if records
        .values()
        .filter(|r| r.kind == Kind::GameObject)
        .any(|r| !games.contains(&r.identity))
        || c.state
            .activation_requests
            .iter()
            .any(|id| !games.contains(id))
        || c.calls.iter().any(|a| {
            !typed(a.input_data, Kind::Data, false)
                || records[&a.input_data].bytes.len() != 0x200
                || !games.contains(&a.gameobject_result)
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    for phase in &c.state.native_phases {
        let data = match phase {
            NativePhase::Entry { input_data } | NativePhase::AlignmentLoad { input_data, .. } => {
                *input_data
            }
            NativePhase::CurrentStateLoad { .. } => continue,
        };
        if !typed(data, Kind::Data, true) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let action = word(owner, 0x180);
    if action != 0 {
        let r = records[&action];
        if word(r, 0x18) != c.callback_entry_bits
            || !typed(word(r, 0x28), Kind::Token, false)
            || !typed(word(r, 0x40), Kind::Token, true)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if c.state
        .callbacks
        .iter()
        .any(|r| !typed(r[0], Kind::Token, true) || !typed(r[1], Kind::Token, false))
        || c.state
            .reveal_requests
            .iter()
            .any(|r| !typed(r[0], Kind::Character, false) || r[1] != 0)
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut state = c.state.clone();
    let mut counts = c.service_counts.clone();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    let poison = c.volatile_return_bits;
    for (index, a) in c.calls.iter().enumerate() {
        state.native_phases.push(NativePhase::Entry {
            input_data: a.input_data,
        });
        let emit = |steps: &mut Vec<Step>,
                    counts: &mut BTreeMap<Service, u64>,
                    state: &State,
                    event: Event,
                    raw_args,
                    rva: Option<u32>| {
            let ordinal = counts.entry(event.service()).or_default();
            *ordinal += 1;
            steps.push(Step {
                call: index,
                ordinal: *ordinal,
                event,
                raw_args,
                caller_return_rva: rva,
                caller_return_bits: rva
                    .map_or(c.caller_return_bits, |v| c.native_base + u64::from(v)),
                state: state.clone(),
            });
        };
        let acted = word(record(&state, c.owner), 0xA8);
        emit(
            &mut steps,
            &mut counts,
            &state,
            Event::GetGameObject {
                acted,
                result: a.gameobject_result,
            },
            [acted, 0, a.entry_r8_bits, a.entry_r9_bits],
            Some(0x365667),
        );
        emit(
            &mut steps,
            &mut counts,
            &state,
            Event::SetActive {
                game: a.gameobject_result,
            },
            [a.gameobject_result, 0, 0, poison[3]],
            Some(0x36567D),
        );
        state
            .games
            .iter_mut()
            .find(|g| g.identity == a.gameobject_result)
            .expect("validated game")
            .active = false;
        state.activation_requests.push(a.gameobject_result);
        for (field, value) in [
            (Field::Bluff, 0),
            (Field::DataRef, a.input_data),
            (Field::RegisterAs, 0),
        ] {
            let owner = state
                .records
                .iter_mut()
                .find(|r| r.identity == c.owner)
                .expect("validated owner");
            owner.bytes[field.offset()..field.offset() + 8].copy_from_slice(&value.to_le_bytes());
            emit(
                &mut steps,
                &mut counts,
                &state,
                Event::Barrier { field, value },
                [c.owner + field.offset() as u64, value, poison[2], poison[3]],
                Some(field.caller()),
            );
        }
        let alignment = dword(record(&state, a.input_data), 0x134);
        let current_state = dword(record(&state, c.owner), 0xE4);
        let owner = state
            .records
            .iter_mut()
            .find(|r| r.identity == c.owner)
            .expect("validated owner");
        for (off, bits) in [
            (0xDC, 1u32),
            (0xF8, alignment),
            (0xE0, current_state),
            (0xE4, 5),
        ] {
            owner.bytes[off..off + 4].copy_from_slice(&bits.to_le_bytes());
        }
        state.native_phases.extend([
            NativePhase::AlignmentLoad {
                input_data: a.input_data,
                bits: alignment,
            },
            NativePhase::CurrentStateLoad {
                bits: current_state,
            },
        ]);
        let action = word(record(&state, c.owner), 0x180);
        if action != 0 {
            let r = record(&state, action);
            let raw = [word(r, 0x40), word(r, 0x28), poison[2], poison[3]];
            emit(
                &mut steps,
                &mut counts,
                &state,
                Event::Callback,
                raw,
                Some(0x3656F8),
            );
            state.callbacks.push(raw);
        }
        let raw = [c.owner, 0, poison[2], poison[3]];
        emit(
            &mut steps,
            &mut counts,
            &state,
            Event::RevealReal,
            raw,
            None,
        );
        state.reveal_requests.push(raw);
        completed.push(state.clone());
    }
    Ok(Replay {
        steps,
        completed,
        final_state: state,
        service_counts: counts,
    })
}
#[cfg(test)]
#[path = "character_reward_initialization_tests.rs"]
mod tests;
