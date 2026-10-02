//! Exact CharacterData constructor caller with inert whole supplied services.
//! Complete diagnostic storage and call ABI are retained. Allocation, generic
//! constructors, metadata, GC, Unity initialization and callbacks stay supplied.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
pub const CHARACTER_DATA_CONSTRUCTOR_NATIVE_V1: &str = "character_data_constructor_native_v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Generic {
    CharacterData,
    SkinData,
    AchievementData,
    ECharacterStatus,
    ECharacterTag,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Data,
    DataClass,
    Skin,
    List(Generic),
    TypeInfo(Generic),
    MethodInfo(Generic),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub identity: Identity,
    pub kind: Kind,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Slot {
    CharacterDataType,
    CharacterDataCtor,
    SkinDataType,
    SkinDataCtor,
    AchievementDataType,
    AchievementDataCtor,
    StatusType,
    StatusCtor,
    TagType,
    TagCtor,
}
impl Slot {
    pub fn rva(self) -> u32 {
        match self {
            Self::CharacterDataType => 0x2700680,
            Self::CharacterDataCtor => 0x2713810,
            Self::SkinDataType => 0x2707880,
            Self::SkinDataCtor => 0x26E1980,
            Self::AchievementDataType => 0x26FF500,
            Self::AchievementDataCtor => 0x270D938,
            Self::StatusType => 0x2701C80,
            Self::StatusCtor => 0x271BCD0,
            Self::TagType => 0x2701D00,
            Self::TagCtor => 0x271C000,
        }
    }
    fn kind(self) -> Kind {
        match self {
            Self::CharacterDataType => Kind::TypeInfo(Generic::CharacterData),
            Self::CharacterDataCtor => Kind::MethodInfo(Generic::CharacterData),
            Self::SkinDataType => Kind::TypeInfo(Generic::SkinData),
            Self::SkinDataCtor => Kind::MethodInfo(Generic::SkinData),
            Self::AchievementDataType => Kind::TypeInfo(Generic::AchievementData),
            Self::AchievementDataCtor => Kind::MethodInfo(Generic::AchievementData),
            Self::StatusType => Kind::TypeInfo(Generic::ECharacterStatus),
            Self::StatusCtor => Kind::MethodInfo(Generic::ECharacterStatus),
            Self::TagType => Kind::TypeInfo(Generic::ECharacterTag),
            Self::TagCtor => Kind::MethodInfo(Generic::ECharacterTag),
        }
    }
}
const METADATA: [Slot; 10] = [
    Slot::TagCtor,
    Slot::SkinDataCtor,
    Slot::StatusCtor,
    Slot::CharacterDataCtor,
    Slot::AchievementDataCtor,
    Slot::StatusType,
    Slot::SkinDataType,
    Slot::TagType,
    Slot::CharacterDataType,
    Slot::AchievementDataType,
];
const METADATA_SITES: [u32; 10] = [
    0x3B50BD, 0x3B50C9, 0x3B50D5, 0x3B50E1, 0x3B50ED, 0x3B50F9, 0x3B5105, 0x3B5111, 0x3B511D,
    0x3B5129,
];
const TYPES: [Slot; 6] = [
    Slot::CharacterDataType,
    Slot::SkinDataType,
    Slot::AchievementDataType,
    Slot::StatusType,
    Slot::TagType,
    Slot::CharacterDataType,
];
const CTORS: [Slot; 6] = [
    Slot::CharacterDataCtor,
    Slot::SkinDataCtor,
    Slot::AchievementDataCtor,
    Slot::StatusCtor,
    Slot::TagCtor,
    Slot::CharacterDataCtor,
];
const GENERICS: [Generic; 6] = [
    Generic::CharacterData,
    Generic::SkinData,
    Generic::AchievementData,
    Generic::ECharacterStatus,
    Generic::ECharacterTag,
    Generic::CharacterData,
];
const OFFSETS: [usize; 6] = [0x48, 0xC8, 0xD0, 0x118, 0x120, 0x128];
const ALLOCATION_SITES: [u32; 6] = [0x3B513C, 0x3B5169, 0x3B5199, 0x3B51C9, 0x3B51F9, 0x3B5229];
const CTOR_SITES: [u32; 6] = [0x3B514E, 0x3B517B, 0x3B51AB, 0x3B51DB, 0x3B520B, 0x3B523B];
const BARRIER_SITES: [u32; 6] = [0x3B515D, 0x3B518D, 0x3B51BD, 0x3B51ED, 0x3B521D, 0x3B524D];
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub volatile_registers: [u64; 7],
    pub volatile_xmm_hex: [String; 6],
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Service {
    Metadata,
    Allocation,
    ListConstructor,
    ReferenceBarrier,
    BaseConstructor,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Completed {
    Metadata {
        slot: Slot,
        value: Identity,
    },
    Allocation {
        index: u8,
        class: Identity,
        result: Identity,
    },
    ListConstructor {
        index: u8,
        owner: Identity,
        method: Identity,
    },
    ReferenceBarrier {
        index: u8,
        owner: Identity,
        value: Identity,
    },
    BaseConstructor {
        owner: Identity,
    },
}
impl Completed {
    fn service(&self) -> Service {
        match self {
            Self::Metadata { .. } => Service::Metadata,
            Self::Allocation { .. } => Service::Allocation,
            Self::ListConstructor { .. } => Service::ListConstructor,
            Self::ReferenceBarrier { .. } => Service::ReferenceBarrier,
            Self::BaseConstructor { .. } => Service::BaseConstructor,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub metadata_slots: BTreeMap<Slot, Identity>,
    pub metadata_flag: u8,
    pub native_entries: Vec<Entry>,
    pub service_history: BTreeMap<Service, Vec<Completed>>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub entry: Entry,
    pub allocation_results: [Identity; 6],
    pub list_ctor_return_bits: u64,
    pub barrier_return_bits: u64,
    pub base_return_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub calls: Vec<Call>,
    pub native_base: u64,
    pub return_sentinel: u64,
    pub volatile_return_registers: [u64; 7],
    pub volatile_return_xmm_hex: [String; 6],
    pub storage_verified: bool,
    pub inert_services_verified: bool,
    pub normal_completion_verified: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub call: usize,
    pub ordinal: usize,
    pub event: Completed,
    pub volatile_registers: [u64; 7],
    pub volatile_xmm_hex: [String; 6],
    /// None identifies the tail's original fixture return sentinel.
    pub native_site_rva: Option<u32>,
    pub caller_return_bits: u64,
    pub state: State,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replay {
    pub steps: Vec<Step>,
    pub completed: Vec<State>,
    pub final_registers: Vec<[u64; 7]>,
    pub final_xmm_hex: Vec<[String; 6]>,
    pub final_state: State,
}
fn xmm_valid(v: &[String; 6]) -> bool {
    v.iter()
        .all(|s| s.len() == 32 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}
fn units(c: &Context) -> Option<usize> {
    let mut n = 128usize.checked_add(c.version.len())?;
    for r in &c.state.records {
        n = n.checked_add(4)?.checked_add(r.bytes.len())?;
    }
    for e in c
        .state
        .native_entries
        .iter()
        .chain(c.calls.iter().map(|call| &call.entry))
    {
        n = n.checked_add(8)?;
        for x in &e.volatile_xmm_hex {
            n = n.checked_add(x.len())?;
        }
    }
    for x in &c.volatile_return_xmm_hex {
        n = n.checked_add(x.len())?;
    }
    n = n.checked_add(c.state.metadata_slots.len().checked_mul(3)?)?;
    for h in c.state.service_history.values() {
        n = n.checked_add(2)?.checked_add(h.len().checked_mul(6)?)?;
    }
    n.checked_add(c.calls.len().checked_mul(10)?)
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let units = units(c);
    // A future entry retains 8 integer/shape units plus 192 hex characters.
    // All 29 possible service-history records are bounded by 6 units each;
    // 29 steps and one completed snapshot per call retain the complete state.
    let work = units.and_then(|n| {
        n.checked_add(c.calls.len().checked_mul(374)?)?
            .checked_mul(c.calls.len().checked_mul(30)?.checked_add(2)?)
    });
    if c.state.records.len() > 64
        || c.calls.len() > 8
        || c.state.native_entries.len() > 64
        || c.state.service_history.len() > 5
        || c.state.service_history.values().any(|v| v.len() > 512)
        || units.is_none_or(|n| n > 65536)
        || work.is_none_or(|n| n > 4_194_304)
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != CHARACTER_DATA_CONSTRUCTOR_NATIVE_V1
        || !c.storage_verified
        || !c.inert_services_verified
        || !c.normal_completion_verified
        || c.native_base == 0
        || c.native_base.checked_add(0x288C4E4).is_none()
        || c.return_sentinel == 0
        || c.state.service_history.len() != 5
        || !xmm_valid(&c.volatile_return_xmm_hex)
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        let size = match r.kind {
            Kind::Data => 512,
            Kind::DataClass | Kind::TypeInfo(_) | Kind::MethodInfo(_) => 256,
            _ => 128,
        };
        if r.identity == 0 || r.bytes.len() != size || records.insert(r.identity, r).is_some() {
            return Err(LedgerError::InvalidContext);
        }
        let Some(end) = r.identity.checked_add(size as u64) else {
            return Err(LedgerError::InvalidContext);
        };
        if c.state.records.iter().any(|o| {
            o.identity != r.identity
                && o.identity < end
                && o.identity
                    .checked_add(o.bytes.len() as u64)
                    .is_none_or(|e| e > r.identity)
        }) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let reference = |id: Identity, kind: Kind| records.get(&id).is_some_and(|r| r.kind == kind);
    if c.state.metadata_slots.len() != 10
        || METADATA.iter().any(|s| {
            !c.state
                .metadata_slots
                .get(s)
                .is_some_and(|id| reference(*id, s.kind()))
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    let entry_valid = |e: &Entry, nullable: bool| {
        xmm_valid(&e.volatile_xmm_hex)
            && (nullable && e.volatile_registers[1] == 0
                || reference(e.volatile_registers[1], Kind::Data))
    };
    if c.state.native_entries.iter().any(|e| !entry_valid(e, true))
        || c.calls.iter().any(|v| {
            !entry_valid(&v.entry, false)
                || v.allocation_results
                    .iter()
                    .enumerate()
                    .any(|(i, id)| !reference(*id, Kind::List(GENERICS[i])))
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    for (service, history) in &c.state.service_history {
        for h in history {
            let valid = h.service() == *service
                && match h {
                    Completed::Metadata { slot, value } => reference(*value, slot.kind()),
                    Completed::Allocation {
                        index,
                        class,
                        result,
                    } => {
                        usize::from(*index) < 6
                            && reference(*class, Kind::TypeInfo(GENERICS[usize::from(*index)]))
                            && reference(*result, Kind::List(GENERICS[usize::from(*index)]))
                    }
                    Completed::ListConstructor {
                        index,
                        owner,
                        method,
                    } => {
                        usize::from(*index) < 6
                            && reference(*owner, Kind::List(GENERICS[usize::from(*index)]))
                            && reference(*method, Kind::MethodInfo(GENERICS[usize::from(*index)]))
                    }
                    Completed::ReferenceBarrier {
                        index,
                        owner,
                        value,
                    } => {
                        usize::from(*index) < 6
                            && reference(*owner, Kind::Data)
                            && reference(*value, Kind::List(GENERICS[usize::from(*index)]))
                    }
                    Completed::BaseConstructor { owner } => reference(*owner, Kind::Data),
                };
            if !valid {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    Ok(())
}
fn emit(
    steps: &mut Vec<Step>,
    s: &mut State,
    c: &Context,
    call: usize,
    counts: &mut BTreeMap<Service, usize>,
    event: Completed,
    regs: [u64; 7],
    xmm: [String; 6],
    site: Option<u32>,
) {
    let service = event.service();
    let ordinal = counts.entry(service).or_default();
    *ordinal += 1;
    steps.push(Step {
        call,
        ordinal: *ordinal,
        event: event.clone(),
        volatile_registers: regs,
        volatile_xmm_hex: xmm,
        native_site_rva: site,
        caller_return_bits: site.map_or(c.return_sentinel, |r| c.native_base + u64::from(r) + 5),
        state: s.clone(),
    });
    s.service_history.entry(service).or_default().push(event);
}
fn write(s: &mut State, id: Identity, off: usize, bytes: &[u8]) {
    s.records
        .iter_mut()
        .find(|r| r.identity == id)
        .expect("validated owner")
        .bytes[off..off + bytes.len()]
        .copy_from_slice(bytes);
}
/// Replay only normal completion with nominal supplied allocations and inert
/// constructors/barriers/base. Validate all calls before cloning any state.
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut s = c.state.clone();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    let mut final_registers = Vec::new();
    let mut final_xmm_hex = Vec::new();
    for (i, call) in c.calls.iter().enumerate() {
        s.native_entries.push(call.entry.clone());
        let owner = call.entry.volatile_registers[1];
        let mut regs = call.entry.volatile_registers;
        let mut xmm = call.entry.volatile_xmm_hex.clone();
        let mut counts = BTreeMap::new();
        if s.metadata_flag == 0 {
            for (slot, site) in METADATA.into_iter().zip(METADATA_SITES) {
                regs[1] = c.native_base + u64::from(slot.rva());
                let value = s.metadata_slots[&slot];
                emit(
                    &mut steps,
                    &mut s,
                    c,
                    i,
                    &mut counts,
                    Completed::Metadata { slot, value },
                    regs,
                    xmm,
                    Some(site),
                );
                regs = c.volatile_return_registers;
                regs[0] = value;
                xmm = c.volatile_return_xmm_hex.clone();
            }
            s.metadata_flag = 1;
        }
        for n in 0..6 {
            let class = s.metadata_slots[&TYPES[n]];
            let captured = call.allocation_results[n];
            regs[1] = class;
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::Allocation {
                    index: n as u8,
                    class,
                    result: captured,
                },
                regs,
                xmm,
                Some(ALLOCATION_SITES[n]),
            );
            regs = c.volatile_return_registers;
            regs[0] = captured;
            xmm = c.volatile_return_xmm_hex.clone();
            let method = s.metadata_slots[&CTORS[n]];
            regs[1] = captured;
            regs[2] = method;
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::ListConstructor {
                    index: n as u8,
                    owner: captured,
                    method,
                },
                regs,
                xmm,
                Some(CTOR_SITES[n]),
            );
            regs = c.volatile_return_registers;
            regs[0] = call.list_ctor_return_bits;
            xmm = c.volatile_return_xmm_hex.clone();
            regs[1] = owner + OFFSETS[n] as u64;
            regs[2] = captured;
            write(&mut s, owner, OFFSETS[n], &captured.to_le_bytes());
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::ReferenceBarrier {
                    index: n as u8,
                    owner,
                    value: captured,
                },
                regs,
                xmm,
                Some(BARRIER_SITES[n]),
            );
            regs = c.volatile_return_registers;
            regs[0] = call.barrier_return_bits;
            xmm = c.volatile_return_xmm_hex.clone();
        }
        regs[2] = 0;
        write(&mut s, owner, 0x13C, &[1]);
        regs[1] = owner;
        write(&mut s, owner, 0x13E, &[1]);
        emit(
            &mut steps,
            &mut s,
            c,
            i,
            &mut counts,
            Completed::BaseConstructor { owner },
            regs,
            xmm,
            None,
        );
        regs = c.volatile_return_registers;
        regs[0] = call.base_return_bits;
        final_registers.push(regs);
        final_xmm_hex.push(c.volatile_return_xmm_hex.clone());
        completed.push(s.clone());
    }
    Ok(Replay {
        steps,
        completed,
        final_registers,
        final_xmm_hex,
        final_state: s,
    })
}
#[cfg(test)]
#[path = "character_data_constructor_tests.rs"]
mod tests;
