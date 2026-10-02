//! Exact normal CharacterData.LoadPreferences caller with inert whole services.
//! Save lookup, enumeration, string comparison and LoadSkin remain supplied.
//! Complete authored windows, chronological histories and volatile ABI persist;
//! callbacks, cleanup, faults, persistence and runtime admission are excluded.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHARACTER_DATA_PREFERENCES_NATIVE_V1: &str = "character_data_preferences_native_v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Slot {
    Dispose,
    MoveNext,
    Current,
    GetEnumerator,
}
impl Slot {
    pub fn rva(self) -> u32 {
        match self {
            Self::Dispose => 0x26D66E8,
            Self::MoveNext => 0x26D6768,
            Self::Current => 0x26D67F0,
            Self::GetEnumerator => 0x27141A0,
        }
    }
}
const SLOTS: [Slot; 4] = [
    Slot::Dispose,
    Slot::MoveNext,
    Slot::Current,
    Slot::GetEnumerator,
];
const META_SITES: [u32; 4] = [0x3B4DD2, 0x3B4DDE, 0x3B4DEA, 0x3B4DF6];
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Owner,
    DataClass,
    SavedCharacters,
    List,
    Array,
    Preference,
    String,
    StringClass,
    Skin,
    Exception,
    Enumerator,
    MethodInfo(Slot),
    Scratch,
}
impl Kind {
    fn size(self) -> usize {
        match self {
            Self::Owner => 512,
            Self::List | Self::Array => 256,
            Self::Scratch => 64,
            _ => 128,
        }
    }
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
pub struct Entry {
    pub owner: Identity,
    pub synthetic_cleanup: bool,
    pub volatile_registers: [u64; 7],
    pub volatile_xmm_hex: [String; 6],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IteratorState {
    pub list: Identity,
    pub entries: Vec<Identity>,
    pub cursor: u32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Service {
    Metadata,
    ReferenceBarrier,
    CharacterPreferences,
    GetEnumerator,
    MoveNext,
    StringEquality,
    LoadSkin,
    Dispose,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Completed {
    Metadata {
        slot: Slot,
        value: Identity,
    },
    ReferenceBarrier {
        owner: Identity,
    },
    CharacterPreferences {
        result: Identity,
    },
    GetEnumerator {
        output: Identity,
        list: Identity,
        method: Identity,
        bytes: [u8; 24],
        entries: Vec<Identity>,
    },
    MoveNext {
        owner: Identity,
        method: Identity,
        current: Identity,
        result_bits: u64,
    },
    StringEquality {
        a: Identity,
        b: Identity,
        result_bits: u64,
    },
    LoadSkin {
        owner: Identity,
        skin_id: Identity,
    },
    Dispose {
        owner: Identity,
        method: Identity,
    },
}
impl Completed {
    pub fn service(&self) -> Service {
        match self {
            Self::Metadata { .. } => Service::Metadata,
            Self::ReferenceBarrier { .. } => Service::ReferenceBarrier,
            Self::CharacterPreferences { .. } => Service::CharacterPreferences,
            Self::GetEnumerator { .. } => Service::GetEnumerator,
            Self::MoveNext { .. } => Service::MoveNext,
            Self::StringEquality { .. } => Service::StringEquality,
            Self::LoadSkin { .. } => Service::LoadSkin,
            Self::Dispose { .. } => Service::Dispose,
        }
    }
    fn units(&self) -> Option<usize> {
        96usize.checked_add(match self {
            Self::GetEnumerator { entries, .. } => entries.len().checked_mul(8)?,
            _ => 0,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub metadata_slots: BTreeMap<Slot, Identity>,
    pub metadata_flag: u8,
    pub native_entries: Vec<Entry>,
    pub service_history: Vec<Completed>,
    pub iterator: Option<IteratorState>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveOutput {
    pub current: Identity,
    pub result_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub entry: Entry,
    pub saved_result: Identity,
    pub enumerator_bytes: [u8; 24],
    pub enumerator_entries: Vec<Identity>,
    pub moves: Vec<MoveOutput>,
    pub equality_results: Vec<u64>,
    pub barrier_return_bits: u64,
    pub get_return_bits: u64,
    pub load_skin_return_bits: u64,
    pub dispose_return_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub calls: Vec<Call>,
    pub native_base: u64,
    pub scratch: Identity,
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
    pub native_site_rva: u32,
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
    v.iter().all(|s| {
        s.len() == 32
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}
fn entry_units(e: &Entry) -> Option<usize> {
    e.volatile_xmm_hex
        .iter()
        .try_fold(224usize, |n, x| n.checked_add(x.len()))
}
fn state_units(s: &State) -> Option<usize> {
    let mut n = 256usize.checked_add(s.metadata_slots.len().checked_mul(24)?)?;
    for r in &s.records {
        n = n.checked_add(64)?.checked_add(r.bytes.len())?;
    }
    for e in &s.native_entries {
        n = n.checked_add(entry_units(e)?)?;
    }
    for h in &s.service_history {
        n = n.checked_add(h.units()?)?;
    }
    if let Some(i) = &s.iterator {
        n = n
            .checked_add(48)?
            .checked_add(i.entries.len().checked_mul(8)?)?;
    }
    Some(n)
}
fn budget(c: &Context) -> Option<(usize, usize)> {
    let mut input = state_units(&c.state)?
        .checked_add(256)?
        .checked_add(c.version.len())?;
    for x in &c.volatile_return_xmm_hex {
        input = input.checked_add(x.len())?;
    }
    let mut future = state_units(&c.state)?;
    let mut snapshots = 2usize;
    let mut trace = 0usize;
    for call in &c.calls {
        let supplied = call
            .enumerator_entries
            .len()
            .checked_mul(8)?
            .checked_add(call.moves.len().checked_mul(16)?)?
            .checked_add(call.equality_results.len().checked_mul(8)?)?;
        input = input
            .checked_add(entry_units(&call.entry)?)?
            .checked_add(160)?
            .checked_add(supplied)?;
        // Every cold call can add four metadata, clear/barrier, save getter,
        // enumerator, each MoveNext/equality/LoadSkin and final Dispose.
        let events = 8usize
            .checked_add(call.moves.len())?
            .checked_add(call.equality_results.len().checked_mul(2)?)?;
        future = future
            .checked_add(entry_units(&call.entry)?)?
            .checked_add(events.checked_mul(128)?)?
            .checked_add(80)?;
        snapshots = snapshots.checked_add(events)?.checked_add(1)?;
        trace = trace.checked_add(events.checked_mul(768)?)?;
    }
    Some((
        input,
        future
            .checked_mul(snapshots)?
            .checked_add(trace)?
            .checked_add(input)?,
    ))
}
fn record(s: &State, id: Identity) -> &Record {
    s.records
        .iter()
        .find(|r| r.identity == id)
        .expect("validated identity")
}
fn q(s: &State, id: Identity, off: usize) -> u64 {
    u64::from_le_bytes(
        record(s, id).bytes[off..off + 8]
            .try_into()
            .expect("validated field"),
    )
}
fn d(s: &State, id: Identity, off: usize) -> u32 {
    u32::from_le_bytes(
        record(s, id).bytes[off..off + 4]
            .try_into()
            .expect("validated field"),
    )
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let b = budget(c);
    if c.state.records.len() > 64
        || c.calls.len() > 8
        || c.state.native_entries.len() > 64
        || c.state.service_history.len() > 512
        || c.calls.iter().any(|v| {
            v.enumerator_entries.len() > 4 || v.moves.len() > 5 || v.equality_results.len() > 4
        })
        || c.state
            .iterator
            .as_ref()
            .is_some_and(|v| v.entries.len() > 4)
        || b.is_none_or(|(i, w)| i > 65536 || w > 8_388_608)
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != CHARACTER_DATA_PREFERENCES_NATIVE_V1
        || !c.storage_verified
        || !c.inert_services_verified
        || !c.normal_completion_verified
        || c.native_base == 0
        || c.native_base.checked_add(0x288C4DE).is_none()
        || !xmm_valid(&c.volatile_return_xmm_hex)
        || c.state.metadata_slots.len() != 4
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut rs = BTreeMap::new();
    for r in &c.state.records {
        if r.identity == 0
            || r.bytes.len() != r.kind.size()
            || rs.insert(r.identity, r).is_some()
            || r.identity.checked_add(r.bytes.len() as u64).is_none()
        {
            return Err(LedgerError::InvalidContext);
        }
        let end = r.identity + r.bytes.len() as u64;
        if SLOTS.iter().any(|slot| {
            let address = c.native_base + u64::from(slot.rva());
            r.identity < address + 8 && address < end
        }) || {
            let flag = c.native_base + 0x288C4DD;
            r.identity <= flag && flag < end
        } {
            return Err(LedgerError::InvalidContext);
        }
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
    let rf =
        |id, kind, nullable| id == 0 && nullable || rs.get(&id).is_some_and(|r| r.kind == kind);
    if !rf(c.scratch, Kind::Scratch, false)
        || SLOTS.iter().any(|s| {
            !c.state
                .metadata_slots
                .get(s)
                .is_some_and(|id| rf(*id, Kind::MethodInfo(*s), false))
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    let output = c.scratch + 8;
    let active = c.scratch + 32;
    let entries_valid =
        |v: &[Identity]| v.len() <= 4 && v.iter().all(|id| rf(*id, Kind::Preference, true));
    let iterator_valid = |i: &IteratorState| {
        rf(i.list, Kind::List, false) && entries_valid(&i.entries) && i.cursor <= 5
    };
    if c.state
        .iterator
        .as_ref()
        .is_some_and(|i| !iterator_valid(i))
    {
        return Err(LedgerError::InvalidContext);
    }
    for r in &c.state.records {
        let valid = match r.kind {
            Kind::Owner => {
                rf(q(&c.state, r.identity, 0), Kind::DataClass, false)
                    && rf(q(&c.state, r.identity, 0x18), Kind::String, true)
                    && rf(q(&c.state, r.identity, 0xC0), Kind::Skin, true)
            }
            Kind::SavedCharacters => rf(q(&c.state, r.identity, 0x10), Kind::List, true),
            Kind::List => {
                let a = q(&c.state, r.identity, 0x10);
                let n = d(&c.state, r.identity, 0x18);
                rf(a, Kind::Array, false)
                    && n <= 4
                    && (0..n).all(|i| {
                        rf(
                            q(&c.state, a, 0x20 + i as usize * 8),
                            Kind::Preference,
                            true,
                        )
                    })
            }
            Kind::Preference => {
                rf(q(&c.state, r.identity, 0x10), Kind::String, true)
                    && rf(q(&c.state, r.identity, 0x18), Kind::String, true)
            }
            Kind::String => {
                rf(q(&c.state, r.identity, 0), Kind::StringClass, false)
                    && d(&c.state, r.identity, 0x10) <= 54
            }
            _ => true,
        };
        if !valid {
            return Err(LedgerError::InvalidContext);
        }
    }
    let ev = |e: &Entry, nullable| {
        !e.synthetic_cleanup
            && e.owner == e.volatile_registers[1]
            && rf(e.owner, Kind::Owner, nullable)
            && xmm_valid(&e.volatile_xmm_hex)
    };
    if c.state.native_entries.iter().any(|e| !ev(e, true)) {
        return Err(LedgerError::InvalidContext);
    }
    for h in &c.state.service_history {
        let valid = match h {
            Completed::Metadata { slot, value } => rf(*value, Kind::MethodInfo(*slot), false),
            Completed::ReferenceBarrier { owner } => rf(*owner, Kind::Owner, false),
            Completed::CharacterPreferences { result } => rf(*result, Kind::SavedCharacters, true),
            Completed::GetEnumerator {
                output: o,
                list,
                method,
                bytes,
                entries,
            } => {
                *o == output
                    && rf(*list, Kind::List, false)
                    && rf(*method, Kind::MethodInfo(Slot::GetEnumerator), false)
                    && entries_valid(entries)
                    && rf(
                        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                        Kind::List,
                        false,
                    )
                    && rf(
                        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
                        Kind::Preference,
                        true,
                    )
            }
            Completed::MoveNext {
                owner,
                method,
                current,
                ..
            } => {
                *owner == active
                    && rf(*method, Kind::MethodInfo(Slot::MoveNext), false)
                    && rf(*current, Kind::Preference, true)
            }
            Completed::StringEquality { a, b, .. } => {
                rf(*a, Kind::String, true) && rf(*b, Kind::String, true)
            }
            Completed::LoadSkin { owner, skin_id } => {
                rf(*owner, Kind::Owner, false) && rf(*skin_id, Kind::String, true)
            }
            Completed::Dispose { owner, method } => {
                *owner == active && rf(*method, Kind::MethodInfo(Slot::Dispose), false)
            }
        };
        if !valid {
            return Err(LedgerError::InvalidContext);
        }
    }
    for call in &c.calls {
        if !ev(&call.entry, false)
            || !rf(call.saved_result, Kind::SavedCharacters, false)
            || !entries_valid(&call.enumerator_entries)
            || call.moves.is_empty()
        {
            return Err(LedgerError::InvalidContext);
        }
        let list = q(&c.state, call.saved_result, 0x10);
        let enum_list = u64::from_le_bytes(call.enumerator_bytes[0..8].try_into().unwrap());
        let enum_current = u64::from_le_bytes(call.enumerator_bytes[16..24].try_into().unwrap());
        if !rf(list, Kind::List, false)
            || !rf(enum_list, Kind::List, false)
            || !rf(enum_current, Kind::Preference, true)
        {
            return Err(LedgerError::InvalidContext);
        }
        // The whole iterator supplies one finite, normal stream. Native only
        // consumes AL; early false and noncanonical true bytes remain legal.
        if call.moves.last().unwrap().result_bits as u8 != 0
            || call.equality_results.len() + 1 != call.moves.len()
        {
            return Err(LedgerError::InvalidContext);
        }
        for (i, m) in call.moves.iter().enumerate() {
            let expected = call.enumerator_entries.get(i).copied().unwrap_or(0);
            if m.current != expected
                || !rf(m.current, Kind::Preference, true)
                || (i + 1 < call.moves.len() && (m.result_bits as u8 == 0 || m.current == 0))
            {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    Ok(())
}
fn write(s: &mut State, id: Identity, off: usize, bytes: &[u8]) {
    s.records
        .iter_mut()
        .find(|r| r.identity == id)
        .expect("validated identity")
        .bytes[off..off + bytes.len()]
        .copy_from_slice(bytes);
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
    site: u32,
) {
    let ordinal = counts.entry(event.service()).or_default();
    *ordinal += 1;
    steps.push(Step {
        call,
        ordinal: *ordinal,
        event: event.clone(),
        volatile_registers: regs,
        volatile_xmm_hex: xmm,
        native_site_rva: site,
        caller_return_bits: c.native_base + u64::from(site) + 5,
        state: s.clone(),
    });
    s.service_history.push(event);
}
/// Validate all identities, normal streams and complete future clone budgets
/// before cloning state. No externally supplied callback is executed here.
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut s = c.state.clone();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    let mut finals = Vec::new();
    let mut final_xmm = Vec::new();
    for (i, call) in c.calls.iter().enumerate() {
        s.native_entries.push(call.entry.clone());
        let owner = call.entry.owner;
        let mut regs = call.entry.volatile_registers;
        let mut xmm = call.entry.volatile_xmm_hex.clone();
        let mut counts = BTreeMap::new();
        if s.metadata_flag == 0 {
            for (slot, site) in SLOTS.into_iter().zip(META_SITES) {
                let v = s.metadata_slots[&slot];
                regs[1] = c.native_base + u64::from(slot.rva());
                emit(
                    &mut steps,
                    &mut s,
                    c,
                    i,
                    &mut counts,
                    Completed::Metadata { slot, value: v },
                    regs,
                    xmm,
                    site,
                );
                regs = c.volatile_return_registers;
                regs[0] = v;
                xmm = c.volatile_return_xmm_hex.clone();
            }
            s.metadata_flag = 1;
        }
        write(&mut s, owner, 0xC0, &0u64.to_le_bytes());
        regs[1] = owner + 0xC0;
        regs[2] = 0;
        emit(
            &mut steps,
            &mut s,
            c,
            i,
            &mut counts,
            Completed::ReferenceBarrier { owner },
            regs,
            xmm,
            0x3B4E10,
        );
        regs = c.volatile_return_registers;
        regs[0] = call.barrier_return_bits;
        xmm = c.volatile_return_xmm_hex.clone();
        regs[1] = 0;
        emit(
            &mut steps,
            &mut s,
            c,
            i,
            &mut counts,
            Completed::CharacterPreferences {
                result: call.saved_result,
            },
            regs,
            xmm,
            0x3B4E17,
        );
        regs = c.volatile_return_registers;
        regs[0] = call.saved_result;
        xmm = c.volatile_return_xmm_hex.clone();
        let list = q(&s, call.saved_result, 0x10);
        regs[1] = c.scratch + 8;
        regs[2] = list;
        regs[3] = s.metadata_slots[&Slot::GetEnumerator];
        emit(
            &mut steps,
            &mut s,
            c,
            i,
            &mut counts,
            Completed::GetEnumerator {
                output: c.scratch + 8,
                list,
                method: regs[3],
                bytes: call.enumerator_bytes,
                entries: call.enumerator_entries.clone(),
            },
            regs,
            xmm,
            0x3B4E3E,
        );
        write(&mut s, c.scratch, 8, &call.enumerator_bytes);
        s.iterator = Some(IteratorState {
            list,
            entries: call.enumerator_entries.clone(),
            cursor: 0,
        });
        regs = c.volatile_return_registers;
        regs[0] = call.get_return_bits;
        xmm = c.volatile_return_xmm_hex.clone();
        let raw = record(&s, c.scratch).bytes[8..32].to_vec();
        xmm[0] = format!(
            "{:032x}",
            u128::from_le_bytes(raw[..16].try_into().unwrap())
        );
        xmm[1] = format!("{:032x}", u64::from_le_bytes(raw[16..].try_into().unwrap()));
        write(&mut s, c.scratch, 32, &raw);
        write(&mut s, c.scratch, 8, &0u64.to_le_bytes());
        write(&mut s, c.scratch, 16, &(c.scratch + 32).to_le_bytes());
        for (j, m) in call.moves.iter().enumerate() {
            regs[1] = c.scratch + 32;
            regs[2] = s.metadata_slots[&Slot::MoveNext];
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::MoveNext {
                    owner: regs[1],
                    method: regs[2],
                    current: m.current,
                    result_bits: m.result_bits,
                },
                regs,
                xmm,
                0x3B4E7C,
            );
            s.iterator.as_mut().unwrap().cursor += 1;
            write(&mut s, c.scratch, 40, &((j + 1) as u32).to_le_bytes());
            write(&mut s, c.scratch, 48, &m.current.to_le_bytes());
            regs = c.volatile_return_registers;
            regs[0] = m.result_bits;
            xmm = c.volatile_return_xmm_hex.clone();
            if m.result_bits as u8 == 0 {
                break;
            }
            let pref = q(&s, c.scratch, 48);
            regs[3] = 0;
            regs[2] = q(&s, owner, 0x18);
            regs[1] = q(&s, pref, 0x10);
            let equal = call.equality_results[j];
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::StringEquality {
                    a: regs[1],
                    b: regs[2],
                    result_bits: equal,
                },
                regs,
                xmm,
                0x3B4E9A,
            );
            regs = c.volatile_return_registers;
            regs[0] = equal;
            xmm = c.volatile_return_xmm_hex.clone();
            if equal as u8 == 0 {
                continue;
            }
            regs[3] = 0;
            regs[2] = q(&s, pref, 0x18);
            regs[1] = owner;
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::LoadSkin {
                    owner,
                    skin_id: regs[2],
                },
                regs,
                xmm,
                0x3B4EAD,
            );
            regs = c.volatile_return_registers;
            regs[0] = call.load_skin_return_bits;
            xmm = c.volatile_return_xmm_hex.clone();
        }
        regs[1] = c.scratch + 32;
        regs[2] = s.metadata_slots[&Slot::Dispose];
        emit(
            &mut steps,
            &mut s,
            c,
            i,
            &mut counts,
            Completed::Dispose {
                owner: regs[1],
                method: regs[2],
            },
            regs,
            xmm,
            0x3B4EBE,
        );
        regs = c.volatile_return_registers;
        regs[0] = call.dispose_return_bits;
        xmm = c.volatile_return_xmm_hex.clone();
        completed.push(s.clone());
        finals.push(regs);
        final_xmm.push(xmm);
    }
    Ok(Replay {
        steps,
        completed,
        final_registers: finals,
        final_xmm_hex: final_xmm,
        final_state: s,
    })
}
#[cfg(test)]
#[path = "character_data_preferences_tests.rs"]
mod tests;
