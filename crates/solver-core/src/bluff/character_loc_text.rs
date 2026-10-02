//! Exact CharacterLoc getters with inert supplied locale search/emptiness.
//! Diagnostic records, complete volatile ABI and prior histories are retained;
//! locale/string algorithms, callbacks and runtime admission remain excluded.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
pub const CHARACTER_LOC_TEXT_NATIVE_V1: &str = "character_loc_text_native_v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Owner,
    LocaleLoc,
    Class,
    String,
    Entries,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub identity: Identity,
    pub kind: Kind,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    GetTranslatedName,
    GetIWasTranslated,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub method: Method,
    pub volatile_registers: [u64; 7],
    pub volatile_xmm_hex: [String; 6],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Completed {
    FindLocale {
        owner: Identity,
        code: Identity,
        result: Identity,
    },
    IsNullOrEmpty {
        input: Identity,
        result_bits: u64,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Service {
    FindLocale,
    IsNullOrEmpty,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub native_entries: Vec<Entry>,
    pub service_history: Vec<Completed>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub entry: Entry,
    pub search_result: Identity,
    pub empty_result_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub calls: Vec<Call>,
    pub service_counts: BTreeMap<Service, u64>,
    pub native_base: u64,
    pub volatile_return_registers: [u64; 7],
    pub volatile_return_xmm_hex: [String; 6],
    pub storage_verified: bool,
    pub inert_services_verified: bool,
    pub normal_completion_verified: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub call: usize,
    pub ordinal: u64,
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
    pub return_bits: Vec<Identity>,
    pub final_registers: Vec<[u64; 7]>,
    pub final_xmm_hex: Vec<[String; 6]>,
    pub service_counts: BTreeMap<Service, u64>,
    pub final_state: State,
}
fn word(r: &Record, off: usize) -> u64 {
    u64::from_le_bytes(r.bytes[off..off + 8].try_into().expect("validated record"))
}
fn record(s: &State, id: Identity) -> &Record {
    s.records
        .iter()
        .find(|r| r.identity == id)
        .expect("validated identity")
}
fn xmm_valid(values: &[String; 6]) -> bool {
    values
        .iter()
        .all(|v| v.len() == 32 && v.bytes().all(|c| c.is_ascii_hexdigit()))
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let units = (|| {
        let mut n = 128usize.checked_add(c.version.len())?;
        for r in &c.state.records {
            n = n.checked_add(4)?.checked_add(r.bytes.len())?;
        }
        for entry in c
            .state
            .native_entries
            .iter()
            .chain(c.calls.iter().map(|v| &v.entry))
        {
            n = n.checked_add(8)?;
            for x in &entry.volatile_xmm_hex {
                n = n.checked_add(x.len())?;
            }
        }
        for x in &c.volatile_return_xmm_hex {
            n = n.checked_add(x.len())?;
        }
        n = n
            .checked_add(c.state.service_history.len().checked_mul(5)?)?
            .checked_add(c.service_counts.len().checked_mul(2)?)?;
        n.checked_add(c.calls.len().checked_mul(2)?)
    })();
    // Every new entry retains seven integers and six complete hex registers.
    let work = units.and_then(|n| {
        n.checked_add(c.calls.len().checked_mul(212)?)?
            .checked_mul(c.calls.len().checked_mul(4)?.checked_add(2)?)
    });
    if c.state.records.len() > 32
        || c.calls.len() > 16
        || c.state.native_entries.len() > 128
        || c.state.service_history.len() > 128
        || units.is_none_or(|n| n > 8192)
        || work.is_none_or(|n| n > 262_144)
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != CHARACTER_LOC_TEXT_NATIVE_V1
        || !c.storage_verified
        || !c.inert_services_verified
        || !c.normal_completion_verified
        || c.native_base == 0
        || c.native_base.checked_add(0x3F5957).is_none()
        || !xmm_valid(&c.volatile_return_xmm_hex)
        || c.service_counts
            .values()
            .any(|v| v.checked_add(c.calls.len() as u64).is_none())
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        let size = if r.kind == Kind::Class { 256 } else { 128 };
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
    let reference = |id: Identity, kind: Kind, nullable: bool| {
        id == 0 && nullable || records.get(&id).is_some_and(|r| r.kind == kind)
    };
    for call in &c.calls {
        if !xmm_valid(&call.entry.volatile_xmm_hex)
            || !reference(call.entry.volatile_registers[1], Kind::Owner, true)
            || !reference(call.entry.volatile_registers[2], Kind::String, true)
            || !reference(call.search_result, Kind::LocaleLoc, true)
        {
            return Err(LedgerError::InvalidContext);
        }
        if call.search_result != 0 {
            let field = if call.entry.method == Method::GetTranslatedName {
                0x18
            } else {
                0x20
            };
            if !reference(
                word(records[&call.search_result], field),
                Kind::String,
                true,
            ) {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    if c.state
        .native_entries
        .iter()
        .any(|e| !xmm_valid(&e.volatile_xmm_hex))
    {
        return Err(LedgerError::InvalidContext);
    }
    for h in &c.state.service_history {
        let valid = match h {
            Completed::FindLocale {
                owner,
                code,
                result,
            } => {
                reference(*owner, Kind::Owner, true)
                    && reference(*code, Kind::String, true)
                    && reference(*result, Kind::LocaleLoc, true)
            }
            Completed::IsNullOrEmpty { input, .. } => reference(*input, Kind::String, true),
        };
        if !valid {
            return Err(LedgerError::InvalidContext);
        }
    }
    Ok(())
}
fn emit(
    steps: &mut Vec<Step>,
    s: &mut State,
    counts: &mut BTreeMap<Service, u64>,
    c: &Context,
    i: usize,
    event: Completed,
    regs: [u64; 7],
    xmm: [String; 6],
    site: u32,
) {
    let service = match event {
        Completed::FindLocale { .. } => Service::FindLocale,
        Completed::IsNullOrEmpty { .. } => Service::IsNullOrEmpty,
    };
    let ordinal = counts.entry(service).or_default();
    *ordinal += 1;
    steps.push(Step {
        call: i,
        ordinal: *ordinal,
        event: event.clone(),
        volatile_registers: regs,
        volatile_xmm_hex: xmm,
        native_site_rva: site,
        caller_return_bits: c.native_base + site as u64 + 5,
        state: s.clone(),
    });
    s.service_history.push(event);
}
/// Atomic normal replay under explicit whole-service outputs and inert storage.
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut s = c.state.clone();
    let mut counts = c.service_counts.clone();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    let mut returns = Vec::new();
    let mut final_regs = Vec::new();
    let mut final_xmm = Vec::new();
    for (i, call) in c.calls.iter().enumerate() {
        s.native_entries.push(call.entry.clone());
        let mut regs = call.entry.volatile_registers;
        let mut xmm = call.entry.volatile_xmm_hex.clone();
        let start = if call.entry.method == Method::GetTranslatedName {
            0x3F5920
        } else {
            0x3F5690
        };
        regs[3] = 0;
        emit(
            &mut steps,
            &mut s,
            &mut counts,
            c,
            i,
            Completed::FindLocale {
                owner: regs[1],
                code: regs[2],
                result: call.search_result,
            },
            regs,
            xmm,
            start + 9,
        );
        regs = c.volatile_return_registers;
        regs[0] = call.search_result;
        xmm = c.volatile_return_xmm_hex.clone();
        if call.search_result == 0 {
            regs[0] = 0;
        } else {
            let field = if call.entry.method == Method::GetTranslatedName {
                0x18
            } else {
                0x20
            };
            let text = word(record(&s, call.search_result), field);
            regs[1] = text;
            regs[2] = 0;
            emit(
                &mut steps,
                &mut s,
                &mut counts,
                c,
                i,
                Completed::IsNullOrEmpty {
                    input: text,
                    result_bits: call.empty_result_bits,
                },
                regs,
                xmm,
                start + 28,
            );
            regs = c.volatile_return_registers;
            regs[0] = if call.empty_result_bits as u8 != 0 {
                0
            } else {
                text
            };
            xmm = c.volatile_return_xmm_hex.clone();
        }
        returns.push(regs[0]);
        final_regs.push(regs);
        final_xmm.push(xmm);
        completed.push(s.clone());
    }
    Ok(Replay {
        steps,
        completed,
        return_bits: returns,
        final_registers: final_regs,
        final_xmm_hex: final_xmm,
        service_counts: counts,
        final_state: s,
    })
}
#[cfg(test)]
#[path = "character_loc_text_tests.rs"]
mod tests;
