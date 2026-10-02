//! Exact normal CharacterData.GetDescription with inert supplied conversion.
//! Full authored storage and raw service-entry ABI are retained. Callbacks,
//! guards/faults, localization implementation and runtime admission are excluded.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHARACTER_DATA_DESCRIPTION_NATIVE_V1: &str = "character_data_description_native_v1";
const SLOT: u32 = 0x271F268;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Data,
    Class,
    Statics,
    ProjectContext,
    GameData,
    String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub identity: Identity,
    pub kind: Kind,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Service {
    Metadata,
    Converter,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Completed {
    Metadata { class: Identity },
    Converter { input: Identity, result: Identity },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub metadata_flag: u8,
    pub project_class: Identity,
    pub native_entries: Vec<[u64; 4]>,
    pub service_history: Vec<Completed>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub converter_results: [Identity; 2],
    pub entry_method_bits: u64,
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
    pub volatile_return_bits: [u64; 4],
    pub storage_verified: bool,
    pub inert_services_verified: bool,
    pub normal_completion_verified: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub call: usize,
    pub ordinal: u64,
    pub event: Completed,
    pub raw_args: [u64; 4],
    pub native_site_rva: u32,
    pub caller_return_bits: u64,
    pub state: State,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replay {
    pub steps: Vec<Step>,
    pub completed: Vec<State>,
    pub return_bits: Vec<Identity>,
    pub service_counts: BTreeMap<Service, u64>,
    pub final_state: State,
}
fn word(r: &Record, off: usize) -> u64 {
    u64::from_le_bytes(r.bytes[off..off + 8].try_into().expect("validated record"))
}
fn dword(r: &Record, off: usize) -> u32 {
    u32::from_le_bytes(r.bytes[off..off + 4].try_into().expect("validated record"))
}
fn record(s: &State, id: Identity) -> &Record {
    s.records
        .iter()
        .find(|r| r.identity == id)
        .expect("validated identity")
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    // Bound the complete input, all future history growth and every full snapshot
    // before constructing reference maps or cloning authored storage.
    let units = (|| {
        let mut n = 96usize.checked_add(c.version.len())?;
        for r in &c.state.records {
            n = n.checked_add(4)?.checked_add(r.bytes.len())?;
        }
        n = n
            .checked_add(c.state.native_entries.len().checked_mul(4)?)?
            .checked_add(c.state.service_history.len().checked_mul(4)?)?
            .checked_add(c.service_counts.len().checked_mul(2)?)?;
        n.checked_add(c.calls.len().checked_mul(6)?)
    })();
    let work = units.and_then(|n| {
        n.checked_add(c.calls.len().checked_mul(16)?)?
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
    if c.version != CHARACTER_DATA_DESCRIPTION_NATIVE_V1
        || !c.storage_verified
        || !c.inert_services_verified
        || !c.normal_completion_verified
        || c.native_base == 0
        || c.native_base.checked_add(SLOT as u64).is_none()
        || c.service_counts
            .values()
            .any(|&v| v.checked_add((c.calls.len() * 3) as u64).is_none())
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        let length = match r.kind {
            Kind::Data => 512,
            Kind::Class => 256,
            _ => 128,
        };
        if r.identity == 0 || r.bytes.len() != length || records.insert(r.identity, r).is_some() {
            return Err(LedgerError::InvalidContext);
        }
        let Some(end) = r.identity.checked_add(length as u64) else {
            return Err(LedgerError::InvalidContext);
        };
        if c.state.records.iter().any(|other| {
            other.identity != r.identity
                && other.identity < end
                && other
                    .identity
                    .checked_add(other.bytes.len() as u64)
                    .is_none_or(|e| e > r.identity)
        }) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let reference = |id: Identity, kind: Kind, nullable: bool| {
        id == 0 && nullable || records.get(&id).is_some_and(|r| r.kind == kind)
    };
    if !reference(c.owner, Kind::Data, false)
        || !reference(c.state.project_class, Kind::Class, false)
    {
        return Err(LedgerError::InvalidContext);
    }
    let static_id = word(records[&c.state.project_class], 0xB8);
    if !reference(static_id, Kind::Statics, false) {
        return Err(LedgerError::InvalidContext);
    }
    let context_id = word(records[&static_id], 0);
    if !reference(context_id, Kind::ProjectContext, false) {
        return Err(LedgerError::InvalidContext);
    }
    let game_id = word(records[&context_id], 0x20);
    if !reference(game_id, Kind::GameData, false)
        || !reference(word(records[&c.owner], 0x50), Kind::String, true)
    {
        return Err(LedgerError::InvalidContext);
    }
    for call in &c.calls {
        if call
            .converter_results
            .iter()
            .any(|&id| !reference(id, Kind::String, true))
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    for h in &c.state.service_history {
        let valid = match h {
            Completed::Metadata { class } => reference(*class, Kind::Class, false),
            Completed::Converter { input, result } => {
                reference(*input, Kind::String, true) && reference(*result, Kind::String, true)
            }
        };
        if !valid {
            return Err(LedgerError::InvalidContext);
        }
    }
    Ok(())
}
fn emit(
    steps: &mut Vec<Step>,
    counts: &mut BTreeMap<Service, u64>,
    state: &mut State,
    call: usize,
    event: Completed,
    raw_args: [u64; 4],
    site: u32,
    base: u64,
) {
    let service = match event {
        Completed::Metadata { .. } => Service::Metadata,
        Completed::Converter { .. } => Service::Converter,
    };
    let ordinal = counts.entry(service).or_default();
    *ordinal += 1;
    steps.push(Step {
        call,
        ordinal: *ordinal,
        event: event.clone(),
        raw_args,
        native_site_rva: site,
        caller_return_bits: base + site as u64 + 5,
        state: state.clone(),
    });
    state.service_history.push(event);
}

/// Atomic all-or-fallback replay. Supplied converter outputs are explicit
/// nominal identities; no text conversion or engine admission is inferred.
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut state = c.state.clone();
    let mut counts = c.service_counts.clone();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    let mut returns = Vec::new();
    for (i, call) in c.calls.iter().enumerate() {
        let mut regs = [
            c.owner,
            call.entry_method_bits,
            call.entry_r8_bits,
            call.entry_r9_bits,
        ];
        state.native_entries.push(regs);
        if state.metadata_flag == 0 {
            regs[0] = c.native_base + SLOT as u64;
            let class = state.project_class;
            emit(
                &mut steps,
                &mut counts,
                &mut state,
                i,
                Completed::Metadata { class },
                regs,
                0x3B4C09,
                c.native_base,
            );
            state.metadata_flag = 1;
            regs = c.volatile_return_bits;
        }
        let input = word(record(&state, c.owner), 0x50);
        regs[0] = input;
        regs[1] = 0;
        emit(
            &mut steps,
            &mut counts,
            &mut state,
            i,
            Completed::Converter {
                input,
                result: call.converter_results[0],
            },
            regs,
            0x3B4C1B,
            c.native_base,
        );
        regs = c.volatile_return_bits;
        let static_id = word(record(&state, state.project_class), 0xB8);
        let context_id = word(record(&state, static_id), 0);
        let game_id = word(record(&state, context_id), 0x20);
        let language = dword(record(&state, game_id), 0x20);
        let mut result = call.converter_results[0];
        if matches!(language, 0 | 10) {
            regs[0] = input;
            regs[1] = 0;
            regs[2] = state.project_class;
            let site = if language == 0 { 0x3B4C4E } else { 0x3B4C81 };
            emit(
                &mut steps,
                &mut counts,
                &mut state,
                i,
                Completed::Converter {
                    input,
                    result: call.converter_results[1],
                },
                regs,
                site,
                c.native_base,
            );
            result = call.converter_results[1];
        }
        returns.push(result);
        completed.push(state.clone());
    }
    Ok(Replay {
        steps,
        completed,
        return_bits: returns,
        service_counts: counts,
        final_state: state,
    })
}

#[cfg(test)]
#[path = "character_data_description_tests.rs"]
mod tests;
