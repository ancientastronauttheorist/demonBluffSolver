//! Exact normal CharacterData text callers with inert supplied services.
//! Full diagnostic storage and service-entry ABI are preserved. Localization,
//! RNG implementation, callbacks, guard paths and runtime admission are excluded.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHARACTER_DATA_TEXT_NATIVE_V1: &str = "character_data_text_consumers_native_v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Data,
    Class,
    Array,
    Translation,
    String,
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
    GetFlavorText,
    GetTranslatedName,
    GetIWasTranslated,
    GetIfLies,
    GetHints,
    UpdateCharacterName,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub method: Method,
    pub raw_args: [u64; 4],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Completed {
    Random {
        maximum: u32,
        result_bits: u64,
    },
    Translation {
        method: Method,
        receiver: Identity,
        locale: Identity,
        result: Identity,
    },
    Name {
        owner: Identity,
        result: Identity,
    },
    Converter {
        input: Identity,
        result: Identity,
    },
    Barrier {
        owner: Identity,
        value: Identity,
    },
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
    pub method: Method,
    pub entry_raw_args: [u64; 4],
    pub random_result_bits: u64,
    pub translation_result: Identity,
    pub name_result: Identity,
    pub converter_result: Identity,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub calls: Vec<Call>,
    pub native_base: u64,
    pub caller_return_bits: u64,
    pub volatile_return_bits: [u64; 4],
    pub storage_verified: bool,
    pub inert_services_verified: bool,
    pub normal_completion_verified: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub call: usize,
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
    pub return_bits: Vec<u64>,
    pub final_state: State,
}
fn record(s: &State, p: Identity) -> &Record {
    s.records
        .iter()
        .find(|r| r.identity == p)
        .expect("validated identity")
}
fn word(r: &Record, off: usize) -> u64 {
    u64::from_le_bytes(r.bytes[off..off + 8].try_into().expect("validated record"))
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let units = (|| {
        let mut n = 96usize.checked_add(c.version.len())?;
        for r in &c.state.records {
            n = n.checked_add(4)?.checked_add(r.bytes.len())?;
        }
        n = n
            .checked_add(c.state.native_entries.len().checked_mul(5)?)?
            .checked_add(c.state.service_history.len().checked_mul(6)?)?;
        n.checked_add(c.calls.len().checked_mul(9)?)
    })();
    let work = units.and_then(|n| {
        n.checked_add(c.calls.len().checked_mul(17)?)?
            .checked_mul(c.calls.len().checked_mul(4)?.checked_add(2)?)
    });
    if c.state.records.len() > 32
        || c.calls.len() > 16
        || c.state.native_entries.len() > 128
        || c.state.service_history.len() > 128
        || units.is_none_or(|v| v > 8192)
        || work.is_none_or(|v| v > 262_144)
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != CHARACTER_DATA_TEXT_NATIVE_V1
        || !c.storage_verified
        || !c.inert_services_verified
        || !c.normal_completion_verified
        || c.native_base == 0
        || c.native_base.checked_add(0x3B5094).is_none()
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        let len = match r.kind {
            Kind::Data => 512,
            Kind::Class | Kind::Array => 256,
            _ => 128,
        };
        if r.identity == 0 || r.bytes.len() != len || records.insert(r.identity, r).is_some() {
            return Err(LedgerError::InvalidContext);
        }
        let Some(end) = r.identity.checked_add(len as u64) else {
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
    let reference = |p: Identity, k: Kind, null: bool| {
        p == 0 && null || records.get(&p).is_some_and(|r| r.kind == k)
    };
    for call in &c.calls {
        let owner = call.entry_raw_args[0];
        if !reference(owner, Kind::Data, false)
            || !reference(call.translation_result, Kind::String, true)
            || !reference(call.name_result, Kind::String, true)
            || !reference(call.converter_result, Kind::String, true)
        {
            return Err(LedgerError::InvalidContext);
        }
        let data = records[&owner];
        match call.method {
            Method::GetFlavorText => {
                let array = word(data, 0x70);
                if !reference(array, Kind::Array, false) {
                    return Err(LedgerError::InvalidContext);
                }
                let r = records[&array];
                let length = word(r, 0x18);
                if length == 0 {
                    if !reference(word(data, 0x68), Kind::String, true) {
                        return Err(LedgerError::InvalidContext);
                    }
                } else {
                    let index = call.random_result_bits as u32;
                    if index >= length as u32
                        || index >= 3
                        || !reference(word(r, 0x20 + index as usize * 8), Kind::String, true)
                    {
                        return Err(LedgerError::InvalidContext);
                    }
                }
            }
            Method::GetTranslatedName | Method::GetIWasTranslated => {
                if !reference(word(data, 0x148), Kind::Translation, true)
                    || !reference(call.entry_raw_args[1], Kind::String, true)
                {
                    return Err(LedgerError::InvalidContext);
                }
            }
            Method::GetHints | Method::GetIfLies => {
                let off = if call.method == Method::GetHints {
                    0x78
                } else {
                    0x80
                };
                if !reference(word(data, off), Kind::String, true) {
                    return Err(LedgerError::InvalidContext);
                }
            }
            Method::UpdateCharacterName => {}
        }
    }
    for h in &c.state.service_history {
        let valid = match h {
            Completed::Random { .. } => true,
            Completed::Translation {
                method,
                receiver,
                locale,
                result,
            } => {
                matches!(
                    method,
                    Method::GetTranslatedName | Method::GetIWasTranslated
                ) && reference(*receiver, Kind::Translation, false)
                    && reference(*locale, Kind::String, true)
                    && reference(*result, Kind::String, true)
            }
            Completed::Name { owner, result } => {
                reference(*owner, Kind::Data, false) && reference(*result, Kind::String, true)
            }
            Completed::Converter { input, result } => {
                reference(*input, Kind::String, true) && reference(*result, Kind::String, true)
            }
            Completed::Barrier { owner, value } => {
                reference(*owner, Kind::Data, false) && reference(*value, Kind::String, true)
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
    s: &mut State,
    c: &Context,
    call: usize,
    event: Completed,
    args: [u64; 4],
    site: u32,
    tail: bool,
) {
    steps.push(Step {
        call,
        event: event.clone(),
        raw_args: args,
        native_site_rva: site,
        caller_return_bits: if tail {
            c.caller_return_bits
        } else {
            c.native_base + site as u64 + 5
        },
        state: s.clone(),
    });
    s.service_history.push(event);
}
/// Validate every call before cloning; unsupported graphs fall back atomically.
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut s = c.state.clone();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    let mut returns = Vec::new();
    for (i, call) in c.calls.iter().enumerate() {
        let mut regs = call.entry_raw_args;
        let owner = regs[0];
        s.native_entries.push(Entry {
            method: call.method,
            raw_args: regs,
        });
        let result = match call.method {
            Method::GetFlavorText => {
                let array = word(record(&s, owner), 0x70);
                let length = word(record(&s, array), 0x18);
                if length == 0 {
                    word(record(&s, owner), 0x68)
                } else {
                    regs = [0, length as u32 as u64, 0, regs[3]];
                    emit(
                        &mut steps,
                        &mut s,
                        c,
                        i,
                        Completed::Random {
                            maximum: length as u32,
                            result_bits: call.random_result_bits,
                        },
                        regs,
                        0x3B4CC8,
                        false,
                    );
                    word(
                        record(&s, array),
                        0x20 + (call.random_result_bits as u32 as usize) * 8,
                    )
                }
            }
            Method::GetTranslatedName | Method::GetIWasTranslated => {
                let translation = word(record(&s, owner), 0x148);
                let mut result = 0;
                if translation != 0 {
                    regs = [translation, regs[1], 0, regs[3]];
                    emit(
                        &mut steps,
                        &mut s,
                        c,
                        i,
                        Completed::Translation {
                            method: call.method,
                            receiver: translation,
                            locale: regs[1],
                            result: call.translation_result,
                        },
                        regs,
                        if call.method == Method::GetTranslatedName {
                            0x3B4D78
                        } else {
                            0x3B4D28
                        },
                        false,
                    );
                    result = call.translation_result;
                    regs = c.volatile_return_bits;
                }
                if result == 0 {
                    regs[0] = owner;
                    regs[1] = 0;
                    emit(
                        &mut steps,
                        &mut s,
                        c,
                        i,
                        Completed::Name {
                            owner,
                            result: call.name_result,
                        },
                        regs,
                        if call.method == Method::GetTranslatedName {
                            0x3B4D8C
                        } else {
                            0x3B4D3C
                        },
                        true,
                    );
                    result = call.name_result;
                }
                result
            }
            Method::GetHints | Method::GetIfLies => {
                regs[0] = word(
                    record(&s, owner),
                    if call.method == Method::GetHints {
                        0x78
                    } else {
                        0x80
                    },
                );
                regs[1] = 0;
                emit(
                    &mut steps,
                    &mut s,
                    c,
                    i,
                    Completed::Converter {
                        input: regs[0],
                        result: call.converter_result,
                    },
                    regs,
                    if call.method == Method::GetHints {
                        0x3B4D06
                    } else {
                        0x3B4D59
                    },
                    true,
                );
                call.converter_result
            }
            Method::UpdateCharacterName => {
                regs[1] = 0;
                emit(
                    &mut steps,
                    &mut s,
                    c,
                    i,
                    Completed::Name {
                        owner,
                        result: call.name_result,
                    },
                    regs,
                    0x3B507B,
                    false,
                );
                s.records
                    .iter_mut()
                    .find(|r| r.identity == owner)
                    .expect("validated owner")
                    .bytes[0x28..0x30]
                    .copy_from_slice(&call.name_result.to_le_bytes());
                regs = c.volatile_return_bits;
                regs[0] = owner + 0x28;
                regs[1] = call.name_result;
                emit(
                    &mut steps,
                    &mut s,
                    c,
                    i,
                    Completed::Barrier {
                        owner,
                        value: call.name_result,
                    },
                    regs,
                    0x3B508F,
                    true,
                );
                0
            }
        };
        returns.push(result);
        completed.push(s.clone());
    }
    Ok(Replay {
        steps,
        completed,
        return_bits: returns,
        final_state: s,
    })
}
#[cfg(test)]
#[path = "character_data_text_consumers_tests.rs"]
mod tests;
