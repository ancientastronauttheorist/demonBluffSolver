//! Exact nominal CharacterData consumers with inert supplied Unity/runtime services.
//! Complete authored records and service-entry state are retained. Callback writes,
//! native guard/failure paths, engine liveness and localization remain excluded.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHARACTER_DATA_CONSUMERS_NATIVE_V1: &str = "character_data_consumers_native_v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Data,
    Skin,
    Class,
    String,
    Sprite,
    Translation,
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
    GetCharacterName,
    GetIWas,
    GetGender,
    GetTranslation,
    GetArt,
    GetAnimatedArt,
    GetArtType,
    GetArtistName,
}
impl Method {
    fn flag(self) -> Option<usize> {
        match self {
            Self::GetArt => Some(0),
            Self::GetAnimatedArt => Some(1),
            Self::GetArtType => Some(2),
            Self::GetArtistName => Some(3),
            _ => None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Comparison {
    pub inequality: bool,
    pub skin: Identity,
    pub method_info: u64,
    pub result_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub metadata_flags: [u8; 4],
    pub comparisons: Vec<Comparison>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub method: Method,
    pub comparison_return_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub data: Identity,
    pub object_class: Identity,
    pub default_artist: Identity,
    pub state: State,
    pub calls: Vec<Call>,
    pub storage_verified: bool,
    pub supplied_inert_verified: bool,
    pub normal_completion_verified: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    ObjectMetadata,
    ArtistMetadata,
    InitializeObject(Identity),
    Compare(Comparison),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub event: Event,
    pub state: State,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replay {
    pub steps: Vec<Step>,
    pub return_bits: Vec<u64>,
    pub final_state: State,
}
fn word(r: &Record, off: usize) -> u64 {
    u64::from_le_bytes(r.bytes[off..off + 8].try_into().expect("validated record"))
}
fn dword(r: &Record, off: usize) -> u32 {
    u32::from_le_bytes(r.bytes[off..off + 4].try_into().expect("validated record"))
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    // Reserve every future full state copy and comparison growth before maps/clones.
    let units = (|| {
        let mut n = 64usize.checked_add(c.version.len())?;
        for r in &c.state.records {
            n = n.checked_add(4)?.checked_add(r.bytes.len())?;
        }
        n = n.checked_add(c.state.comparisons.len().checked_mul(4)?)?;
        n.checked_add(c.calls.len().checked_mul(2)?)
    })();
    let work = units.and_then(|n| {
        n.checked_add(c.calls.len().checked_mul(4)?)?
            .checked_mul(c.calls.len().checked_mul(5)?.checked_add(2)?)
    });
    if c.state.records.len() > 32
        || c.calls.len() > 16
        || c.state.comparisons.len() > 128
        || units.is_none_or(|n| n > 8192)
        || work.is_none_or(|n| n > 262_144)
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != CHARACTER_DATA_CONSUMERS_NATIVE_V1
        || !c.storage_verified
        || !c.supplied_inert_verified
        || !c.normal_completion_verified
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        let len = match r.kind {
            Kind::Data => 0x180,
            Kind::Skin | Kind::Class => 0x100,
            _ => 0x80,
        };
        if r.identity == 0 || r.bytes.len() != len || records.insert(r.identity, r).is_some() {
            return Err(LedgerError::InvalidContext);
        }
    }
    let reference = |p: u64, k: Kind, nullable: bool| {
        p == 0 && nullable || records.get(&p).is_some_and(|r| r.kind == k)
    };
    if !reference(c.data, Kind::Data, false)
        || !reference(c.object_class, Kind::Class, false)
        || !reference(c.default_artist, Kind::String, false)
    {
        return Err(LedgerError::InvalidContext);
    }
    for r in &c.state.records {
        let fields: &[(usize, Kind)] = match r.kind {
            Kind::Data => &[
                (0x28, Kind::String),
                (0x30, Kind::String),
                (0x98, Kind::Sprite),
                (0xA8, Kind::Sprite),
                (0xC0, Kind::Skin),
                (0x148, Kind::Translation),
            ],
            Kind::Skin => &[
                (0x20, Kind::String),
                (0x38, Kind::Sprite),
                (0x40, Kind::Sprite),
            ],
            _ => &[],
        };
        for &(off, kind) in fields {
            if !reference(word(r, off), kind, true) {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    for p in &c.state.comparisons {
        if p.method_info != 0 || !reference(p.skin, Kind::Skin, true) {
            return Err(LedgerError::InvalidContext);
        }
    }
    // AL choosing the skin path requires a physical receiver. Reject before replay.
    let skin = word(records[&c.data], 0xC0);
    for call in &c.calls {
        if call.method.flag().is_some() {
            let use_skin = if call.method == Method::GetArtistName {
                call.comparison_return_bits as u8 != 0
            } else {
                call.comparison_return_bits as u8 == 0
            };
            if use_skin && skin == 0 {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    Ok(())
}
fn record(s: &State, p: u64) -> &Record {
    s.records
        .iter()
        .find(|r| r.identity == p)
        .expect("validated identity")
}
fn emit(steps: &mut Vec<Step>, s: &State, event: Event) {
    steps.push(Step {
        event,
        state: s.clone(),
    });
}
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut s = c.state.clone();
    let mut steps = Vec::new();
    let mut returns = Vec::new();
    for call in &c.calls {
        let method = call.method;
        let bits = if let Some(flag) = method.flag() {
            if s.metadata_flags[flag] == 0 {
                emit(&mut steps, &s, Event::ObjectMetadata);
                if method == Method::GetArtistName {
                    emit(&mut steps, &s, Event::ArtistMetadata);
                }
                s.metadata_flags[flag] = 1;
            }
            let skin = word(record(&s, c.data), 0xC0);
            let artist = c.default_artist;
            if dword(record(&s, c.object_class), 0xE0) == 0 {
                emit(&mut steps, &s, Event::InitializeObject(c.object_class));
                let r = s
                    .records
                    .iter_mut()
                    .find(|r| r.identity == c.object_class)
                    .expect("validated class");
                r.bytes[0xE0..0xE4].copy_from_slice(&1u32.to_le_bytes());
            }
            let compare = Comparison {
                inequality: method == Method::GetArtistName,
                skin,
                method_info: 0,
                result_bits: call.comparison_return_bits,
            };
            emit(&mut steps, &s, Event::Compare(compare.clone()));
            s.comparisons.push(compare);
            let default = if method == Method::GetArtistName {
                call.comparison_return_bits as u8 == 0
            } else {
                call.comparison_return_bits as u8 != 0
            };
            if default {
                match method {
                    Method::GetArt => word(record(&s, c.data), 0x98),
                    Method::GetAnimatedArt => word(record(&s, c.data), 0xA8),
                    Method::GetArtType => 0,
                    Method::GetArtistName => artist,
                    _ => unreachable!(),
                }
            } else {
                let current = word(record(&s, c.data), 0xC0);
                let r = record(&s, current);
                match method {
                    Method::GetArt => word(r, 0x38),
                    Method::GetAnimatedArt => word(r, 0x40),
                    Method::GetArtType => dword(r, 0x50) as u64,
                    Method::GetArtistName => word(r, 0x20),
                    _ => unreachable!(),
                }
            }
        } else {
            let r = record(&s, c.data);
            match method {
                Method::GetCharacterName => word(r, 0x28),
                Method::GetIWas => word(r, 0x30),
                Method::GetGender => dword(r, 0x38) as u64,
                Method::GetTranslation => word(r, 0x148),
                _ => unreachable!(),
            }
        };
        returns.push(bits);
    }
    Ok(Replay {
        steps,
        return_bits: returns,
        final_state: s,
    })
}

#[cfg(test)]
#[path = "character_data_consumers_tests.rs"]
mod tests;
