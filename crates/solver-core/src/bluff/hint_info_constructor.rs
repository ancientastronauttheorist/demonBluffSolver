//! Exact HintInfo constructor with nominal authored storage and inert barriers.
//! No allocation, callback mutation, failed access, real GC or renderer is modeled.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
pub const HINT_INFO_CONSTRUCTOR_NATIVE_V1: &str = "hint_info_constructor_native_v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Hint,
    String,
    Sprite,
    Color,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub identity: Identity,
    pub kind: Kind,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Field {
    Text,
    Title,
    Image,
    Hints,
    Flavor,
}
impl Field {
    fn offset(self) -> usize {
        match self {
            Self::Title => 0x10,
            Self::Text => 0x18,
            Self::Hints => 0x20,
            Self::Flavor => 0x28,
            Self::Image => 0x30,
        }
    }
    fn caller(self) -> u32 {
        match self {
            Self::Text => 0x3BC236,
            Self::Title => 0x3BC247,
            Self::Image => 0x3BC256,
            Self::Hints => 0x3BC265,
            Self::Flavor => 0x3BC276,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    pub owner: Identity,
    pub text: Identity,
    pub image: Identity,
    pub hints: Identity,
    pub flavor: Identity,
    pub title: Identity,
    pub color: Identity,
    pub method_info: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Barrier {
    pub owner: Identity,
    pub field: Field,
    pub value_bits: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeEntry {
    HintConstructor,
    FoldedObjectReturn,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub arguments: Arguments,
    pub completed_barriers: Vec<Barrier>,
    pub native_entries: Vec<NativeEntry>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub calls: Vec<Arguments>,
    pub storage_verified: bool,
    pub inert_barriers_verified: bool,
    pub normal_completion_verified: bool,
    pub barrier_volatile_r8_bits: u64,
    pub barrier_volatile_r9_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub barrier: Barrier,
    pub raw_args: [u64; 4],
    pub caller_return_rva: u32,
    pub state: State,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replay {
    pub steps: Vec<Step>,
    pub final_state: State,
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let units = (|| {
        let mut n = 80usize.checked_add(c.version.len())?;
        for r in &c.state.records {
            n = n.checked_add(4)?.checked_add(r.bytes.len())?;
        }
        n = n
            .checked_add(c.state.completed_barriers.len().checked_mul(4)?)?
            .checked_add(c.state.native_entries.len())?;
        n.checked_add(c.calls.len().checked_mul(8)?)
    })();
    let work = units.and_then(|n| {
        n.checked_add(c.calls.len().checked_mul(22)?)?
            .checked_mul(c.calls.len().checked_mul(6)?.checked_add(2)?)
    });
    if c.state.records.len() > 32
        || c.calls.len() > 16
        || c.state.completed_barriers.len() > 128
        || c.state.native_entries.len() > 128
        || units.is_none_or(|n| n > 8192)
        || work.is_none_or(|n| n > 262_144)
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != HINT_INFO_CONSTRUCTOR_NATIVE_V1
        || !c.storage_verified
        || !c.inert_barriers_verified
        || !c.normal_completion_verified
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        if r.identity == 0 || r.bytes.len() != 128 || records.insert(r.identity, r).is_some() {
            return Err(LedgerError::InvalidContext);
        }
    }
    let mut previous_end = 0;
    for (&id, _) in &records {
        let Some(end) = id.checked_add(128) else {
            return Err(LedgerError::InvalidContext);
        };
        if id < previous_end {
            return Err(LedgerError::InvalidContext);
        }
        previous_end = end;
    }
    let valid = |p: u64, k: Kind, nullable: bool| {
        p == 0 && nullable || records.get(&p).is_some_and(|r| r.kind == k)
    };
    for a in std::iter::once(&c.state.arguments).chain(&c.calls) {
        if !valid(a.owner, Kind::Hint, false)
            || !valid(a.color, Kind::Color, false)
            || !valid(a.image, Kind::Sprite, true)
            || [a.text, a.hints, a.flavor, a.title]
                .into_iter()
                .any(|p| !valid(p, Kind::String, true))
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if c.calls.first().is_some_and(|a| a != &c.state.arguments) {
        return Err(LedgerError::InvalidContext);
    }
    for b in &c.state.completed_barriers {
        if !valid(b.owner, Kind::Hint, false)
            || !valid(
                b.value_bits,
                if b.field == Field::Image {
                    Kind::Sprite
                } else {
                    Kind::String
                },
                true,
            )
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    Ok(())
}
fn record(s: &State, id: u64) -> &Record {
    s.records
        .iter()
        .find(|r| r.identity == id)
        .expect("validated identity")
}
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut state = c.state.clone();
    let mut steps = Vec::new();
    for a in &c.calls {
        state.arguments = a.clone();
        state.native_entries.extend([
            NativeEntry::HintConstructor,
            NativeEntry::FoldedObjectReturn,
        ]);
        for (i, (field, value)) in [
            (Field::Text, a.text),
            (Field::Title, a.title),
            (Field::Image, a.image),
            (Field::Hints, a.hints),
            (Field::Flavor, a.flavor),
        ]
        .into_iter()
        .enumerate()
        {
            let r = state
                .records
                .iter_mut()
                .find(|r| r.identity == a.owner)
                .expect("validated receiver");
            r.bytes[field.offset()..field.offset() + 8].copy_from_slice(&value.to_le_bytes());
            let barrier = Barrier {
                owner: a.owner,
                field,
                value_bits: value,
            };
            steps.push(Step {
                barrier: barrier.clone(),
                raw_args: [
                    a.owner + field.offset() as u64,
                    value,
                    if i == 0 {
                        a.image
                    } else {
                        c.barrier_volatile_r8_bits
                    },
                    if i == 0 {
                        a.hints
                    } else {
                        c.barrier_volatile_r9_bits
                    },
                ],
                caller_return_rva: field.caller(),
                state: state.clone(),
            });
            state.completed_barriers.push(barrier);
        }
        let color: [u8; 16] = record(&state, a.color).bytes[..16]
            .try_into()
            .expect("validated color");
        let hint = state
            .records
            .iter_mut()
            .find(|r| r.identity == a.owner)
            .expect("validated receiver");
        hint.bytes[0x38..0x48].copy_from_slice(&color);
    }
    Ok(Replay {
        steps,
        final_state: state,
    })
}
#[cfg(test)]
#[path = "hint_info_constructor_tests.rs"]
mod tests;
