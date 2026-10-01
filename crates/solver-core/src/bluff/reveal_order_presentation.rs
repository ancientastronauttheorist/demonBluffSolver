//! Standalone normal RevealOrder.Init/Hide caller replay with inert supplied
//! Component/GameObject, Int32.ToString and TMP virtual services. Complete
//! supplied physical records are retained; their sentinel bytes are opaque
//! diagnostics, not valid typed fields beyond the explicitly consumed layout.
//! No constructor, formatter, Unity/TMP internals, mutations, stops or rendering
//! is implemented. Each call supplies its caller-owned stack slot independently.
//! Formatter observations use the fixed native fixture-relative stack offset
//! 0x18018. The actual ABI slot is entry RSP+0x10; that offset is not universal.

use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REVEAL_ORDER_PRESENTATION_NATIVE_V1: &str = "reveal_order_presentation_native_v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Owner,
    Text,
    Class,
    GameObject,
    String,
    MethodInfo,
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
pub struct Value {
    pub identity: Identity,
    pub text: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub caller_order_slot_bits: u64,
    pub game_active: bool,
    pub text_values: Vec<Value>,
    pub formatted_values: Vec<Value>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    Init,
    Hide,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub method: Method,
    pub order_register_bits: u64,
    pub initial_order_slot_bits: u64,
    pub formatted: Option<Identity>,
    /// Independently supplied formatter bookkeeping; no conversion is inferred.
    pub formatted_text: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub storage_verified: bool,
    pub unity_verified_inert: bool,
    pub formatter_verified_inert: bool,
    pub tmp_verified_inert: bool,
    pub normal_completion_verified: bool,
    pub game_object_return_rdx_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub owner: Identity,
    pub game_object: Identity,
    pub callables: Vec<Identity>,
    pub state: State,
    pub calls: Vec<Call>,
    pub services: Services,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    ComponentGameObjectService {
        owner: Identity,
        result: Identity,
        method_bits: u64,
    },
    SetActiveService {
        game: Identity,
        rdx_bits: u64,
        value: u8,
        method_bits: u64,
    },
    Int32ToStringService {
        /// Fixed native fixture projection of entry RSP+0x10.
        slot_offset: u32,
        bits: u32,
        signed: i32,
        method_bits: u64,
        result: Option<Identity>,
    },
    TmpTextSetterService {
        text: Identity,
        value: Option<Identity>,
        method: Identity,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Step {
    pub call: usize,
    pub event: Event,
    pub state: State,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub state: State,
    pub steps: Vec<Step>,
    pub completed: Vec<State>,
}

fn word(r: &Record, offset: usize) -> u64 {
    u64::from_le_bytes(
        r.bytes[offset..offset + 8]
            .try_into()
            .expect("validated layout"),
    )
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    // Reserve all future text copies, values and whole-state snapshots before
    // allocating maps or cloning records. Init has four services plus final.
    let units = (|| {
        let mut n = 64usize.checked_add(c.version.len())?;
        for r in &c.state.records {
            n = n.checked_add(4)?.checked_add(r.bytes.len())?;
        }
        for v in c.state.text_values.iter().chain(&c.state.formatted_values) {
            n = n
                .checked_add(2)?
                .checked_add(v.text.as_ref().map_or(0, String::len))?;
        }
        for call in &c.calls {
            n = n.checked_add(8)?.checked_add(
                call.formatted_text
                    .as_ref()
                    .map_or(0, String::len)
                    .checked_mul(2)?,
            )?;
        }
        n.checked_add(c.callables.len())
    })();
    let work = units.and_then(|n| c.calls.len().checked_mul(5)?.checked_add(2)?.checked_mul(n));
    if c.calls.len() > 16 || units.is_none_or(|n| n > 16_384) || work.is_none_or(|n| n > 262_144) {
        return Err(LedgerError::Capacity);
    }
    let s = &c.services;
    if c.version != REVEAL_ORDER_PRESENTATION_NATIVE_V1
        || !s.storage_verified
        || !s.unity_verified_inert
        || !s.formatter_verified_inert
        || !s.tmp_verified_inert
        || !s.normal_completion_verified
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        if r.identity == 0
            || r.bytes.len() != if r.kind == Kind::Class { 0x600 } else { 0x80 }
            || records.insert(r.identity, r).is_some()
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let typed = |id, kind| records.get(&id).is_some_and(|r| r.kind == kind);
    if !typed(c.owner, Kind::Owner) || !typed(c.game_object, Kind::GameObject) {
        return Err(LedgerError::InvalidContext);
    }
    let mut callables = BTreeSet::new();
    for &id in &c.callables {
        if id == 0 || records.contains_key(&id) || !callables.insert(id) {
            return Err(LedgerError::InvalidContext);
        }
    }
    for r in records.values().filter(|r| r.kind == Kind::Class) {
        if !callables.contains(&word(r, 0x558)) || !typed(word(r, 0x560), Kind::MethodInfo) {
            return Err(LedgerError::InvalidContext);
        }
    }
    for r in records.values().filter(|r| r.kind == Kind::Text) {
        if !typed(word(r, 0), Kind::Class) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let text = word(records[&c.owner], 0x20);
    if text != 0 && !typed(text, Kind::Text) {
        return Err(LedgerError::InvalidContext);
    }
    let mut values = BTreeSet::new();
    for v in &c.state.text_values {
        if !typed(v.identity, Kind::Text) || !values.insert(v.identity) {
            return Err(LedgerError::InvalidContext);
        }
    }
    if records
        .values()
        .filter(|r| r.kind == Kind::Text)
        .any(|r| !values.contains(&r.identity))
    {
        return Err(LedgerError::InvalidContext);
    }
    values.clear();
    for v in &c.state.formatted_values {
        if !typed(v.identity, Kind::String) || v.text.is_none() || !values.insert(v.identity) {
            return Err(LedgerError::InvalidContext);
        }
    }
    for call in &c.calls {
        match call.method {
            Method::Hide => {
                if call.formatted.is_some() || call.formatted_text.is_some() {
                    return Err(LedgerError::InvalidContext);
                }
            }
            Method::Init => {
                if text == 0
                    || call.formatted.is_some_and(|id| !typed(id, Kind::String))
                    || call.formatted.is_some() != call.formatted_text.is_some()
                {
                    return Err(LedgerError::InvalidContext);
                }
            }
        }
    }
    Ok(())
}
fn step(out: &mut Replay, call: usize, event: Event) {
    out.steps.push(Step {
        call,
        event,
        state: out.state.clone(),
    });
}
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut out = Replay {
        state: c.state.clone(),
        steps: vec![],
        completed: vec![],
    };
    for (index, call) in c.calls.iter().enumerate() {
        out.state.caller_order_slot_bits = call.initial_order_slot_bits;
        if call.method == Method::Init {
            out.state.caller_order_slot_bits = (call.initial_order_slot_bits
                & 0xffff_ffff_0000_0000)
                | (call.order_register_bits & 0xffff_ffff);
        }
        step(
            &mut out,
            index,
            Event::ComponentGameObjectService {
                owner: c.owner,
                result: c.game_object,
                method_bits: 0,
            },
        );
        let dx = if call.method == Method::Init {
            (c.services.game_object_return_rdx_bits & !255) | 1
        } else {
            0
        };
        step(
            &mut out,
            index,
            Event::SetActiveService {
                game: c.game_object,
                rdx_bits: dx,
                value: (dx & 255) as u8,
                method_bits: 0,
            },
        );
        out.state.game_active = call.method == Method::Init;
        if call.method == Method::Init {
            let text = word(
                out.state
                    .records
                    .iter()
                    .find(|r| r.identity == c.owner)
                    .unwrap(),
                0x20,
            );
            let bits = out.state.caller_order_slot_bits as u32;
            step(
                &mut out,
                index,
                Event::Int32ToStringService {
                    slot_offset: 0x18018,
                    bits,
                    signed: bits as i32,
                    method_bits: 0,
                    result: call.formatted,
                },
            );
            if let Some(identity) = call.formatted {
                if let Some(v) = out
                    .state
                    .formatted_values
                    .iter_mut()
                    .find(|v| v.identity == identity)
                {
                    v.text = call.formatted_text.clone();
                } else {
                    out.state.formatted_values.push(Value {
                        identity,
                        text: call.formatted_text.clone(),
                    });
                }
            }
            let cls = word(
                out.state
                    .records
                    .iter()
                    .find(|r| r.identity == text)
                    .unwrap(),
                0,
            );
            let method = word(
                out.state
                    .records
                    .iter()
                    .find(|r| r.identity == cls)
                    .unwrap(),
                0x560,
            );
            step(
                &mut out,
                index,
                Event::TmpTextSetterService {
                    text,
                    value: call.formatted,
                    method,
                },
            );
            out.state
                .text_values
                .iter_mut()
                .find(|v| v.identity == text)
                .unwrap()
                .text = call.formatted_text.clone();
        }
        out.completed.push(out.state.clone());
    }
    Ok(out)
}
#[cfg(test)]
#[path = "reveal_order_presentation_tests.rs"]
mod tests;
