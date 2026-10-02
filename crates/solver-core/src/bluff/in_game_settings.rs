//! Guarded normal InGameSettings callers with explicitly supplied inert input
//! and active-state services. Shared opaque record/request types are reused
//! from CardTokens; this boundary accepts only Owner/GameObject/Class records.
//! Pointer callbacks, native null/failure paths, lifecycle and rendering are
//! excluded. Every represented byte and retained request remains in snapshots.
pub use super::card_tokens::{
    ActiveRead, ActiveWrite, Event, Game, KeyRead, Kind, Record, Replay, State, Step,
};
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const IN_GAME_SETTINGS_NATIVE_V1: &str = "in_game_settings_native_v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    Update,
    OnEnable,
    ManageShowSettings,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub method: Method,
    pub key_return_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub storage_verified: bool,
    pub supplied_inert_verified: bool,
    pub normal_completion_verified: bool,
    pub active_false_return_bits: u64,
    pub active_true_return_bits: u64,
    pub volatile_rdx_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub owner: Identity,
    pub state: State,
    pub calls: Vec<Call>,
    pub services: Services,
}
fn word(r: &Record, off: usize) -> u64 {
    u64::from_le_bytes(r.bytes[off..off + 8].try_into().expect("validated record"))
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let units = (|| {
        let mut n = 64usize.checked_add(c.version.len())?;
        for r in &c.state.records {
            n = n.checked_add(4)?.checked_add(r.bytes.len())?;
        }
        n = n.checked_add(c.state.games.len().checked_mul(2)?)?;
        n = n.checked_add(c.state.keys.len().checked_mul(2)?)?;
        n = n.checked_add(c.state.active_reads.len().checked_mul(2)?)?;
        n = n.checked_add(c.state.active_writes.len().checked_mul(3)?)?;
        n.checked_add(c.calls.len().checked_mul(2)?)
    })();
    let work = units.and_then(|n| {
        let services = c.calls.len().checked_mul(3)?;
        n.checked_add(services.checked_mul(3)?)?
            .checked_mul(services.checked_add(c.calls.len())?.checked_add(2)?)
    });
    // Future request growth and all full-state copies are reserved before maps
    // or clones, including retained initial ledgers and arbitrary diagnostics.
    if c.calls.len() > 16
        || c.state.records.len() > 32
        || c.state.games.len() > 16
        || c.state.keys.len() > 128
        || c.state.active_reads.len() > 128
        || c.state.active_writes.len() > 128
        || units.is_none_or(|n| n > 8192)
        || work.is_none_or(|n| n > 262_144)
    {
        return Err(LedgerError::Capacity);
    }
    let s = &c.services;
    if c.version != IN_GAME_SETTINGS_NATIVE_V1
        || !s.storage_verified
        || !s.supplied_inert_verified
        || !s.normal_completion_verified
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        if r.identity == 0
            || r.kind == Kind::Character
            || r.bytes.len() != 0x80
            || records.insert(r.identity, r).is_some()
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let typed = |id, kind| records.get(&id).is_some_and(|r| r.kind == kind);
    if !typed(c.owner, Kind::Owner) {
        return Err(LedgerError::InvalidContext);
    }
    let settings = word(records[&c.owner], 0x20);
    if settings != 0 && !typed(settings, Kind::GameObject) {
        return Err(LedgerError::InvalidContext);
    }
    let mut games = BTreeSet::new();
    for g in &c.state.games {
        if !typed(g.identity, Kind::GameObject) || !games.insert(g.identity) {
            return Err(LedgerError::InvalidContext);
        }
    }
    if records
        .values()
        .filter(|r| r.kind == Kind::GameObject)
        .any(|r| !games.contains(&r.identity))
        || c.state.keys.iter().any(|r| r.key != 27)
        || c.state
            .active_reads
            .iter()
            .any(|r| !games.contains(&r.game))
        || c.state
            .active_writes
            .iter()
            .any(|r| !games.contains(&r.game) || r.value > 1 || r.rdx_bits as u8 != r.value)
        || c.calls.iter().any(|call| {
            (call.method != Method::Update || call.key_return_bits as u8 != 0)
                && !games.contains(&settings)
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut state = c.state.clone();
    let owner = state
        .records
        .iter()
        .find(|r| r.identity == c.owner)
        .expect("validated owner");
    let settings = word(owner, 0x20);
    let game_index: BTreeMap<_, _> = state
        .games
        .iter()
        .enumerate()
        .map(|(i, g)| (g.identity, i))
        .collect();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    for (index, call) in c.calls.iter().enumerate() {
        let emit = |steps: &mut Vec<Step>, state: &State, event| {
            steps.push(Step {
                call: index,
                event,
                state: state.clone(),
            })
        };
        if call.method == Method::Update {
            emit(
                &mut steps,
                &state,
                Event::KeyDownService {
                    key: 27,
                    result_bits: call.key_return_bits,
                },
            );
            state.keys.push(KeyRead {
                key: 27,
                result_bits: call.key_return_bits,
            });
            if call.key_return_bits as u8 == 0 {
                completed.push(state.clone());
                continue;
            }
        }
        let (value, rdx_bits) = if call.method == Method::OnEnable {
            (0, 0)
        } else {
            let result_bits = if state.games[game_index[&settings]].active {
                c.services.active_true_return_bits
            } else {
                c.services.active_false_return_bits
            };
            emit(
                &mut steps,
                &state,
                Event::ActiveSelfService {
                    game: settings,
                    result_bits,
                },
            );
            state.active_reads.push(ActiveRead {
                game: settings,
                result_bits,
            });
            let value = u8::from(result_bits as u8 == 0);
            let rdx_bits = if call.method == Method::Update || value != 0 {
                (c.services.volatile_rdx_bits & !255) | u64::from(value)
            } else {
                0
            };
            (value, rdx_bits)
        };
        emit(
            &mut steps,
            &state,
            Event::SetActiveService {
                game: settings,
                rdx_bits,
                value,
            },
        );
        state.games[game_index[&settings]].active = value != 0;
        state.active_writes.push(ActiveWrite {
            game: settings,
            rdx_bits,
            value,
        });
        completed.push(state.clone());
    }
    Ok(Replay {
        state,
        steps,
        completed,
    })
}
#[cfg(test)]
#[path = "in_game_settings_tests.rs"]
mod tests;
