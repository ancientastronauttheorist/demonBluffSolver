//! Bounded normal CardTokens caller replay. Input and GameObject behavior are
//! explicitly supplied, inert services. Diagnostic bytes are retained verbatim;
//! only exact consumed owner/Character slots are interpreted. This does not
//! collect keyboard input, render tags, admit Unity lifecycle calls, mutate
//! pointers, or replay native guard/failure paths.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CARD_TOKENS_NATIVE_V1: &str = "card_tokens_native_v1";
const KEYS: [u32; 5] = [53, 49, 51, 50, 52];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Owner,
    Character,
    GameObject,
    Class,
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
pub struct Game {
    pub identity: Identity,
    pub active: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyRead {
    pub key: u32,
    pub result_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveRead {
    pub game: Identity,
    pub result_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveWrite {
    pub game: Identity,
    pub rdx_bits: u64,
    pub value: u8,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub games: Vec<Game>,
    pub keys: Vec<KeyRead>,
    pub active_reads: Vec<ActiveRead>,
    pub active_writes: Vec<ActiveWrite>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    OnEnable,
    Update,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub method: Method,
    pub key_return_bits: [u64; 5],
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    KeyDownService {
        key: u32,
        result_bits: u64,
    },
    ActiveSelfService {
        game: Identity,
        result_bits: u64,
    },
    SetActiveService {
        game: Identity,
        rdx_bits: u64,
        value: u8,
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

fn word(record: &Record, off: usize) -> u64 {
    u64::from_le_bytes(
        record.bytes[off..off + 8]
            .try_into()
            .expect("validated record"),
    )
}
fn eligible(character: &Record) -> bool {
    u32::from_le_bytes(character.bytes[0xe8..0xec].try_into().unwrap()) == 10
        && character.bytes[0x190] != 0
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    // Bound input and every future full-state clone/log before constructing maps
    // or copying diagnostic records. A normal Update has at most 23 services.
    let units = (|| {
        let mut n = 64usize.checked_add(c.version.len())?;
        for r in &c.state.records {
            n = n.checked_add(4)?.checked_add(r.bytes.len())?;
        }
        n = n.checked_add(c.state.games.len().checked_mul(2)?)?;
        n = n.checked_add(c.state.keys.len().checked_mul(2)?)?;
        n = n.checked_add(c.state.active_reads.len().checked_mul(2)?)?;
        n = n.checked_add(c.state.active_writes.len().checked_mul(3)?)?;
        n.checked_add(c.calls.len().checked_mul(8)?)
    })();
    let work = units.and_then(|n| {
        let services = c.calls.len().checked_mul(26)?;
        n.checked_add(services.checked_mul(3)?)?
            .checked_mul(services.checked_add(c.calls.len())?.checked_add(2)?)
    });
    if c.calls.len() > 16
        || c.state.records.len() > 64
        || c.state.games.len() > 32
        || c.state.keys.len() > 128
        || c.state.active_reads.len() > 128
        || c.state.active_writes.len() > 128
        || units.is_none_or(|n| n > 16_384)
        || work.is_none_or(|n| n > 1_048_576)
    {
        return Err(LedgerError::Capacity);
    }
    let s = &c.services;
    if c.version != CARD_TOKENS_NATIVE_V1
        || !s.storage_verified
        || !s.supplied_inert_verified
        || !s.normal_completion_verified
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        if r.identity == 0
            || r.bytes.len()
                != if r.kind == Kind::Character {
                    0x200
                } else {
                    0x80
                }
            || records.insert(r.identity, r).is_some()
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let typed = |id, kind| records.get(&id).is_some_and(|r| r.kind == kind);
    if !typed(c.owner, Kind::Owner) {
        return Err(LedgerError::InvalidContext);
    }
    let owner = records[&c.owner];
    let character = word(owner, 0x20);
    if character != 0 && !typed(character, Kind::Character) {
        return Err(LedgerError::InvalidContext);
    }
    for off in [0x28, 0x30, 0x38, 0x40] {
        let id = word(owner, off);
        if id != 0 && !typed(id, Kind::GameObject) {
            return Err(LedgerError::InvalidContext);
        }
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
        || c.state.keys.iter().any(|r| !KEYS.contains(&r.key))
        || c.state
            .active_reads
            .iter()
            .any(|r| !games.contains(&r.game))
        || c.state
            .active_writes
            .iter()
            .any(|r| !games.contains(&r.game) || r.value > 1 || r.rdx_bits as u8 != r.value)
    {
        return Err(LedgerError::InvalidContext);
    }
    // Require all actually consumed references to be safe before replay begins.
    // Idle Update and unused tags can retain native null inputs without guessing
    // what a real failing engine service would do.
    for call in &c.calls {
        let mut needed = BTreeSet::new();
        if call.method == Method::OnEnable {
            needed.extend([0x28, 0x30, 0x38, 0x40]);
        } else {
            if !typed(character, Kind::Character) {
                return Err(LedgerError::InvalidContext);
            }
            if eligible(records[&character]) {
                for (index, key) in KEYS.iter().enumerate() {
                    if call.key_return_bits[index] as u8 == 0 {
                        continue;
                    }
                    match key {
                        53 => {
                            needed.insert(0x30);
                        }
                        49 | 51 | 50 => {
                            needed.extend([0x28, 0x38, 0x40]);
                        }
                        _ => {
                            needed.extend([0x28, 0x30, 0x38, 0x40]);
                        }
                    }
                }
            }
        }
        if needed
            .into_iter()
            .any(|off| !games.contains(&word(owner, off)))
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    Ok(())
}
struct Engine<'a> {
    owner: Identity,
    services: &'a Services,
    state: State,
    steps: Vec<Step>,
    call: usize,
    record_index: BTreeMap<Identity, usize>,
    game_index: BTreeMap<Identity, usize>,
}
impl Engine<'_> {
    fn emit(&mut self, event: Event) {
        self.steps.push(Step {
            call: self.call,
            event,
            state: self.state.clone(),
        });
    }
    fn tag(&self, off: usize) -> Identity {
        word(&self.state.records[self.record_index[&self.owner]], off)
    }
    fn write(&mut self, off: usize, value: u8, rdx_bits: u64) {
        let game = self.tag(off);
        self.emit(Event::SetActiveService {
            game,
            rdx_bits,
            value,
        });
        self.state.games[self.game_index[&game]].active = value != 0;
        self.state.active_writes.push(ActiveWrite {
            game,
            rdx_bits,
            value,
        });
    }
    fn toggle(&mut self, off: usize, always_byte_write: bool) {
        let game = self.tag(off);
        let result_bits = if self.state.games[self.game_index[&game]].active {
            self.services.active_true_return_bits
        } else {
            self.services.active_false_return_bits
        };
        self.emit(Event::ActiveSelfService { game, result_bits });
        self.state
            .active_reads
            .push(ActiveRead { game, result_bits });
        let value = u8::from(result_bits as u8 == 0);
        let rdx_bits = if always_byte_write || value != 0 {
            (self.services.volatile_rdx_bits & !255) | u64::from(value)
        } else {
            0
        };
        self.write(off, value, rdx_bits);
    }
    fn run(&mut self, call: &Call) {
        if call.method == Method::OnEnable {
            for off in [0x28, 0x38, 0x40, 0x30] {
                self.write(off, 0, 0);
            }
            return;
        }
        let owner = &self.state.records[self.record_index[&self.owner]];
        let character = word(owner, 0x20);
        if !eligible(&self.state.records[self.record_index[&character]]) {
            return;
        }
        for (index, key) in KEYS.iter().copied().enumerate() {
            let result_bits = call.key_return_bits[index];
            self.emit(Event::KeyDownService { key, result_bits });
            self.state.keys.push(KeyRead { key, result_bits });
            if result_bits as u8 == 0 {
                continue;
            }
            match key {
                53 => self.toggle(0x30, false),
                49 => {
                    self.toggle(0x28, true);
                    self.write(0x38, 0, 0);
                    self.write(0x40, 0, 0);
                }
                51 => {
                    self.write(0x28, 0, 0);
                    self.write(0x38, 0, 0);
                    self.toggle(0x40, false);
                }
                50 => {
                    self.write(0x28, 0, 0);
                    self.write(0x40, 0, 0);
                    self.toggle(0x38, false);
                }
                _ => {
                    for off in [0x28, 0x38, 0x40, 0x30] {
                        self.write(off, 0, 0);
                    }
                }
            }
        }
    }
}
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut engine = Engine {
        owner: c.owner,
        services: &c.services,
        state: c.state.clone(),
        steps: Vec::new(),
        call: 0,
        record_index: c
            .state
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| (r.identity, i))
            .collect(),
        game_index: c
            .state
            .games
            .iter()
            .enumerate()
            .map(|(i, g)| (g.identity, i))
            .collect(),
    };
    let mut completed = Vec::new();
    for (index, call) in c.calls.iter().enumerate() {
        engine.call = index;
        engine.run(call);
        completed.push(engine.state.clone());
    }
    Ok(Replay {
        state: engine.state,
        steps: engine.steps,
        completed,
    })
}

#[cfg(test)]
#[path = "card_tokens_tests.rs"]
mod tests;
