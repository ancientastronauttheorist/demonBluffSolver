//! Offline Characters.PickRoundDuplicates replay for the pinned native build.
//! Filters return distinct, stable candidate lists; Remove uses reference equality.
//! Uniform occurrence support is a service contract, not a Unity PRNG model.
//! Logical writes and failure prefixes are retained; stale backing slots, managed
//! unwinding, filter internals and actual PRNG advancement are outside this scope.
use super::ledger::{LedgerError, Probability};
use serde::{Deserialize, Serialize};

pub const ROUND_DUPLICATES_NATIVE_V1: &str = "round_duplicates_native_v1";
const MAX_ENTRIES: usize = 4096;
const MAX_PATHS: usize = 65_536;
const MAX_RETAINED: usize = 1_048_576;
pub type Items = Vec<Option<u16>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateList {
    pub identity: u16,
    pub items: Items,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gateway {
    ClassInit,
    Script,
    Clear,
    Bluffable,
    RealType,
    DiscardedGoodFilter,
    Rng,
    VillagerStore,
    Remove,
    OutcastAdd,
    Grow,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceFailure {
    pub gateway: Gateway,
    pub occurrence: u16,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundDuplicateContext {
    pub rule_version: String,
    pub stable_services: bool,
    pub reference_removal: bool,
    pub uniform_occurrence_support: bool,
    pub gameplay_initialized: bool,
    pub gameplay_present: bool,
    pub pool_identity: Option<u16>,
    pub initial_pool: Items,
    pub initial_version: u32,
    pub capacity: u16,
    /// Successful growth reserves this many additional slots. This explicit
    /// service result is not an assertion about native List growth policy.
    pub growth_slots: u16,
    pub villagers: Option<CandidateList>,
    pub outcasts: Option<CandidateList>,
    pub failure: Option<ServiceFailure>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Failure {
    Null,
    Bounds,
    Service(Gateway),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Event {
    pub event: Gateway,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
}
impl Event {
    fn new(event: Gateway) -> Self {
        Self {
            event,
            value: None,
            count: None,
            pool: None,
            width: None,
            index: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Draw {
    pub width: usize,
    /// No selected occurrence exists for a failing service or empty range.
    pub index: Option<usize>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoundDuplicatePath {
    pub probability: Probability,
    pub output: Items,
    pub remaining_villagers: Items,
    pub remaining_outcasts: Items,
    pub version: u32,
    pub capacity: u16,
    pub gameplay_initialized: bool,
    pub events: Vec<Event>,
    pub draws: Vec<Draw>,
    pub error: Option<Failure>,
}
fn emit(c: &RoundDuplicateContext, p: &mut RoundDuplicatePath, e: Event) -> bool {
    let gateway = e.event.clone();
    p.events.push(e);
    if c.failure.as_ref().is_some_and(|f| {
        f.gateway == gateway
            && p.events.iter().filter(|e| e.event == gateway).count() == usize::from(f.occurrence)
    }) {
        p.error = Some(Failure::Service(gateway));
        false
    } else {
        true
    }
}
fn simple(c: &RoundDuplicateContext, p: &mut RoundDuplicatePath, g: Gateway) -> bool {
    emit(c, p, Event::new(g))
}
fn cost(p: &RoundDuplicatePath) -> usize {
    1 + p.output.len()
        + p.remaining_villagers.len()
        + p.remaining_outcasts.len()
        + p.events.len()
        + p.draws.len()
}

pub fn replay(c: &RoundDuplicateContext) -> Result<Vec<RoundDuplicatePath>, LedgerError> {
    if c.rule_version != ROUND_DUPLICATES_NATIVE_V1
        || !c.stable_services
        || !c.reference_removal
        || !c.uniform_occurrence_support
        || c.growth_slots == 0
        || c.failure.as_ref().is_some_and(|f| f.occurrence == 0)
        || c.initial_pool.len() > usize::from(c.capacity)
    {
        return Err(LedgerError::InvalidContext);
    }
    let ids = [
        c.pool_identity,
        c.villagers.as_ref().map(|l| l.identity),
        c.outcasts.as_ref().map(|l| l.identity),
    ];
    for i in 0..3 {
        for j in i + 1..3 {
            if ids[i].is_some() && ids[i] == ids[j] {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    if c.initial_pool.len()
        + c.villagers.as_ref().map_or(0, |l| l.items.len())
        + c.outcasts.as_ref().map_or(0, |l| l.items.len())
        > MAX_ENTRIES
    {
        return Err(LedgerError::Capacity);
    }
    let mut p = RoundDuplicatePath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        output: c.initial_pool.clone(),
        remaining_villagers: c
            .villagers
            .as_ref()
            .map_or_else(Vec::new, |l| l.items.clone()),
        remaining_outcasts: c
            .outcasts
            .as_ref()
            .map_or_else(Vec::new, |l| l.items.clone()),
        version: c.initial_version,
        capacity: c.capacity,
        gameplay_initialized: c.gameplay_initialized,
        events: vec![],
        draws: vec![],
        error: None,
    };
    if !p.gameplay_initialized {
        if !simple(c, &mut p, Gateway::ClassInit) {
            return Ok(vec![p]);
        }
        p.gameplay_initialized = true;
    }
    if !c.gameplay_present {
        p.error = Some(Failure::Null);
        return Ok(vec![p]);
    }
    if !simple(c, &mut p, Gateway::Script) {
        return Ok(vec![p]);
    }
    if c.pool_identity.is_none() {
        p.error = Some(Failure::Null);
        return Ok(vec![p]);
    }
    let old = p.output.len();
    p.output.clear();
    p.version = p.version.wrapping_add(1);
    if old > 0 {
        let mut e = Event::new(Gateway::Clear);
        e.count = Some(old);
        if !emit(c, &mut p, e) {
            return Ok(vec![p]);
        }
    }
    if !simple(c, &mut p, Gateway::Bluffable) {
        return Ok(vec![p]);
    }
    for value in [10, 20] {
        let mut e = Event::new(Gateway::RealType);
        e.value = Some(value);
        if !emit(c, &mut p, e) {
            return Ok(vec![p]);
        }
    }
    if !simple(c, &mut p, Gateway::DiscardedGoodFilter) {
        return Ok(vec![p]);
    }
    let mut active = vec![p];
    for stage in 0..5 {
        let mut next = Vec::new();
        let mut retained = 0;
        for mut p in active {
            let out = stage == 4;
            let skip = p.error.is_some() || (!out && stage > 0 && p.remaining_villagers.is_empty());
            let absent = if out {
                c.outcasts.is_none()
            } else {
                c.villagers.is_none()
            };
            let width = if out {
                p.remaining_outcasts.len()
            } else {
                p.remaining_villagers.len()
            };
            if !skip && absent {
                p.error = Some(Failure::Null)
            }
            if skip || absent || (out && width == 0) {
                retained += cost(&p);
                next.push(p);
                continue;
            }
            // A failing range call precedes selection, so its one failed path
            // retains incoming mass; no fictional successful draw is generated.
            let failing = c.failure.as_ref().is_some_and(|f| {
                f.gateway == Gateway::Rng
                    && p.events.iter().filter(|e| e.event == Gateway::Rng).count() + 1
                        == usize::from(f.occurrence)
            });
            let branches = if failing || width == 0 { 1 } else { width };
            let estimated = branches
                .checked_mul(cost(&p) + 10)
                .ok_or(LedgerError::Capacity)?;
            if next.len() + branches > MAX_PATHS || retained + estimated > MAX_RETAINED {
                return Err(LedgerError::Capacity);
            }
            for index in 0..branches {
                let mut q = p.clone();
                let selected = (!failing && width > 0).then_some(index);
                if selected.is_some() {
                    q.probability = q.probability.multiply(1, width as u64)?;
                }
                q.draws.push(Draw {
                    width,
                    index: selected,
                });
                let mut e = Event::new(Gateway::Rng);
                e.width = Some(width);
                e.index = selected;
                if emit(c, &mut q, e) {
                    if width == 0 {
                        q.error = Some(Failure::Bounds)
                    } else {
                        let value = if out {
                            q.remaining_outcasts[index]
                        } else {
                            q.remaining_villagers[index]
                        };
                        let mut e = Event::new(if out {
                            Gateway::OutcastAdd
                        } else if q.output.len() == usize::from(q.capacity) {
                            Gateway::Grow
                        } else {
                            Gateway::VillagerStore
                        });
                        e.value = value;
                        let growing = q.output.len() == usize::from(q.capacity);
                        if !out {
                            q.version = q.version.wrapping_add(1);
                        }
                        let mut added = false;
                        if !out && !growing {
                            q.output.push(value);
                            added = emit(c, &mut q, e)
                        } else if emit(c, &mut q, e) {
                            if growing {
                                q.capacity = q
                                    .capacity
                                    .checked_add(c.growth_slots)
                                    .ok_or(LedgerError::Capacity)?;
                            }
                            if out {
                                q.version = q.version.wrapping_add(1);
                            }
                            q.output.push(value);
                            added = true;
                        }
                        if added {
                            let mut e = Event::new(Gateway::Remove);
                            e.value = value;
                            e.pool = Some(if out { "outcast" } else { "villager" }.into());
                            if emit(c, &mut q, e) {
                                let list = if out {
                                    &mut q.remaining_outcasts
                                } else {
                                    &mut q.remaining_villagers
                                };
                                let first = list
                                    .iter()
                                    .position(|v| *v == value)
                                    .expect("selected occurrence retained");
                                list.remove(first);
                            }
                        }
                    }
                }
                retained += cost(&q);
                next.push(q);
            }
        }
        if next.len() > MAX_PATHS || retained > MAX_RETAINED {
            return Err(LedgerError::Capacity);
        }
        active = next;
    }
    Ok(active)
}

#[cfg(test)]
#[path = "round_duplicate_tests.rs"]
mod tests;
