//! Actual candidate preparation plus the audited occurrence sampler.
//! Lists have distinct logical identities and ample capacity; services mutate only
//! declared destinations. No backing-slot, managed unwind, or Unity PRNG model.
use super::{
    ledger::{LedgerError, Probability},
    round_duplicates as sampler,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
pub const ROUND_CANDIDATE_NATIVE_V1: &str = "round_candidate_native_v1";
const MAX_ITEMS: usize = 32;
const MAX_PATHS: usize = 1024;
const MAX_RETAINED: usize = 1_048_576;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub real_type: i32,
    pub alignment: i32,
    pub bluffable: u8,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gateway {
    ClassInit,
    Allocate,
    ListCtor,
    AppendRange,
    Enumerator,
    MoveNext,
    Dispose,
    Clear,
    FilterAdd,
    Rng,
    VillagerStore,
    Remove,
    OutcastAdd,
}
impl Gateway {
    fn label(&self) -> String {
        serde_json::to_value(self)
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceFailure {
    pub gateway: Gateway,
    pub occurrence: u16,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub metadata_initialized: bool,
    pub stable_services: bool,
    pub distinct_lists_sufficient_capacity: bool,
    pub reference_removal: bool,
    pub uniform_occurrence_support: bool,
    pub gameplay_initialized: bool,
    pub gameplay_present: bool,
    pub pool_present: bool,
    pub rosters: [Option<sampler::Items>; 4],
    pub assets: BTreeMap<u16, Asset>,
    pub initial_pool: sampler::Items,
    pub initial_version: u32,
    pub failure: Option<ServiceFailure>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListState {
    pub items: sampler::Items,
    pub version: u32,
}
pub type Snapshot = BTreeMap<String, ListState>;
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Trace {
    pub lists: Snapshot,
    pub events: Vec<Value>,
    pub entries: Vec<Value>,
    pub draws: Vec<Value>,
    pub error: Option<String>,
    pub gameplay_initialized: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WeightedTrace {
    pub probability: Probability,
    pub trace: Trace,
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    if c.version != ROUND_CANDIDATE_NATIVE_V1
        || !c.metadata_initialized
        || !c.stable_services
        || !c.distinct_lists_sufficient_capacity
        || !c.reference_removal
        || !c.uniform_occurrence_support
        || c.failure.as_ref().is_some_and(|f| f.occurrence == 0)
    {
        return Err(LedgerError::InvalidContext);
    }
    let count = c.rosters.iter().flatten().map(Vec::len).sum::<usize>() + c.initial_pool.len();
    if count > MAX_ITEMS || c.assets.len() > MAX_ITEMS {
        return Err(LedgerError::Capacity);
    }
    if c.rosters
        .iter()
        .flatten()
        .flatten()
        .chain(c.initial_pool.iter())
        .flatten()
        .any(|id| !c.assets.contains_key(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}
struct Run<'a> {
    c: &'a Context,
    trace: Trace,
    counts: BTreeMap<String, u16>,
}
impl<'a> Run<'a> {
    fn new(c: &'a Context) -> Self {
        let mut lists = Snapshot::new();
        lists.insert(
            "duplicates".into(),
            ListState {
                items: c.initial_pool.clone(),
                version: c.initial_version,
            },
        );
        for (n, roster) in c.rosters.iter().enumerate() {
            lists.insert(
                format!("roster{n}"),
                ListState {
                    items: roster.clone().unwrap_or_default(),
                    version: 0,
                },
            );
        }
        Self {
            c,
            trace: Trace {
                lists,
                events: vec![],
                entries: vec![],
                draws: vec![],
                error: None,
                gameplay_initialized: c.gameplay_initialized,
            },
            counts: BTreeMap::new(),
        }
    }
    fn event(&mut self, g: Gateway, mut args: Value) -> Result<(), String> {
        let label = g.label();
        args["kind"] = json!(label);
        args["snapshot"] = json!(self.trace.lists);
        self.trace.events.push(args);
        let count = self.counts.entry(label.clone()).or_default();
        *count += 1;
        if self
            .c
            .failure
            .as_ref()
            .is_some_and(|f| f.gateway == g && f.occurrence == *count)
        {
            Err(label)
        } else {
            Ok(())
        }
    }
    fn entry(&mut self, method: &str, typ: Option<i32>) {
        self.trace.entries.push(json!({"method":method,"type":typ}));
    }
    fn allocate(&mut self, name: &str) -> Result<(), String> {
        self.event(Gateway::Allocate, json!({"list":name}))?;
        self.trace.lists.insert(
            name.into(),
            ListState {
                items: vec![],
                version: 0,
            },
        );
        self.event(Gateway::ListCtor, json!({"list":name}))
    }
    fn append(&mut self, name: &str, item: Option<u16>) {
        let l = self.trace.lists.get_mut(name).unwrap();
        l.items.push(item);
        l.version = l.version.wrapping_add(1);
    }
    fn filter(&mut self, name: &str, source: &str, typ: Option<i32>) -> Result<(), String> {
        self.entry(
            if name == "bluffable" {
                "FilterBluffableCharacters"
            } else if name == "discarded_good" {
                "FilterAlignmentCharacters"
            } else {
                "FilterRealCharacterType"
            },
            typ,
        );
        self.allocate(name)?;
        self.event(Gateway::Enumerator, json!({"source":source}))?;
        let input = self.trace.lists[source].items.clone();
        for item in input {
            self.event(Gateway::MoveNext, json!({"source":source}))?;
            let id = item.ok_or("null")?;
            let asset = &self.c.assets[&id];
            let accept = if name == "bluffable" {
                asset.bluffable != 0
            } else if name == "discarded_good" {
                asset.alignment == typ.unwrap()
            } else {
                asset.real_type == typ.unwrap()
            };
            if accept {
                self.event(Gateway::FilterAdd, json!({"list":name,"value":id}))?;
                self.append(name, item);
            }
        }
        self.event(Gateway::MoveNext, json!({"source":source}))?;
        self.event(Gateway::Dispose, json!({}))
    }
    fn prepare(&mut self) -> Result<(), String> {
        self.entry("PickRoundDuplicates", None);
        if !self.trace.gameplay_initialized {
            self.event(Gateway::ClassInit, json!({}))?;
            self.trace.gameplay_initialized = true;
        }
        if !self.c.gameplay_present {
            return Err("null".into());
        }
        self.entry("GetScriptCharacters", None);
        self.allocate("script")?;
        for n in 0..4 {
            let source = format!("roster{n}");
            let present = self.c.rosters[n].is_some();
            self.event(Gateway::AppendRange,json!({"source":if present {Some(source.clone())}else{None},"destination":"script"}))?;
            if !present {
                return Err("null_collection".into());
            }
            let values = self.trace.lists[&source].items.clone();
            let l = self.trace.lists.get_mut("script").unwrap();
            l.items.extend(values);
            l.version = l.version.wrapping_add(1);
        }
        if !self.c.pool_present {
            return Err("null".into());
        }
        let l = self.trace.lists.get_mut("duplicates").unwrap();
        let old = l.items.len();
        l.items.clear();
        l.version = l.version.wrapping_add(1);
        if old != 0 {
            self.event(Gateway::Clear, json!({"count":old}))?;
        }
        self.filter("bluffable", "script", None)?;
        self.filter("villagers", "bluffable", Some(10))?;
        self.filter("outcasts", "bluffable", Some(20))?;
        self.filter("discarded_good", "bluffable", Some(10))
    }
    fn select(&mut self, choices: &[usize]) -> Result<(), String> {
        for stage in 0..5 {
            let out = stage == 4;
            let name = if out { "outcasts" } else { "villagers" };
            let width = self.trace.lists[name].items.len();
            if (out || stage > 0) && width == 0 {
                continue;
            }
            let index = choices.get(self.trace.draws.len()).copied().unwrap_or(0);
            self.trace.draws.push(json!({"width":width,"index":index}));
            self.event(Gateway::Rng, json!({"width":width,"index":index}))?;
            let value = *self.trace.lists[name].items.get(index).ok_or("bounds")?;
            if out {
                self.event(
                    Gateway::OutcastAdd,
                    json!({"list":"duplicates","value":value}),
                )?;
                self.append("duplicates", value);
            } else {
                self.append("duplicates", value);
                self.event(Gateway::VillagerStore, json!({"value":value}))?;
            }
            self.event(Gateway::Remove, json!({"list":name,"value":value}))?;
            let l = self.trace.lists.get_mut(name).unwrap();
            let first = l.items.iter().position(|v| *v == value).unwrap();
            l.items.remove(first);
            l.version = l.version.wrapping_add(1);
        }
        Ok(())
    }
}
/// Exact supplied-index trace, including every preparation and sampling failure.
pub fn replay_trace(c: &Context, choices: &[usize]) -> Result<Trace, LedgerError> {
    validate(c)?;
    if choices.len() > 5 || choices.iter().any(|n| *n > u32::MAX as usize) {
        return Err(LedgerError::InvalidContext);
    }
    let mut r = Run::new(c);
    if let Err(e) = r.prepare().and_then(|_| r.select(choices)) {
        r.trace.error = Some(e)
    };
    Ok(r.trace)
}
fn retained(t: &Trace) -> usize {
    let state = t.lists.values().map(|l| 1 + l.items.len()).sum::<usize>();
    // Count the actual snapshot lengths: preparation may clear a larger old pool,
    // and later candidate removals make final lengths smaller than earlier ones.
    let snapshots = t
        .events
        .iter()
        .map(|event| {
            event["snapshot"]
                .as_object()
                .unwrap()
                .values()
                .map(|list| 1 + list["items"].as_array().unwrap().len())
                .sum::<usize>()
        })
        .sum::<usize>();
    state + snapshots + 16 * t.events.len() + 4 * t.draws.len() + 4 * t.entries.len()
}
/// Uniform occurrence composition. Preparation failures retain probability one;
/// sampling failures retain their incoming mass and are never renormalized away.
pub fn replay_weighted(c: &Context) -> Result<Vec<WeightedTrace>, LedgerError> {
    validate(c)?;
    let mut prep = Run::new(c);
    if let Err(e) = prep.prepare() {
        prep.trace.error = Some(e);
        return Ok(vec![WeightedTrace {
            probability: Probability {
                numerator: 1,
                denominator: 1,
            },
            trace: prep.trace,
        }]);
    }
    let failure = c.failure.as_ref().and_then(|f| {
        let gateway = match &f.gateway {
            Gateway::Rng => sampler::Gateway::Rng,
            Gateway::VillagerStore => sampler::Gateway::VillagerStore,
            Gateway::Remove => sampler::Gateway::Remove,
            Gateway::OutcastAdd => sampler::Gateway::OutcastAdd,
            _ => return None,
        };
        Some(sampler::ServiceFailure {
            gateway,
            occurrence: f.occurrence,
        })
    });
    let sc = sampler::RoundDuplicateContext {
        rule_version: sampler::ROUND_DUPLICATES_NATIVE_V1.into(),
        stable_services: true,
        reference_removal: true,
        uniform_occurrence_support: true,
        gameplay_initialized: true,
        gameplay_present: true,
        pool_identity: Some(1),
        initial_pool: c.initial_pool.clone(),
        initial_version: c.initial_version,
        capacity: 128,
        growth_slots: 128,
        villagers: Some(sampler::CandidateList {
            identity: 2,
            items: prep.trace.lists["villagers"].items.clone(),
        }),
        outcasts: Some(sampler::CandidateList {
            identity: 3,
            items: prep.trace.lists["outcasts"].items.clone(),
        }),
        failure,
    };
    let support = sampler::replay(&sc)?;
    if support.len() > MAX_PATHS {
        return Err(LedgerError::Capacity);
    }
    let support_cost = support
        .iter()
        .map(|p| {
            1 + p.output.len()
                + p.remaining_villagers.len()
                + p.remaining_outcasts.len()
                + p.events.len()
                + p.draws.len()
        })
        .sum::<usize>();
    let mut result = Vec::new();
    let mut cost = retained(&prep.trace)
        .checked_add(support_cost)
        .ok_or(LedgerError::Capacity)?;
    if cost > MAX_RETAINED {
        return Err(LedgerError::Capacity);
    }
    for path in support {
        let choices: Vec<_> = path.draws.iter().map(|d| d.index.unwrap_or(0)).collect();
        let trace = replay_trace(c, &choices)?;
        if trace.lists["duplicates"].items != path.output
            || trace.lists["duplicates"].version != path.version
            || trace.lists["villagers"].items != path.remaining_villagers
            || trace.lists["outcasts"].items != path.remaining_outcasts
        {
            return Err(LedgerError::InvalidContext);
        }
        let expected_error = path.error.as_ref().map(|e| match e {
            sampler::Failure::Null => "null".to_owned(),
            sampler::Failure::Bounds => "bounds".to_owned(),
            sampler::Failure::Service(g) => serde_json::to_value(g)
                .unwrap()
                .as_str()
                .unwrap()
                .to_owned(),
        });
        if trace.error != expected_error {
            return Err(LedgerError::InvalidContext);
        }
        cost = cost
            .checked_add(retained(&trace))
            .ok_or(LedgerError::Capacity)?;
        if cost > MAX_RETAINED {
            return Err(LedgerError::Capacity);
        }
        result.push(WeightedTrace {
            probability: path.probability,
            trace,
        });
    }
    Ok(result)
}
#[cfg(test)]
#[path = "round_candidate_composition_tests.rs"]
mod tests;
