//! Bounded unique-pool caller and standalone captured predicate replay.
//! RemoveAll/filter/list bodies are explicit preserving services, not native List
//! composition. Uniform occurrence support does not model Unity PRNG advancement.
use super::{
    ledger::{LedgerError, Probability},
    round_candidate_composition::{Asset, ListState},
    round_duplicates::Items,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
pub const ROUND_BLUFFS_NATIVE_V1: &str = "round_bluffs_native_v1";
const MAX_ITEMS: usize = 32;
const MAX_PATHS: usize = 1024;
const MAX_RETAINED: usize = 1_048_576;
const LISTS: [&str; 10] = [
    "unique",
    "all",
    "script",
    "bluffable",
    "villagers",
    "outcasts",
    "fallback",
    "fallback_bluffable",
    "fallback_good",
    "fallback_villagers",
];
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gateway {
    Allocate,
    ObjectCtor,
    ClassInit,
    Getter,
    Capture,
    Clear,
    PredicateCtor,
    RemoveAll,
    Filter,
    Rng,
    Add,
    Remove,
    Contains,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceFailure {
    pub gateway: Gateway,
    pub occurrence: u16,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Predicate {
    pub captured: bool,
    pub candidate: Option<u16>,
    pub contains_return: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub metadata_initialized: bool,
    pub stable_services: bool,
    pub reference_membership_and_removal: bool,
    pub distinct_lists_sufficient_capacity: bool,
    pub uniform_occurrence_support: bool,
    pub gameplay_initialized: bool,
    pub gameplay_present: bool,
    pub pool_present: bool,
    pub all: Items,
    pub script: Items,
    pub fallback: Items,
    pub initial_pool: Items,
    pub initial_version: u32,
    pub assets: BTreeMap<u16, Asset>,
    pub null_getter: Option<String>,
    pub null_filter: Option<String>,
    /// Controlled singleton disappearance after this successful getter, before its caller resumes.
    pub replace_after_getter: Option<String>,
    pub failure: Option<ServiceFailure>,
    pub predicate: Option<Predicate>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Snapshot {
    pub lists: BTreeMap<String, ListState>,
    pub captured_script: Option<String>,
    pub project_present: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Trace {
    pub state: Snapshot,
    pub events: Vec<Value>,
    pub draws: Vec<Value>,
    pub error: Option<String>,
    pub return_al: Option<u8>,
    pub gameplay_initialized: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WeightedTrace {
    pub probability: Probability,
    pub trace: Trace,
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    if c.version != ROUND_BLUFFS_NATIVE_V1
        || !c.metadata_initialized
        || !c.stable_services
        || !c.reference_membership_and_removal
        || !c.distinct_lists_sufficient_capacity
        || !c.uniform_occurrence_support
        || c.failure.as_ref().is_some_and(|f| f.occurrence == 0)
    {
        return Err(LedgerError::InvalidContext);
    }
    for name in [&c.null_getter, &c.replace_after_getter]
        .into_iter()
        .flatten()
    {
        if !["all", "script", "fallback"].contains(&name.as_str()) {
            return Err(LedgerError::InvalidContext);
        }
    }
    if c.null_filter.as_ref().is_some_and(|n| {
        ![
            "bluffable",
            "villagers",
            "outcasts",
            "fallback_bluffable",
            "fallback_good",
            "fallback_villagers",
        ]
        .contains(&n.as_str())
    }) {
        return Err(LedgerError::InvalidContext);
    }
    if c.all.len() + c.script.len() + c.fallback.len() + c.initial_pool.len() > MAX_ITEMS
        || c.assets.len() > MAX_ITEMS
    {
        return Err(LedgerError::Capacity);
    }
    if c.all
        .iter()
        .chain(&c.script)
        .chain(&c.fallback)
        .chain(&c.initial_pool)
        .flatten()
        .any(|id| !c.assets.contains_key(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}
enum Stop {
    Failure(String),
    Choice(usize),
}
impl From<&str> for Stop {
    fn from(s: &str) -> Self {
        Self::Failure(s.into())
    }
}
struct Run<'a> {
    c: &'a Context,
    trace: Trace,
    counts: BTreeMap<String, u16>,
}
impl<'a> Run<'a> {
    fn new(c: &'a Context) -> Self {
        let lists = LISTS
            .iter()
            .map(|name| {
                (
                    (*name).into(),
                    ListState {
                        items: match *name {
                            "unique" => c.initial_pool.clone(),
                            "all" => c.all.clone(),
                            "script" => c.script.clone(),
                            "fallback" => c.fallback.clone(),
                            _ => vec![],
                        },
                        version: if *name == "unique" {
                            c.initial_version
                        } else {
                            0
                        },
                    },
                )
            })
            .collect();
        Self {
            c,
            trace: Trace {
                state: Snapshot {
                    lists,
                    captured_script: c
                        .predicate
                        .as_ref()
                        .filter(|p| p.captured)
                        .map(|_| "script".into()),
                    project_present: c.gameplay_present,
                },
                events: vec![],
                draws: vec![],
                error: None,
                return_al: None,
                gameplay_initialized: c.gameplay_initialized,
            },
            counts: BTreeMap::new(),
        }
    }
    fn event(&mut self, g: Gateway, mut args: Value) -> Result<(), Stop> {
        let name = serde_json::to_value(&g)
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        args["kind"] = json!(name);
        args["snapshot"] = json!(self.trace.state);
        self.trace.events.push(args);
        let count = self.counts.entry(name.clone()).or_default();
        *count += 1;
        if self
            .c
            .failure
            .as_ref()
            .is_some_and(|f| f.gateway == g && f.occurrence == *count)
        {
            Err(Stop::Failure(name))
        } else {
            Ok(())
        }
    }
    fn getter(&mut self, name: &str) -> Result<Option<String>, Stop> {
        if !self.trace.state.project_present {
            return Err("null".into());
        }
        self.event(Gateway::Getter, json!({"source":name}))?;
        if self.c.replace_after_getter.as_deref() == Some(name) {
            self.trace.state.project_present = false;
        }
        Ok((self.c.null_getter.as_deref() != Some(name)).then(|| name.to_owned()))
    }
    fn filter(
        &mut self,
        name: &str,
        source: Option<&str>,
        requested: Option<i32>,
    ) -> Result<Option<String>, Stop> {
        // The native fixture labels these shared gateways by the supplied
        // source pointer. Null loses the fallback source identity, so preserve
        // its diagnostic label before the same immediate null failure.
        let destination = if source.is_none() {
            match name {
                "fallback_bluffable" => "bluffable",
                "fallback_villagers" => "villagers",
                _ => name,
            }
        } else {
            name
        };
        self.event(
            Gateway::Filter,
            json!({"destination":destination,"source":source,"requested":requested}),
        )?;
        let source = source.ok_or("null")?;
        let items = self.trace.state.lists[source].items.clone();
        if items.contains(&None) {
            return Err("null".into());
        }
        let selected = items
            .into_iter()
            .filter(|id| {
                let a = &self.c.assets[&id.unwrap()];
                if name.ends_with("bluffable") {
                    a.bluffable != 0
                } else if name == "fallback_good" {
                    a.alignment == requested.unwrap()
                } else {
                    a.real_type == requested.unwrap()
                }
            })
            .collect();
        self.trace.state.lists.insert(
            name.into(),
            ListState {
                items: selected,
                version: 0,
            },
        );
        Ok((self.c.null_filter.as_deref() != Some(name)).then(|| name.to_owned()))
    }
    fn append(&mut self, value: Option<u16>) -> Result<(), Stop> {
        self.event(Gateway::Add, json!({"value":value}))?;
        let l = self.trace.state.lists.get_mut("unique").unwrap();
        l.items.push(value);
        l.version = l.version.wrapping_add(1);
        Ok(())
    }
    fn draw(&mut self, name: &str, choices: &[usize], frontier: bool) -> Result<Option<u16>, Stop> {
        let width = self.trace.state.lists[name].items.len();
        let at = self.trace.draws.len();
        let failing = self
            .c
            .failure
            .as_ref()
            .is_some_and(|f| f.gateway == Gateway::Rng && usize::from(f.occurrence) == at + 1);
        if frontier && at >= choices.len() && width > 0 && !failing {
            return Err(Stop::Choice(width));
        }
        let index = choices.get(at).copied().unwrap_or(0);
        self.trace.draws.push(json!({"width":width,"index":index}));
        self.event(Gateway::Rng, json!({"width":width,"index":index}))?;
        self.trace.state.lists[name]
            .items
            .get(index)
            .copied()
            .ok_or_else(|| Stop::from("bounds"))
    }
    fn remove(&mut self, name: &str, value: Option<u16>) -> Result<(), Stop> {
        self.event(Gateway::Remove, json!({"source":name,"value":value}))?;
        let l = self.trace.state.lists.get_mut(name).unwrap();
        let index = l.items.iter().position(|id| *id == value).unwrap();
        l.items.remove(index);
        l.version = l.version.wrapping_add(1);
        Ok(())
    }
    fn execute(&mut self, choices: &[usize], frontier: bool) -> Result<(), Stop> {
        if let Some(p) = self.c.predicate.clone() {
            if !p.captured {
                return Err("null".into());
            }
            self.event(Gateway::Contains, json!({"value":p.candidate}))?;
            self.trace.return_al = Some(p.contains_return as u8);
            return Ok(());
        }
        self.event(Gateway::Allocate, json!({"object":"closure"}))?;
        self.event(Gateway::ObjectCtor, json!({}))?;
        if !self.trace.gameplay_initialized {
            self.event(Gateway::ClassInit, json!({}))?;
            self.trace.gameplay_initialized = true;
        }
        let all = self.getter("all")?;
        let script = self.getter("script")?;
        self.trace.state.captured_script = script;
        self.event(
            Gateway::Capture,
            json!({"source":self.trace.state.captured_script}),
        )?;
        if !self.c.pool_present {
            return Err("null".into());
        }
        let l = self.trace.state.lists.get_mut("unique").unwrap();
        let count = l.items.len();
        l.items.clear();
        l.version = l.version.wrapping_add(1);
        if count > 0 {
            self.event(Gateway::Clear, json!({"count":count}))?;
        }
        self.event(Gateway::Allocate, json!({"object":"predicate"}))?;
        self.event(Gateway::PredicateCtor, json!({}))?;
        let all = all.ok_or("null")?;
        self.event(Gateway::RemoveAll, json!({}))?;
        let before = self.trace.state.lists[&all].items.clone();
        if self.trace.state.captured_script.is_none() && !before.is_empty() {
            return Err("null".into());
        }
        let members = self
            .trace
            .state
            .captured_script
            .as_ref()
            .map(|name| self.trace.state.lists[name].items.clone())
            .unwrap_or_default();
        let after: Items = before
            .iter()
            .filter(|id| !members.contains(id))
            .copied()
            .collect();
        let l = self.trace.state.lists.get_mut(&all).unwrap();
        if before.len() != after.len() {
            l.version = l.version.wrapping_add(1)
        };
        l.items = after;
        let bluffable = self.filter("bluffable", Some(&all), None)?;
        let villagers = self.filter("villagers", bluffable.as_deref(), Some(10))?;
        let outcasts = self.filter("outcasts", bluffable.as_deref(), Some(20))?;
        let villagers = villagers.ok_or("null")?;
        for _ in 0..4 {
            if self.trace.state.lists[&villagers].items.is_empty() {
                break;
            }
            let value = self.draw(&villagers, choices, frontier)?;
            self.append(value)?;
            self.remove(&villagers, value)?;
        }
        let outcasts = outcasts.ok_or("null")?;
        if !self.trace.state.lists[&outcasts].items.is_empty() {
            let value = self.draw(&outcasts, choices, frontier)?;
            self.append(value)?;
            self.remove(&outcasts, value)?;
        }
        if self.trace.state.lists["unique"].items.len() <= 1 {
            let fallback = self.getter("fallback")?;
            let b = self.filter("fallback_bluffable", fallback.as_deref(), None)?;
            let g = self.filter("fallback_good", b.as_deref(), Some(10))?;
            let v = self.filter("fallback_villagers", g.as_deref(), Some(10))?;
            let v = v.ok_or("null")?;
            let value = self.draw(&v, choices, frontier)?;
            self.append(value)?;
        }
        Ok(())
    }
}
pub fn replay_trace(c: &Context, choices: &[usize]) -> Result<Trace, LedgerError> {
    validate(c)?;
    if choices.len() > 6 || choices.iter().any(|n| *n > u32::MAX as usize) {
        return Err(LedgerError::InvalidContext);
    }
    let mut r = Run::new(c);
    match r.execute(choices, false) {
        Err(Stop::Failure(e)) => r.trace.error = Some(e),
        Err(Stop::Choice(_)) => unreachable!(),
        Ok(()) => {}
    };
    Ok(r.trace)
}
fn cost(t: &Trace) -> usize {
    let lists = |s: &Value| {
        s["lists"]
            .as_object()
            .unwrap()
            .values()
            .map(|l| 1 + l["items"].as_array().unwrap().len())
            .sum::<usize>()
    };
    lists(&json!(t.state))
        + t.events
            .iter()
            .map(|e| 16 + lists(&e["snapshot"]))
            .sum::<usize>()
        + 4 * t.draws.len()
}
/// Enumerates occurrence indices by resuming a pure, bounded request trace.
/// Empty/failing Range attempts do not split probability or advance a modeled PRNG.
pub fn replay_weighted(c: &Context) -> Result<Vec<WeightedTrace>, LedgerError> {
    validate(c)?;
    let mut pending = VecDeque::from([(
        Vec::<usize>::new(),
        Probability {
            numerator: 1,
            denominator: 1,
        },
    )]);
    let mut done = vec![];
    let mut retained = 0usize;
    while let Some((choices, probability)) = pending.pop_front() {
        let mut r = Run::new(c);
        match r.execute(&choices, true) {
            Err(Stop::Choice(width)) => {
                if pending.len() + done.len() + width > MAX_PATHS {
                    return Err(LedgerError::Capacity);
                }
                let pending_cost = pending.iter().map(|(p, _)| 4 + p.len()).sum::<usize>();
                let new_cost = width
                    .checked_mul(5 + choices.len())
                    .ok_or(LedgerError::Capacity)?;
                if retained
                    .checked_add(pending_cost)
                    .and_then(|v| v.checked_add(new_cost))
                    .and_then(|v| v.checked_add(cost(&r.trace)))
                    .ok_or(LedgerError::Capacity)?
                    > MAX_RETAINED
                {
                    return Err(LedgerError::Capacity);
                }
                let probability = probability.multiply(1, width as u64)?;
                for index in 0..width {
                    let mut next = choices.clone();
                    next.push(index);
                    pending.push_back((next, probability.clone()));
                }
            }
            result => {
                if let Err(Stop::Failure(e)) = result {
                    r.trace.error = Some(e)
                }
                let next = retained
                    .checked_add(cost(&r.trace))
                    .ok_or(LedgerError::Capacity)?;
                let pending_cost = pending.iter().map(|(p, _)| 4 + p.len()).sum::<usize>();
                if next
                    .checked_add(pending_cost)
                    .ok_or(LedgerError::Capacity)?
                    > MAX_RETAINED
                {
                    return Err(LedgerError::Capacity);
                }
                retained = next;
                done.push(WeightedTrace {
                    probability,
                    trace: r.trace,
                });
            }
        }
    }
    Ok(done)
}
#[cfg(test)]
#[path = "round_bluffs_tests.rs"]
mod tests;
