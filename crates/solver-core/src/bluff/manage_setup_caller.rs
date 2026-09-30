//! Bounded replay of the audited complete ManageCharacters caller.
//!
//! Registries name object identities, not public role names. Collection gateways
//! supply stable occurrence lists (no managed version checking); Init supplies
//! only a dataRef write. Layout/builders, publication, Act and callback bodies,
//! Unity equality and coroutine scheduling are not reconstructed. Explicit
//! after-gateway mutations model caller-visible reentry effects, not recursion.
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const MANAGE_SETUP_CALLER_NATIVE_V1: &str = "manage_setup_caller_native_v1";
const MAX_OBJECTS: usize = 32;
const MAX_EVENTS: usize = 4096;
const MAX_RETAINED: usize = 1_048_576;
const ALL_MATCH: [&str; 3] = ["Alchemist", "Poisoner", "Puzzlemaster"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gateway {
    Metadata,
    ClassInit,
    Positions,
    Unique,
    Duplicates,
    Enumerator,
    MoveNext,
    GetItem,
    Init,
    Publish,
    ActInit,
    ActStart,
    Equal,
    Dispose,
    Callback,
    Allocate,
    IteratorCtor,
    StartCoroutine,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub gateway: Gateway,
    pub occurrence: u16,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct List {
    pub items: Vec<Option<String>>,
    /// Native signed Count read independently of supplied enumeration. A
    /// mismatch is an explicit adversarial service input, not a managed List.
    pub count: i32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Array {
    pub items: Vec<Option<String>>,
    /// Signed low 32 bits used by this caller. Must not exceed backing items.
    pub length: i32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Mutation {
    Board { value: Option<String> },
    Order { value: Option<String> },
    Callback { value: Option<String> },
    ListCount { list: String, value: i32 },
    ArrayLength { array: String, value: i32 },
    Identity { card: String, data: Option<String> },
    Role { data: String, role: Option<String> },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfterGateway {
    pub gateway: Gateway,
    pub occurrence: u16,
    pub mutation: Mutation,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub board: Option<String>,
    pub order: Option<String>,
    pub callback: Option<String>,
    pub lists: BTreeMap<String, List>,
    pub arrays: BTreeMap<String, Array>,
    pub identities: BTreeMap<String, Option<String>>,
    pub data_roles: BTreeMap<String, Option<String>>,
    pub iterator_state: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub pinned_class_hierarchies: bool,
    pub stable_occurrence_services: bool,
    pub reference_equality: bool,
    pub supplied_init_data_write_only: bool,
    pub supplied_gateway_effects_only: bool,
    pub caller_metadata_initialized: bool,
    pub shuffle_metadata_initialized: bool,
    pub math_initialized: bool,
    pub gameplay_initialized: bool,
    pub object_initialized: bool,
    pub state: State,
    pub roster: Option<Vec<Option<String>>>,
    /// Action role identity -> native class identity.
    pub roles: BTreeMap<String, String>,
    /// Class identity -> root-to-self hierarchy, retaining exact managed class
    /// identity. The three special identities use their exact metadata names.
    pub classes: BTreeMap<String, Vec<String>>,
    pub callbacks: BTreeSet<String>,
    pub failure: Option<Failure>,
    pub after: Vec<AfterGateway>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Snapshot {
    pub state: State,
    pub effects: Vec<Gateway>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Event {
    pub kind: Gateway,
    #[serde(flatten)]
    pub arguments: BTreeMap<String, Value>,
    pub snapshot: Snapshot,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub events: Vec<Event>,
    pub final_state: Snapshot,
    pub error: Option<String>,
    pub returned: bool,
}
fn bounded_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64
}
fn present<T>(id: &Option<String>, registry: &BTreeMap<String, T>) -> bool {
    id.as_ref().is_none_or(|id| registry.contains_key(id))
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let s = &c.state;
    if c.version != MANAGE_SETUP_CALLER_NATIVE_V1
        || !c.pinned_class_hierarchies
        || !c.stable_occurrence_services
        || !c.reference_equality
        || !c.supplied_init_data_write_only
        || !c.supplied_gateway_effects_only
        || c.failure.as_ref().is_some_and(|f| f.occurrence == 0)
        || !present(&s.board, &s.lists)
        || !present(&s.order, &s.arrays)
        || s.callback
            .as_ref()
            .is_some_and(|id| !c.callbacks.contains(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    let sizes = [
        s.lists.len(),
        s.arrays.len(),
        s.identities.len(),
        s.data_roles.len(),
        c.roles.len(),
        c.classes.len(),
        c.callbacks.len(),
    ];
    if sizes.iter().any(|n| *n > MAX_OBJECTS)
        || c.after.len() > 64
        || c.roster.as_ref().is_some_and(|v| v.len() > MAX_OBJECTS)
    {
        return Err(LedgerError::Capacity);
    }
    let keys = s
        .lists
        .keys()
        .chain(s.arrays.keys())
        .chain(s.identities.keys())
        .chain(s.data_roles.keys())
        .chain(c.roles.keys())
        .chain(c.classes.keys())
        .chain(c.callbacks.iter());
    if keys.clone().any(|id| !bounded_id(id)) {
        return Err(LedgerError::InvalidContext);
    }
    // Object families cannot silently alias across incompatible supplied types.
    let object_keys = s
        .lists
        .keys()
        .chain(s.arrays.keys())
        .chain(s.identities.keys())
        .chain(s.data_roles.keys())
        .chain(c.roles.keys())
        .chain(c.callbacks.iter());
    let mut unique = BTreeSet::new();
    if object_keys.into_iter().any(|id| !unique.insert(id)) {
        return Err(LedgerError::InvalidContext);
    }
    if s.lists.values().any(|v| v.items.len() > MAX_OBJECTS)
        || s.arrays.values().any(|v| v.items.len() > MAX_OBJECTS)
        || c.classes.values().any(|v| v.len() > MAX_OBJECTS)
    {
        return Err(LedgerError::Capacity);
    }
    if s.lists
        .values()
        .flat_map(|v| &v.items)
        .any(|id| !present(id, &s.identities))
        || s.arrays
            .values()
            .flat_map(|v| &v.items)
            .any(|id| !present(id, &s.data_roles))
        || c.roster
            .iter()
            .flatten()
            .any(|id| !present(id, &s.data_roles))
        || s.identities.values().any(|id| !present(id, &s.data_roles))
        || s.data_roles.values().any(|id| !present(id, &c.roles))
        || c.roles.values().any(|id| !c.classes.contains_key(id))
        || s.arrays.values().any(|v| v.length > v.items.len() as i32)
        || c.classes.iter().any(|(id, ancestors)| {
            ancestors.last() != Some(id)
                || ancestors.iter().any(|a| !c.classes.contains_key(a))
                || ancestors.iter().collect::<BTreeSet<_>>().len() != ancestors.len()
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    // A supplied hierarchy must agree with each ancestor's own prefix.
    for ancestors in c.classes.values() {
        for (index, a) in ancestors.iter().enumerate() {
            if c.classes[a] != ancestors[..=index] {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    for a in &c.after {
        if a.occurrence == 0 {
            return Err(LedgerError::InvalidContext);
        }
        let valid = match &a.mutation {
            Mutation::Board { value } => present(value, &s.lists),
            Mutation::Order { value } => present(value, &s.arrays),
            Mutation::Callback { value } => {
                value.as_ref().is_none_or(|id| c.callbacks.contains(id))
            }
            Mutation::ListCount { list, .. } => s.lists.contains_key(list),
            Mutation::ArrayLength { array, value } => s
                .arrays
                .get(array)
                .is_some_and(|v| *value <= v.items.len() as i32),
            Mutation::Identity { card, data } => {
                s.identities.contains_key(card) && present(data, &s.data_roles)
            }
            Mutation::Role { data, role } => {
                s.data_roles.contains_key(data) && present(role, &c.roles)
            }
        };
        if !valid {
            return Err(LedgerError::InvalidContext);
        }
    }
    Ok(())
}
#[derive(Debug)]
enum Stop {
    Failure(String),
    Capacity,
}
impl From<&str> for Stop {
    fn from(s: &str) -> Self {
        Self::Failure(s.into())
    }
}
struct Run<'a> {
    c: &'a Context,
    state: State,
    effects: Vec<Gateway>,
    events: Vec<Event>,
    counts: BTreeMap<Gateway, u16>,
    math: bool,
    gameplay: bool,
    object: bool,
    retained: usize,
}
impl<'a> Run<'a> {
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            state: self.state.clone(),
            effects: self.effects.clone(),
        }
    }
    fn snapshot_units(&self) -> usize {
        1 + self.effects.len()
            + self.state.identities.len()
            + self.state.data_roles.len()
            + self
                .state
                .lists
                .values()
                .map(|l| 1 + l.items.len())
                .sum::<usize>()
            + self
                .state
                .arrays
                .values()
                .map(|a| 1 + a.items.len())
                .sum::<usize>()
    }
    fn emit(&mut self, kind: Gateway, args: Value) -> Result<(), Stop> {
        if self.events.len() >= MAX_EVENTS {
            return Err(Stop::Capacity);
        }
        let units = self.snapshot_units();
        self.retained = self.retained.checked_add(units).ok_or(Stop::Capacity)?;
        if self.retained > MAX_RETAINED {
            return Err(Stop::Capacity);
        }
        let occurrence = self.counts.entry(kind).or_default();
        *occurrence += 1;
        let failed = self
            .c
            .failure
            .as_ref()
            .is_some_and(|f| f.gateway == kind && f.occurrence == *occurrence);
        self.events.push(Event {
            kind,
            arguments: serde_json::from_value(args).expect("internal object"),
            snapshot: self.snapshot(),
        });
        if failed {
            return Err(Stop::Failure(
                serde_json::to_value(kind).unwrap().as_str().unwrap().into(),
            ));
        }
        Ok(())
    }
    fn after(&mut self, kind: Gateway) {
        for a in self
            .c
            .after
            .iter()
            .filter(|a| a.gateway == kind && Some(&a.occurrence) == self.counts.get(&kind))
        {
            match &a.mutation {
                Mutation::Board { value } => self.state.board = value.clone(),
                Mutation::Order { value } => self.state.order = value.clone(),
                Mutation::Callback { value } => self.state.callback = value.clone(),
                Mutation::ListCount { list, value } => {
                    self.state.lists.get_mut(list).unwrap().count = *value
                }
                Mutation::ArrayLength { array, value } => {
                    self.state.arrays.get_mut(array).unwrap().length = *value
                }
                Mutation::Identity { card, data } => {
                    *self.state.identities.get_mut(card).unwrap() = data.clone()
                }
                Mutation::Role { data, role } => {
                    *self.state.data_roles.get_mut(data).unwrap() = role.clone()
                }
            }
        }
    }
    fn effect(&mut self, kind: Gateway, args: Value) -> Result<(), Stop> {
        self.emit(kind, args)?;
        self.effects.push(kind);
        self.after(kind);
        Ok(())
    }
    fn service(&mut self, kind: Gateway, args: Value) -> Result<(), Stop> {
        self.emit(kind, args)?;
        self.after(kind);
        Ok(())
    }
    fn init_class(&mut self, name: &str) -> Result<(), Stop> {
        let initialized = match name {
            "System.Math_TypeInfo" => self.math,
            "Gameplay_TypeInfo" => self.gameplay,
            _ => self.object,
        };
        if !initialized {
            self.service(Gateway::ClassInit, json!({"name":name}))?;
            match name {
                "System.Math_TypeInfo" => self.math = true,
                "Gameplay_TypeInfo" => self.gameplay = true,
                _ => self.object = true,
            }
        }
        Ok(())
    }
    fn enumerate(&mut self) -> Result<String, Stop> {
        let source = self.state.board.clone().ok_or("null")?;
        self.service(Gateway::Enumerator, json!({"source":source}))?;
        Ok(source)
    }
    fn next(&mut self, source: &str, index: usize) -> Result<Option<Option<String>>, Stop> {
        // Capture current item before after-gateway mutations, matching the
        // supplied native MoveNext gateway's enumerator write ordering.
        let item = self.state.lists[source].items.get(index).cloned();
        self.service(Gateway::MoveNext, json!({"source":source,"index":index}))?;
        Ok(item)
    }
    fn body(&mut self) -> Result<(), Stop> {
        if !self.c.caller_metadata_initialized {
            for _ in 0..12 {
                self.service(Gateway::Metadata, json!({}))?;
            }
        }
        for g in [Gateway::Positions, Gateway::Unique, Gateway::Duplicates] {
            self.effect(g, json!({}))?;
        }
        let source = self.enumerate()?;
        let mut i = 0usize;
        while let Some(card) = self.next(&source, i)? {
            let board = self.state.board.as_ref().ok_or("null")?;
            let count = self.state.lists[board].count;
            self.init_class("System.Math_TypeInfo")?;
            let id = count.wrapping_sub(i as i32).wrapping_abs() as u32;
            let roster = self.c.roster.as_ref().ok_or("null")?;
            self.service(Gateway::GetItem, json!({"index":i}))?;
            let data = roster.get(i).ok_or("bounds")?.clone();
            let card = card.ok_or("null")?;
            self.emit(
                Gateway::Init,
                json!({"card":card,"data":data,"display_id":id}),
            )?;
            *self.state.identities.get_mut(&card).unwrap() = data;
            self.effects.push(Gateway::Init);
            self.after(Gateway::Init);
            i += 1;
        }
        self.service(Gateway::Dispose, json!({}))?;
        let publication = self.state.board.clone();
        self.init_class("Gameplay_TypeInfo")?;
        self.effect(Gateway::Publish, json!({"source":publication}))?;
        let source = self.enumerate()?;
        let mut i = 0;
        while let Some(card) = self.next(&source, i)? {
            let card = card.ok_or("null")?;
            self.effect(Gateway::ActInit, json!({"card":card}))?;
            i += 1;
        }
        self.service(Gateway::Dispose, json!({}))?;
        let order = self.state.order.clone().ok_or("null")?;
        let mut index = 0i32;
        while index < self.state.arrays[&order].length {
            let data = self.state.arrays[&order].items[index as usize].clone();
            let source = self.enumerate()?;
            let mut i = 0;
            while let Some(card) = self.next(&source, i)? {
                let card = card.ok_or("null")?;
                let right = self.state.identities[&card].clone();
                self.init_class("UnityEngine.Object_TypeInfo")?;
                self.service(Gateway::Equal, json!({"left":data,"right":right}))?;
                if data == right {
                    self.effect(Gateway::ActStart, json!({"card":card}))?;
                    let data = data.as_ref().ok_or("null")?;
                    let all = self.state.data_roles[data].as_ref().is_some_and(|role| {
                        self.c.classes[&self.c.roles[role]]
                            .iter()
                            .any(|class| ALL_MATCH.contains(&class.as_str()))
                    });
                    if !all {
                        break;
                    }
                }
                i += 1;
            }
            self.service(Gateway::Dispose, json!({}))?;
            index += 1;
        }
        if self.state.callback.is_some() {
            self.effect(Gateway::Callback, json!({}))?;
        }
        if !self.c.shuffle_metadata_initialized {
            self.service(Gateway::Metadata, json!({}))?;
        }
        self.service(Gateway::Allocate, json!({}))?;
        self.service(Gateway::IteratorCtor, json!({}))?;
        self.state.iterator_state = 0;
        self.effect(Gateway::StartCoroutine, json!({}))?;
        Ok(())
    }
}
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut r = Run {
        c,
        state: c.state.clone(),
        effects: vec![],
        events: vec![],
        counts: BTreeMap::new(),
        math: c.math_initialized,
        gameplay: c.gameplay_initialized,
        object: c.object_initialized,
        retained: 0,
    };
    let error = match r.body() {
        Ok(()) => None,
        Err(Stop::Failure(s)) => Some(s),
        Err(Stop::Capacity) => return Err(LedgerError::Capacity),
    };
    if r.retained
        .checked_add(r.snapshot_units())
        .ok_or(LedgerError::Capacity)?
        > MAX_RETAINED
    {
        return Err(LedgerError::Capacity);
    }
    Ok(Replay {
        final_state: r.snapshot(),
        events: r.events,
        error: error.clone(),
        returned: error.is_none(),
    })
}
#[cfg(test)]
#[path = "manage_setup_caller_tests.rs"]
mod tests;
