//! ManageCharacters prefix through both pool builders with one source/cache/RNG history.
//! Services preserve distinct collections and have sufficient capacity. RemoveAll
//! invokes the capture predicate per occurrence and commits only after all return;
//! native List compaction, custom-script selection and Unity PRNG are not modeled.
use super::{
    ledger::{LedgerError, Probability},
    round_candidate_composition::{Asset, ListState},
    round_duplicates::Items,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};

pub const MANAGE_POOL_NATIVE_V1: &str = "manage_pool_native_v1";
const MAX_ITEMS: usize = 96;
const MAX_PATHS: usize = 1024;
const MAX_RETAINED: usize = 2_097_152;
pub type Pools = [Option<Items>; 4];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gateway {
    Positions,
    DuplicateStore,
    GetItem,
    Allocate,
    ObjectCtor,
    ClassInit,
    ListCtor,
    AppendRange,
    ToArray,
    CacheStore,
    Capture,
    Clear,
    PredicateCtor,
    RemoveAll,
    Contains,
    Enumerator,
    MoveNext,
    Dispose,
    FilterAdd,
    PoolAdd,
    Rng,
    Remove,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallbackKind {
    Profile,
    Project,
    Settings,
    Game,
    Roster,
    Board,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Callback {
    /// Getter name and global AppendRange ordinal, after its successful commit.
    pub at: (String, u16),
    pub kind: CallbackKind,
    pub value: Option<usize>,
    #[serde(default)]
    pub items: Option<Items>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub cold: bool,
    pub null_game: bool,
    pub null_pool: bool,
    pub cached: bool,
    pub null_duplicate_pool: bool,
    pub null_board: bool,
    pub replace_after_positions: bool,
    pub replace_on_move: bool,
    pub null_after_move: bool,
    pub board: Vec<Option<Actor>>,
    pub alternate_board: Vec<Option<Actor>>,
    pub manage_roster: Option<Items>,
    pub roster_aliases: [usize; 4],
    /// Two known ScriptInfo identities; a selected null permits later reselection.
    /// CustomScriptData is explicitly empty in this bounded composition.
    pub inline: Vec<Option<usize>>,
    pub alternate_starting: Option<Pools>,
    pub alternate_fallback: Option<Pools>,
    pub callbacks: Vec<Callback>,
    pub fail: Option<(Gateway, u16)>,
    /// Test supplied indices; replay_weighted ignores this field.
    pub choices: Vec<usize>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            cold: false,
            null_game: false,
            null_pool: false,
            cached: false,
            null_duplicate_pool: false,
            null_board: false,
            replace_after_positions: false,
            replace_on_move: false,
            null_after_move: false,
            board: vec![],
            alternate_board: vec![],
            manage_roster: None,
            roster_aliases: [0, 1, 2, 3],
            inline: vec![],
            alternate_starting: None,
            alternate_fallback: None,
            callbacks: vec![],
            fail: None,
            choices: vec![],
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    /// Faction order 10,20,30,100; these are the profile's direct arrays.
    pub starting: Pools,
    pub rosters: Pools,
    pub fallback: Pools,
    pub options: Options,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub metadata_initialized: bool,
    pub distinct_pools_and_intermediates_sufficient_capacity: bool,
    pub stable_services_reference_equality: bool,
    pub deferred_remove_all_commit: bool,
    pub uniform_occurrence_support: bool,
    pub assets: BTreeMap<u16, Asset>,
    pub initial_pool: Items,
    pub initial_version: u32,
    pub initial_duplicate_pool: Items,
    pub initial_duplicate_version: u32,
    pub input: Input,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Actor {
    Character,
    OtherCharacter,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoardState {
    pub current: Option<String>,
    pub items: BTreeMap<String, Vec<Option<Actor>>>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Snapshot {
    /// Logical collection views. Source arrays have an inert version zero.
    pub lists: BTreeMap<String, ListState>,
    pub profile: Option<usize>,
    pub project: bool,
    pub settings: bool,
    pub game: bool,
    pub captured_script: Option<String>,
    pub cache: [Option<usize>; 2],
    pub roster_sources: [Option<String>; 4],
    pub classes: BTreeMap<String, bool>,
    pub board: BoardState,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Trace {
    pub state: Snapshot,
    pub entries: Vec<String>,
    pub typed: Vec<Value>,
    pub events: Vec<Value>,
    pub draws: Vec<Value>,
    pub callbacks: Vec<Value>,
    pub predicate_results: Vec<u8>,
    pub error: Option<String>,
    pub boundary: Option<String>,
    pub init_arguments: Option<Value>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WeightedTrace {
    pub probability: Probability,
    pub trace: Trace,
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let o = &c.input.options;
    if c.version != MANAGE_POOL_NATIVE_V1
        || !c.metadata_initialized
        || !c.distinct_pools_and_intermediates_sufficient_capacity
        || !c.stable_services_reference_equality
        || !c.deferred_remove_all_commit
        || !c.uniform_occurrence_support
        || o.fail.as_ref().is_some_and(|(_, n)| *n == 0)
        || o.inline.iter().flatten().any(|i| *i >= 2)
        || o.roster_aliases.iter().any(|i| *i >= 4)
        || o.callbacks.iter().any(|cb| {
            !["starting", "script", "fallback", "duplicate_script"].contains(&cb.at.0.as_str())
                || cb.at.1 == 0
                || cb.at.1 > 16
                || match cb.kind {
                    CallbackKind::Profile => cb.value.is_some_and(|v| v >= 2) || cb.items.is_some(),
                    CallbackKind::Roster => cb.value.is_none_or(|v| v >= 4),
                    CallbackKind::Board => cb.value.is_some_and(|v| v > 1) || cb.items.is_some(),
                    _ => cb.value.is_some() || cb.items.is_some(),
                }
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    let collections = c
        .input
        .starting
        .iter()
        .chain(&c.input.rosters)
        .chain(&c.input.fallback)
        .chain(o.alternate_starting.iter().flatten())
        .chain(o.alternate_fallback.iter().flatten())
        .chain(std::iter::once(&o.manage_roster))
        .chain(o.callbacks.iter().map(|cb| &cb.items));
    let mut count = c.initial_pool.len() + c.initial_duplicate_pool.len();
    for items in collections.flatten() {
        count = count
            .checked_add(items.len())
            .ok_or(LedgerError::Capacity)?;
        if items.iter().flatten().any(|id| !c.assets.contains_key(id)) {
            return Err(LedgerError::InvalidContext);
        }
    }
    if c.initial_pool
        .iter()
        .chain(&c.initial_duplicate_pool)
        .flatten()
        .any(|id| !c.assets.contains_key(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    // Non-null roster writes require an existing supplied collection identity.
    // Reject sequences that would manufacture a replacement for a null pointer.
    for cb in &o.callbacks {
        if cb.kind == CallbackKind::Roster && cb.items.is_some() {
            let target = cb.value.unwrap();
            if c.input.rosters[o.roster_aliases[target]].is_none()
                || o.callbacks.iter().any(|earlier| {
                    earlier.kind == CallbackKind::Roster
                        && earlier.value == cb.value
                        && earlier.items.is_none()
                        && earlier.at.1 <= cb.at.1
                })
            {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    if count > MAX_ITEMS
        || c.assets.len() > MAX_ITEMS
        || o.inline.len() > 8
        || o.callbacks.len() > 16
        || o.board.len() > 8
        || o.alternate_board.len() > 8
    {
        return Err(LedgerError::Capacity);
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
        let mut lists = BTreeMap::from([(
            "unique".into(),
            ListState {
                items: c.initial_pool.clone(),
                version: c.initial_version,
            },
        )]);
        lists.insert(
            "duplicates".into(),
            ListState {
                items: c.initial_duplicate_pool.clone(),
                version: c.initial_duplicate_version,
            },
        );
        lists.insert(
            "manage_roster".into(),
            ListState {
                items: c.input.options.manage_roster.clone().unwrap_or_default(),
                version: 0,
            },
        );
        let mut add = |name: String, items: &Option<Items>| {
            if let Some(items) = items {
                lists.insert(
                    name,
                    ListState {
                        items: items.clone(),
                        version: 0,
                    },
                );
            }
        };
        for (i, items) in c.input.rosters.iter().enumerate() {
            add(format!("roster{i}"), items);
        }
        for p in 0..2 {
            let starts = if p == 0 {
                &c.input.starting
            } else {
                c.input
                    .options
                    .alternate_starting
                    .as_ref()
                    .unwrap_or(&c.input.starting)
            };
            let falls = if p == 0 {
                &c.input.fallback
            } else {
                c.input
                    .options
                    .alternate_fallback
                    .as_ref()
                    .unwrap_or(&c.input.fallback)
            };
            for i in 0..4 {
                add(format!("profile{p}.starting{i}"), &starts[i]);
                add(format!("profile{p}.all{i}"), &falls[i]);
                add(format!("script{p}.{i}"), &starts[i]);
            }
        }
        Self {
            c,
            counts: BTreeMap::new(),
            trace: Trace {
                state: Snapshot {
                    roster_sources: std::array::from_fn(|i| {
                        let target = c.input.options.roster_aliases[i];
                        c.input.rosters[target]
                            .as_ref()
                            .map(|_| format!("roster{target}"))
                    }),
                    classes: BTreeMap::from([
                        ("Gameplay_TypeInfo".into(), !c.input.options.cold),
                        ("System.Math_TypeInfo".into(), !c.input.options.cold),
                    ]),
                    board: BoardState {
                        current: (!c.input.options.null_board).then(|| "board".into()),
                        items: BTreeMap::from([
                            ("board".into(), c.input.options.board.clone()),
                            ("alternate".into(), c.input.options.alternate_board.clone()),
                        ]),
                    },
                    lists,
                    profile: Some(0),
                    project: true,
                    settings: true,
                    game: !c.input.options.null_game,
                    captured_script: None,
                    cache: if c.input.options.cached {
                        [Some(0), Some(1)]
                    } else {
                        [None, None]
                    },
                },
                entries: vec![],
                typed: vec![],
                events: vec![],
                draws: vec![],
                callbacks: vec![],
                predicate_results: vec![],
                error: None,
                boundary: None,
                init_arguments: None,
            },
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
            .input
            .options
            .fail
            .as_ref()
            .is_some_and(|(gateway, n)| *gateway == g && *n == *count)
        {
            Err(Stop::Failure(name))
        } else {
            Ok(())
        }
    }
    fn entry(&mut self, name: &str) {
        self.trace.entries.push(name.into());
    }
    fn allocate(&mut self, name: &str) -> Result<(), Stop> {
        self.event(Gateway::Allocate, json!({"object":name}))?;
        self.trace.state.lists.insert(
            name.into(),
            ListState {
                items: vec![],
                version: 0,
            },
        );
        self.event(Gateway::ListCtor, json!({"list":name}))
    }
    fn graph(&self) -> Result<usize, Stop> {
        if !self.trace.state.project || !self.trace.state.settings {
            return Err("null".into());
        }
        self.trace.state.profile.ok_or_else(|| "null".into())
    }
    fn append_range(
        &mut self,
        destination: &str,
        source: Option<String>,
        items: Option<Items>,
    ) -> Result<(), Stop> {
        self.event(
            Gateway::AppendRange,
            json!({"destination":destination,"source":source}),
        )?;
        let items = items.ok_or("null_collection")?;
        let list = self.trace.state.lists.get_mut(destination).unwrap();
        list.items.extend(items);
        list.version = list.version.wrapping_add(1);
        let ordinal = self.counts["append_range"];
        for cb in self
            .c
            .input
            .options
            .callbacks
            .iter()
            .filter(|cb| cb.at.0 == destination && cb.at.1 == ordinal)
        {
            match cb.kind {
                CallbackKind::Profile => self.trace.state.profile = cb.value,
                CallbackKind::Project => self.trace.state.project = false,
                CallbackKind::Settings => self.trace.state.settings = false,
                CallbackKind::Game => self.trace.state.game = false,
                CallbackKind::Board => {
                    self.trace.state.board.current =
                        cb.value.filter(|v| *v != 0).map(|_| "alternate".into())
                }
                CallbackKind::Roster => {
                    let target = cb.value.unwrap();
                    if let Some(items) = &cb.items {
                        let name = self.trace.state.roster_sources[target].as_ref().unwrap();
                        let list = self.trace.state.lists.get_mut(name).unwrap();
                        list.items = items.clone();
                        list.version = list.version.wrapping_add(1);
                    } else {
                        self.trace.state.roster_sources[target] = None;
                    }
                }
            }
            let mut value = json!({"at":cb.at,"kind":cb.kind,"value":cb.value});
            if cb.kind == CallbackKind::Roster {
                value["items"] = json!(cb.items);
            }
            self.trace.callbacks.push(value);
        }
        Ok(())
    }
    fn draw(
        &mut self,
        source: &str,
        width: usize,
        choices: &[usize],
        frontier: bool,
    ) -> Result<usize, Stop> {
        let at = self.trace.draws.len();
        let failing = self
            .c
            .input
            .options
            .fail
            .as_ref()
            .is_some_and(|(g, n)| *g == Gateway::Rng && usize::from(*n) == at + 1);
        if frontier && at >= choices.len() && width > 0 && !failing {
            return Err(Stop::Choice(width));
        }
        let index = choices.get(at).copied().unwrap_or(0);
        self.trace
            .draws
            .push(json!({"source":source,"width":width,"index":index}));
        self.event(
            Gateway::Rng,
            json!({"source":source,"width":width,"index":index}),
        )?;
        if index >= width {
            Err("bounds".into())
        } else {
            Ok(index)
        }
    }
    fn typed(
        &mut self,
        p: usize,
        faction: usize,
        choices: &[usize],
        frontier: bool,
    ) -> Result<(Option<String>, Option<Items>), Stop> {
        self.entry("GetStartingtCharactersOfType");
        self.trace
            .typed
            .push(json!({"profile":p,"type":([10,20,30,100][faction])}));
        if self.trace.state.cache[p].is_none() {
            let next = if self.c.input.options.inline.is_empty() {
                None
            } else {
                let index = self.draw(
                    "inline",
                    self.c.input.options.inline.len(),
                    choices,
                    frontier,
                )?;
                self.c.input.options.inline[index]
            };
            self.trace.state.cache[p] = next;
            self.event(Gateway::CacheStore, json!({"profile":p,"script":next}))?;
        }
        if let Some(script) = self.trace.state.cache[p] {
            let source = format!("script{script}.{faction}");
            let items = self
                .trace
                .state
                .lists
                .get(&source)
                .ok_or("null")?
                .items
                .clone();
            self.event(Gateway::ToArray, json!({"source":source}))?;
            Ok((Some(format!("{source}.snapshot")), Some(items)))
        } else {
            let source = format!("profile{p}.starting{faction}");
            let items = self.trace.state.lists.get(&source).map(|l| l.items.clone());
            Ok((items.as_ref().map(|_| source), items))
        }
    }
    fn source_getter(
        &mut self,
        fallback: bool,
        choices: &[usize],
        frontier: bool,
    ) -> Result<(), Stop> {
        if !self.trace.state.game {
            return Err("null".into());
        }
        let destination = if fallback { "fallback" } else { "starting" };
        self.entry(if fallback {
            "GetAllAscensionCharacters"
        } else {
            "GetAscensionAllStartingCharacters"
        });
        self.allocate(destination)?;
        for faction in if fallback { [0, 1, 2, 0] } else { [3, 1, 2, 0] } {
            let profile = self.graph()?;
            let (source, items) = if fallback {
                let source = format!("profile{profile}.all{faction}");
                let items = self.trace.state.lists.get(&source).map(|l| l.items.clone());
                (items.as_ref().map(|_| source), items)
            } else {
                self.typed(profile, faction, choices, frontier)?
            };
            self.append_range(destination, source, items)?;
        }
        Ok(())
    }
    fn script(&mut self, destination: &str) -> Result<(), Stop> {
        if !self.trace.state.game {
            return Err("null".into());
        }
        self.entry("GetScriptCharacters");
        self.allocate(destination)?;
        for i in 0..4 {
            let source = self.trace.state.roster_sources[i].clone();
            let items = source
                .as_ref()
                .map(|s| self.trace.state.lists[s].items.clone());
            self.append_range(destination, source, items)?;
        }
        Ok(())
    }
    fn filter(&mut self, name: &str, source: &str, field: u8, requested: i32) -> Result<(), Stop> {
        self.entry(match field {
            0 => "FilterBluffableCharacters",
            1 => "FilterRealCharacterType",
            _ => "FilterAlignmentCharacters",
        });
        self.allocate(name)?;
        let items = self.trace.state.lists[source].items.clone();
        self.event(Gateway::Enumerator, json!({"source":source}))?;
        for item in items {
            self.event(Gateway::MoveNext, json!({"source":source}))?;
            let asset = &self.c.assets[&item.ok_or("null")?];
            let accepted = match field {
                0 => asset.bluffable != 0,
                1 => asset.real_type == requested,
                _ => asset.alignment == requested,
            };
            if accepted {
                self.event(Gateway::FilterAdd, json!({"list":name,"value":item}))?;
                self.append(name, item);
            }
        }
        self.event(Gateway::MoveNext, json!({"source":source}))?;
        self.event(Gateway::Dispose, json!({}))
    }
    fn append(&mut self, name: &str, value: Option<u16>) {
        let list = self.trace.state.lists.get_mut(name).unwrap();
        list.items.push(value);
        list.version = list.version.wrapping_add(1);
    }
    fn select(
        &mut self,
        name: &str,
        remove: bool,
        choices: &[usize],
        frontier: bool,
    ) -> Result<(), Stop> {
        let index = self.draw(
            "unique",
            self.trace.state.lists[name].items.len(),
            choices,
            frontier,
        )?;
        let value = self.trace.state.lists[name].items[index];
        self.event(Gateway::PoolAdd, json!({"list":"unique","value":value}))?;
        self.append("unique", value);
        if remove {
            self.event(Gateway::Remove, json!({"list":name,"value":value}))?;
            let l = self.trace.state.lists.get_mut(name).unwrap();
            let first = l.items.iter().position(|v| *v == value).unwrap();
            l.items.remove(first);
            l.version = l.version.wrapping_add(1);
        }
        Ok(())
    }
    fn unique(&mut self, choices: &[usize], frontier: bool) -> Result<(), Stop> {
        self.entry("PickRoundBluffs");
        self.event(Gateway::Allocate, json!({"object":"closure"}))?;
        self.event(Gateway::ObjectCtor, json!({}))?;
        self.class_init("Gameplay_TypeInfo")?;
        self.source_getter(false, choices, frontier)?;
        self.script("script")?;
        self.trace.state.captured_script = Some("script".into());
        self.event(Gateway::Capture, json!({"source":"script"}))?;
        if self.c.input.options.null_pool {
            return Err("null".into());
        }
        let l = self.trace.state.lists.get_mut("unique").unwrap();
        let count = l.items.len();
        l.items.clear();
        l.version = l.version.wrapping_add(1);
        if count > 0 {
            self.event(Gateway::Clear, json!({"list":"unique","count":count}))?;
        }
        self.event(Gateway::Allocate, json!({"object":"predicate"}))?;
        self.event(Gateway::PredicateCtor, json!({}))?;
        self.event(Gateway::RemoveAll, json!({}))?;
        let before = self.trace.state.lists["starting"].items.clone();
        let mut after = vec![];
        for value in &before {
            self.entry("ContainsScriptCharacter");
            self.event(Gateway::Contains, json!({"source":"script","value":value}))?;
            let contained = self.trace.state.lists["script"].items.contains(value);
            self.trace.predicate_results.push(u8::from(contained));
            if !contained {
                after.push(*value);
            }
        }
        let l = self.trace.state.lists.get_mut("starting").unwrap();
        if before.len() != after.len() {
            l.version = l.version.wrapping_add(1);
        }
        l.items = after;
        self.filter("bluffable", "starting", 0, 0)?;
        self.filter("villagers", "bluffable", 1, 10)?;
        self.filter("outcasts", "bluffable", 1, 20)?;
        for _ in 0..4 {
            if self.trace.state.lists["villagers"].items.is_empty() {
                break;
            }
            self.select("villagers", true, choices, frontier)?;
        }
        if !self.trace.state.lists["outcasts"].items.is_empty() {
            self.select("outcasts", true, choices, frontier)?;
        }
        if self.trace.state.lists["unique"].items.len() <= 1 {
            self.source_getter(true, choices, frontier)?;
            self.filter("fallback_bluffable", "fallback", 0, 0)?;
            self.filter("fallback_good", "fallback_bluffable", 2, 10)?;
            self.filter("fallback_villagers", "fallback_good", 1, 10)?;
            self.select("fallback_villagers", false, choices, frontier)?;
        }
        Ok(())
    }
    fn class_init(&mut self, name: &str) -> Result<(), Stop> {
        if !self.trace.state.classes[name] {
            self.event(Gateway::ClassInit, json!({"class_name":name}))?;
            self.trace.state.classes.insert(name.into(), true);
        }
        Ok(())
    }
    fn duplicates(&mut self, choices: &[usize], frontier: bool) -> Result<(), Stop> {
        self.entry("PickRoundDuplicates");
        self.class_init("Gameplay_TypeInfo")?;
        self.script("duplicate_script")?;
        if self.c.input.options.null_duplicate_pool {
            return Err("null".into());
        }
        let list = self.trace.state.lists.get_mut("duplicates").unwrap();
        let count = list.items.len();
        list.items.clear();
        list.version = list.version.wrapping_add(1);
        if count > 0 {
            self.event(Gateway::Clear, json!({"list":"duplicates","count":count}))?;
        }
        self.filter("duplicate_bluffable", "duplicate_script", 0, 0)?;
        self.filter("duplicate_villagers", "duplicate_bluffable", 1, 10)?;
        self.filter("duplicate_outcasts", "duplicate_bluffable", 1, 20)?;
        self.filter("duplicate_discarded_good", "duplicate_bluffable", 2, 10)?;
        for phase in 0..5 {
            let outcast = phase == 4;
            let source = if outcast {
                "duplicate_outcasts"
            } else {
                "duplicate_villagers"
            };
            let width = self.trace.state.lists[source].items.len();
            // Native duplicates requests a first Villager index even when empty.
            if width == 0 && phase > 0 {
                continue;
            }
            let index = self.draw("duplicates", width, choices, frontier)?;
            let value = self.trace.state.lists[source].items[index];
            if outcast {
                self.event(Gateway::PoolAdd, json!({"list":"duplicates","value":value}))?;
                self.append("duplicates", value);
            } else {
                self.append("duplicates", value);
                self.event(Gateway::DuplicateStore, json!({"value":value}))?;
            }
            self.event(Gateway::Remove, json!({"list":source,"value":value}))?;
            let local = self.trace.state.lists.get_mut(source).unwrap();
            let first = local.items.iter().position(|v| *v == value).unwrap();
            local.items.remove(first);
            local.version = local.version.wrapping_add(1);
        }
        Ok(())
    }
    fn execute(&mut self, choices: &[usize], frontier: bool) -> Result<(), Stop> {
        self.entry("ManageCharacters");
        self.event(Gateway::Positions, json!({}))?;
        if self.c.input.options.replace_after_positions {
            self.trace.state.board.current = Some("alternate".into());
        }
        self.unique(choices, frontier)?;
        self.duplicates(choices, frontier)?;
        let source = self.trace.state.board.current.clone().ok_or("null")?;
        self.event(Gateway::Enumerator, json!({"source":source}))?;
        self.event(Gateway::MoveNext, json!({"source":source}))?;
        let first = self.trace.state.board.items[&source].first().copied();
        if self.c.input.options.replace_on_move {
            self.trace.state.board.current = Some("alternate".into());
        }
        if self.c.input.options.null_after_move {
            self.trace.state.board.current = None;
        }
        let Some(actor) = first else {
            self.event(Gateway::Dispose, json!({}))?;
            self.trace.boundary = Some("empty_before_publication".into());
            return Ok(());
        };
        let current = self.trace.state.board.current.as_ref().ok_or("null")?;
        let display_id = self.trace.state.board.items[current].len();
        self.class_init("System.Math_TypeInfo")?;
        if self.c.input.options.manage_roster.is_none() {
            return Err("null".into());
        }
        self.event(Gateway::GetItem, json!({"index":0}))?;
        let data = self.trace.state.lists["manage_roster"]
            .items
            .first()
            .ok_or("bounds")?;
        let actor = actor.ok_or("null")?;
        self.trace.init_arguments =
            Some(json!({"character":actor,"data":data,"display_id_bits":display_id}));
        self.trace.boundary = Some("before_first_init".into());
        Ok(())
    }
}
pub fn replay_trace(c: &Context, choices: &[usize]) -> Result<Trace, LedgerError> {
    validate(c)?;
    if choices.len() > 14 || choices.iter().any(|v| *v > u32::MAX as usize) {
        return Err(LedgerError::InvalidContext);
    }
    let mut run = Run::new(c);
    match run.execute(choices, false) {
        Err(Stop::Failure(e)) => run.trace.error = Some(e),
        Err(Stop::Choice(_)) => unreachable!(),
        Ok(()) => {}
    }
    Ok(run.trace)
}
fn retained(t: &Trace) -> usize {
    let cost = |s: &Value| {
        s["lists"]
            .as_object()
            .unwrap()
            .values()
            .map(|l| 1 + l["items"].as_array().unwrap().len())
            .sum::<usize>()
            + s["board"]["items"]
                .as_object()
                .unwrap()
                .values()
                .map(|v| v.as_array().unwrap().len() + 1)
                .sum::<usize>()
            + s["roster_sources"].as_array().unwrap().len()
            + s["classes"].as_object().unwrap().len()
    };
    cost(&json!(t.state))
        + t.events
            .iter()
            .map(|e| 16 + cost(&e["snapshot"]))
            .sum::<usize>()
        + t.entries.len()
        + 4 * t.typed.len()
        + 4 * t.draws.len()
        + t.callbacks
            .iter()
            .map(|c| 8 + c["items"].as_array().map_or(0, Vec::len))
            .sum::<usize>()
        + t.predicate_results.len()
}
/// Bounded uniform occurrence support, retaining pre-sampling failures and a
/// single chronology for inline source selection and subsequent pool draws.
pub fn replay_weighted(c: &Context) -> Result<Vec<WeightedTrace>, LedgerError> {
    validate(c)?;
    let mut pending = VecDeque::from([(
        vec![],
        Probability {
            numerator: 1,
            denominator: 1,
        },
    )]);
    let mut done = vec![];
    let mut used = 0usize;
    while let Some((choices, probability)) = pending.pop_front() {
        let mut run = Run::new(c);
        let result = run.execute(&choices, true);
        let live = retained(&run.trace) + pending.iter().map(|(c, _)| 4 + c.len()).sum::<usize>();
        if used.checked_add(live).ok_or(LedgerError::Capacity)? > MAX_RETAINED {
            return Err(LedgerError::Capacity);
        }
        if let Err(Stop::Choice(width)) = result {
            if pending.len() + done.len() + width > MAX_PATHS
                || used + live + width * (5 + choices.len()) > MAX_RETAINED
            {
                return Err(LedgerError::Capacity);
            }
            let probability = probability.multiply(1, width as u64)?;
            for index in 0..width {
                let mut next = choices.clone();
                next.push(index);
                pending.push_back((next, probability.clone()));
            }
        } else {
            if let Err(Stop::Failure(e)) = result {
                run.trace.error = Some(e);
            }
            used += retained(&run.trace);
            done.push(WeightedTrace {
                probability,
                trace: run.trace,
            });
        }
    }
    Ok(done)
}

#[cfg(test)]
#[path = "manage_pool_composition_tests.rs"]
mod tests;
