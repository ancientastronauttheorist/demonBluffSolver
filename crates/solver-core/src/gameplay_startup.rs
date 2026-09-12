//! Bounded initialized Gameplay startup with native saved/current list identity.
//! Init refreshes saved lists; Restart copies the existing saved lists.
use crate::score_replay::ScoreState;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const GAMEPLAY_STARTUP_NATIVE_V1: &str = "gameplay_startup_native_v1";
const MAX_LISTS: usize = 64;
const MAX_ELEMENTS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartupList {
    pub elements: Vec<i32>,
    pub version: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartupState {
    pub lists: BTreeMap<u64, StartupList>,
    pub scores: BTreeMap<u64, ScoreState>,
    pub saved: [Option<u64>; 4],
    pub current: [Option<u64>; 4],
    pub deck: Option<u64>,
    pub relics: Option<u64>,
    pub score: Option<u64>,
    pub current_level: i32,
    pub current_day: i32,
    pub starting_level: i32,
    pub project_present: bool,
    /// UnityEngine.Object, Gameplay, GameData runtime class flags, respectively.
    pub classes_initialized: [bool; 3],
    /// Copy lineage is retained separately from equal element content.
    pub copy_sources: BTreeMap<u64, u64>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupOperation {
    Init,
    RestartGame,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectReplacement {
    None,
    AfterEquality,
    AfterFirstPool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupGateway {
    ClassInit,
    UnityNullCheck,
    TypedPool,
    Allocate,
    ListCopy,
    Store,
    ChangeState,
    ArrayClear,
    ResetPlayer,
    HealthReset,
    StartingLevel,
    ResetLevel,
    ModeKind,
    ObjectCtor,
    LoadCharacters,
    StartCoroutine,
}
impl StartupGateway {
    fn bound(self) -> u32 {
        match self {
            Self::ClassInit => 3,
            Self::TypedPool => 4,
            Self::Allocate => 10,
            Self::ListCopy => 8,
            Self::Store => 9,
            Self::ArrayClear | Self::StartingLevel | Self::ObjectCtor => 2,
            _ => 1,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartupFailurePoint {
    pub gateway: StartupGateway,
    pub occurrence: u32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupFailure {
    NullReference,
    NullCollection,
    Gateway(StartupFailurePoint),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupClass {
    UnityObject,
    Gameplay,
    GameData,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupAllocationKind {
    CharacterList,
    ScoreOld,
    InitCoroutine,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupSlot {
    Saved(u8),
    Current(u8),
    Score,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupTarget {
    Receiver,
    Object(u64),
    StateValue(i32),
    DeckArray,
    RelicArray,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameplayStartupContext {
    pub rule_version: String,
    pub operation: StartupOperation,
    pub initial: StartupState,
    pub metadata_initialized: bool,
    /// External callbacks/services preserve all modeled state except explicit
    /// copies, zero allocations, publication and the selected project mutation.
    pub preserving_services_and_audited_slots: bool,
    pub zero_initialized_allocations: bool,
    pub first_allocation_id: u64,
    pub unity_null_result: bool,
    pub project_game_present: bool,
    pub replacement: ProjectReplacement,
    /// Typed 10,20,30,100 provider identities; null reaches copy-constructor failure.
    pub pools: [Option<u64>; 4],
    pub mode_present: bool,
    pub player_present: bool,
    pub health_group_present: bool,
    pub health_present: bool,
    pub starting_levels: [i32; 2],
    pub reset_level: i32,
    pub mode_kind: i32,
    pub fail_at: Option<StartupFailurePoint>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StartupEvent {
    pub gateway: StartupGateway,
    pub occurrence: u32,
    pub class: Option<StartupClass>,
    pub allocation_kind: Option<StartupAllocationKind>,
    pub target: Option<StartupTarget>,
    pub source: Option<u64>,
    pub slot: Option<StartupSlot>,
    pub character_type: Option<i32>,
    pub project_present: Option<bool>,
    pub value: Option<u64>,
    pub state: StartupState,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GameplayStartupResult {
    pub state: StartupState,
    pub events: Vec<StartupEvent>,
    pub failure: Option<StartupFailure>,
    pub allocations: BTreeMap<u64, StartupAllocationKind>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidGameplayStartupContext;
struct Replay<'a> {
    ctx: &'a GameplayStartupContext,
    out: GameplayStartupResult,
    counts: [u32; 16],
    next_id: u64,
}
impl Replay<'_> {
    fn emit(&mut self, mut event: StartupEvent) -> Result<(), StartupFailure> {
        self.counts[event.gateway as usize] += 1;
        event.occurrence = self.counts[event.gateway as usize];
        event.state = self.out.state.clone();
        let point = StartupFailurePoint {
            gateway: event.gateway,
            occurrence: event.occurrence,
        };
        self.out.events.push(event);
        if self.ctx.fail_at == Some(point) {
            Err(StartupFailure::Gateway(point))
        } else {
            Ok(())
        }
    }
    fn event(&self, gateway: StartupGateway) -> StartupEvent {
        StartupEvent {
            gateway,
            occurrence: 0,
            class: None,
            allocation_kind: None,
            target: None,
            source: None,
            slot: None,
            character_type: None,
            project_present: None,
            value: None,
            state: self.out.state.clone(),
        }
    }
    fn simple(&mut self, gateway: StartupGateway) -> Result<(), StartupFailure> {
        self.emit(self.event(gateway))
    }
    fn target(
        &mut self,
        gateway: StartupGateway,
        target: StartupTarget,
    ) -> Result<(), StartupFailure> {
        let mut event = self.event(gateway);
        event.target = Some(target);
        self.emit(event)
    }
    fn require(value: bool) -> Result<(), StartupFailure> {
        if value {
            Ok(())
        } else {
            Err(StartupFailure::NullReference)
        }
    }
    fn initialize(&mut self, class: StartupClass) -> Result<(), StartupFailure> {
        let index = class as usize;
        if !self.out.state.classes_initialized[index] {
            let mut e = self.event(StartupGateway::ClassInit);
            e.class = Some(class);
            self.emit(e)?;
            self.out.state.classes_initialized[index] = true;
        }
        Ok(())
    }
    fn allocate(&mut self, kind: StartupAllocationKind) -> Result<u64, StartupFailure> {
        let mut event = self.event(StartupGateway::Allocate);
        event.allocation_kind = Some(kind);
        self.emit(event)?;
        let id = self.next_id;
        self.next_id += 1;
        self.out.allocations.insert(id, kind);
        if kind == StartupAllocationKind::ScoreOld {
            self.out.state.scores.insert(
                id,
                ScoreState {
                    completed_stages: 0,
                    points_for_completing: 0,
                    completed_days: 0,
                    multiplier: 0,
                    round_points: 0,
                    overall_points: 0,
                    killed_goods: 0,
                    temp_unrevealed_cards: 0,
                    unrevealed_cards: 0,
                    killed_evils: 0,
                    temp_killed_evils: 0,
                    point_per_kill: 0,
                    points_per_unrevealed: 0,
                },
            );
        }
        Ok(id)
    }
    fn copy(&mut self, source: Option<u64>) -> Result<u64, StartupFailure> {
        let id = self.allocate(StartupAllocationKind::CharacterList)?;
        let mut event = self.event(StartupGateway::ListCopy);
        event.target = Some(StartupTarget::Object(id));
        event.source = source;
        self.emit(event)?;
        let source = source.ok_or(StartupFailure::NullCollection)?;
        let elements = self.out.state.lists[&source].elements.clone();
        self.out.state.lists.insert(
            id,
            StartupList {
                elements,
                version: 0,
            },
        );
        self.out.state.copy_sources.insert(id, source);
        Ok(id)
    }
    fn store(&mut self, slot: StartupSlot, id: u64) -> Result<(), StartupFailure> {
        match slot {
            StartupSlot::Saved(i) => self.out.state.saved[i as usize] = Some(id),
            StartupSlot::Current(i) => self.out.state.current[i as usize] = Some(id),
            StartupSlot::Score => self.out.state.score = Some(id),
        }
        let mut event = self.event(StartupGateway::Store);
        event.slot = Some(slot);
        event.value = Some(id);
        self.emit(event)
    }
    fn reset_saved(&mut self) -> Result<(), StartupFailure> {
        self.initialize(StartupClass::UnityObject)?;
        let mut event = self.event(StartupGateway::UnityNullCheck);
        event.project_present = Some(self.out.state.project_present);
        self.emit(event)?;
        if self.ctx.replacement == ProjectReplacement::AfterEquality {
            self.out.state.project_present = false;
        }
        if self.ctx.unity_null_result {
            return Ok(());
        }
        Self::require(self.out.state.project_present && self.ctx.project_game_present)?;
        // GameData is captured here. Later global replacement does not redirect
        // any of the four provider calls or alter the supplied pool identities.
        for (i, typ) in [10, 20, 30, 100].into_iter().enumerate() {
            let mut event = self.event(StartupGateway::TypedPool);
            event.character_type = Some(typ);
            self.emit(event)?;
            if self.ctx.replacement == ProjectReplacement::AfterFirstPool {
                self.out.state.project_present = false;
            }
            let id = self.copy(self.ctx.pools[i])?;
            self.store(StartupSlot::Saved(i as u8), id)?;
        }
        Ok(())
    }
    fn clear(&mut self, id: Option<u64>, target: StartupTarget) -> Result<(), StartupFailure> {
        let id = id.ok_or(StartupFailure::NullReference)?;
        let list = self.out.state.lists.get_mut(&id).unwrap();
        let occupied = !list.elements.is_empty();
        list.elements.clear();
        list.version = list.version.wrapping_add(1);
        if occupied {
            self.target(StartupGateway::ArrayClear, target)?;
        }
        Ok(())
    }
    fn copy_current(&mut self) -> Result<(), StartupFailure> {
        for i in 0..4 {
            let id = self.copy(self.out.state.saved[i])?;
            self.store(StartupSlot::Current(i as u8), id)?;
        }
        Ok(())
    }
    fn score(&mut self) -> Result<u64, StartupFailure> {
        let id = self.allocate(StartupAllocationKind::ScoreOld)?;
        let s = self.out.state.scores.get_mut(&id).unwrap();
        s.points_for_completing = 100;
        s.point_per_kill = 50;
        s.points_per_unrevealed = 10;
        self.target(StartupGateway::ObjectCtor, StartupTarget::Object(id))?;
        Ok(id)
    }
    fn mode_and_load(&mut self) -> Result<(), StartupFailure> {
        Self::require(self.ctx.mode_present)?;
        self.simple(StartupGateway::ModeKind)?;
        if self.ctx.mode_kind == 0 {
            self.target(StartupGateway::LoadCharacters, StartupTarget::Receiver)?;
        }
        Ok(())
    }
    fn run(&mut self) -> Result<(), StartupFailure> {
        if self.ctx.operation == StartupOperation::Init {
            self.reset_saved()?;
            self.initialize(StartupClass::Gameplay)?;
            // The native static ChangeGameplayState gateway receives integer 1;
            // its internal state/event effects are excluded by the contract.
            self.target(StartupGateway::ChangeState, StartupTarget::StateValue(1))?;
            self.out.state.current_day = 0;
            self.clear(self.out.state.deck, StartupTarget::DeckArray)?;
            self.clear(self.out.state.relics, StartupTarget::RelicArray)?;
            self.copy_current()?;
            let score = self.score()?;
            self.store(StartupSlot::Score, score)?;
            self.initialize(StartupClass::GameData)?;
            Self::require(self.ctx.mode_present)?;
            self.simple(StartupGateway::StartingLevel)?;
            self.out.state.current_level = self.ctx.starting_levels[0];
            Self::require(self.ctx.mode_present)?;
            self.simple(StartupGateway::StartingLevel)?;
            self.out.state.starting_level = self.ctx.starting_levels[1];
            self.mode_and_load()?;
            let iterator = self.allocate(StartupAllocationKind::InitCoroutine)?;
            self.target(StartupGateway::ObjectCtor, StartupTarget::Object(iterator))?;
            self.target(StartupGateway::StartCoroutine, StartupTarget::Receiver)?;
        } else {
            self.target(StartupGateway::ResetPlayer, StartupTarget::Receiver)?;
            Self::require(
                self.ctx.player_present && self.ctx.health_group_present && self.ctx.health_present,
            )?;
            self.simple(StartupGateway::HealthReset)?;
            self.copy_current()?;
            self.initialize(StartupClass::GameData)?;
            Self::require(self.ctx.mode_present)?;
            self.simple(StartupGateway::ResetLevel)?;
            self.out.state.current_level = self.ctx.reset_level;
            let score = self.score()?;
            self.initialize(StartupClass::Gameplay)?;
            self.store(StartupSlot::Score, score)?;
            self.out.state.current_day = 0;
            self.mode_and_load()?;
        }
        Ok(())
    }
}
/// Unsupported contexts reject atomically; native failures expose every completed
/// publication and allocated-but-unpublished Score object.
pub fn replay_gameplay_startup(
    ctx: &GameplayStartupContext,
) -> Result<GameplayStartupResult, InvalidGameplayStartupContext> {
    let state = &ctx.initial;
    let known = |id: Option<u64>| id.is_none_or(|id| id != 0 && state.lists.contains_key(&id));
    if ctx.rule_version != GAMEPLAY_STARTUP_NATIVE_V1
        || !ctx.metadata_initialized
        || !ctx.preserving_services_and_audited_slots
        || !ctx.zero_initialized_allocations
        || state.lists.len() > MAX_LISTS - 10
        || state.scores.len() > 15
        || state.lists.contains_key(&0)
        || state.scores.contains_key(&0)
        || state.lists.keys().any(|id| state.scores.contains_key(id))
        || state
            .lists
            .values()
            .any(|l| l.elements.len() > MAX_ELEMENTS)
        || state
            .lists
            .values()
            .map(|l| l.elements.len())
            .sum::<usize>()
            > 1024
        || state
            .saved
            .iter()
            .chain(state.current.iter())
            .chain(ctx.pools.iter())
            .any(|id| !known(*id))
        || !known(state.deck)
        || !known(state.relics)
        || state
            .score
            .is_some_and(|id| !state.scores.contains_key(&id))
        || !state.copy_sources.is_empty()
        || ctx.first_allocation_id == 0
        || ctx.first_allocation_id.checked_add(10).is_none()
        || (0..10).any(|offset| {
            ctx.first_allocation_id
                .checked_add(offset)
                .is_some_and(|id| state.lists.contains_key(&id) || state.scores.contains_key(&id))
        })
        || ctx
            .fail_at
            .is_some_and(|p| p.occurrence == 0 || p.occurrence > p.gateway.bound())
    {
        return Err(InvalidGameplayStartupContext);
    }
    let mut replay = Replay {
        ctx,
        next_id: ctx.first_allocation_id,
        counts: [0; 16],
        out: GameplayStartupResult {
            state: state.clone(),
            events: Vec::new(),
            failure: None,
            allocations: BTreeMap::new(),
        },
    };
    replay.out.failure = replay.run().err();
    Ok(replay.out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    fn id(label: &str) -> Option<u64> {
        if label == "null" {
            None
        } else if label == "previous_score" {
            Some(1)
        } else if let Some(hex) = label.strip_prefix("previous_instance_") {
            Some(0x1000 + u64::from_str_radix(hex, 16).unwrap())
        } else if let Some(hex) = label.strip_prefix("previous_static_") {
            Some(0x2000 + u64::from_str_radix(hex, 16).unwrap())
        } else if let Some(typ) = label.strip_prefix("pool") {
            Some(0x3000 + typ.parse::<u64>().unwrap())
        } else if let Some(rest) = label.strip_prefix("allocation") {
            Some(0x4000 + rest.split(':').next().unwrap().parse::<u64>().unwrap())
        } else {
            panic!("unmapped fixture object {label}")
        }
    }
    fn label(id: Option<u64>, allocations: &BTreeMap<u64, StartupAllocationKind>) -> String {
        match id {
            None => "null".into(),
            Some(1) => "previous_score".into(),
            Some(n) if (0x1000..0x2000).contains(&n) => {
                format!("previous_instance_{:X}", n - 0x1000)
            }
            Some(n) if (0x2000..0x3000).contains(&n) => format!("previous_static_{:X}", n - 0x2000),
            Some(n) if (0x3000..0x4000).contains(&n) => format!("pool{}", n - 0x3000),
            Some(n) => format!(
                "allocation{}:{}",
                n - 0x4000,
                allocation_name(allocations[&n])
            ),
        }
    }
    fn allocation_name(kind: StartupAllocationKind) -> &'static str {
        match kind {
            StartupAllocationKind::CharacterList => {
                "System.Collections.Generic.List<CharacterData>"
            }
            StartupAllocationKind::ScoreOld => "ScoreOld",
            StartupAllocationKind::InitCoroutine => "Gameplay.<InitCoroutine>d__41",
        }
    }
    fn score(defaults: &Value) -> ScoreState {
        ScoreState {
            completed_stages: 0,
            points_for_completing: defaults["14"].as_u64().unwrap() as i32,
            completed_days: 0,
            multiplier: 0,
            round_points: 0,
            overall_points: 0,
            killed_goods: 0,
            temp_unrevealed_cards: 0,
            unrevealed_cards: 0,
            killed_evils: 0,
            temp_killed_evils: 0,
            point_per_kill: defaults["3C"].as_u64().unwrap() as i32,
            points_per_unrevealed: defaults["40"].as_u64().unwrap() as i32,
        }
    }
    fn context(case: &Value) -> GameplayStartupContext {
        let initial = &case["initial"];
        let input = &case["input"];
        let reference = |key: &str| id(initial["references"][key].as_str().unwrap());
        let cold = |name: &str| {
            input["cold"]
                .as_array()
                .is_some_and(|values| values.contains(&json!(name)))
        };
        let lists = initial["list_contents"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, values)| {
                (
                    id(name).unwrap(),
                    StartupList {
                        elements: serde_json::from_value(values.clone()).unwrap(),
                        version: initial["list_headers"][name][1].as_u64().unwrap() as u32,
                    },
                )
            })
            .collect();
        let missing = input["null"].as_str();
        GameplayStartupContext {
            rule_version: GAMEPLAY_STARTUP_NATIVE_V1.into(),
            operation: if case["method"] == "Init" {
                StartupOperation::Init
            } else {
                StartupOperation::RestartGame
            },
            initial: StartupState {
                lists,
                scores: [(1, score(&initial["score_defaults"]))]
                    .into_iter()
                    .collect(),
                saved: ["instance+48", "instance+50", "instance+58", "instance+60"].map(reference),
                current: ["instance+28", "instance+30", "instance+38", "instance+40"]
                    .map(reference),
                deck: reference("instance+20"),
                relics: reference("static+0"),
                score: reference("static+8"),
                current_level: initial["counters"]["currentLevel"].as_u64().unwrap() as i32,
                current_day: initial["counters"]["currentDay"].as_u64().unwrap() as i32,
                starting_level: initial["counters"]["startingLevel"].as_u64().unwrap() as i32,
                project_present: initial["project_present"].as_bool().unwrap(),
                classes_initialized: [
                    !cold("UnityEngine.Object"),
                    !cold("Gameplay"),
                    !cold("GameData"),
                ],
                copy_sources: BTreeMap::new(),
            },
            metadata_initialized: true,
            preserving_services_and_audited_slots: true,
            zero_initialized_allocations: true,
            first_allocation_id: 0x4001,
            unity_null_result: input["equality_null"].as_bool().unwrap_or(false),
            project_game_present: missing != Some("project_game"),
            replacement: if input["replace_after_equality"] == true {
                ProjectReplacement::AfterEquality
            } else if input["replace_after_first_pool"] == true {
                ProjectReplacement::AfterFirstPool
            } else {
                ProjectReplacement::None
            },
            pools: [10, 20, 30, 100].map(|typ| {
                if input["null_pool"] == typ {
                    None
                } else {
                    Some(0x3000 + typ as u64)
                }
            }),
            mode_present: missing != Some("mode"),
            player_present: missing != Some("player"),
            health_group_present: missing != Some("health_group"),
            health_present: missing != Some("health"),
            starting_levels: input["levels"]
                .as_array()
                .map(|v| [v[0].as_i64().unwrap() as i32, v[1].as_i64().unwrap() as i32])
                .unwrap_or([2, 5]),
            reset_level: input["levels"][0].as_i64().unwrap_or(2) as i32,
            mode_kind: input["mode_kind"].as_i64().unwrap_or(0) as i32,
            fail_at: input["fail"].as_array().map(|v| StartupFailurePoint {
                gateway: serde_json::from_value(v[0].clone()).unwrap(),
                occurrence: v[1].as_u64().unwrap() as u32,
            }),
        }
    }
    fn snapshot(
        state: &StartupState,
        initial: &Value,
        allocations: &BTreeMap<u64, StartupAllocationKind>,
    ) -> Value {
        let mut value = initial.clone();
        for (index, offset) in [0x48, 0x50, 0x58, 0x60].into_iter().enumerate() {
            value["references"][format!("instance+{offset:X}")] =
                json!(label(state.saved[index], allocations));
        }
        for (index, offset) in [0x28, 0x30, 0x38, 0x40].into_iter().enumerate() {
            value["references"][format!("instance+{offset:X}")] =
                json!(label(state.current[index], allocations));
        }
        value["references"]["instance+20"] = json!(label(state.deck, allocations));
        value["references"]["static+0"] = json!(label(state.relics, allocations));
        value["references"]["static+8"] = json!(label(state.score, allocations));
        for (name, v) in [
            ("currentLevel", state.current_level),
            ("currentDay", state.current_day),
            ("startingLevel", state.starting_level),
        ] {
            value["counters"][name] = json!(v as u32);
        }
        value["project_present"] = json!(state.project_present);
        value["list_headers"] = json!({});
        value["list_contents"] = json!({});
        for (id, list) in &state.lists {
            let name = label(Some(*id), allocations);
            value["list_headers"][&name] = json!([list.elements.len(), list.version]);
            value["list_contents"][&name] = json!(list.elements);
        }
        value["copy_sources"] = json!({});
        for (target, source) in &state.copy_sources {
            value["copy_sources"][label(Some(*target), allocations)] =
                json!(label(Some(*source), allocations));
        }
        let defaults = |s: &ScoreState| json!({"14":s.points_for_completing as u32,"3C":s.point_per_kill as u32,"40":s.points_per_unrevealed as u32});
        value["score_defaults"] = state
            .score
            .map(|id| defaults(&state.scores[&id]))
            .unwrap_or(Value::Null);
        value["allocated_scores"] = json!({});
        for (id, fields) in &state.scores {
            if *id >= 0x4001 {
                value["allocated_scores"][label(Some(*id), allocations)] = defaults(fields);
            }
        }
        value
    }
    fn event(
        event: &StartupEvent,
        initial: &Value,
        allocations: &BTreeMap<u64, StartupAllocationKind>,
    ) -> Value {
        let mut value =
            json!({"kind":event.gateway,"snapshot":snapshot(&event.state,initial,allocations)});
        if let Some(class) = event.class {
            value["type"] = json!(match class {
                StartupClass::UnityObject => "UnityEngine.Object",
                StartupClass::Gameplay => "Gameplay",
                StartupClass::GameData => "GameData",
            });
        }
        if let Some(kind) = event.allocation_kind {
            value["type"] = json!(allocation_name(kind));
        }
        if let Some(target) = event.target {
            value["target"] = json!(match target {
                StartupTarget::Receiver => "receiver".into(),
                StartupTarget::Object(id) => label(Some(id), allocations),
                StartupTarget::StateValue(v) => format!("unknown:{v:X}"),
                StartupTarget::DeckArray => "unknown:200020100".into(),
                StartupTarget::RelicArray => "unknown:200030100".into(),
            });
        }
        if event.gateway == StartupGateway::ListCopy {
            value["source"] = json!(label(event.source, allocations));
        }
        if let Some(slot) = event.slot {
            value["slot"] = json!(match slot {
                StartupSlot::Saved(i) => format!("instance+{:X}", 0x48 + i * 8),
                StartupSlot::Current(i) => format!("instance+{:X}", 0x28 + i * 8),
                StartupSlot::Score => "static+8".into(),
            });
        }
        if let Some(v) = event.value {
            value["value"] = json!(label(Some(v), allocations));
        }
        if let Some(v) = event.character_type {
            value["character_type"] = json!(v);
        }
        if let Some(v) = event.project_present {
            value["project_present"] = json!(v);
        }
        value
    }
    fn corpus() -> Value {
        serde_json::from_str(include_str!("../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_gameplay_startup_composition.json")).unwrap()
    }
    #[test]
    fn all_111_native_startup_compositions() {
        let report = corpus();
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 111);
        for (index, case) in cases.iter().enumerate() {
            let ctx = context(case);
            let out = replay_gameplay_startup(&ctx).unwrap();
            assert_eq!(
                snapshot(&out.state, &case["initial"], &out.allocations),
                case["final"],
                "case {index}"
            );
            let expected_events = case["events"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    let mut v = v.clone();
                    let n = v
                        .as_object_mut()
                        .unwrap()
                        .remove("snapshot_index")
                        .unwrap()
                        .as_u64()
                        .unwrap();
                    v["snapshot"] = report["snapshot_table"][n as usize].clone();
                    v
                })
                .collect::<Vec<_>>();
            assert_eq!(
                out.events
                    .iter()
                    .map(|e| event(e, &case["initial"], &out.allocations))
                    .collect::<Vec<_>>(),
                expected_events,
                "case {index}"
            );
            let error = match out.failure {
                None => Value::Null,
                Some(StartupFailure::NullReference) => json!("null"),
                Some(StartupFailure::NullCollection) => json!("null_list"),
                Some(StartupFailure::Gateway(point)) => json!(point.gateway),
            };
            assert_eq!(error, case["error"], "case {index}");
        }
    }
    #[test]
    fn unsupported_contracts_and_aliasing_allocation_ids_reject_atomically() {
        let report = corpus();
        let base = context(&report["cases"][0]);
        for which in 0..8 {
            let mut ctx = base.clone();
            match which {
                0 => ctx.metadata_initialized = false,
                1 => ctx.preserving_services_and_audited_slots = false,
                2 => ctx.zero_initialized_allocations = false,
                3 => ctx.first_allocation_id = 1,
                4 => ctx.first_allocation_id = u64::MAX,
                5 => ctx.pools[0] = Some(999),
                6 => ctx.initial.lists.values_mut().next().unwrap().elements = vec![1; 65],
                _ => {
                    ctx.fail_at = Some(StartupFailurePoint {
                        gateway: StartupGateway::Store,
                        occurrence: 10,
                    })
                }
            }
            let before = ctx.clone();
            assert_eq!(
                replay_gameplay_startup(&ctx),
                Err(InvalidGameplayStartupContext)
            );
            assert_eq!(ctx, before);
        }
    }
    #[test]
    fn saved_refresh_is_init_only_and_score_failure_keeps_allocated_defaults() {
        let report = corpus();
        let mut ctx = context(&report["cases"][0]);
        ctx.fail_at = Some(StartupFailurePoint {
            gateway: StartupGateway::ObjectCtor,
            occurrence: 1,
        });
        let out = replay_gameplay_startup(&ctx).unwrap();
        assert_eq!(out.state.score, ctx.initial.score);
        assert_ne!(out.state.saved, ctx.initial.saved);
        let score_id = *out
            .allocations
            .iter()
            .find(|(_, k)| **k == StartupAllocationKind::ScoreOld)
            .unwrap()
            .0;
        assert_eq!(out.state.scores[&score_id].point_per_kill, 50);
        assert_eq!(out.state.scores[&score_id].points_per_unrevealed, 10);
        assert_eq!(out.state.scores[&score_id].points_for_completing, 100);
        ctx.operation = StartupOperation::RestartGame;
        ctx.fail_at = None;
        let out = replay_gameplay_startup(&ctx).unwrap();
        assert_eq!(out.state.saved, ctx.initial.saved);
        assert_ne!(out.state.current, ctx.initial.current);
    }
}
