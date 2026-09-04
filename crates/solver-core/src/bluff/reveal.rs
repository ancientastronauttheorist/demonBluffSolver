//! Bounded offline gameplay projection of delayed Character.Reveal callbacks.
//!
//! Starts at a caller-proven continuation resume and ends after AfterRoundStart.
//! Supports Lilis/Twin/Drunk data, plus explicit Spy role caches in v2, and
//! Scout/Witness/Confessor bluff assets only. V3 adds an explicit Start latch
//! and HealthyBluff re-entry for status-only and inert callbacks.
//! It does not reconstruct coroutine order, native object graphs, view updates,
//! or subscribers. The caller must exclude intervening mutations, including
//! omitted resumes and view epilogues, from the modeled state. No live GameState
//! bridge consumes this module.

use super::ledger::{
    replay_selectors, LedgerError, Probability, Selector, SelectorEvent, SelectorLedger,
    SelectorPools, SelectorTrace, SELECTOR_LEDGER_NATIVE_V1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REVEAL_CALLBACKS_NATIVE_V1: &str = "bounded_reveal_callbacks_native_v1";
pub const REVEAL_CALLBACKS_SPY_NATIVE_V2: &str = "bounded_reveal_callbacks_spy_native_v2";
pub const REVEAL_CALLBACKS_START_NATIVE_V3: &str = "bounded_reveal_callbacks_start_native_v3";
const MAX_RESUMES: usize = 16;
const MAX_PATHS: usize = 65_536;
const MAX_ENTRIES: usize = 1_048_576;
const CORRUPTED: i32 = 10;
const APPEAR_TRUTHFUL: i32 = 25;
const APPEAR_LYING: i32 = 26;
const HEALTHY_BLUFF: i32 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DataRole {
    Lilis,
    TwinMinion,
    Drunk,
    /// Identifies the actual dataRef.role object, not the per-card role clone.
    Spy {
        cache_key: u16,
    },
}

/// These role classes have no Init/AfterRoundStart gameplay effect except
/// Confessor.OnInit. Spy's action-role clone cache is inert at these triggers;
/// only dataRef.role's cache participates in register-as and bluff selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallbackRole {
    Lilis,
    TwinMinion,
    Drunk,
    Scout,
    Witness,
    Confessor,
    Spy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BluffRole {
    Scout,
    Witness,
    Confessor,
}

impl BluffRole {
    fn name(self) -> &'static str {
        match self {
            Self::Scout => "Scout",
            Self::Witness => "Witness",
            Self::Confessor => "Confessor",
        }
    }
    fn parse(name: &str) -> Option<Self> {
        match name {
            "Scout" => Some(Self::Scout),
            "Witness" => Some(Self::Witness),
            "Confessor" => Some(Self::Confessor),
            _ => None,
        }
    }
    fn callback(self) -> CallbackRole {
        match self {
            Self::Scout => CallbackRole::Scout,
            Self::Witness => CallbackRole::Witness,
            Self::Confessor => CallbackRole::Confessor,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BluffReference {
    Null,
    Destroyed { role: BluffRole },
    Live { role: BluffRole },
}

impl BluffReference {
    fn is_live(self) -> bool {
        matches!(self, Self::Live { .. })
    }
}

/// Native ordered status/resistance lists and the single shared target pointer.
/// Other enum values are preserved, not interpreted as alternate lie flags.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusState {
    pub values: Vec<i32>,
    pub resistance: Vec<i32>,
    pub target_position: Option<u8>,
}

impl StatusState {
    fn apply(&mut self, status: i32, target: Option<u8>) -> StatusApplication {
        let accepted = !self.resistance.contains(&status);
        let inserted = accepted && !self.values.contains(&status);
        if inserted {
            self.values.push(status);
        }
        if accepted {
            self.target_position = target;
        }
        StatusApplication {
            status,
            accepted,
            inserted,
            target_after: self.target_position,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealActor {
    pub position: u8,
    pub data_role: DataRole,
    pub action_role: CallbackRole,
    pub runtime_evil: bool,
    pub bluff: BluffReference,
    pub bluff_role: Option<CallbackRole>,
    /// Non-Spy data roles overwrite this with null; Spy uses its role cache.
    pub register_as: Option<String>,
    pub statuses: StatusState,
    pub remaining_continuations: u16,
    pub on_trigger_subscribed: bool,
    /// Explicit native characterStartActed provenance; required in v3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_start_acted: Option<bool>,
}

impl RevealActor {
    pub fn is_lying(&self) -> bool {
        self.statuses.values.contains(&CORRUPTED)
            || (!self.statuses.values.contains(&HEALTHY_BLUFF)
                && (self.runtime_evil || self.bluff.is_live()))
    }

    pub fn appears_lying(&self) -> bool {
        self.statuses.values.contains(&APPEAR_LYING)
            || (!self.statuses.values.contains(&APPEAR_TRUTHFUL) && self.is_lying())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeEvent {
    pub position: u8,
    pub resume_ordinal: u16,
    /// Independent acquisition provenance. Must be absent when raw bluff is
    /// live; resume ordinal is never substituted for acquisition ordinal.
    pub acquisition_ordinal: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealContext {
    pub rule_version: String,
    pub board_size: u8,
    pub trailer_mode: bool,
    pub pools: SelectorPools,
    pub actors: Vec<RevealActor>,
    pub resumes: Vec<ResumeEvent>,
    /// Required provenance for every distinct Spy dataRef.role object. No
    /// missing cache is inferred to be null. Equal keys explicitly share state.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub spy_caches: BTreeMap<u16, BluffReference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StatusApplication {
    pub status: i32,
    pub accepted: bool,
    pub inserted: bool,
    pub target_after: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    Start,
    Init,
    AfterRoundStart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Dispatch {
    Act,
    BluffAct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleSlot {
    Real,
    Bluff,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallbackTrace {
    pub trigger: Trigger,
    pub slot: RoleSlot,
    pub role: CallbackRole,
    pub dispatch: Dispatch,
    pub status_application: Option<StatusApplication>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResumeTrace {
    pub event: ResumeEvent,
    pub previous_register_as: Option<String>,
    pub acquisition: Option<SelectorTrace>,
    pub selector_status: Option<StatusApplication>,
    pub callbacks: Vec<CallbackTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spy_register_as: Option<SpyRegisterTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spy_acquisition: Option<SpyAcquisitionTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpyRegisterSource {
    LiveCache,
    ScriptVillager { occurrence_index: u16 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpyRegisterTrace {
    pub cache_key: u16,
    pub previous_cache: BluffReference,
    pub role: BluffRole,
    pub source: SpyRegisterSource,
    pub rng_draw_count: u8,
}

/// Spy selects from the cache just populated/read by register-as. This call
/// consumes no additional RNG, removes no pool item, and registers no asset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpyAcquisitionTrace {
    pub acquisition_ordinal: u16,
    pub role: BluffRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RevealPath {
    pub probability: Probability,
    pub pools: SelectorPools,
    pub actors: Vec<RevealActor>,
    pub trace: Vec<ResumeTrace>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub spy_caches: BTreeMap<u16, BluffReference>,
}

fn selector_ledger(pools: SelectorPools, events: Vec<SelectorEvent>) -> SelectorLedger {
    SelectorLedger {
        rule_version: SELECTOR_LEDGER_NATIVE_V1.into(),
        pools,
        events,
    }
}

fn validate(context: &RevealContext) -> Result<(), LedgerError> {
    if ![
        REVEAL_CALLBACKS_NATIVE_V1,
        REVEAL_CALLBACKS_SPY_NATIVE_V2,
        REVEAL_CALLBACKS_START_NATIVE_V3,
    ]
    .contains(&context.rule_version.as_str())
        || context.board_size == 0
        || context.trailer_mode
        || context.resumes.len() > MAX_RESUMES
        || context.actors.len() > usize::from(context.board_size)
        || [
            &context.pools.unique,
            &context.pools.duplicate,
            &context.pools.must_include,
        ]
        .iter()
        .any(|pool| pool.iter().any(|role| BluffRole::parse(role).is_none()))
    {
        return Err(LedgerError::InvalidContext);
    }
    // Reuse the selector ledger's canonical asset, faction and occurrence caps.
    replay_selectors(&selector_ledger(context.pools.clone(), Vec::new()))?;
    let cache_keys: BTreeSet<_> = context
        .actors
        .iter()
        .filter_map(|actor| {
            if let DataRole::Spy { cache_key } = actor.data_role {
                Some(cache_key)
            } else {
                None
            }
        })
        .collect();
    if cache_keys != context.spy_caches.keys().copied().collect()
        || (context.rule_version == REVEAL_CALLBACKS_NATIVE_V1
            && (!cache_keys.is_empty()
                || context.actors.iter().any(|a| {
                    a.action_role == CallbackRole::Spy || a.bluff_role == Some(CallbackRole::Spy)
                })))
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut seen = [false; 256];
    for actor in &context.actors {
        if actor.position == 0
            || actor.position > context.board_size
            || seen[usize::from(actor.position)]
            || actor.on_trigger_subscribed
            || (context.rule_version == REVEAL_CALLBACKS_START_NATIVE_V3
                && actor.character_start_acted.is_none())
            || (context.rule_version != REVEAL_CALLBACKS_START_NATIVE_V3
                && (actor.character_start_acted.is_some()
                    || actor.statuses.values.contains(&HEALTHY_BLUFF)))
            || actor.statuses.values.len() > 256
            || actor.statuses.resistance.len() > 256
            || actor
                .statuses
                .target_position
                .is_some_and(|p| p == 0 || p > context.board_size)
            || actor.register_as.as_ref().is_some_and(|role| {
                crate::knowledge_base::get_card(role).is_none_or(|card| card.name != role)
            })
        {
            return Err(LedgerError::InvalidContext);
        }
        seen[usize::from(actor.position)] = true;
    }
    let mut last_resume = None;
    let mut last_acquisition = None;
    for event in &context.resumes {
        if !seen[usize::from(event.position)]
            || last_resume.is_some_and(|last| last >= event.resume_ordinal)
            || event
                .acquisition_ordinal
                .is_some_and(|a| last_acquisition.is_some_and(|last| last >= a))
        {
            return Err(LedgerError::InvalidContext);
        }
        last_resume = Some(event.resume_ordinal);
        if event.acquisition_ordinal.is_some() {
            last_acquisition = event.acquisition_ordinal;
        }
    }
    Ok(())
}

fn callbacks(actor: &mut RevealActor) -> Result<Vec<CallbackTrace>, LedgerError> {
    let mut result = Vec::new();
    for trigger in [Trigger::Start, Trigger::Init, Trigger::AfterRoundStart] {
        if trigger == Trigger::Start {
            if !actor.statuses.values.contains(&HEALTHY_BLUFF)
                || actor.character_start_acted == Some(true)
            {
                continue;
            }
            // onTrigger subscribers are excluded even when the latch is set:
            // native invokes them before checking this guard.
            if actor.character_start_acted != Some(false)
                || [Some(actor.action_role), actor.bluff_role]
                    .iter()
                    .any(|role| matches!(role, Some(CallbackRole::TwinMinion | CallbackRole::Spy)))
            {
                return Err(LedgerError::InvalidContext);
            }
            actor.character_start_acted = Some(true);
        }
        // Character.Act computes this once before real then copied dispatch.
        let lying = actor.is_lying();
        let real_dispatch = if !lying || (actor.runtime_evil && actor.bluff_role.is_some()) {
            Dispatch::Act
        } else {
            Dispatch::BluffAct
        };
        for (slot, role, dispatch) in [
            (RoleSlot::Real, Some(actor.action_role), real_dispatch),
            (
                RoleSlot::Bluff,
                actor.bluff_role,
                if lying {
                    Dispatch::BluffAct
                } else {
                    Dispatch::Act
                },
            ),
        ] {
            if let Some(role) = role {
                let status_application = match (trigger, role) {
                    (Trigger::Init, CallbackRole::Confessor) => {
                        Some(actor.statuses.apply(APPEAR_TRUTHFUL, None))
                    }
                    (Trigger::Start, CallbackRole::Drunk) => {
                        Some(actor.statuses.apply(CORRUPTED, Some(actor.position)))
                    }
                    (Trigger::Start, CallbackRole::Lilis) => Some(actor.statuses.apply(60, None)),
                    _ => None,
                };
                result.push(CallbackTrace {
                    trigger,
                    slot,
                    role,
                    dispatch,
                    status_application,
                });
            }
        }
    }
    Ok(result)
}

fn push_bounded(
    paths: &mut Vec<RevealPath>,
    path: RevealPath,
    entries: &mut usize,
) -> Result<(), LedgerError> {
    let script = &path.pools.script;
    let count = path.spy_caches.len()
        + path.pools.unique.len()
        + path.pools.duplicate.len()
        + path.pools.must_include.len()
        + script.villagers.len()
        + script.outcasts.len()
        + script.minions.len()
        + script.demons.len()
        + path
            .actors
            .iter()
            .map(|a| 1 + a.statuses.values.len() + a.statuses.resistance.len())
            .sum::<usize>()
        + path
            .trace
            .iter()
            .map(|t| {
                1 + t.callbacks.len()
                    + usize::from(t.spy_register_as.is_some())
                    + usize::from(t.spy_acquisition.is_some())
            })
            .sum::<usize>();
    *entries = entries.checked_add(count).ok_or(LedgerError::Capacity)?;
    if paths.len() >= MAX_PATHS || *entries > MAX_ENTRIES {
        return Err(LedgerError::Capacity);
    }
    paths.push(path);
    Ok(())
}

/// Called after consuming the continuation. The raw-bluff guard is checked by
/// the caller, but never used to skip Spy's earlier register-as invocation.
fn spy_resume(
    path: &RevealPath,
    actor_index: usize,
    event: &ResumeEvent,
    previous_register_as: &Option<String>,
    cache_key: u16,
) -> Result<Vec<RevealPath>, LedgerError> {
    let previous_cache = *path
        .spy_caches
        .get(&cache_key)
        .ok_or(LedgerError::InvalidContext)?;
    let choices = match previous_cache {
        BluffReference::Live { role } => vec![(role, SpyRegisterSource::LiveCache)],
        BluffReference::Null | BluffReference::Destroyed { .. } => {
            if path.pools.script.villagers.is_empty() {
                return Err(LedgerError::EmptySupport);
            }
            // Validate the whole source before branching; never discard an
            // unsupported Villager and renormalize the remaining occurrences.
            path.pools
                .script
                .villagers
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    Ok((
                        BluffRole::parse(name).ok_or(LedgerError::InvalidContext)?,
                        SpyRegisterSource::ScriptVillager {
                            occurrence_index: u16::try_from(index)
                                .map_err(|_| LedgerError::Capacity)?,
                        },
                    ))
                })
                .collect::<Result<Vec<_>, LedgerError>>()?
        }
    };
    let probability = path.probability.multiply(1, choices.len() as u64)?;
    let mut paths = Vec::new();
    let mut entries = 0;
    for (role, source) in choices {
        let mut branch = path.clone();
        branch.probability = probability;
        branch
            .spy_caches
            .insert(cache_key, BluffReference::Live { role });
        let actor = &mut branch.actors[actor_index];
        actor.register_as = Some(role.name().into());
        let spy_acquisition = event.acquisition_ordinal.map(|acquisition_ordinal| {
            actor.bluff = BluffReference::Live { role };
            actor.bluff_role = Some(role.callback());
            SpyAcquisitionTrace {
                acquisition_ordinal,
                role,
            }
        });
        let callbacks = callbacks(actor)?;
        let rng_draw_count = u8::from(matches!(source, SpyRegisterSource::ScriptVillager { .. }));
        branch.trace.push(ResumeTrace {
            event: event.clone(),
            previous_register_as: previous_register_as.clone(),
            acquisition: None,
            selector_status: None,
            callbacks,
            spy_register_as: Some(SpyRegisterTrace {
                cache_key,
                previous_cache,
                role,
                source,
                rng_draw_count,
            }),
            spy_acquisition,
        });
        push_bounded(&mut paths, branch, &mut entries)?;
    }
    Ok(paths)
}

/// Compose proven resumes through the synchronous gameplay callbacks. Repeated
/// bodies consume another pending continuation but do not reacquire a live bluff.
/// Any unsupported branch rejects the whole invocation; no partial worlds leak.
pub fn replay_reveal_callbacks(context: &RevealContext) -> Result<Vec<RevealPath>, LedgerError> {
    validate(context)?;
    let mut paths = vec![RevealPath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        pools: context.pools.clone(),
        actors: context.actors.clone(),
        trace: Vec::new(),
        spy_caches: context.spy_caches.clone(),
    }];
    for event in &context.resumes {
        let mut next = Vec::new();
        let mut entries = 0;
        for mut path in paths {
            let actor_index = path
                .actors
                .iter()
                .position(|a| a.position == event.position)
                .ok_or(LedgerError::InvalidContext)?;
            let actor = &mut path.actors[actor_index];
            if actor.remaining_continuations == 0
                || actor.bluff.is_live() == event.acquisition_ordinal.is_some()
            {
                return Err(LedgerError::InvalidContext);
            }
            actor.remaining_continuations -= 1;
            let previous_register_as = actor.register_as.take();
            if let DataRole::Spy { cache_key } = actor.data_role {
                for branch in
                    spy_resume(&path, actor_index, event, &previous_register_as, cache_key)?
                {
                    push_bounded(&mut next, branch, &mut entries)?;
                }
                continue;
            }
            if let Some(acquisition_ordinal) = event.acquisition_ordinal {
                let selector = match actor.data_role {
                    DataRole::Lilis => Selector::Demon,
                    DataRole::TwinMinion => Selector::Minion,
                    DataRole::Drunk => Selector::Drunk {
                        corruption_resistant: actor.statuses.resistance.contains(&CORRUPTED),
                    },
                    DataRole::Spy { .. } => unreachable!("Spy register-as handled above"),
                };
                let selector_status = if actor.data_role == DataRole::Drunk {
                    Some(actor.statuses.apply(CORRUPTED, Some(actor.position)))
                } else {
                    None
                };
                let draws = replay_selectors(&selector_ledger(
                    path.pools.clone(),
                    vec![SelectorEvent {
                        position: event.position,
                        acquisition_ordinal,
                        selector,
                    }],
                ))?;
                for draw in draws {
                    let mut branch = path.clone();
                    branch.probability = branch
                        .probability
                        .multiply(draw.probability.numerator, draw.probability.denominator)?;
                    branch.pools = draw.pools;
                    let acquisition = draw
                        .trace
                        .into_iter()
                        .next()
                        .ok_or(LedgerError::InvalidContext)?;
                    let role = BluffRole::parse(&acquisition.bluff_role)
                        .ok_or(LedgerError::InvalidContext)?;
                    let actor = &mut branch.actors[actor_index];
                    actor.bluff = BluffReference::Live { role };
                    actor.bluff_role = Some(role.callback());
                    let callbacks = callbacks(actor)?;
                    branch.trace.push(ResumeTrace {
                        event: event.clone(),
                        previous_register_as: previous_register_as.clone(),
                        acquisition: Some(acquisition),
                        selector_status: selector_status.clone(),
                        callbacks,
                        spy_register_as: None,
                        spy_acquisition: None,
                    });
                    push_bounded(&mut next, branch, &mut entries)?;
                }
            } else {
                let callbacks = callbacks(actor)?;
                path.trace.push(ResumeTrace {
                    event: event.clone(),
                    previous_register_as,
                    acquisition: None,
                    selector_status: None,
                    callbacks,
                    spy_register_as: None,
                    spy_acquisition: None,
                });
                push_bounded(&mut next, path, &mut entries)?;
            }
        }
        paths = next;
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bluff::ledger::ScriptLists;

    fn names(roles: &[&str]) -> Vec<String> {
        roles.iter().map(|r| r.to_string()).collect()
    }

    fn actor(position: u8, data_role: DataRole) -> RevealActor {
        RevealActor {
            position,
            data_role,
            action_role: match data_role {
                DataRole::Lilis => CallbackRole::Lilis,
                DataRole::TwinMinion => CallbackRole::TwinMinion,
                DataRole::Drunk => CallbackRole::Drunk,
                DataRole::Spy { .. } => CallbackRole::Spy,
            },
            runtime_evil: data_role == DataRole::Lilis,
            bluff: BluffReference::Null,
            bluff_role: None,
            register_as: Some("Bard".into()),
            statuses: StatusState {
                values: vec![],
                resistance: vec![],
                target_position: None,
            },
            remaining_continuations: 2,
            on_trigger_subscribed: false,
            character_start_acted: None,
        }
    }

    fn resume(position: u8, resume_ordinal: u16, acquisition_ordinal: Option<u16>) -> ResumeEvent {
        ResumeEvent {
            position,
            resume_ordinal,
            acquisition_ordinal,
        }
    }

    fn context(actors: Vec<RevealActor>, resumes: Vec<ResumeEvent>) -> RevealContext {
        RevealContext {
            rule_version: REVEAL_CALLBACKS_NATIVE_V1.into(),
            board_size: 8,
            trailer_mode: false,
            pools: SelectorPools {
                unique: names(&["Witness"]),
                duplicate: names(&["Scout"]),
                must_include: vec![],
                script: ScriptLists {
                    villagers: names(&["Bard"]),
                    outcasts: names(&["Drunk"]),
                    minions: names(&["Twin Minion"]),
                    demons: names(&["Lilis"]),
                },
            },
            actors,
            resumes,
            spy_caches: BTreeMap::new(),
        }
    }

    fn spy_context(actors: Vec<RevealActor>, resumes: Vec<ResumeEvent>) -> RevealContext {
        let mut input = context(actors, resumes);
        input.rule_version = REVEAL_CALLBACKS_SPY_NATIVE_V2.into();
        input.pools.script.villagers = names(&["Scout", "Scout", "Confessor"]);
        for actor in &input.actors {
            if let DataRole::Spy { cache_key } = actor.data_role {
                input.spy_caches.insert(cache_key, BluffReference::Null);
            }
        }
        input
    }

    fn healthy_context(role: DataRole) -> RevealContext {
        let mut input = context(
            vec![actor(1, role)],
            vec![resume(1, 0, None), resume(1, 1, None)],
        );
        input.rule_version = REVEAL_CALLBACKS_START_NATIVE_V3.into();
        let actor = &mut input.actors[0];
        actor.character_start_acted = Some(false);
        actor.statuses.values = vec![HEALTHY_BLUFF];
        actor.bluff = BluffReference::Live {
            role: BluffRole::Witness,
        };
        actor.bluff_role = Some(CallbackRole::Witness);
        input
    }

    #[test]
    fn healthy_drunk_start_freezes_dispatch_then_init_rechecks_corruption() {
        let path = replay_reveal_callbacks(&healthy_context(DataRole::Drunk))
            .unwrap()
            .remove(0);
        let first = &path.trace[0].callbacks;
        assert_eq!(first.len(), 6);
        assert_eq!(first[0].trigger, Trigger::Start);
        assert_eq!(first[0].dispatch, Dispatch::Act);
        assert_eq!(
            first[0].status_application.as_ref().unwrap().target_after,
            Some(1)
        );
        assert_eq!(first[1].dispatch, Dispatch::Act);
        assert_eq!(first[2].dispatch, Dispatch::BluffAct);
        assert_eq!(first[3].dispatch, Dispatch::BluffAct);
        assert_eq!(path.trace[1].callbacks.len(), 4);
        assert_eq!(path.actors[0].character_start_acted, Some(true));
        assert_eq!(path.actors[0].remaining_continuations, 0);
        assert_eq!(
            path.actors[0].statuses.values,
            vec![HEALTHY_BLUFF, CORRUPTED]
        );
    }

    #[test]
    fn healthy_drunk_resistance_keeps_truth_and_consumes_start_latch() {
        let mut input = healthy_context(DataRole::Drunk);
        input.actors[0].statuses.resistance.push(CORRUPTED);
        input.actors[0].statuses.target_position = Some(8);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        let application = path.trace[0].callbacks[0]
            .status_application
            .as_ref()
            .unwrap();
        assert!(!application.accepted);
        assert_eq!(application.target_after, Some(8));
        assert!(path
            .trace
            .iter()
            .flat_map(|t| &t.callbacks)
            .all(|c| c.dispatch == Dispatch::Act));
        assert_eq!(path.actors[0].character_start_acted, Some(true));
    }

    #[test]
    fn healthy_lilis_duplicate_status_clears_target_unless_resisted() {
        for resisted in [false, true] {
            let mut input = healthy_context(DataRole::Lilis);
            input.actors[0].statuses.values.push(60);
            input.actors[0].statuses.target_position = Some(8);
            if resisted {
                input.actors[0].statuses.resistance.push(60);
            }
            let path = replay_reveal_callbacks(&input).unwrap().remove(0);
            let application = path.trace[0].callbacks[0]
                .status_application
                .as_ref()
                .unwrap();
            assert!(!application.inserted);
            assert_eq!(application.accepted, !resisted);
            assert_eq!(
                application.target_after,
                if resisted { Some(8) } else { None }
            );
            assert_eq!(path.trace[1].callbacks.len(), 4);
        }
    }

    #[test]
    fn healthy_acquisition_precedes_start_and_confessor_init() {
        let mut input = healthy_context(DataRole::Drunk);
        input.actors[0].bluff = BluffReference::Null;
        input.actors[0].bluff_role = Some(CallbackRole::TwinMinion);
        input.resumes[0].acquisition_ordinal = Some(7);
        input.pools.unique = names(&["Confessor"]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert!(path.trace[0].selector_status.as_ref().unwrap().inserted);
        assert!(
            !path.trace[0].callbacks[0]
                .status_application
                .as_ref()
                .unwrap()
                .inserted
        );
        assert_eq!(path.trace[0].callbacks[1].role, CallbackRole::Confessor);
        assert_eq!(path.trace[0].callbacks[1].dispatch, Dispatch::BluffAct);
        assert_eq!(path.actors[0].statuses.target_position, None);
        assert!(path.actors[0].statuses.values.contains(&APPEAR_TRUTHFUL));
    }

    #[test]
    fn healthy_copied_status_writer_uses_same_start_decision() {
        let mut input = healthy_context(DataRole::Lilis);
        input.actors[0].action_role = CallbackRole::Scout;
        input.actors[0].bluff_role = Some(CallbackRole::Drunk);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.trace[0].callbacks[1].dispatch, Dispatch::Act);
        assert_eq!(
            path.trace[0].callbacks[1]
                .status_application
                .as_ref()
                .unwrap()
                .status,
            CORRUPTED
        );
        assert_eq!(path.trace[0].callbacks[3].dispatch, Dispatch::BluffAct);
    }

    #[test]
    fn healthy_start_requires_versioned_latch_provenance() {
        let mut input = healthy_context(DataRole::Lilis);
        input.actors[0].character_start_acted = None;
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input.actors[0].character_start_acted = Some(false);
        for version in [REVEAL_CALLBACKS_NATIVE_V1, REVEAL_CALLBACKS_SPY_NATIVE_V2] {
            input.rule_version = version.into();
            input.actors[0].statuses.values.clear();
            assert_eq!(
                replay_reveal_callbacks(&input),
                Err(LedgerError::InvalidContext)
            );
        }
        let old = context(vec![actor(1, DataRole::Lilis)], vec![resume(1, 0, Some(0))]);
        let json = serde_json::to_value(replay_reveal_callbacks(&old).unwrap()).unwrap();
        assert!(json[0]["actors"][0].get("character_start_acted").is_none());
    }

    #[test]
    fn healthy_unsupported_start_only_rejected_when_reached_but_subscribers_always_rejected() {
        for role in [CallbackRole::TwinMinion, CallbackRole::Spy] {
            for copied in [false, true] {
                let mut input = healthy_context(DataRole::Lilis);
                if copied {
                    input.actors[0].bluff_role = Some(role);
                } else {
                    input.actors[0].action_role = role;
                }
                assert_eq!(
                    replay_reveal_callbacks(&input),
                    Err(LedgerError::InvalidContext)
                );
                input.actors[0].character_start_acted = Some(true);
                assert!(replay_reveal_callbacks(&input).is_ok());
                input.actors[0].on_trigger_subscribed = true;
                assert_eq!(
                    replay_reveal_callbacks(&input),
                    Err(LedgerError::InvalidContext)
                );
                input.actors[0].on_trigger_subscribed = false;
                input.actors[0].character_start_acted = Some(false);
                input.actors[0].statuses.values.clear();
                let path = replay_reveal_callbacks(&input).unwrap().remove(0);
                assert_eq!(path.actors[0].character_start_acted, Some(false));
            }
        }
    }

    #[test]
    fn spy_script_occurrences_supply_one_register_draw_and_cached_acquisition() {
        let input = spy_context(
            vec![actor(1, DataRole::Spy { cache_key: 0 })],
            vec![resume(1, 2, Some(0))],
        );
        let paths = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(paths.len(), 3);
        for (index, path) in paths.iter().enumerate() {
            assert_eq!(
                path.probability,
                Probability {
                    numerator: 1,
                    denominator: 3
                }
            );
            let trace = path.trace[0].spy_register_as.as_ref().unwrap();
            assert_eq!(
                trace.source,
                SpyRegisterSource::ScriptVillager {
                    occurrence_index: index as u16
                }
            );
            assert_eq!(trace.rng_draw_count, 1);
            assert_eq!(trace.previous_cache, BluffReference::Null);
            assert_eq!(
                path.actors[0].register_as.as_deref(),
                Some(trace.role.name())
            );
            assert_eq!(
                path.actors[0].bluff,
                BluffReference::Live { role: trace.role }
            );
            assert_eq!(
                path.trace[0].spy_acquisition.as_ref().unwrap().role,
                trace.role
            );
            assert!(path.trace[0].acquisition.is_none());
            assert!(path.trace[0].selector_status.is_none());
            assert_eq!(path.pools, input.pools);
        }
        assert_eq!(paths[0].actors, paths[1].actors);
        assert_eq!(paths[2].actors[0].statuses.values, vec![APPEAR_TRUTHFUL]);
        assert_eq!(input.spy_caches[&0], BluffReference::Null);
    }

    #[test]
    fn spy_same_role_object_shares_cache_across_bodies_and_repeat_resumes() {
        let input = spy_context(
            vec![
                actor(1, DataRole::Spy { cache_key: 9 }),
                actor(2, DataRole::Spy { cache_key: 9 }),
            ],
            vec![
                resume(1, 0, Some(0)),
                resume(2, 1, Some(1)),
                resume(1, 2, None),
            ],
        );
        let paths = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(paths.len(), 3); // not 9: the second physical Spy reads the same cache
        for path in paths {
            assert_eq!(path.actors[0].bluff, path.actors[1].bluff);
            assert_eq!(path.actors[0].register_as, path.actors[1].register_as);
            for trace in &path.trace[1..] {
                let register = trace.spy_register_as.as_ref().unwrap();
                assert_eq!(register.source, SpyRegisterSource::LiveCache);
                assert_eq!(register.rng_draw_count, 0);
            }
            assert!(path.trace[1].spy_acquisition.is_some());
            assert!(path.trace[2].spy_acquisition.is_none());
            assert_eq!(path.trace[2].callbacks.len(), 4);
        }
    }

    #[test]
    fn spy_distinct_role_objects_branch_independently() {
        let input = spy_context(
            vec![
                actor(1, DataRole::Spy { cache_key: 2 }),
                actor(2, DataRole::Spy { cache_key: 7 }),
            ],
            vec![resume(1, 0, Some(0)), resume(2, 1, Some(1))],
        );
        let paths = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(paths.len(), 9);
        assert!(paths.iter().all(|path| path.probability
            == Probability {
                numerator: 1,
                denominator: 9
            }));
        assert!(paths
            .iter()
            .any(|path| path.actors[0].bluff != path.actors[1].bluff));
        assert!(paths.iter().all(|path| path.trace[1]
            .spy_register_as
            .as_ref()
            .unwrap()
            .rng_draw_count
            == 1));
    }

    #[test]
    fn spy_live_bluff_does_not_skip_register_as_or_replace_copied_role() {
        let mut body = actor(1, DataRole::Spy { cache_key: 4 });
        body.bluff = BluffReference::Live {
            role: BluffRole::Scout,
        };
        body.bluff_role = Some(CallbackRole::Confessor);
        let mut input = spy_context(vec![body], vec![resume(1, 0, None)]);
        input.pools.script.villagers = names(&["Witness"]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.actors[0].register_as.as_deref(), Some("Witness"));
        assert_eq!(path.actors[0].bluff, input.actors[0].bluff);
        assert_eq!(path.actors[0].bluff_role, input.actors[0].bluff_role);
        assert_eq!(
            path.trace[0]
                .spy_register_as
                .as_ref()
                .unwrap()
                .rng_draw_count,
            1
        );
        assert!(path.trace[0].spy_acquisition.is_none());
        assert_eq!(path.actors[0].statuses.values, vec![APPEAR_TRUTHFUL]);
    }

    #[test]
    fn spy_live_cache_ignores_script_membership_and_old_register_as() {
        let mut input = spy_context(
            vec![actor(1, DataRole::Spy { cache_key: 1 })],
            vec![resume(1, 0, Some(0))],
        );
        input.spy_caches.insert(
            1,
            BluffReference::Live {
                role: BluffRole::Confessor,
            },
        );
        input.pools.script.villagers.clear();
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.actors[0].register_as.as_deref(), Some("Confessor"));
        assert_eq!(path.trace[0].previous_register_as.as_deref(), Some("Bard"));
        assert_eq!(
            path.trace[0]
                .spy_register_as
                .as_ref()
                .unwrap()
                .rng_draw_count,
            0
        );
        assert_eq!(path.pools, input.pools);
        // A supported live cache is usable even when current script contains
        // callback classes outside this bounded model; it doesn't reread it.
        input.pools.script.villagers = names(&["Bard"]);
        assert!(replay_reveal_callbacks(&input).is_ok());
    }

    #[test]
    fn spy_destroyed_cache_is_resampled_before_bluff_guard() {
        let mut input = spy_context(
            vec![actor(1, DataRole::Spy { cache_key: 6 })],
            vec![resume(1, 0, Some(0))],
        );
        input.spy_caches.insert(
            6,
            BluffReference::Destroyed {
                role: BluffRole::Witness,
            },
        );
        let paths = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(paths.len(), 3);
        assert!(paths.iter().all(
            |p| p.trace[0].spy_register_as.as_ref().unwrap().previous_cache
                == BluffReference::Destroyed {
                    role: BluffRole::Witness
                }
        ));
        assert!(paths
            .iter()
            .all(|p| p.actors[0].register_as.as_deref() != Some("Witness")));
    }

    #[test]
    fn minion_script_addition_changes_later_spy_weights() {
        let mut input = spy_context(
            vec![
                actor(1, DataRole::TwinMinion),
                actor(2, DataRole::Spy { cache_key: 1 }),
            ],
            vec![resume(1, 0, Some(0)), resume(2, 1, Some(1))],
        );
        input.pools.script.villagers = names(&["Scout"]);
        input.pools.duplicate = names(&["Confessor"]);
        input.pools.unique = names(&["Witness"]);
        let paths = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(paths.len(), 3);
        assert_eq!(
            paths.iter().map(|p| p.probability).collect::<Vec<_>>(),
            vec![
                Probability {
                    numerator: 2,
                    denominator: 5
                },
                Probability {
                    numerator: 3,
                    denominator: 10
                },
                Probability {
                    numerator: 3,
                    denominator: 10
                }
            ]
        );
        assert_eq!(paths[0].actors[1].register_as.as_deref(), Some("Scout"));
        assert_eq!(paths[2].actors[1].register_as.as_deref(), Some("Witness"));
        assert!(paths
            .iter()
            .all(|p| p.actors[1].register_as.as_deref() != Some("Confessor")));
    }

    #[test]
    fn later_demon_registration_does_not_refresh_spy_cache() {
        let mut input = spy_context(
            vec![
                actor(1, DataRole::Spy { cache_key: 1 }),
                actor(2, DataRole::Lilis),
            ],
            vec![
                resume(1, 0, Some(0)),
                resume(2, 1, Some(1)),
                resume(1, 2, None),
            ],
        );
        input.pools.script.villagers = names(&["Scout"]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.pools.script.villagers, names(&["Scout", "Witness"]));
        assert_eq!(path.actors[0].register_as.as_deref(), Some("Scout"));
        assert_eq!(
            path.trace[2].spy_register_as.as_ref().unwrap().source,
            SpyRegisterSource::LiveCache
        );
        input.resumes = vec![
            resume(2, 0, Some(0)),
            resume(1, 1, Some(1)),
            resume(1, 2, None),
        ];
        let reordered = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(reordered.len(), 2);
        assert_eq!(
            reordered[1].actors[0].register_as.as_deref(),
            Some("Witness")
        );
    }

    #[test]
    fn spy_empty_or_unsupported_script_never_becomes_a_partial_distribution() {
        let mut body = actor(1, DataRole::Spy { cache_key: 0 });
        body.bluff = BluffReference::Live {
            role: BluffRole::Scout,
        };
        let mut input = spy_context(vec![body], vec![resume(1, 0, None)]);
        input.pools.script.villagers.clear();
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::EmptySupport)
        );
        input.pools.script.villagers = names(&["Scout", "Bard"]);
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn spy_requires_exact_cache_provenance_and_v2_marker() {
        let base = spy_context(
            vec![actor(1, DataRole::Spy { cache_key: 1 })],
            vec![resume(1, 0, Some(0))],
        );
        let mut input = base.clone();
        input.rule_version = REVEAL_CALLBACKS_NATIVE_V1.into();
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base.clone();
        input.spy_caches.clear();
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base;
        input.spy_caches.insert(2, BluffReference::Null);
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn spy_schema_preserves_v1_output_shape_and_v2_cache_keys() {
        let old = context(vec![actor(1, DataRole::Lilis)], vec![resume(1, 0, Some(0))]);
        let old_json = serde_json::to_value(&old).unwrap();
        assert!(old_json.get("spy_caches").is_none());
        let old_paths = serde_json::to_value(replay_reveal_callbacks(&old).unwrap()).unwrap();
        assert!(old_paths[0].get("spy_caches").is_none());
        assert!(old_paths[0]["trace"][0].get("spy_register_as").is_none());
        assert!(old_paths[0]["trace"][0].get("spy_acquisition").is_none());
        let current = spy_context(
            vec![actor(1, DataRole::Spy { cache_key: 42 })],
            vec![resume(1, 0, Some(0))],
        );
        let json = serde_json::to_value(&current).unwrap();
        assert_eq!(
            serde_json::from_value::<RevealContext>(json).unwrap(),
            current
        );
        let output = serde_json::to_value(replay_reveal_callbacks(&current).unwrap()).unwrap();
        assert_eq!(output[0]["spy_caches"]["42"]["kind"], "live");
    }

    #[test]
    fn repeated_twin_resume_acquires_once_but_repeats_both_callback_passes() {
        let input = context(
            vec![actor(4, DataRole::TwinMinion)],
            vec![resume(4, 2, Some(0)), resume(4, 8, None)],
        );
        let paths = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(
            paths.iter().map(|p| p.probability).collect::<Vec<_>>(),
            vec![
                Probability {
                    numerator: 2,
                    denominator: 5
                },
                Probability {
                    numerator: 3,
                    denominator: 5
                }
            ]
        );
        for path in &paths {
            assert_eq!(path.actors[0].remaining_continuations, 0);
            assert_eq!(path.actors[0].register_as, None);
            assert_eq!(path.trace[0].previous_register_as.as_deref(), Some("Bard"));
            assert_eq!(path.trace[1].previous_register_as, None);
            assert!(path.trace[1].acquisition.is_none());
            assert_eq!(path.trace[0].callbacks, path.trace[1].callbacks);
            assert_eq!(
                path.trace[0]
                    .callbacks
                    .iter()
                    .map(|c| (c.trigger, c.slot))
                    .collect::<Vec<_>>(),
                vec![
                    (Trigger::Init, RoleSlot::Real),
                    (Trigger::Init, RoleSlot::Bluff),
                    (Trigger::AfterRoundStart, RoleSlot::Real),
                    (Trigger::AfterRoundStart, RoleSlot::Bluff)
                ]
            );
            assert!(path.trace[0]
                .callbacks
                .iter()
                .all(|c| c.dispatch == Dispatch::BluffAct));
        }
        assert_eq!(input.actors[0].remaining_continuations, 2);
        assert_eq!(input.actors[0].bluff, BluffReference::Null);
    }

    #[test]
    fn mixed_acquisitions_compose_pool_mass_and_ordinal_provenance() {
        let mut input = context(
            vec![actor(1, DataRole::TwinMinion), actor(2, DataRole::Lilis)],
            vec![
                resume(1, 0, Some(0)),
                resume(2, 7, Some(3)),
                resume(1, 10, None),
            ],
        );
        input.pools.must_include = names(&["Scout", "Confessor"]);
        let paths = replay_reveal_callbacks(&input).unwrap();
        assert_eq!(paths.len(), 4);
        assert_eq!(
            paths.iter().map(|p| p.probability).collect::<Vec<_>>(),
            vec![
                Probability {
                    numerator: 1,
                    denominator: 5
                },
                Probability {
                    numerator: 1,
                    denominator: 5
                },
                Probability {
                    numerator: 3,
                    denominator: 10
                },
                Probability {
                    numerator: 3,
                    denominator: 10
                }
            ]
        );
        for path in paths {
            assert_eq!(
                path.trace[1]
                    .acquisition
                    .as_ref()
                    .unwrap()
                    .event
                    .acquisition_ordinal,
                3
            );
            assert_eq!(path.trace[1].event.resume_ordinal, 7);
            assert_eq!(path.trace[2].acquisition, None);
        }
    }

    #[test]
    fn drunk_then_confessor_clears_shared_target_without_changing_actual_truth() {
        let mut input = context(
            vec![actor(1, DataRole::Drunk)],
            vec![resume(1, 0, Some(0)), resume(1, 1, None)],
        );
        input.pools.unique = names(&["Confessor"]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(
            path.actors[0].statuses.values,
            vec![CORRUPTED, APPEAR_TRUTHFUL]
        );
        assert_eq!(
            path.trace[0].selector_status,
            Some(StatusApplication {
                status: CORRUPTED,
                accepted: true,
                inserted: true,
                target_after: Some(1),
            })
        );
        assert_eq!(
            path.trace[0].callbacks[1].status_application,
            Some(StatusApplication {
                status: APPEAR_TRUTHFUL,
                accepted: true,
                inserted: true,
                target_after: None,
            })
        );
        assert_eq!(path.trace[1].selector_status, None);
        assert_eq!(
            path.trace[1].callbacks[1]
                .status_application
                .as_ref()
                .unwrap()
                .inserted,
            false
        );
        assert_eq!(path.actors[0].statuses.target_position, None);
        assert!(path.actors[0].is_lying());
        assert!(!path.actors[0].appears_lying());
        assert!(path
            .trace
            .iter()
            .flat_map(|t| &t.callbacks)
            .all(|c| c.dispatch == Dispatch::BluffAct));
    }

    #[test]
    fn resistance_preserves_existing_status_membership_and_target() {
        let mut body = actor(1, DataRole::Drunk);
        body.statuses = StatusState {
            values: vec![CORRUPTED, APPEAR_TRUTHFUL, APPEAR_LYING],
            resistance: vec![CORRUPTED, APPEAR_TRUTHFUL],
            target_position: Some(7),
        };
        let mut input = context(vec![body.clone()], vec![resume(1, 1, Some(4))]);
        input.pools.unique = names(&["Confessor"]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.actors[0].statuses, body.statuses);
        assert!(!path.trace[0].selector_status.as_ref().unwrap().accepted);
        assert!(
            !path.trace[0].callbacks[1]
                .status_application
                .as_ref()
                .unwrap()
                .accepted
        );
        assert!(path.actors[0].appears_lying()); // AppearLying beats AppearTruthful
    }

    #[test]
    fn already_corrupted_drunk_retargets_self_before_resisted_confessor() {
        let mut body = actor(2, DataRole::Drunk);
        body.statuses = StatusState {
            values: vec![CORRUPTED],
            resistance: vec![APPEAR_TRUTHFUL],
            target_position: Some(7),
        };
        let mut input = context(vec![body], vec![resume(2, 0, Some(0))]);
        input.pools.unique = names(&["Confessor"]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.actors[0].statuses.values, vec![CORRUPTED]);
        assert_eq!(path.actors[0].statuses.target_position, Some(2));
        assert!(!path.trace[0].selector_status.as_ref().unwrap().inserted);
        assert!(path.trace[0].selector_status.as_ref().unwrap().accepted);
    }

    #[test]
    fn resisted_clean_drunk_still_lies_from_live_bluff() {
        let mut body = actor(1, DataRole::Drunk);
        body.statuses.resistance = vec![CORRUPTED];
        let input = context(vec![body], vec![resume(1, 0, Some(0))]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert!(!path.actors[0].statuses.values.contains(&CORRUPTED));
        assert!(path.actors[0].is_lying());
        assert!(path.trace[0]
            .callbacks
            .iter()
            .all(|c| c.dispatch == Dispatch::BluffAct));
    }

    #[test]
    fn live_bluff_skips_selection_and_preserves_independent_copied_role() {
        let mut body = actor(1, DataRole::Drunk);
        body.bluff = BluffReference::Live {
            role: BluffRole::Scout,
        };
        body.bluff_role = Some(CallbackRole::Confessor);
        let mut input = context(vec![body], vec![resume(1, 0, None)]);
        input.pools.unique.clear(); // would fail if Drunk incorrectly acquired again
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(path.pools, input.pools);
        assert_eq!(path.actors[0].bluff, input.actors[0].bluff);
        assert_eq!(path.actors[0].bluff_role, Some(CallbackRole::Confessor));
        assert_eq!(path.actors[0].statuses.values, vec![APPEAR_TRUTHFUL]);
        assert_eq!(path.trace[0].acquisition, None);
        assert_eq!(path.trace[0].selector_status, None);
    }

    #[test]
    fn destroyed_bluff_reacquires_and_overwrites_stale_copied_role() {
        let mut body = actor(1, DataRole::Lilis);
        body.bluff = BluffReference::Destroyed {
            role: BluffRole::Scout,
        };
        body.bluff_role = Some(CallbackRole::Confessor);
        let input = context(vec![body], vec![resume(1, 0, Some(0))]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(
            path.actors[0].bluff,
            BluffReference::Live {
                role: BluffRole::Witness
            }
        );
        assert_eq!(path.actors[0].bluff_role, Some(CallbackRole::Witness));
        assert!(path.actors[0].statuses.values.is_empty());
        assert_eq!(path.trace[0].callbacks[0].dispatch, Dispatch::Act);
        assert_eq!(path.trace[0].callbacks[1].dispatch, Dispatch::BluffAct);
    }

    #[test]
    fn evil_real_dispatch_depends_on_copied_pointer_not_appearance() {
        let mut body = actor(1, DataRole::Lilis);
        body.bluff = BluffReference::Live {
            role: BluffRole::Scout,
        };
        body.bluff_role = None;
        body.statuses.values = vec![CORRUPTED, APPEAR_TRUTHFUL];
        let mut input = context(vec![body], vec![resume(1, 0, None)]);
        let without = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(without.trace[0].callbacks.len(), 2);
        assert!(without.trace[0]
            .callbacks
            .iter()
            .all(|c| c.dispatch == Dispatch::BluffAct));
        input.actors[0].bluff_role = Some(CallbackRole::Scout);
        let with = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(with.trace[0].callbacks.len(), 4);
        assert_eq!(with.trace[0].callbacks[0].dispatch, Dispatch::Act);
        assert_eq!(with.trace[0].callbacks[1].dispatch, Dispatch::BluffAct);
        assert!(!with.actors[0].appears_lying());
    }

    #[test]
    fn data_selector_and_shared_action_role_are_not_conflated() {
        let mut body = actor(1, DataRole::Lilis);
        body.action_role = CallbackRole::Confessor;
        let input = context(vec![body], vec![resume(1, 0, Some(0))]);
        let path = replay_reveal_callbacks(&input).unwrap().remove(0);
        assert_eq!(
            path.trace[0].acquisition.as_ref().unwrap().event.selector,
            Selector::Demon
        );
        assert_eq!(path.trace[0].callbacks[0].role, CallbackRole::Confessor);
        assert_eq!(path.actors[0].statuses.values, vec![APPEAR_TRUTHFUL]);
    }

    #[test]
    fn order_and_continuation_provenance_are_required() {
        let base = context(vec![actor(1, DataRole::Lilis)], vec![resume(1, 0, Some(0))]);
        for resumes in [
            vec![resume(1, 0, None)],
            vec![resume(2, 0, Some(0))],
            vec![resume(1, 0, Some(0)), resume(1, 0, None)],
            vec![resume(1, 0, Some(0)), resume(1, 1, Some(1))],
            vec![
                resume(1, 0, Some(0)),
                resume(1, 1, None),
                resume(1, 2, None),
            ],
        ] {
            assert_eq!(
                replay_reveal_callbacks(&RevealContext {
                    resumes,
                    ..base.clone()
                }),
                Err(LedgerError::InvalidContext)
            );
        }
        let mut input = base.clone();
        input.actors[0].remaining_continuations = 0;
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base;
        input.actors.push(actor(2, DataRole::Drunk));
        input.resumes.push(resume(2, 1, Some(0)));
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn unsupported_hooks_and_assets_reject_wholesale() {
        let base = context(vec![actor(1, DataRole::Lilis)], vec![resume(1, 0, Some(0))]);
        let mut input = base.clone();
        input.actors[0].on_trigger_subscribed = true;
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base.clone();
        input.actors[0].statuses.values.push(HEALTHY_BLUFF);
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base.clone();
        input.pools.unique.push("Rambler".into());
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base;
        input.trailer_mode = true;
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn failing_branch_never_returns_successful_siblings() {
        let mut input = context(
            vec![actor(1, DataRole::TwinMinion), actor(2, DataRole::Lilis)],
            vec![resume(1, 0, Some(0)), resume(2, 1, Some(1))],
        );
        input.pools.unique.clear();
        input.pools.must_include = names(&["Scout"]);
        // Twin duplicate preserves Scout for Lilis, but unique consumes it.
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::EmptySupport)
        );
        assert_eq!(input.actors[0].bluff, BluffReference::Null);
    }

    #[test]
    fn bounded_paths_and_schema_fail_closed() {
        let base = context(vec![actor(1, DataRole::Lilis)], vec![resume(1, 0, Some(0))]);
        let serialized = serde_json::to_string(&base).unwrap();
        assert_eq!(
            serde_json::from_str::<RevealContext>(&serialized).unwrap(),
            base
        );
        let mut value = serde_json::to_value(&base).unwrap();
        value["actors"][0]["data_role"] = "spy".into();
        assert!(serde_json::from_value::<RevealContext>(value).is_err());
        let mut value = serde_json::to_value(&base).unwrap();
        value["live_solver_input"] = true.into();
        assert!(serde_json::from_value::<RevealContext>(value).is_err());
        let mut input = base.clone();
        input.resumes = vec![resume(1, 0, None); 17];
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
        input = base;
        input.pools.unique = vec!["Scout".into(); 2000];
        assert_eq!(replay_reveal_callbacks(&input), Err(LedgerError::Capacity));
    }
}
