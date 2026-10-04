//! Supported setup action passes after exact physical initialization.
//!
//! Init and ordered Start remain distinct. Start scans current data identities
//! after each writer rather than replaying a precomputed list of calls. This
//! bridge projects supported role classes only; modeled role callbacks/UI are
//! inert, no continuation is resumed and generated continuation IDs are logical
//! labels. A registered generic-caller callback records a gateway request; this
//! bridge does not execute that subscriber or project its additional queues.
use super::{
    character_start::{self, CharacterStartContext, StartCallTrace, CHARACTER_START_NATIVE_V1},
    continuation_registry::{ContinuationState, CONTINUATION_REGISTRY_NATIVE_V1},
    ledger::{LedgerError, Probability, SelectorPools},
    reveal::{
        BluffReference, CallbackRole, CallbackTrace, DataRole, Dispatch, RevealActor,
        RevealContext, RoleSlot, StatusState, Trigger, REVEAL_CALLBACKS_START_NATIVE_V3,
        SETUP_CALLBACKS_NATIVE_V4, SETUP_REVEAL_CALLBACKS_NATIVE_V6,
    },
    reveal_writer::{RevealWriterContext, ViewUiState, REVEAL_WRITER_VIEW_NATIVE_V2},
    setup_initialization_batch::{self as initialization, Context as InitializationContext},
    twin_writer::{self, BodyState, TwinWriterContext, TWIN_WRITER_NATIVE_V1},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
pub const SETUP_ACTION_BRIDGE_NATIVE_V1: &str = "setup_action_bridge_native_v1";
/// Includes original N5 data/classes but produces a setup-only registry.
pub const SETUP_ACTION_BRIDGE_NATIVE_V2: &str = "setup_action_bridge_native_v2";
/// Adds source/real Archivist under V6's guarded no-Start acquisition domain.
pub const SETUP_ACTION_BRIDGE_NATIVE_V3: &str = "setup_action_bridge_native_v3";
const MAX_PATHS: usize = 256;
const MAX_RETAINED: usize = 1_048_576;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub initialization: InitializationContext,
    /// Actual dataRef identity -> supported semantic class (Spy retains source cache identity).
    pub data_roles: BTreeMap<u64, DataRole>,
    /// Exact class of every reachable final real/copied action object. Published
    /// clones are additionally checked against the producer's native source class.
    pub action_classes: BTreeMap<u64, String>,
    pub action_classes_and_caches_verified: bool,
    pub on_trigger_absent: bool,
    pub final_services_inert: bool,
    pub post_initialization_ui_verified: bool,
    pub pools: SelectorPools,
    pub spy_caches: BTreeMap<u16, BluffReference>,
    pub ui: BTreeMap<u8, ViewUiState>,
    pub next_logical_id: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ActionCall {
    pub position: u8,
    pub trigger: Trigger,
    pub init_callbacks: Vec<CallbackTrace>,
    pub start_callbacks: Vec<StartCallTrace>,
    pub initial_lying: Option<bool>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Created {
    pub logical_id: u64,
    pub position: u8,
    pub call_index: usize,
    pub callback_index: usize,
    pub replacement_index: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Path {
    pub probability: Probability,
    pub state: ContinuationState,
    pub current_data: BTreeMap<u8, u64>,
    pub calls: Vec<ActionCall>,
    pub created: Vec<Created>,
}
fn class(role: DataRole) -> &'static str {
    match role {
        DataRole::Lilis => "Striga",
        DataRole::TwinMinion => "Marionette",
        DataRole::Drunk => "Drunk",
        DataRole::Spy { .. } => "Spy",
        DataRole::Minion => "Minion",
        DataRole::Confessor => "Confessor",
        DataRole::Lover => "Empath",
        DataRole::Hunter => "Tracker",
        DataRole::Enlightened => "Shugenja",
        DataRole::Gemcrafter => "Archivist",
    }
}
fn callback(class: &str) -> Result<CallbackRole, LedgerError> {
    Ok(match class {
        "Striga" => CallbackRole::Lilis,
        "Marionette" => CallbackRole::TwinMinion,
        "Drunk" => CallbackRole::Drunk,
        "Spy" => CallbackRole::Spy,
        "Scout" => CallbackRole::Scout,
        "Witness" => CallbackRole::Witness,
        "Confessor" => CallbackRole::Confessor,
        "Minion" => CallbackRole::Minion,
        "Empath" => CallbackRole::Lover,
        "Tracker" => CallbackRole::Hunter,
        "Shugenja" => CallbackRole::Enlightened,
        "Archivist" => CallbackRole::Gemcrafter,
        _ => return Err(LedgerError::InvalidContext),
    })
}
fn physical_positions(c: &initialization::Replay) -> Result<Vec<u8>, LedgerError> {
    c.current_order
        .iter()
        .map(|id| {
            c.positions
                .get(id)
                .copied()
                .ok_or(LedgerError::InvalidContext)
        })
        .collect()
}
fn project(c: &Context) -> Result<Path, LedgerError> {
    if ![
        SETUP_ACTION_BRIDGE_NATIVE_V1,
        SETUP_ACTION_BRIDGE_NATIVE_V2,
        SETUP_ACTION_BRIDGE_NATIVE_V3,
    ]
    .contains(&c.version.as_str())
        || !c.action_classes_and_caches_verified
        || !c.on_trigger_absent
        || !c.final_services_inert
        || !c.post_initialization_ui_verified
        || c.initialization.caller.failure.is_some()
        || c.next_logical_id == 0
    {
        return Err(LedgerError::InvalidContext);
    }
    let produced = initialization::replay(&c.initialization)?;
    if !produced.initialization_complete || produced.prefix_error.is_some() {
        return Err(LedgerError::InvalidContext);
    }
    let current_order = physical_positions(&produced)?;
    let physical = produced
        .current_order
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if physical.is_empty()
        || physical.len() > 32
        || c.data_roles.len() > 32
        || c.action_classes.len() > 64
    {
        return Err(LedgerError::Capacity);
    }
    let positions = physical
        .iter()
        .map(|id| produced.positions[id])
        .collect::<BTreeSet<_>>();
    if positions != c.ui.keys().copied().collect() {
        return Err(LedgerError::InvalidContext);
    }
    let board_size = *positions.last().unwrap();
    if positions.iter().copied().collect::<Vec<_>>() != (1..=board_size).collect::<Vec<_>>() {
        return Err(LedgerError::InvalidContext);
    }
    let mut pending = BTreeMap::new();
    let mut counts = BTreeMap::<u64, u16>::new();
    for item in &produced.continuations {
        if !physical.contains(&item.actor)
            || item.state != 1
            || item.current.is_none()
            || item.identity >= c.next_logical_id
            || pending
                .insert(item.identity, produced.positions[&item.actor])
                .is_some()
        {
            return Err(LedgerError::InvalidContext);
        }
        let count = counts.entry(item.actor).or_default();
        *count = count.checked_add(1).ok_or(LedgerError::Capacity)?;
    }
    let actual_data = physical
        .iter()
        .map(|id| produced.actors[id].data.ok_or(LedgerError::InvalidContext))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if actual_data != c.data_roles.keys().copied().collect() {
        return Err(LedgerError::InvalidContext);
    }
    let mut roles_seen = Vec::new();
    let mut cache_sources = BTreeMap::new();
    let mut source_keys = BTreeMap::new();
    for (identity, role) in &c.data_roles {
        if *role == DataRole::Gemcrafter && c.version != SETUP_ACTION_BRIDGE_NATIVE_V3 {
            return Err(LedgerError::InvalidContext);
        }
        if roles_seen.contains(role) {
            return Err(LedgerError::InvalidContext);
        }
        roles_seen.push(*role);
        let source = c
            .initialization
            .data
            .get(identity)
            .and_then(|d| d.source_role)
            .ok_or(LedgerError::InvalidContext)?;
        let alias = c
            .initialization
            .object_identities
            .iter()
            .find(|(_, id)| **id == source)
            .map(|(a, _)| a)
            .ok_or(LedgerError::InvalidContext)?;
        if c.initialization.caller.roles.get(alias).map(String::as_str) != Some(class(*role)) {
            return Err(LedgerError::InvalidContext);
        }
        if let DataRole::Spy { cache_key } = role {
            if cache_sources
                .insert(*cache_key, source)
                .is_some_and(|old| old != source)
                || source_keys
                    .insert(source, *cache_key)
                    .is_some_and(|old| old != *cache_key)
            {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    let action_ids = physical
        .iter()
        .flat_map(|id| [produced.actors[id].role, produced.actors[id].bluff_role])
        .flatten()
        .collect::<BTreeSet<_>>();
    if action_ids != c.action_classes.keys().copied().collect() {
        return Err(LedgerError::InvalidContext);
    }
    for publication in &produced.publications {
        if let Some(clone) = &publication.clone {
            if c.action_classes
                .get(&clone.identity)
                .is_some_and(|name| *name != clone.managed_class)
            {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    let mut actors = vec![];
    let mut bodies = BTreeMap::new();
    let mut current_data = BTreeMap::new();
    for id in physical {
        let a = &produced.actors[&id];
        let position = produced.positions[&id];
        let data = a.data.ok_or(LedgerError::InvalidContext)?;
        // Every current occurrence completed ordinary Init. These reset fields
        // must agree before mapping to a writer kernel with no subscriber model.
        if a.bluff.is_some()
            || a.register_as.is_some()
            || a.state_callback.is_some()
            || a.runtime.is_some()
        {
            return Err(LedgerError::InvalidContext);
        }
        let role = a.role.ok_or(LedgerError::InvalidContext)?;
        let target = a
            .statuses
            .target
            .map(|t| {
                produced
                    .positions
                    .get(&t)
                    .copied()
                    .filter(|p| positions.contains(p))
                    .ok_or(LedgerError::InvalidContext)
            })
            .transpose()?;
        actors.push(RevealActor {
            position,
            data_role: c.data_roles[&data],
            action_role: callback(&c.action_classes[&role])?,
            runtime_evil: a.alignment == 20,
            bluff: BluffReference::Null,
            bluff_role: a
                .bluff_role
                .map(|r| callback(&c.action_classes[&r]))
                .transpose()?,
            register_as: None,
            statuses: StatusState {
                values: a.statuses.active.clone(),
                resistance: a.statuses.resistances.clone(),
                target_position: target,
            },
            remaining_continuations: counts.get(&id).copied().unwrap_or(0),
            on_trigger_subscribed: false,
            character_start_acted: Some(a.started),
        });
        bodies.insert(
            position,
            BodyState {
                state: a.state,
                previous_state: a.previous,
                revealed: a.revealed,
                killed_by_demon: a.killed_demon,
                pickable_uses: a.uses,
                acted_info_count: a.infos.len() as u32,
                created_dead_presentation: a.dead_prefab.as_ref().is_some_and(|p| p.live),
                on_state_change_subscribed: false,
            },
        );
        current_data.insert(position, data);
    }
    let board = TwinWriterContext {
        rule_version: TWIN_WRITER_NATIVE_V1.into(),
        reveal: RevealContext {
            rule_version: if c.version == SETUP_ACTION_BRIDGE_NATIVE_V3 {
                SETUP_REVEAL_CALLBACKS_NATIVE_V6.into()
            } else if c.version == SETUP_ACTION_BRIDGE_NATIVE_V2 {
                SETUP_CALLBACKS_NATIVE_V4.into()
            } else {
                REVEAL_CALLBACKS_START_NATIVE_V3.into()
            },
            board_size,
            trailer_mode: false,
            pools: c.pools.clone(),
            actors,
            resumes: vec![],
            spy_caches: c.spy_caches.clone(),
        },
        current_order,
        position: *positions.first().unwrap(),
        copied_slot: false,
        bodies,
    };
    twin_writer::validate_board(&board)?;
    Ok(Path {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        state: ContinuationState {
            rule_version: CONTINUATION_REGISTRY_NATIVE_V1.into(),
            initial: RevealWriterContext {
                rule_version: REVEAL_WRITER_VIEW_NATIVE_V2.into(),
                board,
                resumes: vec![],
                ui: c.ui.clone(),
            },
            pending,
            next_id: c.next_logical_id,
            batch_ordinal: 0,
        },
        current_data,
        calls: vec![],
        created: vec![],
    })
}
fn init_pass(path: &mut Path) {
    for position in path.state.initial.board.current_order.clone() {
        let actor = path
            .state
            .initial
            .board
            .reveal
            .actors
            .iter_mut()
            .find(|a| a.position == position)
            .unwrap();
        let lying = actor.is_lying();
        let real = if !lying || actor.runtime_evil && actor.bluff_role.is_some() {
            Dispatch::Act
        } else {
            Dispatch::BluffAct
        };
        let mut callbacks = vec![];
        for (slot, role, dispatch) in [
            (RoleSlot::Real, Some(actor.action_role), real),
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
                let status_application = if role == CallbackRole::Confessor {
                    Some(actor.statuses.apply(25, None))
                } else {
                    None
                };
                callbacks.push(CallbackTrace {
                    trigger: Trigger::Init,
                    slot,
                    role,
                    dispatch,
                    status_application,
                });
            }
        }
        path.calls.push(ActionCall {
            position,
            trigger: Trigger::Init,
            init_callbacks: callbacks,
            start_callbacks: vec![],
            initial_lying: Some(lying),
        });
    }
}
fn units(p: &Path) -> usize {
    twin_writer::retained_entries(&p.state.initial.board)
        + p.state.pending.len() * 2
        + p.current_data.len() * 2
        + p.state.initial.ui.len() * 4
        + p.created.len() * 5
        + p.calls
            .iter()
            .map(|c| {
                4 + c.init_callbacks.len() * 3
                    + c.start_callbacks
                        .iter()
                        .map(|s| 3 + s.twin.as_ref().map_or(0, |t| t.replacements.len() * 4))
                        .sum::<usize>()
            })
            .sum::<usize>()
}
fn push(paths: &mut Vec<Path>, p: Path, retained: &mut usize) -> Result<(), LedgerError> {
    *retained = retained
        .checked_add(units(&p))
        .ok_or(LedgerError::Capacity)?;
    if paths.len() >= MAX_PATHS || *retained > MAX_RETAINED || p.state.pending.len() > 4096 {
        return Err(LedgerError::Capacity);
    }
    paths.push(p);
    Ok(())
}
pub fn replay(c: &Context) -> Result<Vec<Path>, LedgerError> {
    let initial = replay_init_prefix(c)?;
    let order = c
        .initialization
        .caller
        .state
        .order
        .as_ref()
        .ok_or(LedgerError::InvalidContext)?;
    let order = &c.initialization.caller.state.arrays[order];
    let mut paths = vec![initial];
    for alias in order.items.iter().take(order.length.max(0) as usize) {
        let Some(alias) = alias else { continue };
        let data = c.initialization.object_identities[alias];
        let mut next = vec![];
        let mut retained = 0;
        for path in paths {
            // First matching physical occurrence wins even when its Start latch
            // suppresses dispatch. All-match role classes are unsupported here.
            let position = path
                .state
                .initial
                .board
                .current_order
                .iter()
                .find(|p| path.current_data[p] == data)
                .copied();
            let Some(position) = position else {
                push(&mut next, path, &mut retained)?;
                continue;
            };
            let mut board = path.state.initial.board.clone();
            board.position = position;
            for started in character_start::replay_character_start(&CharacterStartContext {
                rule_version: CHARACTER_START_NATIVE_V1.into(),
                board,
            })? {
                let mut branch = path.clone();
                branch.probability = branch.probability.multiply(
                    started.probability.numerator,
                    started.probability.denominator,
                )?;
                let mut replaced = BTreeSet::new();
                for (callback_index, call) in started.callbacks.iter().enumerate() {
                    if let Some(twin) = &call.twin {
                        for (replacement_index, replacement) in twin.replacements.iter().enumerate()
                        {
                            let data = c
                                .data_roles
                                .iter()
                                .find(|(_, r)| **r == replacement.new_data)
                                .map(|(id, _)| *id)
                                .ok_or(LedgerError::InvalidContext)?;
                            branch.current_data.insert(replacement.position, data);
                            let logical_id = branch.state.next_id;
                            branch.state.next_id =
                                logical_id.checked_add(1).ok_or(LedgerError::Capacity)?;
                            branch
                                .state
                                .pending
                                .insert(logical_id, replacement.position);
                            branch.created.push(Created {
                                logical_id,
                                position: replacement.position,
                                call_index: branch.calls.len(),
                                callback_index,
                                replacement_index,
                            });
                            let ui = branch
                                .state
                                .initial
                                .ui
                                .get_mut(&replacement.position)
                                .unwrap();
                            if replaced.insert(replacement.position)
                                && path.state.initial.board.bodies[&replacement.position]
                                    .created_dead_presentation
                            {
                                ui.rip_active = false;
                            }
                            if ui.disguise_icon_active.is_some() {
                                ui.disguise_icon_active = Some(false);
                            }
                        }
                    }
                }
                branch.state.initial.board = started.board;
                branch.calls.push(ActionCall {
                    position,
                    trigger: Trigger::Start,
                    init_callbacks: vec![],
                    start_callbacks: started.callbacks,
                    initial_lying: started.initial_lying,
                });
                push(&mut next, branch, &mut retained)?;
            }
        }
        paths = next;
    }
    // Registry validation performs no resumes when the batch is empty.
    for path in &paths {
        super::continuation_registry::validate_registry(&path.state)?;
    }
    Ok(paths)
}

/// Complete only the Init action pass. Ordered Start, onSetup, shuffle,
/// acquisition and public observation admission are outside this boundary.
pub fn replay_init_prefix(c: &Context) -> Result<Path, LedgerError> {
    let mut path = project(c)?;
    init_pass(&mut path);
    super::continuation_registry::validate_registry(&path.state)?;
    Ok(path)
}
#[cfg(test)]
#[path = "setup_action_bridge_tests.rs"]
mod tests;
