//! Join the successful ManageCharacters Init prefix to explicit actor/clone state.
//!
//! Object aliases and physical positions are supplied, never inferred from the
//! displayed ID. Allocation bindings name actual modeled objects; their sequence
//! is registration order, not readiness or coroutine resume order. No Act pass is
//! run here, and unsupported/null initializer dependencies reject the join.

use super::character_initialization::{self as init, Actor, Continuation, Identity};
use super::ledger::LedgerError;
use super::manage_setup_caller::{self as caller, Gateway};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SETUP_INITIALIZATION_BATCH_NATIVE_V1: &str = "setup_initialization_batch_native_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataBinding {
    pub starting_alignment: i32,
    pub source_role: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloneBinding {
    pub identity: Identity,
    pub source_role: Identity,
    pub managed_class: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Allocation {
    /// Zero-based occurrence in the successful native Init prefix.
    pub init_index: usize,
    pub clone: Option<CloneBinding>,
    pub continuation_identity: Identity,
    pub wait_identity: Identity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub caller: caller::Context,
    /// Exact union of caller Character, CharacterData and source Role aliases.
    /// Distinct aliases must have distinct nonzero object identities.
    pub object_identities: BTreeMap<String, Identity>,
    pub actors: BTreeMap<Identity, Actor>,
    /// Explicit Unity liveness for every non-null initial raw bluff reference.
    /// Null is represented by Actor.bluff=None, not by a false liveness value.
    pub raw_bluff_liveness: BTreeMap<Identity, bool>,
    pub positions: BTreeMap<Identity, u8>,
    pub data: BTreeMap<Identity, DataBinding>,
    pub continuations: Vec<Continuation>,
    /// One binding per successfully completed Init, not per attempted call.
    pub allocations: Vec<Allocation>,
    pub required_objects_and_lists_valid: bool,
    pub callbacks_and_ui_inert: bool,
    pub clone_results_verified: bool,
    pub synchronous_first_yield_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Publication {
    pub init_index: usize,
    pub physical_actor: Identity,
    pub position: u8,
    pub data: Identity,
    pub source_role: Identity,
    pub clone: Option<CloneBinding>,
    pub display_id: i32,
    pub continuation_identity: Identity,
    pub wait_identity: Identity,
    pub events: Vec<init::Event>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub actors: BTreeMap<Identity, Actor>,
    pub raw_bluff_liveness: BTreeMap<Identity, bool>,
    pub positions: BTreeMap<Identity, u8>,
    /// Occurrence order, including repeated references to a physical actor.
    pub current_order: Vec<Identity>,
    pub continuations: Vec<Continuation>,
    pub publications: Vec<Publication>,
    /// All initializers completed and the caller reached its Publish gateway.
    /// This does not claim publication itself or the later Act passes succeeded.
    pub initialization_complete: bool,
    pub prefix_error: Option<String>,
}

fn actor_references(actor: &Actor, retained: &mut BTreeSet<Identity>) {
    retained.insert(actor.identity);
    retained.extend(
        [
            actor.data,
            actor.bluff,
            actor.register_as,
            actor.trailer,
            actor.runtime,
            actor.role,
            actor.bluff_role,
            actor.saved_act,
            actor.state_callback,
            actor.statuses.target,
        ]
        .into_iter()
        .flatten(),
    );
    retained.extend(actor.infos.iter().flatten().copied());
    retained.extend(actor.dead_prefab.as_ref().map(|r| r.identity));
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    if c.version != SETUP_INITIALIZATION_BATCH_NATIVE_V1
        || !c.required_objects_and_lists_valid
        || !c.callbacks_and_ui_inert
        || !c.clone_results_verified
        || !c.synchronous_first_yield_verified
        || !c.caller.after.is_empty()
    {
        return Err(LedgerError::InvalidContext);
    }
    if c.actors.len() > 32
        || c.data.len() > 32
        || c.allocations.len() > 32
        || c.continuations.len() > 4096
    {
        return Err(LedgerError::Capacity);
    }
    let aliases: BTreeSet<_> = c
        .caller
        .state
        .identities
        .keys()
        .chain(c.caller.state.data_roles.keys())
        .chain(c.caller.roles.keys())
        .cloned()
        .collect();
    if aliases != c.object_identities.keys().cloned().collect()
        || c.object_identities.values().any(|id| *id == 0)
        || c.object_identities.values().collect::<BTreeSet<_>>().len() != c.object_identities.len()
    {
        return Err(LedgerError::InvalidContext);
    }
    let actor_ids: BTreeSet<_> = c
        .caller
        .state
        .identities
        .keys()
        .map(|key| c.object_identities[key])
        .collect();
    let data_ids: BTreeSet<_> = c
        .caller
        .state
        .data_roles
        .keys()
        .map(|key| c.object_identities[key])
        .collect();
    if actor_ids != c.actors.keys().copied().collect()
        || actor_ids != c.positions.keys().copied().collect()
        || data_ids != c.data.keys().copied().collect()
        || c.actors.iter().any(|(id, actor)| *id != actor.identity)
        || c.positions.values().any(|p| *p == 0)
        || c.positions.values().collect::<BTreeSet<_>>().len() != c.positions.len()
    {
        return Err(LedgerError::InvalidContext);
    }
    if c.actors
        .values()
        .filter_map(|a| a.bluff)
        .collect::<BTreeSet<_>>()
        != c.raw_bluff_liveness.keys().copied().collect()
    {
        return Err(LedgerError::InvalidContext);
    }
    for (alias, data_alias) in &c.caller.state.identities {
        if c.actors[&c.object_identities[alias]].data
            != data_alias.as_ref().map(|key| c.object_identities[key])
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    for (alias, role_alias) in &c.caller.state.data_roles {
        if c.data[&c.object_identities[alias]].source_role
            != role_alias.as_ref().map(|key| c.object_identities[key])
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let mut pending = BTreeSet::new();
    let mut non_iterators: BTreeSet<_> = c.object_identities.values().copied().collect();
    for actor in c.actors.values() {
        actor_references(actor, &mut non_iterators);
    }
    let currents: BTreeSet<_> = c
        .continuations
        .iter()
        .filter_map(|item| item.current)
        .collect();
    for item in &c.continuations {
        if item.identity == 0
            || !pending.insert(item.identity)
            || !c.actors.contains_key(&item.actor)
            || item.current == Some(0)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if pending
        .iter()
        .any(|id| non_iterators.contains(id) || currents.contains(id))
        || currents.iter().any(|id| non_iterators.contains(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    // Validate the caller's referential model before indexing its identity maps.
    let caller_result = caller::replay(&c.caller)?;
    validate(c)?;
    let board = c
        .caller
        .state
        .board
        .as_ref()
        .ok_or(LedgerError::InvalidContext)?;
    let current_order = c.caller.state.lists[board]
        .items
        .iter()
        .map(|alias| {
            alias
                .as_ref()
                .map(|key| c.object_identities[key])
                .ok_or(LedgerError::InvalidContext)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let successful = caller_result
        .final_state
        .effects
        .iter()
        .filter(|g| **g == Gateway::Init)
        .count();
    let prefix: Vec<_> = caller_result
        .events
        .iter()
        .take_while(|event| event.kind != Gateway::Publish)
        .filter(|event| event.kind == Gateway::Init)
        .take(successful)
        .collect();
    if prefix.len() != successful || c.allocations.len() != successful {
        return Err(LedgerError::InvalidContext);
    }
    let mut actors = c.actors.clone();
    let mut continuations = c.continuations.clone();
    let mut publications = Vec::new();
    let mut retained: BTreeSet<_> = c.object_identities.values().copied().collect();
    for actor in actors.values() {
        actor_references(actor, &mut retained);
    }
    for continuation in &continuations {
        retained.extend([continuation.identity, continuation.actor]);
        retained.extend(continuation.current);
    }
    for (index, (event, allocation)) in prefix.iter().zip(&c.allocations).enumerate() {
        let card_alias = event.arguments["card"]
            .as_str()
            .ok_or(LedgerError::InvalidContext)?;
        let data_alias = event.arguments["data"]
            .as_str()
            .ok_or(LedgerError::InvalidContext)?;
        let physical = c.object_identities[card_alias];
        let data = c.object_identities[data_alias];
        let source_role = c.data[&data]
            .source_role
            .ok_or(LedgerError::InvalidContext)?;
        let source_alias = c.caller.state.data_roles[data_alias]
            .as_ref()
            .ok_or(LedgerError::InvalidContext)?;
        let class = &c.caller.roles[source_alias];
        if allocation.init_index != index
            || allocation
                .clone
                .as_ref()
                .is_some_and(|role| role.source_role != source_role || role.managed_class != *class)
        {
            return Err(LedgerError::InvalidContext);
        }
        let new_ids = [
            Some(allocation.continuation_identity),
            Some(allocation.wait_identity),
            allocation.clone.as_ref().map(|r| r.identity),
        ];
        for identity in new_ids.into_iter().flatten() {
            if identity == 0 || !retained.insert(identity) {
                return Err(LedgerError::InvalidContext);
            }
        }
        let displayed = event.arguments["display_id"]
            .as_u64()
            .ok_or(LedgerError::InvalidContext)?;
        let display_id = u32::try_from(displayed).map_err(|_| LedgerError::InvalidContext)? as i32;
        let initialized = init::replay(&init::Context {
            version: init::CHARACTER_INITIALIZATION_NATIVE_V1.into(),
            method: init::Method::Init,
            actor: actors[&physical].clone(),
            data,
            starting_alignment: c.data[&data].starting_alignment,
            id: display_id,
            required_objects_and_lists_valid: c.required_objects_and_lists_valid,
            callbacks_and_ui_inert: c.callbacks_and_ui_inert,
            clone_result_verified: c.clone_results_verified,
            synchronous_first_yield_verified: c.synchronous_first_yield_verified,
            clone_result: allocation.clone.as_ref().map(|r| r.identity),
            continuation_identity: allocation.continuation_identity,
            wait_identity: allocation.wait_identity,
            continuations,
        })?;
        actors.insert(physical, initialized.actor);
        continuations = initialized.continuations;
        publications.push(Publication {
            init_index: index,
            physical_actor: physical,
            position: c.positions[&physical],
            data,
            source_role,
            clone: allocation.clone.clone(),
            display_id,
            continuation_identity: allocation.continuation_identity,
            wait_identity: allocation.wait_identity,
            events: initialized.events,
        });
    }
    let initialization_complete = caller_result
        .events
        .iter()
        .any(|e| e.kind == Gateway::Publish);
    Ok(Replay {
        actors,
        raw_bluff_liveness: c.raw_bluff_liveness.clone(),
        positions: c.positions.clone(),
        current_order,
        continuations,
        publications,
        initialization_complete,
        prefix_error: if initialization_complete {
            None
        } else {
            caller_result.error
        },
    })
}

#[cfg(test)]
#[path = "setup_initialization_batch_tests.rs"]
pub(crate) mod tests;
