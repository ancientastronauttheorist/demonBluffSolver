//! Constructor-produced physical lists through guarded Hidden initialization.
//!
//! The initializer and both refresh APIs expose separate semantic projections;
//! their events are not presented as one reconstructed raw service trace.
//! Serialized components are supplied before construction. Base/List/runtime,
//! callbacks and UI are inert, and synchronous first yield is explicit provenance.
//! Standalone View's valid transform/template bindings remain required on Hidden
//! paths. Retained backing arrays are unaliased; only the two new empty lists
//! share their supplied static array. A retained death object may not alias a
//! used typed component/control. No admission, later resume or failure is inferred.

use super::character_initialization::{
    self as init, Actor, Continuation, Identity, Method, ObjectReference,
};
use super::character_refresh::{self as refresh, AbilityData};
use super::character_refresh_view::{self as view, Transform, UiObject, VectorBits};
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CHARACTER_CONSTRUCTOR_INIT_NATIVE_V1: &str = "character_constructor_init_native_v1";
const MAX_OCCURRENCES: usize = 32;
const MAX_RETAINED: usize = 65_536;
const MAX_WORK: usize = 262_144;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceList {
    pub identity: Identity,
    pub backing_array: Identity,
    pub count: usize,
    pub version: u32,
    /// Complete supplied storage, including unused slots.
    pub slots: Vec<Option<Identity>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constructor {
    pub acted_list: Identity,
    pub hover_list: Identity,
    /// The two constructed empty lists share this explicit static array.
    pub empty_array: Identity,
    pub empty_string: Identity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataBinding {
    pub identity: Identity,
    pub starting_alignment: i32,
    pub ability_usage: i32,
    pub picking: bool,
    pub source_role: Identity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Occurrence {
    pub method: Method,
    pub data: Identity,
    pub id: i32,
    pub clone_source: Identity,
    pub clone_result: Option<Identity>,
    pub continuation: Identity,
    pub wait: Identity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Components {
    pub acted_component: Identity,
    pub acted_game_object: Identity,
    pub number_component: Identity,
    pub statuses_component: Identity,
    pub active_status_list: Identity,
    pub active_status_array: Identity,
    /// Clearing the logical active list retains these native backing values.
    pub active_status_slots: Vec<i32>,
    pub resistance_list: Identity,
    pub picked_array: Identity,
    pub picked_objects: Vec<Identity>,
    pub pickable: Identity,
    pub rip: Identity,
    pub disguise: Option<ObjectReference>,
    /// Standalone View requires these valid bindings even on its unread Hidden path.
    pub icon_transform: Identity,
    pub actor_transform: Identity,
    pub dead_template: Identity,
    pub ui_objects: Vec<UiObject>,
    pub transforms: Vec<Transform>,
    pub vector3_zero: VectorBits,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub native_runtime_and_bindings_verified: bool,
    pub constructor_allocations_and_empty_storage_verified: bool,
    pub empty_literal_verified: bool,
    pub base_and_all_callbacks_inert: bool,
    pub ui_and_other_services_inert: bool,
    pub unity_liveness_verified: bool,
    pub required_initializer_objects_valid: bool,
    pub clone_results_verified: bool,
    pub synchronous_first_yield_verified: bool,
    pub normal_completion_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub actor: Actor,
    pub act: bool,
    pub initial_acted_list: Option<Identity>,
    pub initial_hover_list: Option<Identity>,
    /// Old objects remain retained after constructor reference replacement.
    /// Their backing arrays must be unaliased with each other and all typed objects.
    pub retained_lists: Vec<ReferenceList>,
    pub constructor: Constructor,
    pub data: Vec<DataBinding>,
    pub components: Components,
    pub gameplay_previous_state: i32,
    pub gameplay_current_state: i32,
    pub continuations: Vec<Continuation>,
    pub occurrences: Vec<Occurrence>,
    pub services: Services,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConstructorEvent {
    Uses {
        value: i32,
    },
    AllocateList {
        list: Identity,
    },
    ConstructEmptyList {
        list: Identity,
        backing: Identity,
    },
    StoreActed {
        list: Identity,
    },
    StoreHover {
        list: Identity,
    },
    StoreSavedAct {
        text: Identity,
    },
    Barrier {
        owner: Identity,
        field: String,
        value: Identity,
    },
    Act {
        value: bool,
    },
    BaseConstructor {
        actor: Identity,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Constructed {
    pub actor: Actor,
    pub act: bool,
    pub acted_list: Identity,
    pub hover_list: Identity,
    pub lists: Vec<ReferenceList>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Publication {
    pub occurrence: usize,
    /// Exact intermediate Actor after callback/status handling, before either refresh.
    pub before_refresh: Actor,
    pub cleared_physical_list: Identity,
    pub list_version_before: u32,
    pub list_version_after: u32,
    /// Caller semantic events, including the actual refresh positions.
    pub initialization: init::Replay,
    pub refresh_character: refresh::Replay,
    pub refresh_view: view::Replay,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub constructed: Constructed,
    pub constructor_events: Vec<ConstructorEvent>,
    pub actor: Actor,
    pub act: bool,
    pub acted_list: Identity,
    pub hover_list: Identity,
    pub lists: Vec<ReferenceList>,
    pub components: Components,
    pub continuations: Vec<Continuation>,
    pub publications: Vec<Publication>,
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
    retained.extend(actor.dead_prefab.as_ref().map(|p| p.identity));
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    // Reserve all retained records and every whole-producer projection before cloning.
    let mut retained = 2usize
        .checked_add(
            c.occurrences
                .len()
                .checked_mul(3)
                .ok_or(LedgerError::Capacity)?,
        )
        .ok_or(LedgerError::Capacity)?;
    for count in [
        c.actor.infos.len(),
        c.actor.statuses.active.len(),
        c.actor.statuses.resistances.len(),
        c.retained_lists.len(),
        c.data.len(),
        c.components.picked_objects.len(),
        c.components.ui_objects.len(),
        c.components.transforms.len(),
        c.continuations.len(),
        c.components.active_status_slots.len(),
    ] {
        retained = retained.checked_add(count).ok_or(LedgerError::Capacity)?;
    }
    for list in &c.retained_lists {
        retained = retained
            .checked_add(list.slots.len())
            .ok_or(LedgerError::Capacity)?;
    }
    let multiplier = c
        .occurrences
        .len()
        .checked_mul(10)
        .and_then(|n| n.checked_add(3))
        .ok_or(LedgerError::Capacity)?;
    if c.occurrences.len() > MAX_OCCURRENCES
        || retained > MAX_RETAINED
        || retained
            .checked_mul(multiplier)
            .ok_or(LedgerError::Capacity)?
            > MAX_WORK
    {
        return Err(LedgerError::Capacity);
    }
    if c.actor.infos.len() > 4096
        || c.actor.statuses.active.len() > 4096
        || c.actor.statuses.resistances.len() > 4096
        || c.data.len() > 4096
        || c.components.picked_objects.len() > 4096
        || c.components.ui_objects.len() > 4096
        || c.components.transforms.len() > 4096
        || c.components.active_status_slots.len() > 4096
        || c.continuations
            .len()
            .checked_add(c.occurrences.len())
            .ok_or(LedgerError::Capacity)?
            > 4096
    {
        return Err(LedgerError::Capacity);
    }
    let s = &c.services;
    if c.version != CHARACTER_CONSTRUCTOR_INIT_NATIVE_V1
        || !s.native_runtime_and_bindings_verified
        || !s.constructor_allocations_and_empty_storage_verified
        || !s.empty_literal_verified
        || !s.base_and_all_callbacks_inert
        || !s.ui_and_other_services_inert
        || !s.unity_liveness_verified
        || !s.required_initializer_objects_valid
        || !s.clone_results_verified
        || !s.synchronous_first_yield_verified
        || !s.normal_completion_verified
        || c.actor.identity == 0
        || c.initial_acted_list == Some(0)
        || c.initial_hover_list == Some(0)
    {
        return Err(LedgerError::InvalidContext);
    }
    if [
        c.actor.data,
        c.actor.bluff,
        c.actor.register_as,
        c.actor.trailer,
        c.actor.runtime,
        c.actor.role,
        c.actor.bluff_role,
        c.actor.saved_act,
        c.actor.state_callback,
        c.actor.statuses.target,
    ]
    .contains(&Some(0))
        || c.actor.infos.contains(&Some(0))
        || c.actor
            .dead_prefab
            .as_ref()
            .is_some_and(|d| d.identity == 0)
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut typed = BTreeSet::from([c.actor.identity]);
    let mut insert = |id| {
        if id == 0 || !typed.insert(id) {
            Err(LedgerError::InvalidContext)
        } else {
            Ok(())
        }
    };
    for id in [
        c.components.acted_component,
        c.components.number_component,
        c.components.statuses_component,
        c.components.active_status_list,
        c.components.active_status_array,
        c.components.resistance_list,
        c.components.picked_array,
    ] {
        insert(id)?;
    }
    for id in c
        .data
        .iter()
        .map(|d| d.identity)
        .chain(c.components.ui_objects.iter().map(|o| o.identity))
        .chain(c.components.transforms.iter().map(|t| t.identity))
        .chain(c.retained_lists.iter().map(|l| l.identity))
        .chain(c.retained_lists.iter().map(|l| l.backing_array))
    {
        insert(id)?;
    }
    let ui: BTreeSet<_> = c.components.ui_objects.iter().map(|o| o.identity).collect();
    let transforms: BTreeSet<_> = c.components.transforms.iter().map(|t| t.identity).collect();
    let data: BTreeMap<_, _> = c.data.iter().map(|d| (d.identity, d)).collect();
    if ![
        c.components.acted_game_object,
        c.components.pickable,
        c.components.rip,
    ]
    .iter()
    .all(|id| ui.contains(id))
        || c.components
            .picked_objects
            .iter()
            .any(|id| !ui.contains(id))
        || c.components
            .disguise
            .as_ref()
            .is_some_and(|d| d.identity == 0 || !ui.contains(&d.identity))
        || ![c.components.icon_transform, c.components.actor_transform]
            .iter()
            .all(|id| transforms.contains(id))
        || c.components.dead_template == 0
        || typed.contains(&c.components.dead_template)
        || c.actor.statuses.active.len() > c.components.active_status_slots.len()
        || c.actor.statuses.active
            != c.components.active_status_slots[..c.actor.statuses.active.len()]
        || c.retained_lists
            .iter()
            .any(|l| l.count > l.slots.len() || l.slots.contains(&Some(0)))
    {
        return Err(LedgerError::InvalidContext);
    }
    typed.insert(c.components.dead_template);
    if c.data
        .iter()
        .any(|d| d.source_role == 0 || typed.contains(&d.source_role))
    {
        return Err(LedgerError::InvalidContext);
    }
    // Init invokes this reference as a delegate; consumed objects of another
    // known type cannot supply that callback, even when all services are inert.
    if c.actor.state_callback.is_some_and(|callback| {
        typed.contains(&callback)
            || c.data.iter().any(|d| d.source_role == callback)
            || c.actor.infos.contains(&Some(callback))
            || c.retained_lists
                .iter()
                .any(|l| l.slots.contains(&Some(callback)))
    }) {
        return Err(LedgerError::InvalidContext);
    }
    // Destroying a retained presentation that aliases a used control needs its own lifetime proof.
    if c.actor
        .dead_prefab
        .as_ref()
        .is_some_and(|d| typed.contains(&d.identity))
    {
        return Err(LedgerError::InvalidContext);
    }
    let lists: BTreeMap<_, _> = c.retained_lists.iter().map(|l| (l.identity, l)).collect();
    match c.initial_acted_list {
        Some(id) => {
            let list = lists.get(&id).ok_or(LedgerError::InvalidContext)?;
            if c.actor.infos != list.slots[..list.count] || c.actor.info_version != list.version {
                return Err(LedgerError::InvalidContext);
            }
        }
        None if !c.actor.infos.is_empty() || c.actor.info_version != 0 => {
            return Err(LedgerError::InvalidContext)
        }
        None => {}
    }
    if c.initial_hover_list
        .is_some_and(|id| !lists.contains_key(&id))
    {
        return Err(LedgerError::InvalidContext);
    }
    let literal = c.constructor.empty_string;
    let literal_is_non_string = typed.contains(&literal)
        || [
            c.actor.data,
            c.actor.bluff,
            c.actor.register_as,
            c.actor.trailer,
            c.actor.runtime,
            c.actor.role,
            c.actor.bluff_role,
            c.actor.state_callback,
            c.actor.statuses.target,
        ]
        .contains(&Some(literal))
        || c.actor.infos.contains(&Some(literal))
        || c.actor
            .dead_prefab
            .as_ref()
            .is_some_and(|o| o.identity == literal)
        || c.data.iter().any(|d| d.source_role == literal)
        || c.retained_lists
            .iter()
            .any(|l| l.slots.contains(&Some(literal)))
        || c.continuations
            .iter()
            .any(|p| p.identity == literal || p.current == Some(literal));
    let mut retained_ids = typed;
    actor_references(&c.actor, &mut retained_ids);
    retained_ids.insert(c.components.dead_template);
    retained_ids.extend(c.data.iter().map(|d| d.source_role));
    retained_ids.extend(
        c.retained_lists
            .iter()
            .flat_map(|l| l.slots.iter().flatten().copied()),
    );
    let mut pending = BTreeSet::new();
    for item in &c.continuations {
        if item.identity == 0
            || item.actor != c.actor.identity
            || item.current == Some(0)
            || !pending.insert(item.identity)
            || retained_ids.contains(&item.identity)
            || item
                .current
                .is_some_and(|id| retained_ids.contains(&id) || pending.contains(&id))
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let currents: BTreeSet<_> = c.continuations.iter().filter_map(|p| p.current).collect();
    if !pending.is_disjoint(&currents) {
        return Err(LedgerError::InvalidContext);
    }
    retained_ids.extend(pending);
    retained_ids.extend(currents);
    let ctor = &c.constructor;
    // Literal reuse is valid only in opaque string fields; it cannot be a typed object or array.
    if ctor.empty_string == 0
        || literal_is_non_string
        || retained_ids.contains(&ctor.empty_string) && c.actor.saved_act != Some(ctor.empty_string)
        || ctor.empty_array == 0
        || retained_ids.contains(&ctor.empty_array)
        || ctor.empty_array == ctor.empty_string
    {
        return Err(LedgerError::InvalidContext);
    }
    retained_ids.insert(ctor.empty_string);
    retained_ids.insert(ctor.empty_array);
    for id in [ctor.acted_list, ctor.hover_list] {
        if id == 0 || !retained_ids.insert(id) {
            return Err(LedgerError::InvalidContext);
        }
    }
    for item in &c.occurrences {
        let asset = data.get(&item.data).ok_or(LedgerError::InvalidContext)?;
        if item.clone_source != asset.source_role {
            return Err(LedgerError::InvalidContext);
        }
        for id in [Some(item.continuation), Some(item.wait), item.clone_result]
            .into_iter()
            .flatten()
        {
            if id == 0 || !retained_ids.insert(id) {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    Ok(())
}

fn set_active(components: &mut Components, object: Identity, active: bool) {
    components
        .ui_objects
        .iter_mut()
        .find(|o| o.identity == object)
        .unwrap()
        .active = active;
}

pub fn replay(input: &Context) -> Result<Replay, LedgerError> {
    validate(input)?;
    let mut actor = input.actor.clone();
    let ctor = &input.constructor;
    let mut lists = input.retained_lists.clone();
    let mut events = vec![ConstructorEvent::Uses { value: 1 }];
    actor.uses = 1;
    for (id, acted) in [(ctor.acted_list, true), (ctor.hover_list, false)] {
        events.push(ConstructorEvent::AllocateList { list: id });
        events.push(ConstructorEvent::ConstructEmptyList {
            list: id,
            backing: ctor.empty_array,
        });
        lists.push(ReferenceList {
            identity: id,
            backing_array: ctor.empty_array,
            count: 0,
            version: 0,
            slots: vec![],
        });
        if acted {
            actor.infos.clear();
            actor.info_version = 0;
            events.push(ConstructorEvent::StoreActed { list: id });
        } else {
            events.push(ConstructorEvent::StoreHover { list: id });
        }
        events.push(ConstructorEvent::Barrier {
            owner: actor.identity,
            field: if acted { "acted_infos" } else { "hover_infos" }.into(),
            value: id,
        });
    }
    actor.saved_act = Some(ctor.empty_string);
    events.push(ConstructorEvent::StoreSavedAct {
        text: ctor.empty_string,
    });
    events.push(ConstructorEvent::Barrier {
        owner: actor.identity,
        field: "saved_act".into(),
        value: ctor.empty_string,
    });
    events.push(ConstructorEvent::Act { value: true });
    events.push(ConstructorEvent::BaseConstructor {
        actor: actor.identity,
    });
    let constructed = Constructed {
        actor: actor.clone(),
        act: true,
        acted_list: ctor.acted_list,
        hover_list: ctor.hover_list,
        lists: lists.clone(),
    };
    let mut components = input.components.clone();
    let mut continuations = input.continuations.clone();
    let mut publications = Vec::new();
    let abilities: Vec<_> = input
        .data
        .iter()
        .map(|d| AbilityData {
            identity: d.identity,
            ability_usage: d.ability_usage,
            picking: d.picking,
        })
        .collect();
    for (occurrence, item) in input.occurrences.iter().enumerate() {
        let asset = input.data.iter().find(|d| d.identity == item.data).unwrap();
        let old_role = actor.role;
        let initialized = init::replay(&init::Context {
            version: init::CHARACTER_INITIALIZATION_NATIVE_V1.into(),
            method: item.method,
            actor: actor.clone(),
            data: item.data,
            starting_alignment: asset.starting_alignment,
            id: item.id,
            required_objects_and_lists_valid: true,
            callbacks_and_ui_inert: true,
            clone_result_verified: true,
            synchronous_first_yield_verified: true,
            clone_result: item.clone_result,
            continuation_identity: item.continuation,
            wait_identity: item.wait,
            continuations: continuations.clone(),
        })?;
        // Of all initializer fields, only role changes after the two refreshes.
        // Full intermediate snapshots and both component Actors are verified by native joined fixtures.
        let mut before_refresh = initialized.actor.clone();
        before_refresh.role = old_role;
        for event in &initialized.events {
            match event {
                init::Event::HideActed => {
                    let id = components.acted_game_object;
                    set_active(&mut components, id, false);
                }
                init::Event::HideRip => {
                    let id = components.rip;
                    set_active(&mut components, id, false);
                }
                init::Event::RefreshCharacter => break,
                _ => {}
            }
        }
        let refreshed = refresh::replay(&refresh::Context {
            version: refresh::CHARACTER_REFRESH_NATIVE_V1.into(),
            actor: before_refresh.clone(),
            raw_bluff: None,
            gameplay_previous_state: input.gameplay_previous_state,
            gameplay_current_state: input.gameplay_current_state,
            data: abilities.clone(),
            picked_array_identity: components.picked_array,
            picked_objects: components.picked_objects.clone(),
            pickable_object: components.pickable,
            ui_objects: components
                .ui_objects
                .iter()
                .map(|o| refresh::UiObject {
                    identity: o.identity,
                    active: o.active,
                })
                .collect(),
            native_runtime_provenance_verified: true,
            unity_liveness_verified: true,
            callbacks_and_ui_inert: true,
            normal_completion_verified: true,
        })?;
        if refreshed.context.actor != before_refresh {
            return Err(LedgerError::InvalidContext);
        }
        components.ui_objects = refreshed
            .context
            .ui_objects
            .iter()
            .map(|o| UiObject {
                identity: o.identity,
                active: o.active,
            })
            .collect();
        let viewed = view::replay(&view::Context {
            version: view::CHARACTER_REFRESH_VIEW_NATIVE_V1.into(),
            actor: refreshed.context.actor.clone(),
            raw_bluff: None,
            icon_component: components.icon_transform,
            actor_transform: components.actor_transform,
            icon_transform: components.icon_transform,
            dead_prefab_template: components.dead_template,
            pickable: components.pickable,
            rip: components.rip,
            disguise: components.disguise.clone(),
            ui_objects: components.ui_objects.clone(),
            transforms: components.transforms.clone(),
            vector3_zero: components.vector3_zero,
            creation: None,
            native_runtime_and_bindings_verified: true,
            unity_liveness_verified: true,
            callbacks_and_services_inert: true,
            normal_completion_verified: true,
        })?;
        if viewed.context.actor != before_refresh {
            return Err(LedgerError::InvalidContext);
        }
        components.ui_objects = viewed.context.ui_objects.clone();
        components.transforms = viewed.context.transforms.clone();
        let list = lists
            .iter_mut()
            .find(|l| l.identity == ctor.acted_list)
            .unwrap();
        let version_before = list.version;
        list.count = 0;
        list.version = initialized.actor.info_version;
        let version_after = list.version;
        actor = initialized.actor.clone();
        continuations = initialized.continuations.clone();
        publications.push(Publication {
            occurrence,
            before_refresh,
            cleared_physical_list: ctor.acted_list,
            list_version_before: version_before,
            list_version_after: version_after,
            initialization: initialized,
            refresh_character: refreshed,
            refresh_view: viewed,
        });
    }
    Ok(Replay {
        constructed,
        constructor_events: events,
        actor,
        act: true,
        acted_list: ctor.acted_list,
        hover_list: ctor.hover_list,
        lists,
        components,
        continuations,
        publications,
    })
}

#[cfg(test)]
#[path = "character_constructor_init_tests.rs"]
mod tests;
