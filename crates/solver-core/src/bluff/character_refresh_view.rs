//! Standalone native RefreshView caller with explicit inert presentation services.
//! Physical references and Vector3 bits survive unchanged unless the caller
//! explicitly stores a created object or requests a UI/transform write.

use super::character_initialization::{Actor, Identity, ObjectReference};
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CHARACTER_REFRESH_VIEW_NATIVE_V1: &str = "character_refresh_view_native_v1";
const MAX_ENTRIES: usize = 4096;
pub type VectorBits = [u32; 3];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UiObject {
    pub identity: Identity,
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    pub identity: Identity,
    pub position: VectorBits,
    pub euler_angles: VectorBits,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Creation {
    /// Independently verified successful Instantiate result; must be fresh/live.
    pub instance: ObjectReference,
    /// Explicit outputs of the two distinct GameObject.get_transform calls.
    pub first_transform: Identity,
    pub second_transform: Identity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub actor: Actor,
    pub raw_bluff: Option<ObjectReference>,
    pub icon_component: Identity,
    pub actor_transform: Identity,
    pub icon_transform: Identity,
    /// Template source; actor.dead_prefab is the retained created presentation.
    pub dead_prefab_template: Identity,
    pub pickable: Identity,
    pub rip: Identity,
    pub disguise: Option<ObjectReference>,
    pub ui_objects: Vec<UiObject>,
    pub transforms: Vec<Transform>,
    pub vector3_zero: VectorBits,
    /// Required exactly when this caller reaches Instantiate.
    pub creation: Option<Creation>,
    pub native_runtime_and_bindings_verified: bool,
    pub unity_liveness_verified: bool,
    pub callbacks_and_services_inert: bool,
    pub normal_completion_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    SetActive {
        object: Identity,
        active: bool,
    },
    UnityNull {
        object: Option<Identity>,
        result: bool,
    },
    UnityLive {
        object: Option<Identity>,
        result: bool,
    },
    ComponentTransform {
        component: Identity,
        result: Identity,
    },
    InstantiateGameObject {
        template: Identity,
        parent: Identity,
        result: Identity,
    },
    StoreCreated {
        previous: Option<ObjectReference>,
        current: ObjectReference,
    },
    BarrierCreated {
        actor: Identity,
        value: Identity,
    },
    GameObjectTransform {
        object: Identity,
        result: Identity,
    },
    GetPosition {
        transform: Identity,
        bits: VectorBits,
    },
    SetPosition {
        transform: Identity,
        bits: VectorBits,
    },
    SetEulerAngles {
        transform: Identity,
        bits: VectorBits,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub context: Context,
    pub events: Vec<Event>,
}

fn creates(c: &Context) -> bool {
    c.actor.state == 20 && !c.actor.dead_prefab.as_ref().is_some_and(|o| o.live)
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    if c.version != CHARACTER_REFRESH_VIEW_NATIVE_V1
        || !c.native_runtime_and_bindings_verified
        || !c.unity_liveness_verified
        || !c.callbacks_and_services_inert
        || !c.normal_completion_verified
        || c.actor.bluff == Some(0)
        || c.raw_bluff.as_ref().map(|r| r.identity) != c.actor.bluff
        || c.actor
            .dead_prefab
            .as_ref()
            .is_some_and(|o| o.identity == 0)
        || c.disguise.as_ref().is_some_and(|o| o.identity == 0)
        || [
            c.actor.identity,
            c.icon_component,
            c.actor_transform,
            c.icon_transform,
            c.dead_prefab_template,
            c.pickable,
            c.rip,
        ]
        .contains(&0)
        || creates(c) != c.creation.is_some()
    {
        return Err(LedgerError::InvalidContext);
    }
    if c.ui_objects.len() > MAX_ENTRIES
        || c.transforms.len() > MAX_ENTRIES
        || c.actor.infos.len() > MAX_ENTRIES
        || c.actor.statuses.active.len() > MAX_ENTRIES
        || c.actor.statuses.resistances.len() > MAX_ENTRIES
    {
        return Err(LedgerError::Capacity);
    }
    let mut liveness = BTreeMap::new();
    for reference in [
        c.raw_bluff.as_ref(),
        c.actor.dead_prefab.as_ref(),
        c.disguise.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if liveness
            .insert(reference.identity, reference.live)
            .is_some_and(|previous| previous != reference.live)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let ui: BTreeSet<_> = c.ui_objects.iter().map(|o| o.identity).collect();
    let transforms: BTreeSet<_> = c.transforms.iter().map(|t| t.identity).collect();
    if ui.len() != c.ui_objects.len()
        || transforms.len() != c.transforms.len()
        || ui.contains(&0)
        || transforms.contains(&0)
        || !ui.is_disjoint(&transforms)
        || ui.contains(&c.actor.identity)
        || transforms.contains(&c.actor.identity)
        || transforms.contains(&c.dead_prefab_template)
        || c.dead_prefab_template == c.actor.identity
        || ![c.pickable, c.rip].iter().all(|id| ui.contains(id))
        || c.disguise
            .as_ref()
            .is_some_and(|d| !ui.contains(&d.identity))
        || ![c.icon_component, c.actor_transform, c.icon_transform]
            .iter()
            .all(|id| transforms.contains(id))
        || c.actor
            .dead_prefab
            .as_ref()
            .is_some_and(|d| transforms.contains(&d.identity) || d.identity == c.actor.identity)
    {
        return Err(LedgerError::InvalidContext);
    }
    if let Some(creation) = &c.creation {
        let id = creation.instance.identity;
        let mut retained: BTreeSet<_> = [c.actor.identity, c.dead_prefab_template]
            .into_iter()
            .collect();
        retained.extend(ui);
        retained.extend(transforms.iter().copied());
        retained.extend(
            [
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
            .into_iter()
            .flatten(),
        );
        retained.extend(c.actor.infos.iter().flatten().copied());
        retained.extend(c.actor.dead_prefab.as_ref().map(|d| d.identity));
        if id == 0
            || !creation.instance.live
            || retained.contains(&id)
            || !transforms.contains(&creation.first_transform)
            || !transforms.contains(&creation.second_transform)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    Ok(())
}

pub fn replay(input: &Context) -> Result<Replay, LedgerError> {
    validate(input)?;
    let mut context = input.clone();
    let mut events = Vec::new();
    let ui_indices: BTreeMap<_, _> = context
        .ui_objects
        .iter()
        .enumerate()
        .map(|(index, o)| (o.identity, index))
        .collect();
    let transform_indices: BTreeMap<_, _> = context
        .transforms
        .iter()
        .enumerate()
        .map(|(index, t)| (t.identity, index))
        .collect();
    if context.actor.uses <= 0 {
        context.ui_objects[ui_indices[&context.pickable]].active = false;
        events.push(Event::SetActive {
            object: context.pickable,
            active: false,
        });
    }
    if context.actor.state == 20 {
        let is_null = !context.actor.dead_prefab.as_ref().is_some_and(|o| o.live);
        events.push(Event::UnityNull {
            object: context.actor.dead_prefab.as_ref().map(|o| o.identity),
            result: is_null,
        });
        if is_null {
            let creation = context.creation.as_ref().unwrap().clone();
            events.push(Event::ComponentTransform {
                component: context.actor.identity,
                result: context.actor_transform,
            });
            events.push(Event::InstantiateGameObject {
                template: context.dead_prefab_template,
                parent: context.actor_transform,
                result: creation.instance.identity,
            });
            let previous = context.actor.dead_prefab.replace(creation.instance.clone());
            events.push(Event::StoreCreated {
                previous,
                current: creation.instance.clone(),
            });
            events.push(Event::BarrierCreated {
                actor: context.actor.identity,
                value: creation.instance.identity,
            });
            events.push(Event::GameObjectTransform {
                object: creation.instance.identity,
                result: creation.first_transform,
            });
            events.push(Event::ComponentTransform {
                component: context.icon_component,
                result: context.icon_transform,
            });
            let bits = context.transforms[transform_indices[&context.icon_transform]].position;
            events.push(Event::GetPosition {
                transform: context.icon_transform,
                bits,
            });
            context.transforms[transform_indices[&creation.first_transform]].position = bits;
            events.push(Event::SetPosition {
                transform: creation.first_transform,
                bits,
            });
            // This is a second native object-field reload/getter call, even if
            // both authored getter outputs refer to one physical Transform.
            events.push(Event::GameObjectTransform {
                object: context.actor.dead_prefab.as_ref().unwrap().identity,
                result: creation.second_transform,
            });
            context.transforms[transform_indices[&creation.second_transform]].euler_angles =
                context.vector3_zero;
            events.push(Event::SetEulerAngles {
                transform: creation.second_transform,
                bits: context.vector3_zero,
            });
            context.ui_objects[ui_indices[&context.rip]].active = true;
            events.push(Event::SetActive {
                object: context.rip,
                active: true,
            });
        }
    }
    let icon_live = context.disguise.as_ref().is_some_and(|d| d.live);
    events.push(Event::UnityLive {
        object: context.disguise.as_ref().map(|d| d.identity),
        result: icon_live,
    });
    if icon_live && !context.actor.killed_demon {
        let mut active = false;
        if matches!(context.actor.state, 20 | 30) {
            let bluff_null = !context.raw_bluff.as_ref().is_some_and(|b| b.live);
            events.push(Event::UnityNull {
                object: context.actor.bluff,
                result: bluff_null,
            });
            active = !bluff_null;
        }
        let object = context.disguise.as_ref().unwrap().identity;
        context.ui_objects[ui_indices[&object]].active = active;
        events.push(Event::SetActive { object, active });
    }
    // A replayed result retains the allocation for subsequent refreshes; its
    // consumed fresh-result binding must not be reused as another allocation.
    context.creation = None;
    Ok(Replay { context, events })
}

#[cfg(test)]
#[path = "character_refresh_view_tests.rs"]
mod tests;
