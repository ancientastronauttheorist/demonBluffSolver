//! Bounded standalone Character.RefreshCharacter with inert supplied services.
//!
//! Runtime and Unity liveness provenance are explicit. This preserves physical
//! data/UI identities and does not broaden initializer or scheduler composition.

use super::character_initialization::{Actor, Identity, ObjectReference};
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CHARACTER_REFRESH_NATIVE_V1: &str = "character_refresh_native_v1";
const MAX_ENTRIES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbilityData {
    pub identity: Identity,
    pub ability_usage: i32,
    pub picking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UiObject {
    pub identity: Identity,
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub actor: Actor,
    /// Must agree with actor.bluff, retaining a destroyed non-null identity.
    pub raw_bluff: Option<ObjectReference>,
    pub gameplay_previous_state: i32,
    /// Retained provenance; native RefreshCharacter never reads this field.
    pub gameplay_current_state: i32,
    pub data: Vec<AbilityData>,
    pub picked_array_identity: Identity,
    /// Ordered occurrences; repeated references to one GameObject are legal.
    pub picked_objects: Vec<Identity>,
    pub pickable_object: Identity,
    /// Unique physical records. Unreferenced records are retained unchanged.
    pub ui_objects: Vec<UiObject>,
    pub native_runtime_provenance_verified: bool,
    pub unity_liveness_verified: bool,
    pub callbacks_and_ui_inert: bool,
    pub normal_completion_verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionPass {
    AbilityUsage,
    Picking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DataSource {
    Real,
    Bluff,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    HidePicked {
        array_index: usize,
        object: Identity,
    },
    SelectData {
        pass: SelectionPass,
        source: DataSource,
        identity: Identity,
    },
    ResetUses {
        previous: i32,
        current: i32,
    },
    ActivatePickable {
        object: Identity,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub context: Context,
    pub events: Vec<Event>,
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    if c.version != CHARACTER_REFRESH_NATIVE_V1
        || !c.native_runtime_provenance_verified
        || !c.unity_liveness_verified
        || !c.callbacks_and_ui_inert
        || !c.normal_completion_verified
        || c.actor.identity == 0
        || c.actor.data.is_none()
        || c.actor.data == Some(0)
        || c.actor.bluff == Some(0)
        || c.picked_array_identity == 0
        || c.pickable_object == 0
        || c.raw_bluff.as_ref().map(|r| r.identity) != c.actor.bluff
    {
        return Err(LedgerError::InvalidContext);
    }
    if c.data.len() > MAX_ENTRIES
        || c.ui_objects.len() > MAX_ENTRIES
        || c.picked_objects.len() > MAX_ENTRIES
        || c.actor.infos.len() > MAX_ENTRIES
        || c.actor.statuses.active.len() > MAX_ENTRIES
        || c.actor.statuses.resistances.len() > MAX_ENTRIES
    {
        return Err(LedgerError::Capacity);
    }
    let data_ids: BTreeSet<_> = c.data.iter().map(|d| d.identity).collect();
    let ui_ids: BTreeSet<_> = c.ui_objects.iter().map(|o| o.identity).collect();
    if data_ids.len() != c.data.len()
        || ui_ids.len() != c.ui_objects.len()
        || data_ids.contains(&0)
        || ui_ids.contains(&0)
        || !data_ids.is_disjoint(&ui_ids)
        || data_ids.contains(&c.actor.identity)
        || data_ids.contains(&c.picked_array_identity)
        || ui_ids.contains(&c.actor.identity)
        || ui_ids.contains(&c.picked_array_identity)
        || c.actor.identity == c.picked_array_identity
        || !data_ids.contains(&c.actor.data.unwrap())
        || c.actor.bluff.is_some_and(|b| !data_ids.contains(&b))
        || !ui_ids.contains(&c.pickable_object)
        || c.picked_objects.iter().any(|id| !ui_ids.contains(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

fn select(c: &Context) -> (DataSource, Identity) {
    if !matches!(c.actor.state, 20 | 30)
        && !c.actor.revealed
        && c.raw_bluff.as_ref().is_some_and(|b| b.live)
    {
        (DataSource::Bluff, c.actor.bluff.unwrap())
    } else {
        (DataSource::Real, c.actor.data.unwrap())
    }
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
    for (array_index, object) in context.picked_objects.iter().copied().enumerate() {
        context.ui_objects[ui_indices[&object]].active = false;
        events.push(Event::HidePicked {
            array_index,
            object,
        });
    }
    if context.gameplay_previous_state == 20 {
        let (source, identity) = select(&context);
        events.push(Event::SelectData {
            pass: SelectionPass::AbilityUsage,
            source,
            identity,
        });
        let selected = context
            .data
            .iter()
            .find(|d| d.identity == identity)
            .unwrap();
        if selected.ability_usage == 10 {
            let previous = context.actor.uses;
            context.actor.uses = 1;
            events.push(Event::ResetUses {
                previous,
                current: 1,
            });
            if !matches!(context.actor.state, 5 | 20) {
                // Native selects again after its reset and state checks. Inert
                // callbacks make the identities stable, but both reads remain
                // explicit rather than borrowing the first selection.
                let (source, identity) = select(&context);
                events.push(Event::SelectData {
                    pass: SelectionPass::Picking,
                    source,
                    identity,
                });
                let selected = context
                    .data
                    .iter()
                    .find(|d| d.identity == identity)
                    .unwrap();
                if selected.picking {
                    context.ui_objects[ui_indices[&context.pickable_object]].active = true;
                    events.push(Event::ActivatePickable {
                        object: context.pickable_object,
                    });
                }
            }
        }
    }
    Ok(Replay { context, events })
}

#[cfg(test)]
#[path = "character_refresh_tests.rs"]
mod tests;
