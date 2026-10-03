//! Guarded offline publication of at most two verified suspended role results.
//! All supplied result resumes precede speech resumes; speech runs to completion
//! in registration order. Readiness, elapsed time and real scheduler order are
//! not inferred. Mutating callbacks, growth and failed services are rejected.
//! ReferenceList objects may be shared by ActedInfo references; their nonnull
//! backing arrays must be distinct from each other, the history backing array
//! and all other typed objects. Shared backing storage is outside this version.

use super::character_initialization::{Actor, Identity, ObjectReference};
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CHARACTER_ROLE_PUBLICATION_NATIVE_V1: &str = "character_role_publication_native_v1";
pub const RESULT_WAIT_BITS: u32 = 0;
pub const SPEECH_WAIT_BITS: u32 = 0x3ECCCCCD;
const MAX_RESULTS: usize = 2;
const MAX_RETAINED: usize = 65_536;
const MAX_WORK: usize = 262_144;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedString {
    pub identity: Identity,
    /// Actual UTF16 units, including embedded NUL and unpaired surrogates.
    pub units: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceList {
    pub identity: Identity,
    pub backing_array: Option<Identity>,
    pub version: u32,
    pub entries: Vec<Option<Identity>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActedInfo {
    pub identity: Identity,
    pub description: Option<Identity>,
    pub references: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryStorage {
    pub identity: Identity,
    pub backing_array: Identity,
    pub capacity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataAsset {
    pub identity: Identity,
    pub picking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaitObject {
    pub identity: Identity,
    pub seconds_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultIterator {
    pub identity: Identity,
    pub actor: Identity,
    pub info: Option<Identity>,
    pub trigger_bits: u32,
    pub delay_bits: u32,
    pub state: i32,
    pub current: Identity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Allocation {
    pub result_iterator: Identity,
    pub speech_iterator: Identity,
    /// Required only on the native non-picking, non-state20 wait path.
    pub speech_wait: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UiObject {
    pub identity: Identity,
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ui {
    pub acted_component: Identity,
    pub acted_version: Identity,
    pub blank_text: Identity,
    pub blank_text_value: Option<Identity>,
    pub layout_array: Identity,
    pub layouts: Vec<Identity>,
    pub first_game_object: Identity,
    /// The second getter output is forwarded to Log without a null check.
    pub log_game_object: Option<Identity>,
    pub show_game_object: Identity,
    pub name_text: Identity,
    pub log_text: Identity,
    pub pickable: Identity,
    pub objects: Vec<UiObject>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub runtime_and_metadata_verified: bool,
    pub callback_captures_and_first_yield_verified: bool,
    pub list_storage_and_capacity_verified: bool,
    pub preappend_callbacks_inert: bool,
    pub global_callbacks_inert: bool,
    pub ui_and_other_services_inert: bool,
    pub unity_liveness_verified: bool,
    pub trailer_lookup_stable_verified: bool,
    pub supplied_resume_order_verified: bool,
    pub normal_completion_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub actor: Actor,
    pub act: bool,
    pub raw_bluff: Option<ObjectReference>,
    pub data_assets: Vec<DataAsset>,
    pub history: HistoryStorage,
    pub strings: Vec<ManagedString>,
    pub infos: Vec<ActedInfo>,
    pub reference_lists: Vec<ReferenceList>,
    pub result_iterators: Vec<ResultIterator>,
    pub result_waits: Vec<WaitObject>,
    pub allocations: Vec<Allocation>,
    pub preappend_callback: Option<Identity>,
    pub info_revealed_callback: Option<Identity>,
    pub trailer_mode: bool,
    pub trailer_text: Option<Identity>,
    pub ui: Ui,
    /// Exact permutation of the verified suspended results; all precede speech.
    pub result_resume_order: Vec<Identity>,
    /// Registration order, contiguous one/two resumes per speech until false.
    pub speech_resume_order: Vec<Identity>,
    pub services: Services,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpeechIterator {
    pub identity: Identity,
    pub actor: Identity,
    pub description: Identity,
    pub state: i32,
    pub current: Option<Identity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    State {
        iterator: Identity,
        state: i32,
    },
    StringEmpty {
        text: Option<Identity>,
        empty: bool,
    },
    Preappend {
        callback: Identity,
        info: Identity,
        trigger_bits: u32,
    },
    HistoryVersion {
        list: Identity,
        version: u32,
    },
    HistoryAppend {
        list: Identity,
        info: Identity,
        slot: usize,
    },
    Barrier {
        owner: Identity,
        field: String,
        value: Identity,
    },
    Uses {
        previous_bits: u32,
        current_bits: u32,
    },
    InfoRevealed {
        callback: Identity,
        actor: Identity,
    },
    AllocateSpeech {
        iterator: Identity,
        result_iterator: Identity,
    },
    CaptureActor {
        iterator: Identity,
        actor: Identity,
    },
    CaptureDescription {
        iterator: Identity,
        text: Identity,
    },
    RegisterSpeech {
        actor: Identity,
        iterator: Identity,
    },
    TrailerLookup {
        actor_id: i32,
        text: Option<Identity>,
    },
    GameObject {
        component: Identity,
        result: Option<Identity>,
    },
    Name {
        object: Identity,
        text: Identity,
    },
    ConcatLog {
        name: Identity,
        result: Identity,
    },
    Log {
        text: Identity,
        context: Option<Identity>,
    },
    SetText {
        blank: Identity,
        text: Identity,
    },
    SaveSpeech {
        previous: Option<Identity>,
        current: Identity,
    },
    UnityNull {
        object: Option<Identity>,
        result: bool,
    },
    SelectData {
        asset: Identity,
    },
    AllocateWait {
        wait: Identity,
    },
    ConstructWait {
        wait: Identity,
        bits: u32,
    },
    Current {
        iterator: Identity,
        wait: Identity,
    },
    SetActive {
        object: Identity,
        active: bool,
    },
    Show {
        version: Identity,
        text: Identity,
    },
    Rebuild {
        rect: Identity,
    },
    Return {
        iterator: Identity,
        value: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub context: Context,
    pub speech_iterators: Vec<SpeechIterator>,
    pub waits: Vec<WaitObject>,
    pub events: Vec<Event>,
}

fn string<'a>(c: &'a Context, id: Identity) -> Option<&'a ManagedString> {
    c.strings.iter().find(|s| s.identity == id)
}

fn publishable(c: &Context, r: &ResultIterator) -> bool {
    c.act
        && r.info
            .and_then(|id| c.infos.iter().find(|i| i.identity == id))
            .and_then(|i| i.description)
            .and_then(|id| string(c, id))
            .is_some_and(|s| !s.units.is_empty())
}

fn selected_data(c: &Context) -> Identity {
    if matches!(c.actor.state, 20 | 30)
        || c.actor.revealed
        || !c.raw_bluff.as_ref().is_some_and(|b| b.live)
    {
        c.actor.data.unwrap()
    } else {
        c.actor.bluff.unwrap()
    }
}

fn speech_waits(c: &Context) -> bool {
    c.actor.state != 20
        && !c
            .data_assets
            .iter()
            .find(|a| a.identity == selected_data(c))
            .unwrap()
            .picking
}

fn validate(c: &Context, scheduled: bool) -> Result<(), LedgerError> {
    // Aggregate retained slots, text units and derived work before maps/clones.
    let mut retained = 0usize;
    for count in [
        c.actor.infos.len(),
        c.actor.statuses.active.len(),
        c.actor.statuses.resistances.len(),
        c.data_assets.len(),
        c.strings.len(),
        c.infos.len(),
        c.reference_lists.len(),
        c.result_iterators.len(),
        c.result_waits.len(),
        c.allocations.len(),
        c.ui.layouts.len(),
        c.ui.objects.len(),
        c.result_resume_order.len(),
        c.speech_resume_order.len(),
        c.history.capacity,
    ] {
        retained = retained.checked_add(count).ok_or(LedgerError::Capacity)?;
    }
    for count in c
        .strings
        .iter()
        .map(|s| s.units.len())
        .chain(c.reference_lists.iter().map(|l| l.entries.len()))
    {
        retained = retained.checked_add(count).ok_or(LedgerError::Capacity)?;
    }
    let work = retained.checked_mul(4).ok_or(LedgerError::Capacity)?;
    if retained > MAX_RETAINED || work > MAX_WORK || c.result_iterators.len() > MAX_RESULTS {
        return Err(LedgerError::Capacity);
    }
    let s = &c.services;
    if c.version != CHARACTER_ROLE_PUBLICATION_NATIVE_V1
        || !s.runtime_and_metadata_verified
        || !s.callback_captures_and_first_yield_verified
        || !s.list_storage_and_capacity_verified
        || !s.preappend_callbacks_inert
        || !s.global_callbacks_inert
        || !s.ui_and_other_services_inert
        || !s.unity_liveness_verified
        || !s.trailer_lookup_stable_verified
        || if scheduled {
            s.supplied_resume_order_verified
                || !c.result_resume_order.is_empty()
                || !c.speech_resume_order.is_empty()
        } else {
            !s.supplied_resume_order_verified
        }
        || !s.normal_completion_verified
        || c.actor.data.is_none()
        || c.actor.bluff == Some(0)
        || c.raw_bluff.as_ref().map(|b| b.identity) != c.actor.bluff
        || (!c.trailer_mode && c.trailer_text.is_some())
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut typed = BTreeSet::new();
    let mut add = |id: Identity| -> Result<(), LedgerError> {
        if id == 0 || !typed.insert(id) {
            Err(LedgerError::InvalidContext)
        } else {
            Ok(())
        }
    };
    for id in [
        c.actor.identity,
        c.history.identity,
        c.history.backing_array,
        c.ui.acted_component,
        c.ui.acted_version,
        c.ui.blank_text,
        c.ui.layout_array,
    ] {
        add(id)?;
    }
    for id in c
        .strings
        .iter()
        .map(|o| o.identity)
        .chain(c.infos.iter().map(|o| o.identity))
        .chain(c.reference_lists.iter().map(|o| o.identity))
        .chain(c.data_assets.iter().map(|o| o.identity))
        .chain(c.ui.objects.iter().map(|o| o.identity))
        .chain(c.result_iterators.iter().map(|o| o.identity))
        .chain(c.result_waits.iter().map(|o| o.identity))
    {
        add(id)?;
    }
    for id in c.reference_lists.iter().filter_map(|l| l.backing_array) {
        add(id)?;
    }
    let texts: BTreeSet<_> = c.strings.iter().map(|s| s.identity).collect();
    let info_ids: BTreeSet<_> = c.infos.iter().map(|i| i.identity).collect();
    let lists: BTreeSet<_> = c.reference_lists.iter().map(|l| l.identity).collect();
    let assets: BTreeSet<_> = c.data_assets.iter().map(|a| a.identity).collect();
    let ui: BTreeSet<_> = c.ui.objects.iter().map(|o| o.identity).collect();
    if !assets.contains(&c.actor.data.unwrap())
        || c.actor.bluff.is_some_and(|b| !assets.contains(&b))
        || [c.actor.saved_act, c.ui.blank_text_value, c.trailer_text]
            .into_iter()
            .flatten()
            .any(|id| !texts.contains(&id))
        || c.infos.iter().any(|i| {
            i.description.is_some_and(|id| !texts.contains(&id))
                || i.references.is_some_and(|id| !lists.contains(&id))
        })
        || c.actor
            .infos
            .iter()
            .flatten()
            .any(|id| !info_ids.contains(id))
        || c.reference_lists.iter().any(|l| {
            l.entries.contains(&Some(0)) || (!l.entries.is_empty() && l.backing_array.is_none())
        })
        || c.ui.layouts.contains(&0)
        || ![c.ui.first_game_object, c.ui.show_game_object, c.ui.pickable]
            .into_iter()
            .all(|id| ui.contains(&id))
        || c.ui.log_game_object.is_some_and(|id| !ui.contains(&id))
        || !texts.contains(&c.ui.name_text)
        || !texts.contains(&c.ui.log_text)
        || c.preappend_callback == Some(0)
        || c.info_revealed_callback == Some(0)
        || [
            c.actor.register_as,
            c.actor.trailer,
            c.actor.runtime,
            c.actor.role,
            c.actor.bluff_role,
            c.actor.state_callback,
            c.actor.statuses.target,
        ]
        .contains(&Some(0))
        || c.actor
            .dead_prefab
            .as_ref()
            .is_some_and(|o| o.identity == 0)
        || c.ui.layouts.iter().any(|id| typed.contains(id))
        || [c.preappend_callback, c.info_revealed_callback]
            .into_iter()
            .flatten()
            .any(|id| typed.contains(&id) || c.ui.layouts.contains(&id))
    {
        return Err(LedgerError::InvalidContext);
    }
    let prefix = "Character: ".encode_utf16();
    if !prefix
        .chain(string(c, c.ui.name_text).unwrap().units.iter().copied())
        .eq(string(c, c.ui.log_text).unwrap().units.iter().copied())
    {
        return Err(LedgerError::InvalidContext);
    }
    let results: BTreeSet<_> = c.result_iterators.iter().map(|r| r.identity).collect();
    let order: BTreeSet<_> = c.result_resume_order.iter().copied().collect();
    let mut currents = BTreeSet::new();
    if (!scheduled && (results != order || order.len() != c.result_resume_order.len()))
        || c.result_waits.len() != results.len()
        || c.result_iterators.iter().any(|r| {
            r.actor != c.actor.identity
                || r.state != 1
                || r.delay_bits != RESULT_WAIT_BITS
                || !currents.insert(r.current)
                || !c
                    .result_waits
                    .iter()
                    .any(|w| w.identity == r.current && w.seconds_bits == RESULT_WAIT_BITS)
                || r.info.is_some_and(|id| !info_ids.contains(&id))
                || (c.act && r.info.is_none())
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    let count = c
        .result_iterators
        .iter()
        .filter(|r| publishable(c, r))
        .count();
    if c.actor
        .infos
        .len()
        .checked_add(count)
        .ok_or(LedgerError::Capacity)?
        > c.history.capacity
    {
        return Err(LedgerError::InvalidContext);
    }
    if c.allocations.len() != count {
        return Err(LedgerError::InvalidContext);
    }
    let mut retained_ids = typed;
    retained_ids.extend(c.ui.layouts.iter().copied());
    retained_ids.extend(
        [
            c.actor.register_as,
            c.actor.trailer,
            c.actor.runtime,
            c.actor.role,
            c.actor.bluff_role,
            c.actor.state_callback,
            c.actor.statuses.target,
            c.preappend_callback,
            c.info_revealed_callback,
        ]
        .into_iter()
        .flatten(),
    );
    retained_ids.extend(c.actor.dead_prefab.as_ref().map(|o| o.identity));
    retained_ids.extend(
        c.reference_lists
            .iter()
            .flat_map(|l| l.entries.iter().flatten().copied()),
    );
    let mut allocation_keys = BTreeSet::new();
    for a in &c.allocations {
        if !allocation_keys.insert(a.result_iterator)
            || !c
                .result_iterators
                .iter()
                .any(|r| r.identity == a.result_iterator && publishable(c, r))
            || a.speech_iterator == 0
            || !retained_ids.insert(a.speech_iterator)
            || a.speech_wait.is_some() != speech_waits(c)
        {
            return Err(LedgerError::InvalidContext);
        }
        if let Some(wait) = a.speech_wait {
            if wait == 0 || !retained_ids.insert(wait) {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    let mut expected = Vec::new();
    for id in &c.result_resume_order {
        if let Some(a) = c.allocations.iter().find(|a| a.result_iterator == *id) {
            expected.push(a.speech_iterator);
            if speech_waits(c) {
                expected.push(a.speech_iterator);
            }
        }
    }
    if !scheduled && expected != c.speech_resume_order {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

fn barrier(events: &mut Vec<Event>, owner: Identity, field: &str, value: Identity) {
    events.push(Event::Barrier {
        owner,
        field: field.into(),
        value,
    });
}

fn active(c: &mut Context, events: &mut Vec<Event>, object: Identity, value: bool) {
    c.ui.objects
        .iter_mut()
        .find(|o| o.identity == object)
        .unwrap()
        .active = value;
    events.push(Event::SetActive {
        object,
        active: value,
    });
}

/// Internal retained stepper for explicit replay and separately scheduled steps.
/// Construction checks the original complete object/allocation contract. The
/// caller supplies timing/ownership separately; stepping does not establish UI
/// observation availability or legal gameplay requests.
#[derive(Debug, Clone)]
pub(super) struct PublicationStepper {
    replay: Replay,
}

impl PublicationStepper {
    pub(super) fn new(input: &Context) -> Result<Self, LedgerError> {
        validate(input, false)?;
        Ok(Self::validated(input))
    }

    pub(super) fn new_scheduled(input: &Context) -> Result<Self, LedgerError> {
        validate(input, true)?;
        Ok(Self::validated(input))
    }

    fn validated(input: &Context) -> Self {
        Self {
            replay: Replay {
                context: input.clone(),
                speech_iterators: Vec::new(),
                waits: input.result_waits.clone(),
                events: Vec::new(),
            },
        }
    }

    pub(super) fn snapshot(&self) -> &Replay {
        &self.replay
    }

    pub(super) fn into_replay(self) -> Replay {
        self.replay
    }

    fn result_prefix(&mut self, id: Identity) -> Result<Option<Identity>, LedgerError> {
        let Replay {
            context: c,
            speech_iterators: speeches,
            events,
            ..
        } = &mut self.replay;
        let index = c
            .result_iterators
            .iter()
            .position(|r| r.identity == id)
            .ok_or(LedgerError::InvalidContext)?;
        if c.result_iterators[index].state != 1 {
            return Err(LedgerError::InvalidContext);
        }
        let r = c.result_iterators[index].clone();
        c.result_iterators[index].state = -1;
        events.push(Event::State {
            iterator: id,
            state: -1,
        });
        if !c.act {
            events.push(Event::Return {
                iterator: id,
                value: false,
            });
            return Ok(None);
        }
        let info = c
            .infos
            .iter()
            .find(|i| Some(i.identity) == r.info)
            .unwrap()
            .clone();
        let empty = info
            .description
            .is_none_or(|text| string(c, text).unwrap().units.is_empty());
        events.push(Event::StringEmpty {
            text: info.description,
            empty,
        });
        if empty {
            events.push(Event::Return {
                iterator: id,
                value: false,
            });
            return Ok(None);
        }
        let description = info.description.unwrap();
        if let Some(callback) = c.preappend_callback {
            events.push(Event::Preappend {
                callback,
                info: info.identity,
                trigger_bits: r.trigger_bits,
            });
        }
        c.actor.info_version = c.actor.info_version.wrapping_add(1);
        events.push(Event::HistoryVersion {
            list: c.history.identity,
            version: c.actor.info_version,
        });
        let slot = c.actor.infos.len();
        c.actor.infos.push(Some(info.identity));
        events.push(Event::HistoryAppend {
            list: c.history.identity,
            info: info.identity,
            slot,
        });
        barrier(events, c.history.backing_array, "element", info.identity);
        if r.trigger_bits == 30 {
            let previous_bits = c.actor.uses as u32;
            let current_bits = previous_bits.wrapping_sub(1);
            c.actor.uses = current_bits as i32;
            events.push(Event::Uses {
                previous_bits,
                current_bits,
            });
        }
        if let Some(callback) = c.info_revealed_callback {
            events.push(Event::InfoRevealed {
                callback,
                actor: c.actor.identity,
            });
        }
        let allocation = c
            .allocations
            .iter()
            .find(|a| a.result_iterator == id)
            .unwrap();
        let speech = allocation.speech_iterator;
        events.push(Event::AllocateSpeech {
            iterator: speech,
            result_iterator: id,
        });
        events.push(Event::CaptureActor {
            iterator: speech,
            actor: c.actor.identity,
        });
        events.push(Event::State {
            iterator: speech,
            state: 0,
        });
        barrier(events, speech, "actor", c.actor.identity);
        events.push(Event::CaptureDescription {
            iterator: speech,
            text: description,
        });
        barrier(events, speech, "description", description);
        speeches.push(SpeechIterator {
            identity: speech,
            actor: c.actor.identity,
            description,
            state: 0,
            current: None,
        });
        events.push(Event::RegisterSpeech {
            actor: c.actor.identity,
            iterator: speech,
        });
        Ok(Some(speech))
    }

    fn result_tail(&mut self, id: Identity) {
        let Replay {
            context: c, events, ..
        } = &mut self.replay;
        if c.actor.uses == 0 {
            let pickable = c.ui.pickable;
            active(c, events, pickable, false);
        }
        events.push(Event::Return {
            iterator: id,
            value: false,
        });
    }

    pub(super) fn resume_result(
        &mut self,
        id: Identity,
        immediate_speech_start: bool,
    ) -> Result<Option<WaitObject>, LedgerError> {
        let Some(speech) = self.result_prefix(id)? else {
            return Ok(None);
        };
        let wait = if immediate_speech_start {
            self.resume_speech(speech)?
        } else {
            None
        };
        // Native StartCoroutine is synchronous: its nested first step completes
        // before the result's picker-hide and final return.
        self.result_tail(id);
        Ok(wait)
    }

    pub(super) fn resume_speech(
        &mut self,
        id: Identity,
    ) -> Result<Option<WaitObject>, LedgerError> {
        let Replay {
            context: c,
            speech_iterators: speeches,
            waits,
            events,
        } = &mut self.replay;
        let index = speeches
            .iter()
            .position(|s| s.identity == id)
            .ok_or(LedgerError::InvalidContext)?;
        if !matches!(speeches[index].state, 0 | 1) {
            return Err(LedgerError::InvalidContext);
        }
        let first = speeches[index].state == 0;
        speeches[index].state = -1;
        events.push(Event::State {
            iterator: id,
            state: -1,
        });
        if first {
            if c.trailer_mode {
                events.push(Event::TrailerLookup {
                    actor_id: c.actor.id,
                    text: c.trailer_text,
                });
                let empty = c
                    .trailer_text
                    .is_none_or(|text| string(c, text).unwrap().units.is_empty());
                events.push(Event::StringEmpty {
                    text: c.trailer_text,
                    empty,
                });
                if !empty {
                    events.push(Event::TrailerLookup {
                        actor_id: c.actor.id,
                        text: c.trailer_text,
                    });
                    speeches[index].description = c.trailer_text.unwrap();
                    events.push(Event::CaptureDescription {
                        iterator: id,
                        text: speeches[index].description,
                    });
                    barrier(events, id, "description", speeches[index].description);
                }
            }
            let text = speeches[index].description;
            events.push(Event::GameObject {
                component: c.ui.acted_component,
                result: Some(c.ui.first_game_object),
            });
            events.push(Event::Name {
                object: c.ui.first_game_object,
                text: c.ui.name_text,
            });
            events.push(Event::ConcatLog {
                name: c.ui.name_text,
                result: c.ui.log_text,
            });
            events.push(Event::GameObject {
                component: c.ui.acted_component,
                result: c.ui.log_game_object,
            });
            events.push(Event::Log {
                text: c.ui.log_text,
                context: c.ui.log_game_object,
            });
            c.ui.blank_text_value = Some(text);
            events.push(Event::SetText {
                blank: c.ui.blank_text,
                text,
            });
            let previous = c.actor.saved_act.replace(text);
            events.push(Event::SaveSpeech {
                previous,
                current: text,
            });
            barrier(events, c.actor.identity, "saved_act", text);
            if !matches!(c.actor.state, 20 | 30) && !c.actor.revealed {
                events.push(Event::UnityNull {
                    object: c.actor.bluff,
                    result: !c.raw_bluff.as_ref().is_some_and(|b| b.live),
                });
            }
            events.push(Event::SelectData {
                asset: selected_data(c),
            });
            if speech_waits(c) {
                let wait = c
                    .allocations
                    .iter()
                    .find(|a| a.speech_iterator == id)
                    .unwrap()
                    .speech_wait
                    .unwrap();
                events.push(Event::AllocateWait { wait });
                waits.push(WaitObject {
                    identity: wait,
                    seconds_bits: SPEECH_WAIT_BITS,
                });
                events.push(Event::ConstructWait {
                    wait,
                    bits: SPEECH_WAIT_BITS,
                });
                speeches[index].current = Some(wait);
                events.push(Event::Current { iterator: id, wait });
                barrier(events, id, "current", wait);
                speeches[index].state = 1;
                events.push(Event::State {
                    iterator: id,
                    state: 1,
                });
                events.push(Event::Return {
                    iterator: id,
                    value: true,
                });
                return Ok(Some(WaitObject {
                    identity: wait,
                    seconds_bits: SPEECH_WAIT_BITS,
                }));
            }
        }
        let text = speeches[index].description;
        events.push(Event::GameObject {
            component: c.ui.acted_component,
            result: Some(c.ui.show_game_object),
        });
        let game = c.ui.show_game_object;
        active(c, events, game, true);
        events.push(Event::Show {
            version: c.ui.acted_version,
            text,
        });
        for rect in &c.ui.layouts {
            events.push(Event::Rebuild { rect: *rect });
        }
        events.push(Event::Return {
            iterator: id,
            value: false,
        });
        Ok(None)
    }
}

pub fn replay(input: &Context) -> Result<Replay, LedgerError> {
    let mut stepper = PublicationStepper::new(input)?;
    for id in &input.result_resume_order {
        stepper.resume_result(*id, false)?;
    }
    for id in &input.speech_resume_order {
        stepper.resume_speech(*id)?;
    }
    Ok(stepper.into_replay())
}

#[cfg(test)]
#[path = "character_role_publication_tests.rs"]
mod tests;
