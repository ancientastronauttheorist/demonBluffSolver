//! Offline TutorialsController.OnEnable/OnDisable registration callers only.
//! Combine/Remove and pointer-width casts use independently supplied outcomes;
//! this does not implement CLR multicast semantics or invoke any handler. Native
//! class-header equality is exact for plain Action. Only verified cast failures
//! may return a stopped partial replay; arbitrary runtime stops are rejected.

use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const TUTORIAL_EVENT_WIRING_NATIVE_V1: &str = "tutorial_event_wiring_native_v1";
const MAX_CALLS: usize = 3;
const MAX_RETAINED: usize = 16_384;
const MAX_SNAPSHOT_WORK: usize = 262_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Enable,
    Disable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Flavor {
    Action,
    Character,
    ShowTutorial,
    CloseTutorial,
}

impl Flavor {
    pub fn metadata_name(self) -> &'static str {
        match self {
            Self::Action => "System.Action_TypeInfo",
            Self::Character => "System.Action<Character>_TypeInfo",
            Self::ShowTutorial => "System.Action<ETutorialType, Transform>_TypeInfo",
            Self::CloseTutorial => "System.Action<ETutorialType>_TypeInfo",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Slot {
    GameStart,
    CharacterRevealed,
    CharacterInfoRevealed,
    CharacterKilled,
    ShowTutorial,
    CloseTutorial,
    StartNewLevel,
}

pub const ORDER: [Slot; 7] = [
    Slot::GameStart,
    Slot::CharacterRevealed,
    Slot::CharacterInfoRevealed,
    Slot::CharacterKilled,
    Slot::ShowTutorial,
    Slot::CloseTutorial,
    Slot::StartNewLevel,
];

impl Slot {
    pub fn offset(self) -> u32 {
        match self {
            Self::GameStart => 0,
            Self::CharacterRevealed => 0x50,
            Self::CharacterInfoRevealed => 0x58,
            Self::CharacterKilled => 0x48,
            Self::ShowTutorial => 0x90,
            Self::CloseTutorial => 0x98,
            Self::StartNewLevel => 0x28,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::GameStart => "OnGameStart",
            Self::CharacterRevealed => "OnCharacterRevealed",
            Self::CharacterInfoRevealed => "OnCharacterInfoRevealed",
            Self::CharacterKilled => "OnCharacterKilled",
            Self::ShowTutorial => "OnShowTutorial",
            Self::CloseTutorial => "OnCloseTutorial",
            Self::StartNewLevel => "OnStartNewLevel",
        }
    }
    pub fn handler(self) -> &'static str {
        match self {
            Self::GameStart => "StartTutorials",
            Self::CharacterRevealed => "OnCharacterReveal",
            Self::CharacterInfoRevealed => "CharacterInfoNote",
            Self::CharacterKilled => "CharacterKilledTutorial",
            Self::ShowTutorial => "EnableTutorial",
            Self::CloseTutorial => "CloseTutorialIfAble",
            Self::StartNewLevel => "LevelIdTutorial",
        }
    }
    pub fn flavor(self) -> Flavor {
        match self {
            Self::GameStart | Self::StartNewLevel => Flavor::Action,
            Self::CharacterRevealed | Self::CharacterInfoRevealed | Self::CharacterKilled => {
                Flavor::Character
            }
            Self::ShowTutorial => Flavor::ShowTutorial,
            Self::CloseTutorial => Flavor::CloseTutorial,
        }
    }
    fn index(self) -> usize {
        self.offset() as usize / 8
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Invocation {
    pub target: u64,
    pub method_info: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Delegate {
    pub identity: u64,
    pub class_header: u64,
    /// Supplied logical cast/constructor compatibility, not inferred from header.
    pub flavor: Flavor,
    pub invocations: Vec<Invocation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderBinding {
    pub flavor: Flavor,
    pub identity: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodBinding {
    pub slot: Slot,
    pub method_info: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    pub controller: u64,
    pub gameplay_class: u64,
    pub static_fields: u64,
    pub headers: Vec<HeaderBinding>,
    pub other_class_headers: Vec<u64>,
    pub methods: Vec<MethodBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    /// All 29 fields at offsets 0x00..0xE0 in eight-byte increments. The seven
    /// consumed slots resolve physical delegates; other pointers are opaque.
    pub slots: [Option<u64>; 29],
    pub delegates: Vec<Delegate>,
    pub allocation_order: Vec<u64>,
    pub enable_metadata_byte: u8,
    pub disable_metadata_byte: u8,
    /// Retained exact DWORD; neither native caller reads it or initializes it.
    pub class_initialized_word: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceOutcome {
    pub pointer: Option<u64>,
    /// Optional fresh physical result produced by the supplied Combine/Remove.
    pub created: Option<Delegate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CastOutcome {
    NotCalled,
    /// Actual RAX pointer-width result. Normal V1 casts retain the raw token;
    /// null returns represent the separately verified native failure boundary.
    Return {
        pointer: Option<u64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registration {
    pub slot: Slot,
    /// Supplied allocation/header; native constructor publishes the exact own
    /// receiver/handler invocation, so initial invocations must be empty.
    pub allocation: Delegate,
    pub outcome: ServiceOutcome,
    pub first_cast: CastOutcome,
    pub second_cast: CastOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub operation: Operation,
    pub registrations: Vec<Registration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Completion {
    Normal,
    NativeCastFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub runtime_metadata_and_bindings_verified: bool,
    pub allocation_and_constructor_verified: bool,
    pub combine_remove_outcomes_verified: bool,
    pub pointer_cast_outcomes_verified: bool,
    pub bindings_headers_and_other_services_stable: bool,
    pub normal_completion_verified: bool,
    pub native_cast_failure_boundary_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub runtime: Runtime,
    pub state: State,
    pub calls: Vec<Call>,
    pub completion: Completion,
    pub services: Services,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailurePhase {
    PlainHeader,
    FirstGenericCast,
    SecondGenericCast,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Metadata {
        name: String,
    },
    PublishMetadata {
        operation: Operation,
        byte: u8,
    },
    Allocate {
        token: u64,
        flavor: Flavor,
    },
    Construct {
        token: u64,
        receiver: u64,
        method_info: u64,
        flavor: Flavor,
    },
    CombineRemove {
        operation: Operation,
        left: Option<u64>,
        right: u64,
        result: Option<u64>,
    },
    HeaderGate {
        token: u64,
        expected: u64,
        accepted: bool,
    },
    Cast {
        token: u64,
        expected: u64,
        pointer: Option<u64>,
    },
    Publish {
        slot: Slot,
        pointer: Option<u64>,
    },
    Barrier {
        slot: Slot,
        pointer: Option<u64>,
    },
    NativeCastFailure {
        slot: Slot,
        token: u64,
        expected: u64,
        phase: FailurePhase,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Step {
    pub event: Event,
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CallResult {
    pub operation: Operation,
    pub returned: bool,
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub state: State,
    pub steps: Vec<Step>,
    pub calls: Vec<CallResult>,
    pub stopped: Option<(Slot, FailurePhase)>,
}

fn header(r: &Runtime, flavor: Flavor) -> u64 {
    r.headers
        .iter()
        .find(|h| h.flavor == flavor)
        .unwrap()
        .identity
}
fn method(r: &Runtime, slot: Slot) -> u64 {
    r.methods
        .iter()
        .find(|m| m.slot == slot)
        .unwrap()
        .method_info
}

fn delegate_valid(d: &Delegate, r: &Runtime) -> bool {
    d.identity != 0
        && d.class_header != 0
        && (d.class_header == header(r, d.flavor)
            || r.other_class_headers.contains(&d.class_header))
        && d.invocations.iter().all(|i| i.method_info != 0)
}

fn failure(reg: &Registration, raw: Option<&Delegate>, r: &Runtime) -> Option<FailurePhase> {
    let raw = raw?;
    if reg.slot.flavor() == Flavor::Action {
        (raw.class_header != header(r, Flavor::Action)).then_some(FailurePhase::PlainHeader)
    } else if reg.first_cast == (CastOutcome::Return { pointer: None }) {
        Some(FailurePhase::FirstGenericCast)
    } else if reg.second_cast == (CastOutcome::Return { pointer: None }) {
        Some(FailurePhase::SecondGenericCast)
    } else {
        None
    }
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    // Bound all retained definitions, invocation slots and every possible full
    // state snapshot before constructing maps or cloning replay state.
    let mut retained = 29usize;
    let mut steps = 0usize;
    for n in [
        c.runtime.headers.len(),
        c.runtime.other_class_headers.len(),
        c.runtime.methods.len(),
        c.state.delegates.len(),
        c.state.allocation_order.len(),
        c.calls.len(),
    ] {
        retained = retained.checked_add(n).ok_or(LedgerError::Capacity)?;
    }
    for d in &c.state.delegates {
        retained = retained
            .checked_add(d.invocations.len())
            .ok_or(LedgerError::Capacity)?;
    }
    for call in &c.calls {
        steps = steps.checked_add(13).ok_or(LedgerError::Capacity)?;
        for reg in &call.registrations {
            retained = retained
                .checked_add(3 + reg.allocation.invocations.len())
                .ok_or(LedgerError::Capacity)?;
            if let Some(d) = &reg.outcome.created {
                retained = retained
                    .checked_add(1 + d.invocations.len())
                    .ok_or(LedgerError::Capacity)?;
            }
            steps = steps.checked_add(8).ok_or(LedgerError::Capacity)?;
        }
    }
    let snapshots = steps
        .checked_add(c.calls.len() + 2)
        .ok_or(LedgerError::Capacity)?;
    let work = retained
        .checked_mul(snapshots)
        .ok_or(LedgerError::Capacity)?;
    if retained > MAX_RETAINED || work > MAX_SNAPSHOT_WORK || c.calls.len() > MAX_CALLS {
        return Err(LedgerError::Capacity);
    }
    let s = &c.services;
    if c.version != TUTORIAL_EVENT_WIRING_NATIVE_V1
        || c.calls.is_empty()
        || !s.runtime_metadata_and_bindings_verified
        || !s.allocation_and_constructor_verified
        || !s.combine_remove_outcomes_verified
        || !s.pointer_cast_outcomes_verified
        || !s.bindings_headers_and_other_services_stable
        || (c.completion == Completion::Normal && !s.normal_completion_verified)
        || (c.completion == Completion::NativeCastFailure
            && (!s.native_cast_failure_boundary_verified || s.normal_completion_verified))
        || c.runtime.headers.len() != 4
        || c.runtime.methods.len() != 7
    {
        return Err(LedgerError::InvalidContext);
    }
    let kinds: BTreeSet<_> = c.runtime.headers.iter().map(|h| h.flavor).collect();
    let methods: BTreeSet<_> = c.runtime.methods.iter().map(|m| m.slot).collect();
    if kinds.len() != 4 || methods.len() != 7 {
        return Err(LedgerError::InvalidContext);
    }
    let mut reserved = BTreeSet::new();
    for id in [
        c.runtime.controller,
        c.runtime.gameplay_class,
        c.runtime.static_fields,
    ]
    .into_iter()
    .chain(c.runtime.headers.iter().map(|h| h.identity))
    .chain(c.runtime.methods.iter().map(|m| m.method_info))
    .chain(c.runtime.other_class_headers.iter().copied())
    {
        if id == 0 || !reserved.insert(id) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let mut nodes: BTreeMap<_, _> = BTreeMap::new();
    for d in &c.state.delegates {
        if !delegate_valid(d, &c.runtime)
            || d.invocations.is_empty()
            || reserved.contains(&d.identity)
            || nodes.insert(d.identity, d.clone()).is_some()
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    for slot in ORDER {
        if c.state.slots[slot.index()]
            .is_some_and(|id| nodes.get(&id).is_none_or(|d| d.flavor != slot.flavor()))
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if c.state.slots.contains(&Some(0))
        || c.state
            .allocation_order
            .iter()
            .any(|id| !nodes.contains_key(id))
        || c.state
            .allocation_order
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != c.state.allocation_order.len()
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut all_retained = reserved.clone();
    all_retained.extend(nodes.keys().copied());
    all_retained.extend(c.state.slots.iter().flatten().copied());
    for d in &c.state.delegates {
        all_retained.extend(d.invocations.iter().flat_map(|i| [i.target, i.method_info]));
    }
    let mut planned = BTreeSet::new();
    for call in &c.calls {
        for reg in &call.registrations {
            for d in std::iter::once(&reg.allocation).chain(reg.outcome.created.iter()) {
                if !delegate_valid(d, &c.runtime)
                    || all_retained.contains(&d.identity)
                    || !planned.insert(d.identity)
                {
                    return Err(LedgerError::InvalidContext);
                }
            }
        }
    }
    // Outcome records themselves may retain arbitrary verified foreign targets
    // and MethodInfos; fresh allocations must not alias any such references.
    for call in &c.calls {
        for d in call
            .registrations
            .iter()
            .filter_map(|reg| reg.outcome.created.as_ref())
        {
            if d.invocations
                .iter()
                .any(|i| planned.contains(&i.target) || planned.contains(&i.method_info))
            {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    let mut slots = c.state.slots;
    let mut stopped = false;
    for (call_index, call) in c.calls.iter().enumerate() {
        if call.registrations.is_empty() || call.registrations.len() > 7 || stopped {
            return Err(LedgerError::InvalidContext);
        }
        for (index, reg) in call.registrations.iter().enumerate() {
            if reg.slot != ORDER[index]
                || !reg.allocation.invocations.is_empty()
                || reg.allocation.flavor != reg.slot.flavor()
            {
                return Err(LedgerError::InvalidContext);
            }
            let mut allocated = reg.allocation.clone();
            allocated.invocations.push(Invocation {
                target: c.runtime.controller,
                method_info: method(&c.runtime, reg.slot),
            });
            nodes.insert(allocated.identity, allocated);
            if let Some(created) = &reg.outcome.created {
                if reg.outcome.pointer != Some(created.identity) || created.invocations.is_empty() {
                    return Err(LedgerError::InvalidContext);
                }
                nodes.insert(created.identity, created.clone());
            }
            let raw = reg
                .outcome
                .pointer
                .map(|id| nodes.get(&id).ok_or(LedgerError::InvalidContext))
                .transpose()?;
            if raw.is_some_and(|d| d.flavor != reg.slot.flavor()) {
                return Err(LedgerError::InvalidContext);
            }
            if reg.slot.flavor() == Flavor::Action || raw.is_none() {
                if reg.first_cast != CastOutcome::NotCalled
                    || reg.second_cast != CastOutcome::NotCalled
                {
                    return Err(LedgerError::InvalidContext);
                }
            } else {
                if reg.first_cast
                    != (CastOutcome::Return {
                        pointer: reg.outcome.pointer,
                    })
                    && reg.first_cast != (CastOutcome::Return { pointer: None })
                {
                    return Err(LedgerError::InvalidContext);
                }
                let expected_second = if reg.first_cast == (CastOutcome::Return { pointer: None }) {
                    CastOutcome::NotCalled
                } else {
                    CastOutcome::Return {
                        pointer: reg.outcome.pointer,
                    }
                };
                if reg.second_cast != expected_second
                    && !(expected_second != CastOutcome::NotCalled
                        && reg.second_cast == (CastOutcome::Return { pointer: None }))
                {
                    return Err(LedgerError::InvalidContext);
                }
            }
            if let Some(phase) = failure(reg, raw, &c.runtime) {
                if c.completion != Completion::NativeCastFailure
                    || call_index + 1 != c.calls.len()
                    || index + 1 != call.registrations.len()
                {
                    return Err(LedgerError::InvalidContext);
                }
                if phase == FailurePhase::SecondGenericCast {
                    slots[reg.slot.index()] = reg.outcome.pointer;
                }
                stopped = true;
                break;
            }
            slots[reg.slot.index()] = reg.outcome.pointer;
        }
        if !stopped && call.registrations.len() != 7 {
            return Err(LedgerError::InvalidContext);
        }
    }
    if stopped != (c.completion == Completion::NativeCastFailure) {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

fn step(out: &mut Replay, event: Event) {
    out.steps.push(Step {
        event,
        state: out.state.clone(),
    });
}

fn metadata_names() -> Vec<String> {
    let mut names: Vec<_> = [
        Flavor::CloseTutorial,
        Flavor::Character,
        Flavor::ShowTutorial,
        Flavor::Action,
    ]
    .into_iter()
    .map(|f| f.metadata_name().into())
    .collect();
    names.push("GameplayEvents_TypeInfo".into());
    for slot in [
        Slot::CharacterInfoRevealed,
        Slot::CharacterKilled,
        Slot::CloseTutorial,
        Slot::ShowTutorial,
        Slot::StartNewLevel,
        Slot::CharacterRevealed,
        Slot::GameStart,
    ] {
        names.push(format!("Method$TutorialsController.{}()", slot.handler()));
    }
    names
}

pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut out = Replay {
        state: c.state.clone(),
        steps: vec![],
        calls: vec![],
        stopped: None,
    };
    for call in &c.calls {
        let byte = match call.operation {
            Operation::Enable => out.state.enable_metadata_byte,
            Operation::Disable => out.state.disable_metadata_byte,
        };
        if byte == 0 {
            for name in metadata_names() {
                step(&mut out, Event::Metadata { name });
            }
            match call.operation {
                Operation::Enable => out.state.enable_metadata_byte = 1,
                Operation::Disable => out.state.disable_metadata_byte = 1,
            }
            step(
                &mut out,
                Event::PublishMetadata {
                    operation: call.operation,
                    byte: 1,
                },
            );
        }
        for reg in &call.registrations {
            let flavor = reg.slot.flavor();
            let own = reg.allocation.identity;
            step(&mut out, Event::Allocate { token: own, flavor });
            out.state.delegates.push(reg.allocation.clone());
            out.state.allocation_order.push(own);
            let method_info = method(&c.runtime, reg.slot);
            step(
                &mut out,
                Event::Construct {
                    token: own,
                    receiver: c.runtime.controller,
                    method_info,
                    flavor,
                },
            );
            out.state
                .delegates
                .last_mut()
                .unwrap()
                .invocations
                .push(Invocation {
                    target: c.runtime.controller,
                    method_info,
                });
            let left = out.state.slots[reg.slot.index()];
            step(
                &mut out,
                Event::CombineRemove {
                    operation: call.operation,
                    left,
                    right: own,
                    result: reg.outcome.pointer,
                },
            );
            if let Some(created) = &reg.outcome.created {
                out.state.delegates.push(created.clone());
            }
            let raw = reg
                .outcome
                .pointer
                .and_then(|id| out.state.delegates.iter().find(|d| d.identity == id));
            let failed = failure(reg, raw, &c.runtime);
            let expected = header(&c.runtime, flavor);
            if let Some(token) = reg.outcome.pointer {
                if flavor == Flavor::Action {
                    step(
                        &mut out,
                        Event::HeaderGate {
                            token,
                            expected,
                            accepted: failed.is_none(),
                        },
                    );
                } else {
                    let CastOutcome::Return { pointer } = reg.first_cast else {
                        unreachable!()
                    };
                    step(
                        &mut out,
                        Event::Cast {
                            token,
                            expected,
                            pointer,
                        },
                    );
                }
                if matches!(
                    failed,
                    Some(FailurePhase::PlainHeader | FailurePhase::FirstGenericCast)
                ) {
                    let phase = failed.unwrap();
                    step(
                        &mut out,
                        Event::NativeCastFailure {
                            slot: reg.slot,
                            token,
                            expected,
                            phase,
                        },
                    );
                    out.stopped = Some((reg.slot, phase));
                    break;
                }
            }
            // First generic cast identity is constrained to raw in V1. Native
            // publication precedes the second cast on the original raw token.
            out.state.slots[reg.slot.index()] = reg.outcome.pointer;
            step(
                &mut out,
                Event::Publish {
                    slot: reg.slot,
                    pointer: reg.outcome.pointer,
                },
            );
            if let Some(token) = reg.outcome.pointer {
                if flavor == Flavor::Action {
                    step(
                        &mut out,
                        Event::HeaderGate {
                            token,
                            expected,
                            accepted: true,
                        },
                    );
                } else {
                    let CastOutcome::Return { pointer } = reg.second_cast else {
                        unreachable!()
                    };
                    step(
                        &mut out,
                        Event::Cast {
                            token,
                            expected,
                            pointer,
                        },
                    );
                    if let Some(phase) = failed {
                        step(
                            &mut out,
                            Event::NativeCastFailure {
                                slot: reg.slot,
                                token,
                                expected,
                                phase,
                            },
                        );
                        out.stopped = Some((reg.slot, phase));
                        break;
                    }
                }
            }
            step(
                &mut out,
                Event::Barrier {
                    slot: reg.slot,
                    pointer: reg.outcome.pointer,
                },
            );
        }
        out.calls.push(CallResult {
            operation: call.operation,
            returned: out.stopped.is_none(),
            state: out.state.clone(),
        });
        if out.stopped.is_some() {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "tutorial_event_wiring_tests.rs"]
mod tests;
