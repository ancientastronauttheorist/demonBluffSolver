//! Offline iterator field replay; yielded waits are descriptors, not scheduled records.
//! No elapsed time, coroutine acquisition, engine ownership or cancellation is inferred.
use serde::{Deserialize, Serialize};

pub const GAMEPLAY_ITERATOR_NATIVE_V1: &str = "gameplay_iterator_native_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameplayIteratorKind {
    DelayedDeckIntro,
    InitCoroutine,
    SetupDelay,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameplayIteratorState {
    pub state: i32,
    /// Opaque reference label; zero means null. Completion does not clear it.
    pub current: u64,
    /// Only SetupDelay factories capture their receiver. No ownership is implied.
    pub captured_receiver: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum GameplayIteratorOperation {
    Factory {
        kind: GameplayIteratorKind,
        receiver: u64,
    },
    DelayedDeckMoveNext,
    Reset {
        kind: GameplayIteratorKind,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IteratorGateway {
    Allocate,
    WaitCtor,
    BlindDeck,
    ClassInit,
    ChangeState,
    ExceptionCtor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameplayIteratorContext {
    pub rule_version: String,
    pub metadata_initialized: bool,
    /// Services preserve supplied iterator fields, including during failure.
    pub services_preserve_iterator: bool,
    pub initial: GameplayIteratorState,
    pub operation: GameplayIteratorOperation,
    /// Fresh nonzero opaque allocation label. Known reference aliases are rejected
    /// only when an allocation is reached; no runtime pointer is synthesized.
    pub allocation_id: u64,
    /// Integer returned at the resumption-time Settings gateway, not at creation.
    pub blind_deck: i32,
    pub gameplay_initialized: bool,
    pub fail_at: Option<IteratorGateway>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "gateway", rename_all = "snake_case")]
pub enum IteratorFailure {
    Gateway(IteratorGateway),
    NotSupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum IteratorEvent {
    Allocate,
    WaitCtor { duration_f32_bits: u32 },
    Barrier,
    BlindDeck,
    ClassInit,
    ChangeState { value: i32 },
    ExceptionCtor,
    Throw { iterator_kind: GameplayIteratorKind },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GameplayIteratorResult {
    /// Partial original iterator state. Factory writes are in created_iterator.
    pub iterator: GameplayIteratorState,
    pub created_iterator: Option<GameplayIteratorState>,
    pub created_identity: Option<u64>,
    pub move_next_result: Option<bool>,
    /// Exact WaitForSeconds constructor argument, without producer clock fields.
    pub yielded_wait_f32_bits: Option<u32>,
    pub requested_gameplay_state: Option<i32>,
    pub events: Vec<IteratorEvent>,
    pub failure: Option<IteratorFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidIteratorContext;

pub fn replay_gameplay_iterator(
    c: &GameplayIteratorContext,
) -> Result<GameplayIteratorResult, InvalidIteratorContext> {
    if c.rule_version != GAMEPLAY_ITERATOR_NATIVE_V1
        || !c.metadata_initialized
        || !c.services_preserve_iterator
    {
        return Err(InvalidIteratorContext);
    }
    let allocates = !matches!(c.operation, GameplayIteratorOperation::DelayedDeckMoveNext)
        || c.initial.state == 0;
    if allocates
        && (c.allocation_id == 0
            || c.allocation_id == c.initial.current
            || c.initial.captured_receiver == Some(c.allocation_id)
            || matches!(c.operation, GameplayIteratorOperation::Factory { receiver, .. } if c.allocation_id == receiver))
    {
        return Err(InvalidIteratorContext);
    }
    let mut out = GameplayIteratorResult {
        iterator: c.initial.clone(),
        created_iterator: None,
        created_identity: None,
        move_next_result: None,
        yielded_wait_f32_bits: None,
        requested_gameplay_state: None,
        events: vec![],
        failure: None,
    };
    macro_rules! gateway {
        ($kind:ident, $event:expr) => {{
            out.events.push($event);
            if c.fail_at == Some(IteratorGateway::$kind) {
                out.failure = Some(IteratorFailure::Gateway(IteratorGateway::$kind));
                return Ok(out);
            }
        }};
    }
    match c.operation {
        GameplayIteratorOperation::Factory { kind, receiver } => {
            gateway!(Allocate, IteratorEvent::Allocate);
            out.created_iterator = Some(GameplayIteratorState {
                state: 0,
                current: 0,
                captured_receiver: (kind == GameplayIteratorKind::SetupDelay).then_some(receiver),
            });
            out.created_identity = Some(c.allocation_id);
            if kind == GameplayIteratorKind::SetupDelay {
                out.events.push(IteratorEvent::Barrier);
            }
        }
        GameplayIteratorOperation::DelayedDeckMoveNext => match out.iterator.state {
            0 => {
                out.iterator.state = -1;
                gateway!(Allocate, IteratorEvent::Allocate);
                gateway!(
                    WaitCtor,
                    IteratorEvent::WaitCtor {
                        duration_f32_bits: 1.0f32.to_bits()
                    }
                );
                out.iterator.current = c.allocation_id;
                out.events.push(IteratorEvent::Barrier);
                out.iterator.state = 1;
                out.yielded_wait_f32_bits = Some(1.0f32.to_bits());
                out.move_next_result = Some(true);
            }
            1 => {
                out.iterator.state = -1;
                gateway!(BlindDeck, IteratorEvent::BlindDeck);
                if c.blind_deck != 1 {
                    if !c.gameplay_initialized {
                        gateway!(ClassInit, IteratorEvent::ClassInit);
                    }
                    out.requested_gameplay_state = Some(8);
                    gateway!(ChangeState, IteratorEvent::ChangeState { value: 8 });
                }
                out.move_next_result = Some(false);
            }
            _ => out.move_next_result = Some(false),
        },
        GameplayIteratorOperation::Reset { kind } => {
            gateway!(Allocate, IteratorEvent::Allocate);
            gateway!(ExceptionCtor, IteratorEvent::ExceptionCtor);
            out.events.push(IteratorEvent::Throw {
                iterator_kind: kind,
            });
            out.failure = Some(IteratorFailure::NotSupported);
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "gameplay_iterator_tests.rs"]
mod tests;
