//! Bind one weighted Reveal drain to audited, explicit clock transitions.
//!
//! This is an offline composition boundary. It does not choose a PlayerLoop,
//! infer operating-system timestamps, or model clock mutation inside callbacks.

use super::clock::{
    checked, select_fixed, update_frame, ClockContext, ClockState, FrameUpdatePath,
};
use super::ledger::LedgerError;
use super::scheduled_reveal::{
    replay_scheduled_reveal, RevealCallbackBoundary, ScheduledRevealContext, ScheduledRevealPath,
    ScheduledRevealState, SCHEDULED_REVEAL_NATIVE_V1,
};
use super::wait_eligibility::{WaitDispatchContext, UNITY_WAIT_ELIGIBILITY_NATIVE_V1};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CLOCKED_REVEAL_NATIVE_V1: &str = "clocked_delay_reveal_native_v1";
const MAX_TRANSITIONS: usize = 256;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum ClockTransition {
    UpdateFrame { timestamp: f64 },
    SelectFixed,
}

/// Owner and lifetime facts stay explicit; timestamps are derived from the
/// audited clock state instead of supplied independently for each callback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockedCallbackBoundary {
    pub same_live_owner: bool,
    pub callback_result: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockedRevealContext {
    pub rule_version: String,
    pub initial_clock: ClockContext,
    pub initial: ScheduledRevealState,
    /// Actual clock-operation order before this one drain, not a proposed loop.
    pub transitions: Vec<ClockTransition>,
    pub phase_mask: u32,
    /// Required provenance for the entire synchronous drain, including writer
    /// callbacks. An unsupported clock mutation rejects the complete replay.
    pub clock_stable_during_callbacks: bool,
    pub callbacks: BTreeMap<u64, ClockedCallbackBoundary>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum ClockTransitionResult {
    UpdateFrame {
        path: FrameUpdatePath,
        state: ClockState,
    },
    SelectFixed {
        selected_fixed: bool,
        state: ClockState,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ClockedRevealResult {
    /// All weighted paths share this deterministic clock history. It is stored
    /// once so branch expansion does not duplicate up to 256 clock snapshots.
    pub clock: ClockContext,
    pub clock_trace: Vec<ClockTransitionResult>,
    pub dispatch: WaitDispatchContext,
    pub paths: Vec<ScheduledRevealPath>,
}

/// Apply the supplied clock operations, then execute one native-ordered drain.
/// Public time feeds consumption; retained frame time feeds every new wait.
/// Queue generation comes from the supplied queue itself. Any unsupported
/// weighted branch invalidates the whole call, preserving the original API's
/// atomic fallback contract and occurrence-sensitive RNG probabilities.
pub fn replay_clocked_reveal(
    context: &ClockedRevealContext,
) -> Result<ClockedRevealResult, LedgerError> {
    if context.rule_version != CLOCKED_REVEAL_NATIVE_V1 || !context.clock_stable_during_callbacks {
        return Err(LedgerError::InvalidContext);
    }
    if context.transitions.len() > MAX_TRANSITIONS {
        return Err(LedgerError::Capacity);
    }
    let mut clock = ClockContext {
        rule_version: context.initial_clock.rule_version.clone(),
        state: checked(&context.initial_clock)?,
    };
    let mut clock_trace = Vec::with_capacity(context.transitions.len());
    for transition in &context.transitions {
        match transition {
            ClockTransition::UpdateFrame { timestamp } => {
                let (state, path) = update_frame(&clock, *timestamp)?;
                clock_trace.push(ClockTransitionResult::UpdateFrame {
                    path,
                    state: state.clone(),
                });
                clock.state = state;
            }
            ClockTransition::SelectFixed => {
                let (state, selected_fixed) = select_fixed(&clock)?;
                clock_trace.push(ClockTransitionResult::SelectFixed {
                    selected_fixed,
                    state: state.clone(),
                });
                clock.state = state;
            }
        }
    }
    let dispatch = WaitDispatchContext {
        rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
        sampled_time: clock.state.public.time,
        sampled_frame_counter: clock.state.frame_counter,
        phase_mask: context.phase_mask,
        generation_before: context.initial.queue.generation,
    };
    let callbacks = context
        .callbacks
        .iter()
        .map(|(&id, boundary)| {
            (
                id,
                RevealCallbackBoundary {
                    same_live_owner: boundary.same_live_owner,
                    callback_result: boundary.callback_result,
                    producer_time: clock.state.frame.time,
                    producer_frame_counter: clock.state.frame_counter,
                },
            )
        })
        .collect();
    let paths = replay_scheduled_reveal(&ScheduledRevealContext {
        rule_version: SCHEDULED_REVEAL_NATIVE_V1.into(),
        initial: context.initial.clone(),
        dispatch: dispatch.clone(),
        callbacks,
    })?;
    Ok(ClockedRevealResult {
        clock,
        clock_trace,
        dispatch,
        paths,
    })
}
