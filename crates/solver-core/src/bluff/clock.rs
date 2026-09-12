//! Finite projection of the pinned UnityPlayer clock update and selector.
//!
//! Timestamps, initial fields and policy flags remain explicit offline inputs.
//! This does not infer operating-system time or drive a complete PlayerLoop.

use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};

pub const UNITY_CLOCK_NATIVE_V1: &str = "unity_clock_native_v1";
const MIN_DELTA: f32 = 0.00001;
const SCALE_EPSILON: f32 = 0.000001;

/// The 48 bytes copied between fixed, frame and public timing snapshots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingBlock {
    pub time: f64,
    pub previous_time: f64,
    pub unscaled_time: f64,
    pub delta: f32,
    pub unscaled_delta: f32,
    pub smooth_delta: f32,
    pub smooth_weight: f32,
    pub reciprocal_delta: f32,
    pub opaque_tail: u32,
}

impl TimingBlock {
    fn finite(&self) -> bool {
        [self.time, self.previous_time, self.unscaled_time]
            .iter()
            .all(|v| v.is_finite())
            && [
                self.delta,
                self.unscaled_delta,
                self.smooth_delta,
                self.smooth_weight,
                self.reciprocal_delta,
            ]
            .iter()
            .all(|v| v.is_finite())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockState {
    pub fixed: TimingBlock,
    pub frame: TimingBlock,
    pub public: TimingBlock,
    /// Native +0xC0; cleared only by its non-capture early-return path.
    pub skip_once: bool,
    /// Native +0xC1; uses a 0.02-second step unless capture takes precedence.
    pub first_frame: bool,
    pub first_fixed: bool,
    pub frame_counter: i64,
    pub rendered_counter: u32,
    pub capture_delta: f32,
    pub scaled_offset: f64,
    pub unscaled_offset: f64,
    /// Native +0xF8; counters still increment when this is set.
    pub suppress_update: bool,
    pub in_fixed_step: bool,
    pub time_scale: f32,
    pub maximum_delta: f32,
}

impl ClockState {
    fn finite(&self) -> bool {
        self.fixed.finite()
            && self.frame.finite()
            && self.public.finite()
            && self.scaled_offset.is_finite()
            && self.unscaled_offset.is_finite()
            && [self.capture_delta, self.time_scale, self.maximum_delta]
                .iter()
                .all(|v| v.is_finite())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockContext {
    pub rule_version: String,
    pub state: ClockState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameUpdatePath {
    Suppressed,
    SkipOnce,
    Capture,
    FirstStep,
    Maximum,
    Minimum,
    UnitScale,
    Scaled,
}

pub(super) fn checked(context: &ClockContext) -> Result<ClockState, LedgerError> {
    if context.rule_version != UNITY_CLOCK_NATIVE_V1 || !context.state.finite() {
        return Err(LedgerError::InvalidContext);
    }
    Ok(context.state.clone())
}

/// Update from one explicit native timestamp, retaining single-precision
/// rounding before promotion. Errors leave the caller's input untouched.
pub fn update_frame(
    context: &ClockContext,
    timestamp: f64,
) -> Result<(ClockState, FrameUpdatePath), LedgerError> {
    let mut state = checked(context)?;
    if !timestamp.is_finite() {
        return Err(LedgerError::InvalidContext);
    }
    state.frame_counter = state.frame_counter.wrapping_add(1);
    state.rendered_counter = state.rendered_counter.wrapping_add(1);
    if state.suppress_update {
        return Ok((state, FrameUpdatePath::Suppressed));
    }
    let unscaled = timestamp - state.unscaled_offset;
    let unscaled_delta = (unscaled - state.frame.unscaled_time) as f32;
    if !unscaled.is_finite() || !unscaled_delta.is_finite() {
        return Err(LedgerError::InvalidContext);
    }
    if unscaled_delta >= MIN_DELTA {
        state.frame.unscaled_time = unscaled;
        state.frame.unscaled_delta = unscaled_delta;
    } else {
        state.frame.unscaled_delta = MIN_DELTA;
    }
    let old = state.frame.time;
    let scale = state.time_scale;
    let (updated, path) = if state.capture_delta > 0.0 {
        (
            old + f64::from(state.capture_delta * scale),
            FrameUpdatePath::Capture,
        )
    } else if state.skip_once {
        state.skip_once = false;
        return Ok((state, FrameUpdatePath::SkipOnce));
    } else if state.first_frame {
        (old + f64::from(scale * 0.02), FrameUpdatePath::FirstStep)
    } else {
        let candidate = timestamp - state.scaled_offset;
        let elapsed = candidate - old;
        if !candidate.is_finite() || !elapsed.is_finite() {
            return Err(LedgerError::InvalidContext);
        }
        if elapsed > f64::from(state.maximum_delta) {
            (
                old + f64::from(state.maximum_delta * scale),
                FrameUpdatePath::Maximum,
            )
        } else if elapsed < f64::from(MIN_DELTA) {
            (old + f64::from(scale * MIN_DELTA), FrameUpdatePath::Minimum)
        } else if (scale - 1.0).abs() <= SCALE_EPSILON {
            // Native preserves the double candidate near unit scale.
            (candidate, FrameUpdatePath::UnitScale)
        } else {
            (
                old + f64::from(elapsed as f32 * scale),
                FrameUpdatePath::Scaled,
            )
        }
    };
    state.frame.time = updated;
    state.frame.previous_time = old;
    state.frame.delta = (updated - old) as f32;
    state.frame.reciprocal_delta = if state.frame.delta > MIN_DELTA {
        1.0 / state.frame.delta
    } else {
        1.0
    };
    state.frame.smooth_weight = state.frame.smooth_weight * 0.8 + 0.2;
    let blend = 0.2 / state.frame.smooth_weight;
    state.frame.smooth_delta = (1.0 - blend) * state.frame.smooth_delta + blend * state.frame.delta;
    state.public = state.frame.clone();
    state.scaled_offset = timestamp - updated;
    if state.first_frame {
        state.first_frame = false;
        // Public snapshot retains the pre-reset smoothing weight.
        state.frame.smooth_weight = 0.0;
    }
    if !state.finite() {
        return Err(LedgerError::InvalidContext);
    }
    Ok((state, path))
}

/// Choose the public timing block for a fixed step. The frame clock used by
/// WaitForSeconds production is retained, even when public time becomes fixed.
pub fn select_fixed(context: &ClockContext) -> Result<(ClockState, bool), LedgerError> {
    let mut state = checked(context)?;
    let candidate = state.fixed.time + f64::from(state.fixed.delta);
    if !candidate.is_finite() {
        return Err(LedgerError::InvalidContext);
    }
    if candidate > state.frame.time && !state.first_fixed {
        state.public = state.frame.clone();
        state.in_fixed_step = false;
        return Ok((state, false));
    }
    state.fixed.previous_time = state.fixed.time;
    if !state.first_fixed {
        state.fixed.time = candidate;
    }
    if state.time_scale != 0.0 {
        let unscaled = (state.fixed.time - state.frame.time) / f64::from(state.time_scale)
            + state.frame.unscaled_time;
        state.fixed.unscaled_delta = (unscaled - state.fixed.unscaled_time) as f32;
        state.fixed.unscaled_time = unscaled;
    }
    state.first_fixed = false;
    state.public = state.fixed.clone();
    state.in_fixed_step = true;
    if !state.finite() {
        return Err(LedgerError::InvalidContext);
    }
    Ok((state, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bluff::wait_eligibility::{
        evaluate_wait, make_wait_for_seconds, WaitDispatchContext, WaitEligibility,
        WaitForSecondsContext, UNITY_WAIT_ELIGIBILITY_NATIVE_V1,
    };

    fn report() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_clock_audit.json"
        ))
        .unwrap()
    }

    fn context(state: ClockState) -> ClockContext {
        ClockContext {
            rule_version: UNITY_CLOCK_NATIVE_V1.into(),
            state,
        }
    }

    fn sample() -> ClockState {
        serde_json::from_value(report()["selected_results"][0]["input"].clone()).unwrap()
    }

    #[test]
    fn update_matches_native_full_snapshot_fixtures() {
        let report = report();
        for fixture in report["selected_results"].as_array().unwrap() {
            let input: ClockState = serde_json::from_value(fixture["input"].clone()).unwrap();
            let output: ClockState = serde_json::from_value(fixture["output"].clone()).unwrap();
            let path: FrameUpdatePath = serde_json::from_value(fixture["path"].clone()).unwrap();
            assert_eq!(
                update_frame(&context(input), fixture["timestamp"].as_f64().unwrap()).unwrap(),
                (output, path)
            );
        }
    }

    #[test]
    fn fixed_selection_matches_native_full_snapshot_fixtures() {
        let report = report();
        for fixture in report["selector_fixtures"].as_array().unwrap() {
            let input = serde_json::from_value(fixture["input"].clone()).unwrap();
            let output = serde_json::from_value(fixture["output"].clone()).unwrap();
            assert_eq!(
                select_fixed(&context(input)).unwrap(),
                (output, fixture["selected_fixed"].as_bool().unwrap())
            );
        }
    }

    #[test]
    fn fixed_public_clock_does_not_replace_wait_producer_clock() {
        let mut input = sample();
        input.fixed.time = 8.0;
        input.fixed.delta = 0.5;
        input.frame.time = 10.0;
        let (selected, fixed) = select_fixed(&context(input)).unwrap();
        assert!(fixed);
        let wait = make_wait_for_seconds(&WaitForSecondsContext {
            rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
            duration: 0.0,
            producer_time: selected.frame.time,
            producer_frame_counter: selected.frame_counter,
            insertion_generation: 0,
        })
        .unwrap();
        assert_eq!(wait.deadline, 10.0);
        let dispatch = WaitDispatchContext {
            rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
            sampled_time: selected.public.time,
            sampled_frame_counter: selected.frame_counter.wrapping_add(1),
            phase_mask: 2,
            generation_before: 0,
        };
        assert_eq!(
            evaluate_wait(&wait, &dispatch).unwrap(),
            WaitEligibility::StopAtFutureDeadline
        );
    }

    #[test]
    fn suppression_can_clear_frame_gate_without_changing_public_time() {
        let mut input = sample();
        input.frame.time = 10.0;
        input.public.time = 10.0;
        input.suppress_update = true;
        let wait = make_wait_for_seconds(&WaitForSecondsContext {
            rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
            duration: 0.0,
            producer_time: input.frame.time,
            producer_frame_counter: input.frame_counter,
            insertion_generation: 0,
        })
        .unwrap();
        let (updated, _) = update_frame(&context(input), 500.0).unwrap();
        assert_eq!(updated.public.time, 10.0);
        assert_eq!(
            evaluate_wait(
                &wait,
                &WaitDispatchContext {
                    rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
                    sampled_time: updated.public.time,
                    sampled_frame_counter: updated.frame_counter,
                    phase_mask: 2,
                    generation_before: 0,
                }
            )
            .unwrap(),
            WaitEligibility::Eligible
        );
    }

    #[test]
    fn rejects_unknown_version_nonfinite_and_overflow_without_mutating_input() {
        let mut input = context(sample());
        input.rule_version = "unknown".into();
        assert!(update_frame(&input, 100.0).is_err());
        assert!(select_fixed(&input).is_err());
        input.rule_version = UNITY_CLOCK_NATIVE_V1.into();
        assert!(update_frame(&input, f64::NAN).is_err());
        input.state.frame.time = f64::INFINITY;
        assert!(select_fixed(&input).is_err());
        input.state = sample();
        input.state.capture_delta = f32::MAX;
        input.state.time_scale = f32::MAX;
        let original = input.clone();
        assert!(update_frame(&input, 100.0).is_err());
        assert_eq!(input, original);
    }

    #[test]
    fn serialized_context_requires_provenance() {
        let mut value = serde_json::to_value(context(sample())).unwrap();
        value.as_object_mut().unwrap().remove("rule_version");
        assert!(serde_json::from_value::<ClockContext>(value).is_err());
    }

    #[test]
    fn json_preserves_native_timestamp_bits() {
        let timestamp = 100.0 + f64::from(MIN_DELTA);
        let text = serde_json::to_string(&timestamp).unwrap();
        let parsed: f64 = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed.to_bits(), timestamp.to_bits());
        let original = context(sample());
        let parsed: ClockContext =
            serde_json::from_str(&serde_json::to_string(&original).unwrap()).unwrap();
        assert_eq!(parsed, original);
    }
}
