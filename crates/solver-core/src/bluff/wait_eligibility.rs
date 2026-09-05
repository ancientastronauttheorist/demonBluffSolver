//! Finite-clock projection of the fingerprinted UnityPlayer wait boundary.
//!
//! This evaluates one record at an explicitly supplied dispatch snapshot. It
//! does not infer the producer clock from Time.time, resolve native owners,
//! walk/mutate the queue, or admit any logical Reveal continuation by itself.

use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};

pub const UNITY_WAIT_ELIGIBILITY_NATIVE_V1: &str = "unity_wait_eligibility_native_v1";

/// The producer and consumer use different engine clock fields. This snapshot
/// must come from the producer boundary, not a substituted public time getter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaitForSecondsContext {
    pub rule_version: String,
    pub duration: f32,
    pub producer_time: f64,
    pub producer_frame_counter: i64,
    pub insertion_generation: u32,
}

/// Only the timing/generation fields of a nonrepeating native wait record.
/// Native payload pointers, owner keys and callback addresses are not modeled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaitTimingRecord {
    pub deadline: f64,
    pub frame_threshold: i64,
    pub phase_mask: u32,
    pub insertion_generation: u32,
}

/// Values sampled on entry to the native drain, before its generation increment.
/// `sampled_frame_counter` retains the full native signed 64-bit value even
/// though the public Time.frameCount getter exposes only the low 32 bits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaitDispatchContext {
    pub rule_version: String,
    pub sampled_time: f64,
    pub sampled_frame_counter: i64,
    pub phase_mask: u32,
    pub generation_before: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitEligibility {
    /// Native traversal terminates here; this is not just a skipped record.
    StopAtFutureDeadline,
    SkipPhase,
    SkipCurrentGeneration,
    SkipFutureFrame,
    /// Timing gates passed. Owner resolution and callback dispatch still follow.
    Eligible,
}

/// Project the native finite WaitForSeconds producer: promote its float
/// duration, add it to the producer double, retain the next signed frame value,
/// and tag the record with phase mask 0xA and the current generation.
pub fn make_wait_for_seconds(
    context: &WaitForSecondsContext,
) -> Result<WaitTimingRecord, LedgerError> {
    if context.rule_version != UNITY_WAIT_ELIGIBILITY_NATIVE_V1
        || !context.duration.is_finite()
        || !context.producer_time.is_finite()
    {
        return Err(LedgerError::InvalidContext);
    }
    // MINSD with DBL_MAX is identity on the supported finite producer clock.
    let deadline = context.producer_time + f64::from(context.duration);
    if !deadline.is_finite() {
        return Err(LedgerError::InvalidContext);
    }
    Ok(WaitTimingRecord {
        deadline,
        frame_threshold: context.producer_frame_counter.wrapping_add(1),
        phase_mask: 0xA,
        insertion_generation: context.insertion_generation,
    })
}

/// Evaluate gates in native order, with explicit fixed-width generation/frame
/// semantics. Nonfinite clocks/deadlines are outside this version's contract;
/// they are not silently reinterpreted as ready, missing, or infinitely delayed.
pub fn evaluate_wait(
    record: &WaitTimingRecord,
    context: &WaitDispatchContext,
) -> Result<WaitEligibility, LedgerError> {
    if context.rule_version != UNITY_WAIT_ELIGIBILITY_NATIVE_V1
        || !context.sampled_time.is_finite()
        || !record.deadline.is_finite()
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(if context.sampled_time < record.deadline {
        WaitEligibility::StopAtFutureDeadline
    } else if context.phase_mask & record.phase_mask == 0 {
        WaitEligibility::SkipPhase
    } else if record.insertion_generation == context.generation_before.wrapping_add(1) {
        WaitEligibility::SkipCurrentGeneration
    } else if record.frame_threshold > context.sampled_frame_counter {
        WaitEligibility::SkipFutureFrame
    } else {
        WaitEligibility::Eligible
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn producer() -> WaitForSecondsContext {
        WaitForSecondsContext {
            rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
            duration: 0.1,
            producer_time: 10.0,
            producer_frame_counter: 100,
            insertion_generation: 7,
        }
    }

    fn dispatch() -> WaitDispatchContext {
        WaitDispatchContext {
            rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
            sampled_time: 11.0,
            sampled_frame_counter: 101,
            phase_mask: 2,
            generation_before: 7,
        }
    }

    #[test]
    fn producer_promotes_float_duration_and_does_not_substitute_consumer_clock() {
        let record = make_wait_for_seconds(&producer()).unwrap();
        assert_eq!(record.deadline, 10.0 + f64::from(0.1_f32));
        assert_ne!(record.deadline, 10.1_f64);
        assert_eq!(record.frame_threshold, 101);
        assert_eq!(record.phase_mask, 0xA);
        assert_eq!(record.insertion_generation, 7);
        let mut at = dispatch();
        at.sampled_time = 10.1;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::StopAtFutureDeadline
        );
        at.sampled_time = record.deadline;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::Eligible
        );
    }

    #[test]
    fn zero_and_negative_durations_still_require_the_next_frame_threshold() {
        for duration in [0.0, -0.1] {
            let mut source = producer();
            source.duration = duration;
            let record = make_wait_for_seconds(&source).unwrap();
            let mut at = dispatch();
            at.sampled_frame_counter = 100;
            assert_eq!(
                evaluate_wait(&record, &at).unwrap(),
                WaitEligibility::SkipFutureFrame
            );
            at.sampled_frame_counter = 101;
            assert_eq!(
                evaluate_wait(&record, &at).unwrap(),
                WaitEligibility::Eligible
            );
        }
    }

    #[test]
    fn gating_priority_distinguishes_traversal_stop_from_each_skip() {
        let mut record = make_wait_for_seconds(&producer()).unwrap();
        let mut at = dispatch();
        record.insertion_generation = 8;
        record.frame_threshold = 200;
        at.phase_mask = 4;
        at.sampled_time = 0.0;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::StopAtFutureDeadline
        );
        at.sampled_time = record.deadline;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::SkipPhase
        );
        at.phase_mask = 8;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::SkipCurrentGeneration
        );
        record.insertion_generation = 7;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::SkipFutureFrame
        );
        at.sampled_frame_counter = 200;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::Eligible
        );
        at.phase_mask = 0x12;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::Eligible
        );
    }

    #[test]
    fn generation_wrap_preserves_same_drain_exclusion() {
        let mut record = make_wait_for_seconds(&producer()).unwrap();
        let mut at = dispatch();
        at.generation_before = u32::MAX;
        record.insertion_generation = 0;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::SkipCurrentGeneration
        );
        record.insertion_generation = u32::MAX;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::Eligible
        );
    }

    #[test]
    fn full_width_frame_gates_use_signed_comparisons_and_native_increment() {
        let mut source = producer();
        source.producer_frame_counter = i64::from(u32::MAX);
        let record = make_wait_for_seconds(&source).unwrap();
        assert_eq!(record.frame_threshold, 1_i64 << 32);
        let mut at = dispatch();
        at.sampled_frame_counter = i64::from(u32::MAX);
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::SkipFutureFrame
        );
        at.sampled_frame_counter += 1;
        assert_eq!(
            evaluate_wait(&record, &at).unwrap(),
            WaitEligibility::Eligible
        );
        source.producer_frame_counter = i64::MAX;
        let wrapped = make_wait_for_seconds(&source).unwrap();
        assert_eq!(wrapped.frame_threshold, i64::MIN);
        at.sampled_frame_counter = 0;
        assert_eq!(
            evaluate_wait(&wrapped, &at).unwrap(),
            WaitEligibility::Eligible
        );
        let mut sentinel = record;
        sentinel.frame_threshold = -1;
        assert_eq!(
            evaluate_wait(&sentinel, &at).unwrap(),
            WaitEligibility::Eligible
        );
    }

    #[test]
    fn nonfinite_values_and_unknown_contracts_fail_explicitly() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut source = producer();
            source.producer_time = value;
            assert_eq!(
                make_wait_for_seconds(&source),
                Err(LedgerError::InvalidContext)
            );
            source = producer();
            source.duration = value as f32;
            assert_eq!(
                make_wait_for_seconds(&source),
                Err(LedgerError::InvalidContext)
            );
            let mut record = make_wait_for_seconds(&producer()).unwrap();
            let mut at = dispatch();
            record.deadline = value;
            assert_eq!(
                evaluate_wait(&record, &at),
                Err(LedgerError::InvalidContext)
            );
            record = make_wait_for_seconds(&producer()).unwrap();
            at.sampled_time = value;
            assert_eq!(
                evaluate_wait(&record, &at),
                Err(LedgerError::InvalidContext)
            );
        }
        let mut source = producer();
        source.rule_version.clear();
        assert_eq!(
            make_wait_for_seconds(&source),
            Err(LedgerError::InvalidContext)
        );
        let mut at = dispatch();
        at.rule_version.clear();
        assert_eq!(
            evaluate_wait(&make_wait_for_seconds(&producer()).unwrap(), &at),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn snapshot_schema_requires_every_provenance_field() {
        let source = producer();
        let encoded = serde_json::to_value(&source).unwrap();
        assert_eq!(
            serde_json::from_value::<WaitForSecondsContext>(encoded.clone()).unwrap(),
            source
        );
        for field in [
            "rule_version",
            "duration",
            "producer_time",
            "producer_frame_counter",
            "insertion_generation",
        ] {
            let mut missing = encoded.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<WaitForSecondsContext>(missing).is_err());
        }
        let mut extra = serde_json::to_value(dispatch()).unwrap();
        extra["assumed_ready"] = true.into();
        assert!(serde_json::from_value::<WaitDispatchContext>(extra).is_err());
    }
}
