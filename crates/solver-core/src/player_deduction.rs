//! Conditional finite-world deduction from externally reviewed player history.
//!
//! UI review admits observations, not hidden setup assumptions. This API reports
//! those assumptions explicitly and makes no generation, probability or policy
//! claim. Other domains remain unsupported even if a mechanical adapter exists.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::player_history::{AdmittedPlayerHistory, ProjectionError, HUNTER_BAA_PROJECTION_DOMAIN};
use crate::scenario::build_scenarios;
use crate::solver::solve;
use crate::types::{Scenario, SolverResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionalAssumption {
    DistinctFixedCyclicSeats,
    OneBaaOtherwiseIdenticalHunters,
    FinishedWriterStatusDeathFreeSetup,
    AcquiredHunterBluff,
    SuppliedLegalObservationAvailability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AssumptionStatus {
    AssumedConditional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStatus {
    CompleteFiniteConditionalWorlds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Conclusion {
    Unique,
    Ambiguous,
    Contradiction,
}

/// A complete latent world within this narrow conditional model. Identical
/// Hunters have no additional identity/status alternatives under its assumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ConditionalWorld {
    pub baa_position: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompleteDeduction {
    pub domain_id: String,
    pub assumption_status: AssumptionStatus,
    pub assumptions: Vec<ConditionalAssumption>,
    pub search_status: SearchStatus,
    pub conclusion: Conclusion,
    pub worlds: Vec<ConditionalWorld>,
    pub conditional_definite_evil: Vec<u8>,
    pub conditional_definite_good: Vec<u8>,
    pub based_on_ordinals: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteKind {
    MissingObservation,
    BackendInvariant,
}

/// Serialize-only results cannot manufacture an admitted input capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PlayerDeductionOutcome {
    Complete(CompleteDeduction),
    Incomplete {
        kind: IncompleteKind,
        ordinal: Option<u64>,
        reason: String,
    },
    Unsupported {
        ordinal: Option<u64>,
        reason: String,
    },
}

fn invariant(reason: &str) -> PlayerDeductionOutcome {
    PlayerDeductionOutcome::Incomplete {
        kind: IncompleteKind::BackendInvariant,
        ordinal: None,
        reason: reason.into(),
    }
}

fn canonical_world(scenario: &Scenario, n: u8) -> Option<ConditionalWorld> {
    if scenario.evil_positions.len() != 1 {
        return None;
    }
    let (&position, role) = scenario.evil_positions.iter().next()?;
    if !(1..=n).contains(&position) || role != "Baa" {
        return None;
    }
    let mut expected = Scenario::default();
    expected.evil_positions.insert(position, "Baa".into());
    // Compare all serialized fields, including future populated state. Dropping
    // an unexplained trace would erase correlations and falsely imply completeness.
    if serde_json::to_value(scenario).ok()? != serde_json::to_value(expected).ok()? {
        return None;
    }
    Some(ConditionalWorld {
        baa_position: position,
    })
}

fn canonical_set(scenarios: &[Scenario], n: u8) -> Option<BTreeSet<ConditionalWorld>> {
    let mut worlds = BTreeSet::new();
    for scenario in scenarios {
        if !worlds.insert(canonical_world(scenario, n)?) {
            return None;
        }
    }
    Some(worlds)
}

fn check_backend(
    n: u8,
    candidates: &[Scenario],
    result: &SolverResult,
) -> Result<BTreeSet<ConditionalWorld>, PlayerDeductionOutcome> {
    let expected: BTreeSet<_> = (1..=n)
        .map(|baa_position| ConditionalWorld { baa_position })
        .collect();
    if canonical_set(candidates, n).as_ref() != Some(&expected)
        || result.n_scenarios != expected.len()
    {
        return Err(invariant(
            "candidate enumeration does not cover each conditional world exactly once",
        ));
    }
    let worlds = canonical_set(&result.surviving_scenarios, n)
        .ok_or_else(|| invariant("survivors contain duplicate or unexplained latent state"))?;
    if result.n_surviving != worlds.len() || !worlds.is_subset(&expected) {
        return Err(invariant(
            "survivor counts or candidate membership disagree",
        ));
    }
    let (evil, good) = definite_positions(n, &worlds);
    if result.definite_evil != evil
        || result.definite_good != good
        || !result.bombardier_positions.is_empty()
    {
        return Err(invariant(
            "backend definite positions disagree with complete conditional worlds",
        ));
    }
    Ok(worlds)
}

fn definite_positions(n: u8, worlds: &BTreeSet<ConditionalWorld>) -> (Vec<u8>, Vec<u8>) {
    let evil = (1..=n)
        .filter(|&position| {
            !worlds.is_empty() && worlds.iter().all(|world| world.baa_position == position)
        })
        .collect();
    let good = (1..=n)
        .filter(|&position| {
            !worlds.is_empty() && worlds.iter().all(|world| world.baa_position != position)
        })
        .collect();
    (evil, good)
}

/// Deduce only within the independently enumerated four/five-seat Hunter/Baa
/// conditional model. Review binds exact public history; it does not prove these
/// hidden setup assumptions. No raw JSON or caller-supplied Scenario is accepted.
pub fn deduce_conditional_history(history: &AdmittedPlayerHistory) -> PlayerDeductionOutcome {
    let planner = history.planner_history();
    if planner.domain_id != HUNTER_BAA_PROJECTION_DOMAIN {
        return PlayerDeductionOutcome::Unsupported {
            ordinal: None,
            reason: "no complete conditional deduction model for this domain".into(),
        };
    }
    let state = match history.project_legacy_snapshot() {
        Ok(state) => state,
        Err(ProjectionError::Incomplete { ordinal, reason }) => {
            return PlayerDeductionOutcome::Incomplete {
                kind: IncompleteKind::MissingObservation,
                ordinal,
                reason,
            };
        }
        Err(ProjectionError::Unsupported { ordinal, reason }) => {
            return PlayerDeductionOutcome::Unsupported { ordinal, reason };
        }
    };
    if !matches!(state.n_cards, 4 | 5) {
        return invariant("adapter returned a size outside the finite conditional family");
    }
    let candidates = build_scenarios(&state);
    let result = solve(&state);
    let worlds = match check_backend(state.n_cards, &candidates, &result) {
        Ok(worlds) => worlds,
        Err(error) => return error,
    };
    let (conditional_definite_evil, conditional_definite_good) =
        definite_positions(state.n_cards, &worlds);
    PlayerDeductionOutcome::Complete(CompleteDeduction {
        domain_id: planner.domain_id,
        assumption_status: AssumptionStatus::AssumedConditional,
        assumptions: vec![
            ConditionalAssumption::DistinctFixedCyclicSeats,
            ConditionalAssumption::OneBaaOtherwiseIdenticalHunters,
            ConditionalAssumption::FinishedWriterStatusDeathFreeSetup,
            ConditionalAssumption::AcquiredHunterBluff,
            ConditionalAssumption::SuppliedLegalObservationAvailability,
        ],
        search_status: SearchStatus::CompleteFiniteConditionalWorlds,
        conclusion: match worlds.len() {
            0 => Conclusion::Contradiction,
            1 => Conclusion::Unique,
            _ => Conclusion::Ambiguous,
        },
        worlds: worlds.into_iter().collect(),
        conditional_definite_evil,
        conditional_definite_good,
        based_on_ordinals: planner.events.iter().map(|event| event.ordinal).collect(),
    })
}

#[cfg(test)]
#[path = "player_deduction_tests.rs"]
mod tests;
