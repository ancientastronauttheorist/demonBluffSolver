//! Exact managed Mutant selector, which is distinct from public Mutant/Skinwalker.
//!
//! The caller supplies the combined script occurrence list and live asset fields.
//! This kernel preserves native failure ordering but does not allocate objects,
//! model the Unity RNG state, or claim that this managed role has a shipped asset.
use super::ledger::{LedgerError, Probability};
use super::reveal::{StatusApplication, StatusState};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MUTANT_SELECTOR_NATIVE_V1: &str = "managed_mutant_selector_native_v1";
const MAX_ENTRIES: usize = 4096;
const MAD: i32 = 20;

/// One actual asset identity; repeated occurrences reference the same key.
/// Alignment is the native serialized field, not a public-name faction guess.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MutantAsset {
    pub asset_id: u16,
    pub alignment: i32,
    pub bluffable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MutantSelectorContext {
    pub rule_version: String,
    pub assets: Vec<MutantAsset>,
    /// Already combined by the caller in native order. None is an actual null
    /// list element, not missing provenance. Singletons and receiver are live.
    pub script_occurrences: Vec<Option<u16>>,
    pub statuses: StatusState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MutantSelection {
    Selected {
        asset_id: u16,
        occurrence_index: u16,
    },
    /// Native first filter reaches a null element before AddStatus runs.
    NullAsset { occurrence_index: u16 },
    /// Mad has already been attempted, and a zero-width draw reaches indexed
    /// access. This is an observed failure outcome, never a successful no-bluff.
    EmptySupport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MutantSelectorPath {
    pub probability: Probability,
    pub selection: MutantSelection,
    pub statuses: StatusState,
    pub mad_attempt: Option<StatusApplication>,
    pub rng_draw_count: u8,
}

/// Returns native success/failure paths without mutating caller state. Invalid
/// provenance and capacity violations remain atomic errors. The native method
/// removes no occurrence and does not register the selected asset in a script.
pub fn replay_mutant_selector(
    context: &MutantSelectorContext,
) -> Result<Vec<MutantSelectorPath>, LedgerError> {
    if context.rule_version != MUTANT_SELECTOR_NATIVE_V1
        || context.assets.len() > MAX_ENTRIES
        || context.script_occurrences.len() > MAX_ENTRIES
        || context.statuses.values.len() > 256
        || context.statuses.resistance.len() > 256
        || context.statuses.target_position == Some(0)
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut assets = BTreeMap::new();
    for asset in &context.assets {
        if assets.insert(asset.asset_id, asset).is_some() {
            return Err(LedgerError::InvalidContext);
        }
    }
    if context
        .script_occurrences
        .iter()
        .flatten()
        .any(|id| !assets.contains_key(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut path = MutantSelectorPath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        selection: MutantSelection::EmptySupport,
        statuses: context.statuses.clone(),
        mad_attempt: None,
        rng_draw_count: 0,
    };
    let mut candidates = Vec::new();
    for (index, id) in context.script_occurrences.iter().enumerate() {
        let Some(id) = id else {
            path.selection = MutantSelection::NullAsset {
                occurrence_index: index as u16,
            };
            return Ok(vec![path]);
        };
        let asset = assets[id];
        // Exact Good is 10. The first filter completes before the second one
        // and before the status attempt; even a later null aborts the call.
        if asset.alignment == 10 && asset.bluffable {
            candidates.push((index as u16, *id));
        }
    }
    path.mad_attempt = Some(path.statuses.apply(MAD, None));
    path.rng_draw_count = 1;
    if candidates.is_empty() {
        return Ok(vec![path]);
    }
    // Bound cloned status histories, not just the number of output branches.
    if candidates.len() * (path.statuses.values.len() + path.statuses.resistance.len() + 1)
        > 1_048_576
    {
        return Err(LedgerError::Capacity);
    }
    path.probability.denominator = candidates.len() as u64;
    Ok(candidates
        .into_iter()
        .map(|(occurrence_index, asset_id)| {
            let mut branch = path.clone();
            branch.selection = MutantSelection::Selected {
                asset_id,
                occurrence_index,
            };
            branch
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context() -> MutantSelectorContext {
        MutantSelectorContext {
            rule_version: MUTANT_SELECTOR_NATIVE_V1.into(),
            assets: vec![
                MutantAsset {
                    asset_id: 1,
                    alignment: 10,
                    bluffable: true,
                },
                MutantAsset {
                    asset_id: 2,
                    alignment: 20,
                    bluffable: true,
                },
                MutantAsset {
                    asset_id: 3,
                    alignment: 10,
                    bluffable: false,
                },
                MutantAsset {
                    asset_id: 4,
                    alignment: 10,
                    bluffable: true,
                },
            ],
            script_occurrences: vec![Some(1), Some(2), Some(3), Some(1), Some(4)],
            statuses: StatusState {
                values: vec![10],
                resistance: vec![],
                target_position: Some(7),
            },
        }
    }
    #[test]
    fn exact_good_bluffable_occurrences_preserve_multiplicity_and_source_indices() {
        let c = context();
        let saved = c.clone();
        let paths = replay_mutant_selector(&c).unwrap();
        assert_eq!(paths.len(), 3);
        for (p, (asset_id, occurrence_index)) in paths.iter().zip([(1, 0), (1, 3), (4, 4)]) {
            assert_eq!(
                p.selection,
                MutantSelection::Selected {
                    asset_id,
                    occurrence_index
                }
            );
            assert_eq!(
                p.probability,
                Probability {
                    numerator: 1,
                    denominator: 3
                }
            );
            assert_eq!(p.statuses.values, [10, 20]);
            assert_eq!(p.statuses.target_position, None);
            assert_eq!(p.rng_draw_count, 1);
        }
        assert_eq!(c, saved);
    }
    #[test]
    fn late_null_aborts_before_mad_even_after_eligible_occurrences() {
        let mut c = context();
        c.script_occurrences.push(None);
        let p = replay_mutant_selector(&c).unwrap().remove(0);
        assert_eq!(
            p.selection,
            MutantSelection::NullAsset {
                occurrence_index: 5
            }
        );
        assert_eq!(p.statuses, c.statuses);
        assert!(p.mad_attempt.is_none());
        assert_eq!(p.rng_draw_count, 0);
    }
    #[test]
    fn empty_support_retains_accepted_mad_and_clears_shared_target() {
        let mut c = context();
        c.script_occurrences = vec![Some(2), Some(3)];
        let p = replay_mutant_selector(&c).unwrap().remove(0);
        assert_eq!(p.selection, MutantSelection::EmptySupport);
        assert_eq!(p.statuses.values, [10, 20]);
        assert_eq!(p.statuses.target_position, None);
        assert_eq!(p.rng_draw_count, 1);
        assert!(p.mad_attempt.unwrap().accepted);
    }
    #[test]
    fn resistant_empty_support_preserves_existing_mad_and_target() {
        let mut c = context();
        c.script_occurrences.clear();
        c.statuses.values.push(20);
        c.statuses.resistance.push(20);
        let p = replay_mutant_selector(&c).unwrap().remove(0);
        assert_eq!(p.statuses, c.statuses);
        assert!(!p.mad_attempt.unwrap().accepted);
    }
    #[test]
    fn existing_mad_unique_add_still_overwrites_target() {
        let mut c = context();
        c.statuses.values.push(20);
        let p = replay_mutant_selector(&c).unwrap().remove(0);
        assert_eq!(p.statuses.values, [10, 20]);
        let attempt = p.mad_attempt.unwrap();
        assert!(attempt.accepted);
        assert!(!attempt.inserted);
        assert_eq!(attempt.target_after, None);
    }
    #[test]
    fn invalid_provenance_and_oversized_output_fail_atomically() {
        let c = context();
        for variant in 0..5 {
            let mut bad = c.clone();
            match variant {
                0 => bad.rule_version = "public_mutant".into(),
                1 => bad.assets.push(bad.assets[0].clone()),
                2 => bad.script_occurrences.push(Some(99)),
                3 => bad.statuses.target_position = Some(0),
                _ => bad.script_occurrences = vec![Some(1); 4097],
            }
            assert_eq!(
                replay_mutant_selector(&bad).unwrap_err(),
                LedgerError::InvalidContext
            );
        }
        let mut wide = c.clone();
        wide.script_occurrences = vec![Some(1); 4096];
        wide.statuses.values = vec![10; 256];
        wide.statuses.resistance = vec![99; 256];
        let saved = wide.clone();
        assert_eq!(
            replay_mutant_selector(&wide).unwrap_err(),
            LedgerError::Capacity
        );
        assert_eq!(wide, saved);
    }
    #[test]
    fn agrees_with_pinned_native_selector_caller_cases() {
        let report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_mutant_audit.json"
        ))
        .unwrap();
        assert_eq!(report["build_id"], "f530404b0f3f_807de4a83df4");
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 16);
        for case in cases {
            let context = serde_json::from_value(case["context"].clone()).unwrap();
            let paths = replay_mutant_selector(&context).unwrap();
            let path = &paths[case["draw_index"].as_u64().unwrap() as usize];
            assert_eq!(
                serde_json::to_value(&path.selection).unwrap(),
                case["expected_selection"]
            );
            assert_eq!(
                path.mad_attempt.is_some(),
                case["mad_attempted"].as_bool().unwrap()
            );
            assert_eq!(
                u64::from(path.rng_draw_count),
                case["rng_draw_count"].as_u64().unwrap()
            );
        }
    }
    #[test]
    fn strict_serialized_context_rejects_unknown_fields() {
        let c = context();
        let mut v = serde_json::to_value(&c).unwrap();
        v["infer_alignment_from_role"] = true.into();
        assert!(serde_json::from_value::<MutantSelectorContext>(v).is_err());
    }
}
