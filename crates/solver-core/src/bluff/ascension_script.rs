//! Weighted native AscensionsData.SetupCharactersCount script selection.
//!
//! This offline kernel models the cached script and both ordered RNG draws.
//! It does not select a game mode, clone ScriptInfo, materialize role arrays,
//! or read hidden live state. Null inputs are actual native nulls, not unknowns.
use super::ledger::{LedgerError, Probability};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const ASCENSION_SCRIPT_NATIVE_V1: &str = "ascension_script_native_v1";
const MAX_ENTRIES: usize = 4096;
const MAX_PATHS: usize = 65536;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomScriptRecord {
    pub script_id: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscensionScriptContext {
    pub rule_version: String,
    pub script_ids: Vec<u16>,
    pub cached_script: Option<u16>,
    pub inline_scripts: Option<Vec<Option<u16>>>,
    pub custom_scripts: Option<Vec<Option<CustomScriptRecord>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScriptDrawSource {
    Inline,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScriptDraw {
    pub source: ScriptDrawSource,
    pub occurrence_index: u16,
    pub width: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScriptSelectionFailure {
    NullInlineArray,
    NullCustomArray,
    NullCustomRecord,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AscensionScriptPath {
    pub probability: Probability,
    pub cached_script: Option<u16>,
    pub draws: Vec<ScriptDraw>,
    pub failure: Option<ScriptSelectionFailure>,
}

/// Enumerate individual occurrence paths. Superseded inline draws remain in
/// the trace and probability, including when a later custom record is null.
pub fn replay_ascension_script(
    context: &AscensionScriptContext,
) -> Result<Vec<AscensionScriptPath>, LedgerError> {
    let ids: BTreeSet<_> = context.script_ids.iter().copied().collect();
    let inline_len = context.inline_scripts.as_ref().map_or(0, Vec::len);
    let custom_len = context.custom_scripts.as_ref().map_or(0, Vec::len);
    if context.rule_version != ASCENSION_SCRIPT_NATIVE_V1
        || ids.len() != context.script_ids.len()
        || ids.len() > MAX_ENTRIES
        || inline_len > MAX_ENTRIES
        || custom_len > MAX_ENTRIES
        || context.cached_script.is_some_and(|id| !ids.contains(&id))
        || context
            .inline_scripts
            .iter()
            .flatten()
            .flatten()
            .any(|id| !ids.contains(id))
        || context
            .custom_scripts
            .iter()
            .flatten()
            .flatten()
            .filter_map(|record| record.script_id)
            .any(|id| !ids.contains(&id))
    {
        return Err(LedgerError::InvalidContext);
    }
    let initial = AscensionScriptPath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        cached_script: context.cached_script,
        draws: Vec::new(),
        failure: None,
    };
    if context.cached_script.is_some() {
        return Ok(vec![initial]);
    }
    let Some(inline) = &context.inline_scripts else {
        return Ok(vec![AscensionScriptPath {
            failure: Some(ScriptSelectionFailure::NullInlineArray),
            ..initial
        }]);
    };
    if inline_len.max(1) * custom_len.max(1) > MAX_PATHS {
        return Err(LedgerError::Capacity);
    }
    let mut paths = Vec::new();
    if inline.is_empty() {
        paths.push(initial);
    } else {
        for (index, script_id) in inline.iter().enumerate() {
            paths.push(AscensionScriptPath {
                probability: initial.probability.multiply(1, inline.len() as u64)?,
                cached_script: *script_id,
                draws: vec![ScriptDraw {
                    source: ScriptDrawSource::Inline,
                    occurrence_index: index as u16,
                    width: inline.len() as u16,
                }],
                failure: None,
            });
        }
    }
    let Some(custom) = &context.custom_scripts else {
        for path in &mut paths {
            path.failure = Some(ScriptSelectionFailure::NullCustomArray);
        }
        return Ok(paths);
    };
    if custom.is_empty() {
        return Ok(paths);
    }
    let mut results = Vec::with_capacity(paths.len() * custom.len());
    for path in paths {
        for (index, record) in custom.iter().enumerate() {
            let mut branch = path.clone();
            branch.probability = branch.probability.multiply(1, custom.len() as u64)?;
            branch.draws.push(ScriptDraw {
                source: ScriptDrawSource::Custom,
                occurrence_index: index as u16,
                width: custom.len() as u16,
            });
            match record {
                Some(record) => branch.cached_script = record.script_id,
                None => branch.failure = Some(ScriptSelectionFailure::NullCustomRecord),
            }
            results.push(branch);
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> AscensionScriptContext {
        AscensionScriptContext {
            rule_version: ASCENSION_SCRIPT_NATIVE_V1.into(),
            script_ids: vec![0, 1, 2, 3],
            cached_script: None,
            inline_scripts: Some(vec![Some(0), Some(1)]),
            custom_scripts: Some(vec![
                Some(CustomScriptRecord { script_id: Some(2) }),
                Some(CustomScriptRecord { script_id: Some(3) }),
            ]),
        }
    }

    #[test]
    fn custom_replacement_retains_both_draws_and_unconditional_weights() {
        let paths = replay_ascension_script(&context()).unwrap();
        assert_eq!(paths.len(), 4);
        for path in paths {
            assert_eq!(
                path.probability,
                Probability {
                    numerator: 1,
                    denominator: 4
                }
            );
            assert_eq!(path.draws.len(), 2);
            assert_eq!(path.cached_script, Some(2 + path.draws[1].occurrence_index));
            assert_eq!(path.failure, None);
        }
    }

    #[test]
    fn cache_skips_both_native_null_arrays() {
        let mut ctx = context();
        ctx.cached_script = Some(1);
        ctx.inline_scripts = None;
        ctx.custom_scripts = None;
        let paths = replay_ascension_script(&ctx).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].cached_script, Some(1));
        assert!(paths[0].draws.is_empty());
        assert_eq!(paths[0].failure, None);
    }

    #[test]
    fn null_custom_array_retains_earlier_selection() {
        let mut ctx = context();
        ctx.custom_scripts = None;
        let paths = replay_ascension_script(&ctx).unwrap();
        for (index, path) in paths.iter().enumerate() {
            assert_eq!(path.cached_script, Some(index as u16));
            assert_eq!(path.failure, Some(ScriptSelectionFailure::NullCustomArray));
            assert_eq!(path.draws.len(), 1);
        }
        ctx.inline_scripts = None;
        let paths = replay_ascension_script(&ctx).unwrap();
        assert_eq!(
            paths[0].failure,
            Some(ScriptSelectionFailure::NullInlineArray)
        );
        assert!(paths[0].draws.is_empty());
    }

    #[test]
    fn null_record_fails_but_null_script_clears_cache() {
        let mut ctx = context();
        ctx.inline_scripts = Some(vec![Some(0)]);
        ctx.custom_scripts = Some(vec![None, Some(CustomScriptRecord { script_id: None })]);
        let paths = replay_ascension_script(&ctx).unwrap();
        assert_eq!(paths[0].cached_script, Some(0));
        assert_eq!(
            paths[0].failure,
            Some(ScriptSelectionFailure::NullCustomRecord)
        );
        assert_eq!(paths[1].cached_script, None);
        assert_eq!(paths[1].failure, None);
        ctx.cached_script = paths[1].cached_script;
        assert_eq!(replay_ascension_script(&ctx).unwrap()[1].draws.len(), 2);
    }

    #[test]
    fn repeated_occurrences_and_empty_sources_keep_native_draw_counts() {
        let mut ctx = context();
        ctx.inline_scripts = Some(vec![Some(0), Some(0), Some(1)]);
        ctx.custom_scripts = Some(vec![]);
        let paths = replay_ascension_script(&ctx).unwrap();
        assert_eq!(
            paths.iter().filter(|p| p.cached_script == Some(0)).count(),
            2
        );
        assert!(paths
            .iter()
            .all(|p| p.probability.denominator == 3 && p.draws.len() == 1));
        ctx.inline_scripts = Some(vec![]);
        let paths = replay_ascension_script(&ctx).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].draws.is_empty());
        assert_eq!(paths[0].cached_script, None);
    }

    #[test]
    fn rejects_unknown_provenance_and_excessive_expansion() {
        let mut ctx = context();
        ctx.inline_scripts = Some(vec![Some(99)]);
        assert_eq!(
            replay_ascension_script(&ctx),
            Err(LedgerError::InvalidContext)
        );
        ctx.inline_scripts = Some(vec![Some(0); 257]);
        ctx.custom_scripts = Some(vec![Some(CustomScriptRecord { script_id: Some(1) }); 256]);
        assert_eq!(replay_ascension_script(&ctx), Err(LedgerError::Capacity));
    }

    #[test]
    fn matches_pinned_native_selection_corpus() {
        let report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_ascension_setup_audit.json"
        )).unwrap();
        let cases = report["script_selection_cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let mut ctx = context();
            ctx.inline_scripts
                .as_mut()
                .unwrap()
                .truncate(case["inline_count"].as_u64().unwrap() as usize);
            ctx.custom_scripts
                .as_mut()
                .unwrap()
                .truncate(case["custom_count"].as_u64().unwrap() as usize);
            ctx.cached_script = case["cached"].as_bool().unwrap().then_some(0);
            let paths = replay_ascension_script(&ctx).unwrap();
            let choices: Vec<_> = case["choices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u16)
                .collect();
            let path = paths
                .iter()
                .find(|p| {
                    p.draws
                        .iter()
                        .map(|d| d.occurrence_index)
                        .eq(choices.iter().copied())
                })
                .unwrap();
            assert_eq!(
                path.cached_script,
                case["selected_script_index"].as_u64().map(|n| n as u16)
            );
            assert_eq!(path.failure, None);
            assert_eq!(
                path.draws.len(),
                case["rng_draws"].as_array().unwrap().len()
            );
        }
    }
}
