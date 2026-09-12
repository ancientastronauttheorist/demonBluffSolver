//! Offline native starting-character sequence, with occurrence-weighted reselection.
//!
//! Services preserve input arrays/records/lists. ToArray returns a snapshot;
//! InsertRange appends that array without filtering. Injected service failures
//! stop before that service changes output. Partial output on failure is local
//! diagnostic state, not a list successfully returned to the caller.
use super::ascension_script::{
    replay_ascension_script, AscensionScriptContext, AscensionScriptPath, ScriptDraw,
    ScriptDrawSource, ScriptSelectionFailure, ASCENSION_SCRIPT_NATIVE_V1,
};
use super::ledger::{LedgerError, Probability};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ASCENSION_STARTING_NATIVE_V1: &str = "ascension_starting_native_v1";
const MAX_ENTRIES: usize = 4096;
const MAX_PATHS: usize = 65_536;
const MAX_RETAINED_ENTRIES: usize = 1_048_576;
const FACTIONS: [i32; 4] = [10, 20, 30, 100];
pub type StartingArray = Option<Vec<Option<u16>>>;
pub type StartingLists = [StartingArray; 4];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StartingMethod {
    AllLazy,
    AllStored,
    Typed { character_type: i32 },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartingServiceFailures {
    /// One-based attempted conversion; no output snapshot is produced.
    pub to_array: Option<u8>,
    /// One-based attempted append; the current append makes no changes.
    pub append: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscensionStartingContext {
    pub rule_version: String,
    pub services_preserve_inputs: bool,
    pub method: StartingMethod,
    pub selection: AscensionScriptContext,
    pub script_lists: BTreeMap<u16, StartingLists>,
    pub starting: StartingLists,
    pub service_failures: StartingServiceFailures,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StartingFailure {
    Selection { failure: ScriptSelectionFailure },
    NullSelectedList { script_id: u16, character_type: i32 },
    NullAppendCollection { character_type: i32 },
    ToArrayService { attempt: u8, character_type: i32 },
    AppendService { attempt: u8, character_type: i32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AscensionStartingPath {
    pub probability: Probability,
    pub cached_script: Option<u16>,
    pub draws: Vec<ScriptDraw>,
    /// Includes discarded inline selections and successful null cache writes.
    pub cache_writes: Vec<Option<u16>>,
    pub faction_read_order: Vec<i32>,
    pub output_prefix: Vec<Option<u16>>,
    /// Only Typed uses this field. None is a successful null array when failure
    /// is absent; concatenators instead reject null at the collection service.
    pub typed_result: StartingArray,
    pub to_array_attempts: u8,
    pub append_attempts: u8,
    pub failure: Option<StartingFailure>,
}

fn validate(context: &AscensionStartingContext) -> Result<(), LedgerError> {
    let s = &context.selection;
    let ids: BTreeSet<_> = s.script_ids.iter().copied().collect();
    if context.rule_version != ASCENSION_STARTING_NATIVE_V1
        || !context.services_preserve_inputs
        || s.rule_version != ASCENSION_SCRIPT_NATIVE_V1
        || ids.len() != s.script_ids.len()
        || ids.len() > MAX_ENTRIES
        || context
            .script_lists
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            != ids
        || s.inline_scripts
            .as_ref()
            .is_some_and(|v| v.len() > MAX_ENTRIES)
        || s.custom_scripts
            .as_ref()
            .is_some_and(|v| v.len() > MAX_ENTRIES)
        || s.cached_script.is_some_and(|id| !ids.contains(&id))
        || s.inline_scripts
            .iter()
            .flatten()
            .flatten()
            .any(|id| !ids.contains(id))
        || s.custom_scripts
            .iter()
            .flatten()
            .flatten()
            .filter_map(|r| r.script_id)
            .any(|id| !ids.contains(&id))
        || [
            context.service_failures.to_array,
            context.service_failures.append,
        ]
        .into_iter()
        .flatten()
        .any(|n| !(1..=4).contains(&n))
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut entries = 0usize;
    for list in context
        .starting
        .iter()
        .chain(context.script_lists.values().flatten())
        .flatten()
    {
        if list.len() > MAX_ENTRIES {
            return Err(LedgerError::Capacity);
        }
        entries = entries
            .checked_add(list.len())
            .ok_or(LedgerError::Capacity)?;
        if entries > MAX_RETAINED_ENTRIES {
            return Err(LedgerError::Capacity);
        }
    }
    Ok(())
}

fn selected_writes(
    context: &AscensionScriptContext,
    path: &AscensionScriptPath,
) -> Vec<Option<u16>> {
    let mut writes = Vec::new();
    for draw in &path.draws {
        match draw.source {
            ScriptDrawSource::Inline => writes
                .push(context.inline_scripts.as_ref().unwrap()[draw.occurrence_index as usize]),
            ScriptDrawSource::Custom => {
                if let Some(record) =
                    &context.custom_scripts.as_ref().unwrap()[draw.occurrence_index as usize]
                {
                    writes.push(record.script_id);
                }
            }
        }
    }
    if path.draws.is_empty()
        && path.failure.is_none()
        && context.inline_scripts.as_ref().is_some_and(Vec::is_empty)
        && context.custom_scripts.as_ref().is_some_and(Vec::is_empty)
    {
        writes.push(None);
    }
    writes
}

fn read_and_append(
    context: &AscensionStartingContext,
    path: &mut AscensionStartingPath,
    kind: i32,
) {
    let Some(index) = FACTIONS.iter().position(|v| *v == kind) else {
        return;
    };
    let list = if context.method == StartingMethod::AllStored {
        &context.starting[index]
    } else if let Some(id) = path.cached_script {
        let list = &context.script_lists[&id][index];
        if list.is_none() {
            path.failure = Some(StartingFailure::NullSelectedList {
                script_id: id,
                character_type: kind,
            });
            return;
        }
        path.to_array_attempts += 1;
        if context.service_failures.to_array == Some(path.to_array_attempts) {
            path.failure = Some(StartingFailure::ToArrayService {
                attempt: path.to_array_attempts,
                character_type: kind,
            });
            return;
        }
        list
    } else {
        &context.starting[index]
    };
    if matches!(context.method, StartingMethod::Typed { .. }) {
        path.typed_result = list.clone();
        return;
    }
    path.append_attempts += 1;
    if context.service_failures.append == Some(path.append_attempts) {
        path.failure = Some(StartingFailure::AppendService {
            attempt: path.append_attempts,
            character_type: kind,
        });
    } else if let Some(values) = list {
        path.output_prefix.extend_from_slice(values);
    } else {
        path.failure = Some(StartingFailure::NullAppendCollection {
            character_type: kind,
        });
    }
}

/// Return every consumed occurrence path with unconditional rational mass.
/// Any unsupported provenance, overflow or support limit rejects the entire
/// request; successful paths are never silently renormalized after failures.
pub fn replay_ascension_starting(
    context: &AscensionStartingContext,
) -> Result<Vec<AscensionStartingPath>, LedgerError> {
    validate(context)?;
    let mut paths = vec![AscensionStartingPath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        cached_script: context.selection.cached_script,
        draws: Vec::new(),
        cache_writes: Vec::new(),
        faction_read_order: Vec::new(),
        output_prefix: Vec::new(),
        typed_result: None,
        to_array_attempts: 0,
        append_attempts: 0,
        failure: None,
    }];
    let kinds = match context.method {
        StartingMethod::Typed { character_type } => vec![character_type],
        _ => FACTIONS.to_vec(),
    };
    let mut uncached = None;
    for kind in kinds {
        let mut next = Vec::new();
        let mut retained = 0usize;
        for mut path in paths {
            if path.failure.is_some() {
                push_bounded(&mut next, path, &mut retained)?;
                continue;
            }
            path.faction_read_order.push(kind);
            if context.method == StartingMethod::AllStored || path.cached_script.is_some() {
                read_and_append(context, &mut path, kind);
                push_bounded(&mut next, path, &mut retained)?;
                continue;
            }
            if uncached.is_none() {
                let mut selection = context.selection.clone();
                selection.cached_script = None;
                uncached = Some(replay_ascension_script(&selection)?);
            }
            let branches = uncached.as_ref().unwrap();
            if next
                .len()
                .checked_add(branches.len())
                .ok_or(LedgerError::Capacity)?
                > MAX_PATHS
            {
                return Err(LedgerError::Capacity);
            }
            for selected in branches {
                let mut branch = path.clone();
                branch.probability = branch.probability.multiply(
                    selected.probability.numerator,
                    selected.probability.denominator,
                )?;
                branch.cached_script = selected.cached_script;
                branch.draws.extend_from_slice(&selected.draws);
                branch
                    .cache_writes
                    .extend(selected_writes(&context.selection, selected));
                if let Some(failure) = &selected.failure {
                    branch.failure = Some(StartingFailure::Selection {
                        failure: failure.clone(),
                    });
                } else {
                    read_and_append(context, &mut branch, kind);
                }
                push_bounded(&mut next, branch, &mut retained)?;
            }
        }
        paths = next;
    }
    Ok(paths)
}

fn push_bounded(
    paths: &mut Vec<AscensionStartingPath>,
    path: AscensionStartingPath,
    retained: &mut usize,
) -> Result<(), LedgerError> {
    if paths.len() == MAX_PATHS {
        return Err(LedgerError::Capacity);
    }
    let cost = path.output_prefix.len()
        + path.typed_result.as_ref().map_or(0, Vec::len)
        + path.draws.len()
        + path.cache_writes.len()
        + path.faction_read_order.len();
    *retained = retained.checked_add(cost).ok_or(LedgerError::Capacity)?;
    if *retained > MAX_RETAINED_ENTRIES {
        return Err(LedgerError::Capacity);
    }
    paths.push(path);
    Ok(())
}

#[cfg(test)]
#[path = "ascension_starting_tests.rs"]
mod tests;
