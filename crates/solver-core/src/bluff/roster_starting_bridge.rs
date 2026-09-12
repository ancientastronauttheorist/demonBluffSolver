//! Offline native GetNotInDeck composition through actual lazy starting getters.
//!
//! The fixed temporary profile and reference-comparison/stable-service contract
//! are explicit. Each typed selection runs before its range append; a later
//! getter failure keeps both earlier output and already-published cache writes.
use super::ascension_script::ScriptDraw;
use super::ascension_starting::{
    replay_ascension_starting, AscensionStartingContext, StartingFailure, StartingMethod,
};
use super::ledger::{LedgerError, Probability};
use super::roster::{
    replay_roster, ListId, Occurrences, RosterContext, RosterFailure, RosterList, RosterMethod,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const ROSTER_STARTING_BRIDGE_NATIVE_V1: &str = "roster_starting_bridge_native_v1";
const ORDER: [i32; 4] = [100, 20, 30, 10];
const MAX_PATHS: usize = 65_536;
const MAX_RETAINED: usize = 1_048_576;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RosterStartingBridgeContext {
    pub rule_version: String,
    pub temporary_profile_stable: bool,
    /// AllLazy is required as the source contract; the bridge invokes four
    /// Typed operations in the native deck order. Starting.append must be None;
    /// append failures belong to roster. The ToArray ordinal spans all getters.
    pub starting: AscensionStartingContext,
    /// GetNotInDeck is required. All four starting_pools must be None because
    /// this bridge supplies their actual lazy results, not a second input copy.
    pub roster: RosterContext,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RosterStartingBridgeFailure {
    Getter { failure: StartingFailure },
    Roster { failure: RosterFailure },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RosterStartingBridgePath {
    pub probability: Probability,
    pub cached_script: Option<u16>,
    pub draws: Vec<ScriptDraw>,
    pub cache_writes: Vec<Option<u16>>,
    pub typed_requests: Vec<i32>,
    pub to_array_attempts: u8,
    pub append_attempts: usize,
    pub lists: BTreeMap<ListId, RosterList>,
    pub local_output: ListId,
    pub returned_list: Option<ListId>,
    pub removes: Vec<(ListId, Option<u16>)>,
    pub class_initialized: bool,
    pub failure: Option<RosterStartingBridgeFailure>,
}
#[derive(Clone)]
struct Pending {
    probability: Probability,
    cache: Option<u16>,
    draws: Vec<ScriptDraw>,
    writes: Vec<Option<u16>>,
    to_arrays: u8,
    pools: [Option<Occurrences>; 4],
}
fn faction(kind: i32) -> usize {
    match kind {
        10 => 0,
        20 => 1,
        30 => 2,
        100 => 3,
        _ => unreachable!(),
    }
}
fn validate(context: &RosterStartingBridgeContext) -> Result<(), LedgerError> {
    if context.rule_version != ROSTER_STARTING_BRIDGE_NATIVE_V1
        || !context.temporary_profile_stable
        || context.starting.method != StartingMethod::AllLazy
        || context.starting.service_failures.append.is_some()
        || context.roster.method != RosterMethod::GetNotInDeckCharacters
        || context.roster.starting_pools.iter().any(Option::is_some)
        || context
            .starting
            .starting
            .iter()
            .chain(context.starting.script_lists.values().flatten())
            .flatten()
            .flatten()
            .flatten()
            .any(|id| !context.roster.assets.contains_key(id))
    {
        return Err(LedgerError::InvalidContext);
    }
    // Reuse both strict validators without speculative lazy support expansion.
    let mut source = context.starting.clone();
    source.method = StartingMethod::AllStored;
    replay_ascension_starting(&source)?;
    replay_roster(&context.roster)?;
    Ok(())
}
fn finish(
    context: &RosterStartingBridgeContext,
    pending: Pending,
    getter_failure: Option<(usize, StartingFailure)>,
) -> Result<RosterStartingBridgePath, LedgerError> {
    let mut roster = context.roster.clone();
    roster.starting_pools = pending.pools;
    if let Some((phase, _)) = &getter_failure {
        // Project the elapsed list operations through the public roster kernel.
        // Its typed-service stop reproduces the same prefix; the public bridge
        // reports the actual getter cause instead of this projection marker.
        roster.service_failures.typed_pool = Some((*phase + 1) as u16);
    }
    let mut outcomes = replay_roster(&roster)?;
    assert_eq!(outcomes.len(), 1);
    let output = outcomes.remove(0);
    let failure = if let Some((_, failure)) = getter_failure {
        Some(RosterStartingBridgeFailure::Getter { failure })
    } else {
        output
            .failure
            .map(|failure| RosterStartingBridgeFailure::Roster { failure })
    };
    Ok(RosterStartingBridgePath {
        probability: pending.probability,
        cached_script: pending.cache,
        draws: pending.draws,
        cache_writes: pending.writes,
        typed_requests: output.typed_pool_requests,
        to_array_attempts: pending.to_arrays,
        append_attempts: output.append_ranges.len(),
        local_output: output.allocations[0],
        returned_list: output.returned_list,
        lists: output.lists,
        removes: output.removes,
        class_initialized: output.class_initialized,
        failure,
    })
}
fn add_finished(
    results: &mut Vec<RosterStartingBridgePath>,
    path: RosterStartingBridgePath,
    cost: &mut usize,
    retained_next: usize,
) -> Result<(), LedgerError> {
    if results.len() == MAX_PATHS {
        return Err(LedgerError::Capacity);
    }
    let size = path
        .lists
        .values()
        .map(|v| v.items.len() + 1)
        .sum::<usize>()
        + path.draws.len()
        + path.cache_writes.len()
        + path.removes.len()
        + path.typed_requests.len();
    let updated = cost.checked_add(size).ok_or(LedgerError::Capacity)?;
    let combined = updated
        .checked_add(retained_next)
        .ok_or(LedgerError::Capacity)?;
    if combined > MAX_RETAINED {
        return Err(LedgerError::Capacity);
    }
    *cost = updated;
    results.push(path);
    Ok(())
}

/// Enumerate consumed occurrence paths without dropping failure mass. Typed
/// fallback null is successful until the native caller's append rejects it.
pub fn replay_roster_starting_bridge(
    context: &RosterStartingBridgeContext,
) -> Result<Vec<RosterStartingBridgePath>, LedgerError> {
    validate(context)?;
    let mut active = vec![Pending {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        cache: context.starting.selection.cached_script,
        draws: Vec::new(),
        writes: Vec::new(),
        to_arrays: 0,
        pools: std::array::from_fn(|_| None),
    }];
    let mut done = Vec::new();
    let mut retained_done = 0;
    for (phase, kind) in ORDER.into_iter().enumerate() {
        let mut next = Vec::new();
        let mut retained_next = 0usize;
        for pending in active {
            if !context.roster.project_available
                || context.roster.service_failures.typed_pool == Some((phase + 1) as u16)
            {
                add_finished(
                    &mut done,
                    finish(context, pending, None)?,
                    &mut retained_done,
                    retained_next,
                )?;
                continue;
            }
            let mut source = context.starting.clone();
            source.method = StartingMethod::Typed {
                character_type: kind,
            };
            source.selection.cached_script = pending.cache;
            source.service_failures.to_array =
                if context.starting.service_failures.to_array == Some(pending.to_arrays + 1) {
                    Some(1)
                } else {
                    None
                };
            let typed_paths = replay_ascension_starting(&source)?;
            if done.len() + next.len() + typed_paths.len() > MAX_PATHS {
                return Err(LedgerError::Capacity);
            }
            for typed in typed_paths {
                let mut branch = pending.clone();
                branch.probability = branch
                    .probability
                    .multiply(typed.probability.numerator, typed.probability.denominator)?;
                branch.cache = typed.cached_script;
                branch.draws.extend(typed.draws);
                branch.writes.extend(typed.cache_writes);
                branch.to_arrays += typed.to_array_attempts;
                if let Some(mut failure) = typed.failure {
                    if let StartingFailure::ToArrayService { attempt, .. } = &mut failure {
                        *attempt = branch.to_arrays;
                    }
                    add_finished(
                        &mut done,
                        finish(context, branch, Some((phase, failure)))?,
                        &mut retained_done,
                        retained_next,
                    )?;
                    continue;
                }
                branch.pools[faction(kind)] = typed.typed_result;
                if branch.pools[faction(kind)].is_none()
                    || context.roster.service_failures.append_range == Some((phase + 1) as u16)
                    || phase == 3
                {
                    add_finished(
                        &mut done,
                        finish(context, branch, None)?,
                        &mut retained_done,
                        retained_next,
                    )?;
                } else {
                    let size = branch.pools.iter().flatten().map(Vec::len).sum::<usize>()
                        + branch.draws.len()
                        + branch.writes.len();
                    retained_next = retained_next
                        .checked_add(size)
                        .ok_or(LedgerError::Capacity)?;
                    if retained_next + retained_done > MAX_RETAINED {
                        return Err(LedgerError::Capacity);
                    }
                    next.push(branch);
                }
            }
        }
        active = next;
    }
    assert!(active.is_empty());
    Ok(done)
}

#[cfg(test)]
#[path = "roster_starting_bridge_tests.rs"]
mod tests;
