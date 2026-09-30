//! Ordered composition of the audited round-pool kernels and selector ledger.
//!
//! Getters in the unique kernel remain supplied. This boundary requires a stable
//! shared script, distinct pools, a one-to-one live asset mapping, independently
//! uniform draws and independently established selector chronology. Construction
//! failures retain their unconditional mass; unsupported selector failures reject
//! the whole call. This does not execute ManageCharacters or recover Unity RNG.

use super::{
    ledger::{
        self, LedgerError, Probability, ScriptLists, SelectorEvent, SelectorLedger, SelectorPath,
        SelectorPools,
    },
    round_bluffs::{self, Context as UniqueContext},
    round_candidate_composition::{self as duplicate, Asset, Context as DuplicateContext},
    round_duplicates::Items,
};
use crate::knowledge_base::{get_card, Faction};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const POOL_LEDGER_BRIDGE_NATIVE_V1: &str = "pool_ledger_bridge_native_v1";
const MAX_PATHS: usize = 1024;
const MAX_RETAINED: usize = 1_048_576;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub stable_script_and_assets: bool,
    pub distinct_pool_identities: bool,
    pub independent_uniform_draws: bool,
    pub selector_order_and_no_intervening_writers: bool,
    pub live_current_build_asset_mapping: bool,
    pub unique: UniqueContext,
    pub duplicate: DuplicateContext,
    /// One canonical role per live asset. Distinct same-name assets are rejected.
    pub asset_names: BTreeMap<u16, String>,
    pub must_include: Items,
    pub events: Vec<SelectorEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResultKind {
    UniqueFailure,
    DuplicateFailure,
    Selected { path: SelectorPath },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Outcome {
    pub probability: Probability,
    pub unique_path: usize,
    pub duplicate_path: Option<usize>,
    pub result: ResultKind,
}

/// Construction traces are retained once and referenced by each joint outcome.
/// The Selected path probability is conditional on its construction pair; the
/// outer probability additionally includes both pool-construction histories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub unique: Vec<round_bluffs::WeightedTrace>,
    pub duplicate: Vec<duplicate::WeightedTrace>,
    pub outcomes: Vec<Outcome>,
}

pub(super) fn names_valid(assets: &BTreeMap<u16, Asset>, names: &BTreeMap<u16, String>) -> bool {
    let mut seen = BTreeSet::new();
    assets.len() == names.len()
        && assets.iter().all(|(id, asset)| {
            let Some(name) = names.get(id) else {
                return false;
            };
            let Some(card) = get_card(name) else {
                return false;
            };
            let real_type = match card.faction {
                Faction::Villager => 10,
                Faction::Outcast => 20,
                Faction::Minion => 30,
                Faction::Demon => 100,
            };
            card.name == name && asset.real_type == real_type && seen.insert(name)
        })
}

pub(super) fn map(items: &Items, names: &BTreeMap<u16, String>) -> Result<Vec<String>, LedgerError> {
    items
        .iter()
        .map(|id| {
            id.and_then(|id| names.get(&id))
                .cloned()
                .ok_or(LedgerError::InvalidContext)
        })
        .collect()
}

fn script(c: &Context) -> Result<ScriptLists, LedgerError> {
    let lists = c.duplicate.rosters.each_ref().map(|list| {
        list.as_ref()
            .ok_or(LedgerError::InvalidContext)
            .and_then(|items| map(items, &c.asset_names))
    });
    let [villagers, outcasts, minions, demons] = lists;
    Ok(ScriptLists {
        villagers: villagers?,
        outcasts: outcasts?,
        minions: minions?,
        demons: demons?,
    })
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    if c.unique.assets.len() > 32
        || c.asset_names.len() > 32
        || c.unique.all.len()
            + c.unique.script.len()
            + c.unique.fallback.len()
            + c.unique.initial_pool.len()
            > 32
        || c.duplicate
            .rosters
            .iter()
            .flatten()
            .map(Vec::len)
            .sum::<usize>()
            + c.duplicate.initial_pool.len()
            > 32
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != POOL_LEDGER_BRIDGE_NATIVE_V1
        || !c.stable_script_and_assets
        || !c.distinct_pool_identities
        || !c.independent_uniform_draws
        || !c.selector_order_and_no_intervening_writers
        || !c.live_current_build_asset_mapping
        || c.unique.predicate.is_some()
        || c.unique.replace_after_getter.is_some()
        || c.unique.assets != c.duplicate.assets
        || c.unique.gameplay_initialized != c.duplicate.gameplay_initialized
        || c.unique.gameplay_present != c.duplicate.gameplay_present
        || !names_valid(&c.unique.assets, &c.asset_names)
        || c.events.len() > 16
    {
        return Err(LedgerError::InvalidContext);
    }
    if c.must_include.len() > 32 {
        return Err(LedgerError::Capacity);
    }
    let mut flat = Vec::new();
    for roster in &c.duplicate.rosters {
        flat.extend(
            roster
                .as_ref()
                .ok_or(LedgerError::InvalidContext)?
                .iter()
                .copied(),
        );
    }
    if flat != c.unique.script {
        return Err(LedgerError::InvalidContext);
    }
    let mut positions = BTreeSet::new();
    for (index, event) in c.events.iter().enumerate() {
        if event.position == 0
            || !positions.insert(event.position)
            || (index > 0 && c.events[index - 1].acquisition_ordinal >= event.acquisition_ordinal)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let script = script(c)?;
    for (roles, faction) in [
        (&script.villagers, Faction::Villager),
        (&script.outcasts, Faction::Outcast),
        (&script.minions, Faction::Minion),
        (&script.demons, Faction::Demon),
    ] {
        if roles
            .iter()
            .any(|name| get_card(name).unwrap().faction != faction)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if !super::pool_is_valid(&map(&c.must_include, &c.asset_names)?) {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

pub(super) fn json_units(value: &Value) -> usize {
    1 + match value {
        Value::Array(values) => values.iter().map(json_units).sum(),
        Value::Object(values) => values.values().map(json_units).sum(),
        _ => 0,
    }
}

fn unique_units(t: &round_bluffs::Trace) -> usize {
    1 + t
        .state
        .lists
        .values()
        .map(|list| 1 + list.items.len())
        .sum::<usize>()
        + t.events
            .iter()
            .chain(&t.draws)
            .map(json_units)
            .sum::<usize>()
}
fn duplicate_units(t: &duplicate::Trace) -> usize {
    1 + t
        .lists
        .values()
        .map(|list| 1 + list.items.len())
        .sum::<usize>()
        + t.events
            .iter()
            .chain(&t.entries)
            .chain(&t.draws)
            .map(json_units)
            .sum::<usize>()
}
fn selector_units(t: &SelectorPath) -> usize {
    ledger::path_units(t)
}
fn charge(retained: &mut usize, units: usize) -> Result<(), LedgerError> {
    *retained = retained.checked_add(units).ok_or(LedgerError::Capacity)?;
    if *retained > MAX_RETAINED {
        return Err(LedgerError::Capacity);
    }
    Ok(())
}

pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let unique = round_bluffs::replay_weighted(&c.unique)?;
    let mut retained = 0;
    for path in &unique {
        charge(&mut retained, unique_units(&path.trace))?;
    }
    let mut duplicate_context = c.duplicate.clone();
    let duplicate = if let Some(path) = unique.iter().find(|p| p.trace.error.is_none()) {
        // The actual caller's static Gameplay initialization is shared by both
        // builders. Carry its completed write into the second kernel.
        duplicate_context.gameplay_initialized = path.trace.gameplay_initialized;
        duplicate::replay_weighted(&duplicate_context)?
    } else {
        Vec::new()
    };
    for path in &duplicate {
        charge(&mut retained, duplicate_units(&path.trace))?;
    }
    let initial_script = script(c)?;
    let must_include = map(&c.must_include, &c.asset_names)?;
    let mut outcomes = Vec::new();
    for (unique_path, first) in unique.iter().enumerate() {
        if first.trace.error.is_some() {
            if outcomes.len() >= MAX_PATHS {
                return Err(LedgerError::Capacity);
            }
            charge(&mut retained, 1)?;
            outcomes.push(Outcome {
                probability: first.probability,
                unique_path,
                duplicate_path: None,
                result: ResultKind::UniqueFailure,
            });
            continue;
        }
        for (duplicate_path, second) in duplicate.iter().enumerate() {
            let probability = first
                .probability
                .multiply(second.probability.numerator, second.probability.denominator)?;
            if second.trace.error.is_some() {
                if outcomes.len() >= MAX_PATHS {
                    return Err(LedgerError::Capacity);
                }
                charge(&mut retained, 1)?;
                outcomes.push(Outcome {
                    probability,
                    unique_path,
                    duplicate_path: Some(duplicate_path),
                    result: ResultKind::DuplicateFailure,
                });
                continue;
            }
            let ledger = SelectorLedger {
                rule_version: ledger::SELECTOR_LEDGER_NATIVE_V1.into(),
                pools: SelectorPools {
                    unique: map(&first.trace.state.lists["unique"].items, &c.asset_names)?,
                    duplicate: map(&second.trace.lists["duplicates"].items, &c.asset_names)?,
                    must_include: must_include.clone(),
                    script: initial_script.clone(),
                },
                events: c.events.clone(),
            };
            let selected = ledger::replay_selectors_bounded(
                &ledger,
                MAX_PATHS - outcomes.len(),
                MAX_RETAINED - retained,
            )?;
            let working_units = selected.iter().map(selector_units).sum::<usize>();
            if retained
                .checked_add(working_units)
                .ok_or(LedgerError::Capacity)?
                > MAX_RETAINED
                || outcomes
                    .len()
                    .checked_add(selected.len())
                    .ok_or(LedgerError::Capacity)?
                    > MAX_PATHS
            {
                return Err(LedgerError::Capacity);
            }
            for path in selected {
                charge(&mut retained, selector_units(&path) + 1)?;
                outcomes.push(Outcome {
                    probability: probability
                        .multiply(path.probability.numerator, path.probability.denominator)?,
                    unique_path,
                    duplicate_path: Some(duplicate_path),
                    result: ResultKind::Selected { path },
                });
            }
        }
    }
    Ok(Replay {
        unique,
        duplicate,
        outcomes,
    })
}

#[cfg(test)]
#[path = "pool_ledger_bridge_tests.rs"]
mod tests;
