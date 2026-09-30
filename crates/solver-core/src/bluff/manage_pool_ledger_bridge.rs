//! Actual ManageCharacters pool prefix followed by explicit successful selectors.
//! The caller must verify completion of intervening setup with no pool/script
//! writers and supply exact selector dispatch/order. Native prefix failure mass
//! is retained. This does not infer actor dispatch or execute Init/Start/Reveal.
use super::{
    ledger::{
        self, LedgerError, Probability, ScriptLists, SelectorEvent, SelectorLedger, SelectorPath,
        SelectorPools,
    },
    manage_pool_composition::{self as construction, Context as ConstructionContext},
    pool_ledger_bridge::{json_units, map, names_valid},
    round_duplicates::Items,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MANAGE_POOL_LEDGER_NATIVE_V1: &str = "manage_pool_ledger_native_v1";
const MAX_PATHS: usize = 1024;
const MAX_RETAINED: usize = 4_194_304;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub construction: ConstructionContext,
    pub asset_names: BTreeMap<u16, String>,
    pub must_include: Items,
    pub live_current_build_asset_mapping: bool,
    pub completed_setup_without_pool_script_writers: bool,
    pub actual_selector_dispatch_and_order_verified: bool,
    pub independent_uniform_selector_draws: bool,
    pub events: Vec<SelectorEvent>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResultKind {
    PrefixFailure,
    Selected { path: SelectorPath },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Outcome {
    pub probability: Probability,
    pub construction_path: usize,
    pub result: ResultKind,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub construction: Vec<construction::WeightedTrace>,
    pub outcomes: Vec<Outcome>,
}
fn charge(retained: &mut usize, units: usize) -> Result<(), LedgerError> {
    *retained = retained.checked_add(units).ok_or(LedgerError::Capacity)?;
    if *retained > MAX_RETAINED {
        return Err(LedgerError::Capacity);
    }
    Ok(())
}
fn units(t: &construction::Trace) -> usize {
    1 + t
        .state
        .lists
        .values()
        .map(|list| 1 + list.items.len())
        .sum::<usize>()
        + t.state.board.items.values().map(Vec::len).sum::<usize>()
        + t.entries.len()
        + t.predicate_results.len()
        + t.events
            .iter()
            .chain(&t.draws)
            .chain(&t.typed)
            .chain(&t.callbacks)
            .map(json_units)
            .sum::<usize>()
}
fn script(
    t: &construction::Trace,
    names: &BTreeMap<u16, String>,
) -> Result<ScriptLists, LedgerError> {
    let mut lists = Vec::new();
    for source in &t.state.roster_sources {
        let id = source.as_ref().ok_or(LedgerError::InvalidContext)?;
        let list = t.state.lists.get(id).ok_or(LedgerError::InvalidContext)?;
        lists.push(map(&list.items, names)?);
    }
    let mut it = lists.into_iter();
    Ok(ScriptLists {
        villagers: it.next().unwrap(),
        outcasts: it.next().unwrap(),
        minions: it.next().unwrap(),
        demons: it.next().unwrap(),
    })
}
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    if c.version != MANAGE_POOL_LEDGER_NATIVE_V1
        || !c.live_current_build_asset_mapping
        || !c.completed_setup_without_pool_script_writers
        || !c.actual_selector_dispatch_and_order_verified
        || !c.independent_uniform_selector_draws
        || c.events.len() > 16
    {
        return Err(LedgerError::InvalidContext);
    }
    if c.asset_names.len() > 32 || c.must_include.len() > 32 {
        return Err(LedgerError::Capacity);
    }
    if !names_valid(&c.construction.assets, &c.asset_names) {
        return Err(LedgerError::InvalidContext);
    }
    let must = map(&c.must_include, &c.asset_names)?;
    if !super::pool_is_valid(&must) {
        return Err(LedgerError::InvalidContext);
    }
    let mut seen = BTreeSet::new();
    for (i, e) in c.events.iter().enumerate() {
        if e.position == 0
            || !seen.insert(e.position)
            || (i > 0 && c.events[i - 1].acquisition_ordinal >= e.acquisition_ordinal)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let construction = construction::replay_weighted(&c.construction)?;
    let mut retained = 0;
    for p in &construction {
        charge(&mut retained, units(&p.trace))?;
    }
    let mut outcomes = Vec::new();
    for (construction_path, p) in construction.iter().enumerate() {
        if outcomes.len() >= MAX_PATHS {
            return Err(LedgerError::Capacity);
        }
        if p.trace.error.is_some() {
            charge(&mut retained, 1)?;
            outcomes.push(Outcome {
                probability: p.probability,
                construction_path,
                result: ResultKind::PrefixFailure,
            });
            continue;
        }
        let first_init = p.trace.boundary.as_deref() == Some("before_first_init");
        let empty =
            p.trace.boundary.as_deref() == Some("empty_before_publication") && c.events.is_empty();
        if (!first_init && !empty)
            || !p.trace.state.game
            || (first_init
                && p.trace
                    .init_arguments
                    .as_ref()
                    .is_none_or(|args| args["data"].is_null()))
        {
            return Err(LedgerError::InvalidContext);
        }
        let ledger = SelectorLedger {
            rule_version: ledger::SELECTOR_LEDGER_NATIVE_V1.into(),
            pools: SelectorPools {
                unique: map(&p.trace.state.lists["unique"].items, &c.asset_names)?,
                duplicate: map(&p.trace.state.lists["duplicates"].items, &c.asset_names)?,
                must_include: must.clone(),
                script: script(&p.trace, &c.asset_names)?,
            },
            events: c.events.clone(),
        };
        // The construction probability already includes lazy source and both
        // pool histories. Only the conditional selector fraction is multiplied.
        let selected = ledger::replay_selectors_bounded(
            &ledger,
            MAX_PATHS - outcomes.len(),
            MAX_RETAINED - retained,
        )?;
        for path in selected {
            charge(&mut retained, ledger::path_units(&path) + 1)?;
            outcomes.push(Outcome {
                probability: p
                    .probability
                    .multiply(path.probability.numerator, path.probability.denominator)?,
                construction_path,
                result: ResultKind::Selected { path },
            });
        }
    }
    Ok(Replay {
        construction,
        outcomes,
    })
}

#[cfg(test)]
#[path = "manage_pool_ledger_bridge_tests.rs"]
mod tests;
