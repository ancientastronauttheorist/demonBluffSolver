//! Offline composition of successful native bluff-selector invocations.
//!
//! This is a selector kernel, not a Character.Reveal simulator. The caller must
//! establish dispatch, order, live assets, and absence of intervening writers.
//! Register-as, GiveBluff, copied-role hooks and scheduler/RNG-state recovery
//! remain outside this boundary. No GameState or live bridge consumes it.

use super::{gcd, is_supported_villager_bluff, pool_is_valid};
use crate::knowledge_base::{get_card, Faction};
use crate::types::BluffAcquisitionSource;
use serde::{Deserialize, Serialize};

pub const SELECTOR_LEDGER_NATIVE_V1: &str = "bluff_selector_ledger_native_v1";
const MAX_EVENTS: usize = 16;
const MAX_PATHS: usize = 65_536;
// Bound cloned pool/script/history entries as well as path count. A wide pool
// otherwise permits quadratic allocation before reaching the path limit.
const MAX_RETAINED_ENTRIES: usize = 1_048_576;

/// Canonical names stand for one live current-build asset per public role.
/// Lists retain order and repeated occurrences; distinct same-name assets are
/// unsupported. Registration appends only when that asset is absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptLists {
    pub villagers: Vec<String>,
    pub outcasts: Vec<String>,
    pub minions: Vec<String>,
    pub demons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorPools {
    pub unique: Vec<String>,
    pub duplicate: Vec<String>,
    pub must_include: Vec<String>,
    pub script: ScriptLists,
}

/// Exact managed selector, not a faction-based guess: Spy/Mutant overrides
/// must never be routed to Minion just because their assets are Minions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Selector {
    Demon,
    Minion,
    Drunk { corruption_resistant: bool },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorEvent {
    pub position: u8,
    pub acquisition_ordinal: u16,
    pub selector: Selector,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorLedger {
    pub rule_version: String,
    pub pools: SelectorPools,
    /// Explicit successful acquisitions, in independently established order.
    /// Gaps permit unrelated events only if their effects on this kernel are
    /// independently excluded. A second acquisition on one body needs a reset
    /// writer and is rejected by this version.
    pub events: Vec<SelectorEvent>,
}

/// Reduced unconditional path probability. Unlike locally reduced ticket
/// counts, these fractions compose even when earlier draws change pool widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Probability {
    pub numerator: u64,
    pub denominator: u64,
}

impl Probability {
    fn multiply(self, numerator: u64, denominator: u64) -> Result<Self, LedgerError> {
        let left = gcd(self.numerator, denominator);
        let right = gcd(numerator, self.denominator);
        let numerator = (self.numerator / left)
            .checked_mul(numerator / right)
            .ok_or(LedgerError::Capacity)?;
        let denominator = (self.denominator / right)
            .checked_mul(denominator / left)
            .ok_or(LedgerError::Capacity)?;
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CorruptionAttempt {
    /// Native unique-add preserves existing membership and writes the actor as
    /// the shared status target, even when Corrupted was already present.
    AcceptedSelf,
    /// Existing status and target, if any, remain unchanged.
    Resisted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelectorTrace {
    pub event: SelectorEvent,
    pub bluff_role: String,
    /// Index in the original unfiltered source list, before first-equal removal.
    pub source: BluffAcquisitionSource,
    /// Includes the Minion branch roll and discarded must-include probe.
    /// Probe outcomes are marginalized, not recorded as RNG-state evidence.
    pub rng_draw_count: u8,
    pub script_added: bool,
    pub corruption_attempt: Option<CorruptionAttempt>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelectorPath {
    pub probability: Probability,
    pub pools: SelectorPools,
    pub trace: Vec<SelectorTrace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerError {
    InvalidContext,
    /// At least one positive-mass path would fail native indexed selection.
    /// Reject the entire kernel invocation; never renormalize successful paths.
    EmptySupport,
    Capacity,
}

fn valid_script(script: &ScriptLists) -> bool {
    [
        (&script.villagers, Faction::Villager),
        (&script.outcasts, Faction::Outcast),
        (&script.minions, Faction::Minion),
        (&script.demons, Faction::Demon),
    ]
    .iter()
    .all(|(roles, faction)| {
        roles.len() <= MAX_PATHS
            && roles.iter().all(|role| {
                get_card(role).is_some_and(|card| card.name == role && card.faction == *faction)
            })
    })
}

fn register(script: &mut ScriptLists, role: &str) -> bool {
    // All candidates passed pool validation: canonical Villager or Outcast.
    let list = match get_card(role).expect("validated asset").faction {
        Faction::Villager => &mut script.villagers,
        Faction::Outcast => &mut script.outcasts,
        _ => unreachable!("validated bluff faction"),
    };
    if list.iter().any(|entry| entry == role) {
        false
    } else {
        list.push(role.to_string());
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolSource {
    Duplicate,
    Unique,
    MustInclude,
}

struct Draw<'a> {
    candidates: Vec<(usize, &'a String)>,
    source: PoolSource,
    branch_numerator: u64,
    branch_denominator: u64,
    rng_draw_count: u8,
}

fn draws<'a>(pools: &'a SelectorPools, selector: &Selector) -> Result<Vec<Draw<'a>>, LedgerError> {
    let minion = matches!(selector, Selector::Minion);
    let eligible = |pool: &'a [String]| -> Vec<(usize, &'a String)> {
        pool.iter()
            .enumerate()
            .filter(|(_, role)| minion || is_supported_villager_bluff(role))
            .collect()
    };
    let must = eligible(&pools.must_include);
    let uses_must = !must.is_empty();
    let unique = if uses_must {
        must
    } else {
        eligible(&pools.unique)
    };
    if unique.is_empty() || (minion && pools.duplicate.is_empty()) {
        return Err(LedgerError::EmptySupport);
    }
    let mut result = Vec::new();
    if minion {
        result.push(Draw {
            candidates: pools.duplicate.iter().enumerate().collect(),
            source: PoolSource::Duplicate,
            branch_numerator: 2,
            branch_denominator: 5,
            rng_draw_count: 2,
        });
    }
    result.push(Draw {
        candidates: unique,
        source: if uses_must {
            PoolSource::MustInclude
        } else {
            PoolSource::Unique
        },
        branch_numerator: if minion { 3 } else { 1 },
        branch_denominator: if minion { 5 } else { 1 },
        rng_draw_count: 1 + u8::from(uses_must) + u8::from(minion),
    });
    Ok(result)
}

/// Replay an ordered sequence of supported selector calls without altering the
/// input. Any unsupported/failing branch rejects the whole operation. Failure
/// side effects (notably Drunk corruption before a failed draw) are not emulated.
pub fn replay_selectors(ledger: &SelectorLedger) -> Result<Vec<SelectorPath>, LedgerError> {
    if ledger.rule_version != SELECTOR_LEDGER_NATIVE_V1
        || ledger.events.len() > MAX_EVENTS
        || !pool_is_valid(&ledger.pools.unique)
        || !pool_is_valid(&ledger.pools.duplicate)
        || !pool_is_valid(&ledger.pools.must_include)
        || !valid_script(&ledger.pools.script)
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut seen = [false; 256];
    for (index, event) in ledger.events.iter().enumerate() {
        if event.position == 0
            || seen[usize::from(event.position)]
            || (index > 0
                && ledger.events[index - 1].acquisition_ordinal >= event.acquisition_ordinal)
        {
            return Err(LedgerError::InvalidContext);
        }
        seen[usize::from(event.position)] = true;
    }
    let mut paths = vec![SelectorPath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        pools: ledger.pools.clone(),
        trace: Vec::new(),
    }];
    for event in &ledger.events {
        let mut next = Vec::new();
        let mut retained_entries = 0usize;
        for path in paths {
            for draw in draws(&path.pools, &event.selector)? {
                let probability = path.probability.multiply(
                    draw.branch_numerator,
                    draw.branch_denominator * draw.candidates.len() as u64,
                )?;
                for (index, role) in draw.candidates {
                    if next.len() >= MAX_PATHS {
                        return Err(LedgerError::Capacity);
                    }
                    let script = &path.pools.script;
                    let entry_budget = path.pools.unique.len()
                        + path.pools.duplicate.len()
                        + path.pools.must_include.len()
                        + script.villagers.len()
                        + script.outcasts.len()
                        + script.minions.len()
                        + script.demons.len()
                        + path.trace.len()
                        + 2; // next trace and possible script append
                    retained_entries = retained_entries
                        .checked_add(entry_budget)
                        .ok_or(LedgerError::Capacity)?;
                    if retained_entries > MAX_RETAINED_ENTRIES {
                        return Err(LedgerError::Capacity);
                    }
                    let mut branch = path.clone();
                    branch.probability = probability;
                    let occurrence_index = index as u16; // pool validation bounds indices
                    let source = match draw.source {
                        PoolSource::Duplicate => {
                            BluffAcquisitionSource::DuplicatePool { occurrence_index }
                        }
                        PoolSource::Unique => {
                            BluffAcquisitionSource::UniquePool { occurrence_index }
                        }
                        PoolSource::MustInclude => {
                            let removed = branch
                                .pools
                                .must_include
                                .iter()
                                .position(|entry| entry == role)
                                .expect("selected live asset");
                            branch.pools.must_include.remove(removed);
                            BluffAcquisitionSource::BluffMustInclude { occurrence_index }
                        }
                    };
                    let script_added = draw.source != PoolSource::Duplicate
                        && register(&mut branch.pools.script, role);
                    let corruption_attempt = match event.selector {
                        Selector::Drunk {
                            corruption_resistant: true,
                        } => Some(CorruptionAttempt::Resisted),
                        Selector::Drunk {
                            corruption_resistant: false,
                        } => Some(CorruptionAttempt::AcceptedSelf),
                        _ => None,
                    };
                    branch.trace.push(SelectorTrace {
                        event: event.clone(),
                        bluff_role: role.clone(),
                        source,
                        rng_draw_count: draw.rng_draw_count,
                        script_added,
                        corruption_attempt,
                    });
                    next.push(branch);
                }
            }
        }
        paths = next;
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bluff::{
        enumerate_twin_recipient_bluffs_after_one_lilis, TWIN_RECIPIENT_BLUFF_NATIVE_V1,
        TWIN_RECIPIENT_BLUFF_ONE_LILIS_PREFIX_NATIVE_V1,
    };
    use crate::types::{
        DelayedRevealAcquisitionEvent, TwinRecipientBluffContext, TwinRecipientBluffPrefixContext,
    };

    fn names(roles: &[&str]) -> Vec<String> {
        roles.iter().map(|role| role.to_string()).collect()
    }

    fn input(selectors: Vec<Selector>, must: &[&str]) -> SelectorLedger {
        SelectorLedger {
            rule_version: SELECTOR_LEDGER_NATIVE_V1.into(),
            pools: SelectorPools {
                unique: names(&["Witness"]),
                duplicate: names(&["Confessor"]),
                must_include: names(must),
                script: ScriptLists {
                    villagers: names(&["Bard"]),
                    outcasts: vec![],
                    minions: names(&["Minion"]),
                    demons: names(&["Lilis"]),
                },
            },
            events: selectors
                .into_iter()
                .enumerate()
                .map(|(i, selector)| SelectorEvent {
                    position: (i + 1) as u8,
                    acquisition_ordinal: (i * 3) as u16,
                    selector,
                })
                .collect(),
        }
    }

    fn p(numerator: u64, denominator: u64) -> Probability {
        Probability {
            numerator,
            denominator,
        }
    }

    #[test]
    fn minion_then_demon_keeps_unconditional_mass_across_changed_widths() {
        let paths = replay_selectors(&input(
            vec![Selector::Minion, Selector::Demon],
            &["Scout", "Bard"],
        ))
        .unwrap();
        // Duplicate branch leaves two Demon options: 2/5 * 1/2 each.
        // Unique branch has two first draws, each leaving one: 3/5 * 1/2 each.
        assert_eq!(paths.len(), 4);
        assert_eq!(
            paths
                .iter()
                .map(|path| path.probability)
                .collect::<Vec<_>>(),
            vec![p(1, 5), p(1, 5), p(3, 10), p(3, 10)]
        );
        assert_eq!(paths[0].pools.must_include, names(&["Bard"]));
        assert_eq!(paths[1].pools.must_include, names(&["Scout"]));
        assert!(paths[2..]
            .iter()
            .all(|path| path.pools.must_include.is_empty()));
        assert!(!paths[0].trace[0].script_added);
        assert!(!paths[0]
            .pools
            .script
            .villagers
            .contains(&"Confessor".into()));
        assert_eq!(paths[0].trace[0].rng_draw_count, 2);
        assert_eq!(paths[2].trace[0].rng_draw_count, 3);
    }

    #[test]
    fn demon_drunk_minion_drain_only_their_eligible_occurrences() {
        let ledger = input(
            vec![
                Selector::Demon,
                Selector::Drunk {
                    corruption_resistant: false,
                },
                Selector::Minion,
            ],
            &["Bombardier", "Scout"],
        );
        let paths = replay_selectors(&ledger).unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].probability, p(2, 5));
        assert_eq!(paths[1].probability, p(3, 5));
        for path in &paths {
            assert_eq!(path.trace[0].bluff_role, "Scout");
            assert_eq!(path.trace[1].bluff_role, "Witness");
            assert_eq!(path.trace[0].rng_draw_count, 2);
            assert_eq!(path.trace[1].rng_draw_count, 1);
            assert_eq!(
                path.trace[1].corruption_attempt,
                Some(CorruptionAttempt::AcceptedSelf)
            );
            assert_eq!(path.pools.unique, ledger.pools.unique);
            assert_eq!(path.pools.duplicate, ledger.pools.duplicate);
        }
        assert_eq!(paths[0].pools.must_include, names(&["Bombardier"]));
        assert!(paths[1].pools.must_include.is_empty());
        assert_eq!(paths[1].pools.script.outcasts, names(&["Bombardier"]));
    }

    #[test]
    fn separated_duplicates_keep_source_indices_but_remove_first_equal_asset() {
        let paths = replay_selectors(&input(
            vec![Selector::Demon],
            &["Scout", "Bombardier", "Scout", "Bard"],
        ))
        .unwrap();
        assert_eq!(paths.len(), 3);
        assert!(paths.iter().all(|path| path.probability == p(1, 3)));
        assert_eq!(
            paths[0].pools.must_include,
            names(&["Bombardier", "Scout", "Bard"])
        );
        assert_eq!(paths[0].pools, paths[1].pools);
        assert_eq!(
            paths[1].trace[0].source,
            BluffAcquisitionSource::BluffMustInclude {
                occurrence_index: 2
            }
        );
        assert_eq!(
            paths[2].trace[0].source,
            BluffAcquisitionSource::BluffMustInclude {
                occurrence_index: 3
            }
        );
    }

    #[test]
    fn existing_script_membership_does_not_remove_draws_or_reorder_lists() {
        let mut ledger = input(
            vec![
                Selector::Drunk {
                    corruption_resistant: true,
                },
                Selector::Drunk {
                    corruption_resistant: false,
                },
            ],
            &[],
        );
        ledger.pools.unique = names(&["Bard", "Bard"]);
        ledger.pools.script.villagers = names(&["Bard", "Bard", "Scout"]);
        let paths = replay_selectors(&ledger).unwrap();
        assert_eq!(paths.len(), 4);
        for path in paths {
            assert_eq!(path.probability, p(1, 4));
            assert_eq!(path.pools, ledger.pools);
            assert!(path.trace.iter().all(|trace| !trace.script_added));
            assert_eq!(
                path.trace[0].corruption_attempt,
                Some(CorruptionAttempt::Resisted)
            );
            assert_eq!(
                path.trace[1].corruption_attempt,
                Some(CorruptionAttempt::AcceptedSelf)
            );
        }
    }

    #[test]
    fn minion_duplicate_outcast_never_registers_or_consumes() {
        let mut ledger = input(vec![Selector::Minion], &[]);
        ledger.pools.duplicate = names(&["Bombardier", "Bombardier"]);
        let paths = replay_selectors(&ledger).unwrap();
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0].probability, p(1, 5));
        assert_eq!(paths[1].probability, p(1, 5));
        assert_eq!(paths[2].probability, p(3, 5));
        assert_eq!(paths[0].pools, ledger.pools);
        assert_eq!(paths[1].pools, ledger.pools);
        assert!(!paths[0].trace[0].script_added);
    }

    #[test]
    fn changing_event_order_changes_results() {
        let demon_first =
            replay_selectors(&input(vec![Selector::Demon, Selector::Minion], &["Scout"])).unwrap();
        let minion_first =
            replay_selectors(&input(vec![Selector::Minion, Selector::Demon], &["Scout"])).unwrap();
        assert!(demon_first
            .iter()
            .all(|path| path.trace[0].bluff_role == "Scout"));
        assert_eq!(minion_first[1].trace[1].bluff_role, "Witness");
        assert_eq!(demon_first[1].trace[1].bluff_role, "Witness");
        assert_eq!(minion_first[0].trace[1].bluff_role, "Scout");
    }

    #[test]
    fn exhausted_positive_mass_branch_rejects_whole_replay() {
        let mut ledger = input(vec![Selector::Minion, Selector::Demon], &["Scout"]);
        ledger.pools.unique = names(&["Bombardier"]);
        // Minion duplicate -> Demon succeeds, but Minion unique drains Scout
        // and leaves no Demon fallback. Returning only the former biases mass.
        assert_eq!(replay_selectors(&ledger), Err(LedgerError::EmptySupport));
        assert_eq!(ledger.pools.must_include, names(&["Scout"]));
    }

    #[test]
    fn unused_empty_pools_are_legal_but_required_empty_support_is_not() {
        let mut ledger = input(vec![Selector::Demon], &["Scout"]);
        ledger.pools.unique.clear();
        ledger.pools.duplicate.clear();
        assert_eq!(replay_selectors(&ledger).unwrap()[0].probability, p(1, 1));
        ledger.events[0].selector = Selector::Minion;
        assert_eq!(replay_selectors(&ledger), Err(LedgerError::EmptySupport));
        ledger.events[0].selector = Selector::Drunk {
            corruption_resistant: false,
        };
        ledger.pools.must_include.clear();
        assert_eq!(replay_selectors(&ledger), Err(LedgerError::EmptySupport));
    }

    #[test]
    fn invalid_provenance_assets_and_script_factions_are_rejected() {
        let base = input(vec![Selector::Demon, Selector::Minion], &[]);
        let mut bad = base.clone();
        bad.rule_version = "future".into();
        assert_eq!(replay_selectors(&bad), Err(LedgerError::InvalidContext));
        for events in [
            vec![SelectorEvent {
                position: 0,
                ..base.events[0].clone()
            }],
            vec![
                base.events[0].clone(),
                SelectorEvent {
                    position: 1,
                    ..base.events[1].clone()
                },
            ],
            vec![base.events[1].clone(), base.events[0].clone()],
            vec![
                base.events[0].clone(),
                SelectorEvent {
                    acquisition_ordinal: 0,
                    ..base.events[1].clone()
                },
            ],
        ] {
            assert_eq!(
                replay_selectors(&SelectorLedger {
                    events,
                    ..base.clone()
                }),
                Err(LedgerError::InvalidContext)
            );
        }
        for role in [
            "scout",
            " Scout",
            "Drunk",
            "Wretch",
            "Doppelganger",
            "Pooka",
        ] {
            let mut bad = base.clone();
            bad.pools.unique = names(&[role]);
            assert_eq!(replay_selectors(&bad), Err(LedgerError::InvalidContext));
        }
        let mut bad = base;
        bad.pools.script.villagers.push("Drunk".into());
        assert_eq!(replay_selectors(&bad), Err(LedgerError::InvalidContext));
    }

    #[test]
    fn capacity_limits_reject_without_partial_results() {
        let mut ledger = input(vec![Selector::Demon; 17], &[]);
        assert_eq!(replay_selectors(&ledger), Err(LedgerError::InvalidContext));
        ledger.events.truncate(1);
        ledger.pools.unique = vec!["Scout".into(); 2000];
        assert_eq!(replay_selectors(&ledger), Err(LedgerError::Capacity));
        ledger.pools.unique = vec!["Scout".into(); 65_537];
        assert_eq!(replay_selectors(&ledger), Err(LedgerError::InvalidContext));
        assert_eq!(p(1, u64::MAX).multiply(1, 2), Err(LedgerError::Capacity));
        assert_eq!(p(u64::MAX, 2).multiply(2, u64::MAX), Ok(p(1, 1)));
    }

    #[test]
    fn offline_schema_roundtrips_and_rejects_unknown_dispatch() {
        let ledger = input(
            vec![Selector::Drunk {
                corruption_resistant: true,
            }],
            &["Scout"],
        );
        let json = serde_json::to_string(&ledger).unwrap();
        assert_eq!(
            serde_json::from_str::<SelectorLedger>(&json).unwrap(),
            ledger
        );
        assert!(
            serde_json::from_str::<SelectorLedger>(&json.replace("\"drunk\"", "\"spy\"")).is_err()
        );
        let mut value = serde_json::to_value(&ledger).unwrap();
        value["live_solver_input"] = true.into();
        assert!(serde_json::from_value::<SelectorLedger>(value).is_err());
        let paths = replay_selectors(&ledger).unwrap();
        assert!(serde_json::to_value(paths).unwrap()[0]["probability"].is_object());
    }

    #[test]
    fn empty_ledger_is_identity() {
        let ledger = input(vec![], &[]);
        let paths = replay_selectors(&ledger).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].pools, ledger.pools);
        assert_eq!(paths[0].probability, p(1, 1));
        assert!(paths[0].trace.is_empty());
    }

    #[test]
    fn conditioned_ledger_matches_existing_one_lilis_ticket_distribution() {
        let mut ledger = input(
            vec![Selector::Demon, Selector::Minion],
            &["Scout", "Witness", "Scout"],
        );
        ledger.events[0].position = 2;
        ledger.events[1].position = 4;
        ledger.events[1].acquisition_ordinal = 7;
        let context = TwinRecipientBluffContext {
            rule_version: TWIN_RECIPIENT_BLUFF_NATIVE_V1.into(),
            recipient_position: 4,
            acquisition_ordinal: 7,
            duplicate_pool: ledger.pools.duplicate.clone(),
            unique_pool: ledger.pools.unique.clone(),
            bluff_must_include_at_recipient: names(&["Witness", "Scout"]),
        };
        let prefix = TwinRecipientBluffPrefixContext {
            rule_version: TWIN_RECIPIENT_BLUFF_ONE_LILIS_PREFIX_NATIVE_V1.into(),
            acquisition_order: vec![
                DelayedRevealAcquisitionEvent {
                    position: 2,
                    acquisition_ordinal: 0,
                },
                DelayedRevealAcquisitionEvent {
                    position: 4,
                    acquisition_ordinal: 7,
                },
                DelayedRevealAcquisitionEvent {
                    position: 6,
                    acquisition_ordinal: 9,
                },
            ],
            bluff_must_include_before_prefix: ledger.pools.must_include.clone(),
        };
        let old = enumerate_twin_recipient_bluffs_after_one_lilis(&context, &prefix, 2, 6).unwrap();
        let paths = replay_selectors(&ledger).unwrap();
        // Captured remainder selects the two Scout draws (total prior mass 2/3).
        let retained: Vec<_> = paths
            .iter()
            .filter(|path| path.trace[0].bluff_role == "Scout")
            .collect();
        let total_tickets: u64 = old.iter().map(|outcome| outcome.tickets).sum();
        assert_eq!(retained.len(), old.len());
        for (path, outcome) in retained.iter().zip(old) {
            assert_eq!(
                path.trace[0].source,
                outcome.trace.prior_acquisitions[0].source
            );
            assert_eq!(path.trace[1].source, outcome.trace.source);
            assert_eq!(path.trace[1].bluff_role, outcome.trace.bluff_role);
            assert_eq!(
                path.probability.multiply(3, 2).unwrap(),
                p(1, 1).multiply(outcome.tickets, total_tickets).unwrap()
            );
        }
    }
}
