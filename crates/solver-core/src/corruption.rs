//! Native-ordered Start-phase corruption simulation.
//!
//! The shipped build does not compute one immutable corruption snapshot. Its
//! serialized Start order mutates the live status lists in this order:
//! Pooka, every Poisoner, Drunk, Puppeteer conversion, every Plague Doctor,
//! then every Alchemist. This module keeps those mutations together so target
//! eligibility and later Alchemist clues observe the correct prior state.

use std::collections::{HashMap, HashSet};

use crate::geometry::adjacent_positions;

/// Role/type facts that are already fixed by a placement hypothesis.
#[derive(Debug, Clone, Default)]
pub struct StartCorruptionContext {
    /// Real `dataRef.type == Villager` positions after Chancellor acts but
    /// before Puppeteer replaces its target with Puppet data.
    pub real_villagers_before_puppet: HashSet<u8>,

    /// Positions satisfying Plague Doctor's
    /// `(registerAs ?? dataRef).type == Villager` predicate at its Start slot.
    /// During the initial Start pass, delayed Reveal has not populated ordinary
    /// register-as values, so this is normally the post-conversion real-
    /// Villager set. It remains a separate field to preserve the native type
    /// boundary for future retrigger modeling.
    pub registered_villagers_at_pd_call: HashSet<u8>,

    /// Positions whose Init hook installed exact Corrupted resistance.
    pub corruption_resistant_at_init: HashSet<u8>,

    /// Positions whose Init hook installed exact `MessedUpByEvil`
    /// resistance. Status resistances are enum-specific; this must not be
    /// inferred from Corrupted resistance.
    pub messed_up_resistant_at_init: HashSet<u8>,

    /// True Alchemist actors that still own the role at the ordered Alchemist
    /// Start slot. Apparent Alchemist bluffs do not belong here.
    pub true_alchemist_positions: Vec<u8>,

    /// `MessedUpByEvil` statuses already present when Pooka begins. Chancellor
    /// installs one on its chosen real-Outcast anchor before every corruption
    /// producer in the serialized Start order.
    pub initial_messed_up_by_evil: HashSet<u8>,

    pub drunk_position: Option<u8>,
    pub puppet_position: Option<u8>,
    pub plague_doctor_acts: bool,
}

/// One observable outcome of Start-phase random target selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartCorruptionOutcome {
    /// Live Corrupted statuses after all Alchemist cure attempts.
    pub corrupted: HashSet<u8>,
    /// Plague Doctor's selected Start target, retained even if later cured.
    pub pd_target: Option<u8>,
    /// Per-Alchemist live scan/attempt count at that actor's Start turn.
    pub alchemist_counts: HashMap<u8, u8>,
    /// Persistent evil-effect marker used by Witness. Alchemist cures do not
    /// remove this status.
    pub messed_up_by_evil: HashSet<u8>,
}

/// Enumerate native-order corruption outcomes for a single role-placement
/// hypothesis. Random target choices branch; deterministic status mutations do
/// not.
pub fn enumerate_start_corruption(
    n_cards: u8,
    full_evil: &HashMap<u8, String>,
    context: &StartCorruptionContext,
    known_pd_target: Option<u8>,
) -> Vec<StartCorruptionOutcome> {
    let mut corrupted = HashSet::new();
    let mut messed_up_by_evil: HashSet<u8> = context
        .initial_messed_up_by_evil
        .iter()
        .copied()
        .filter(|position| !context.messed_up_resistant_at_init.contains(position))
        .collect();

    // Ordinary Start roles stop after the first match in CurrentCharacters,
    // whose construction order is highest displayed ID first.
    if let Some(pooka_position) = full_evil
        .iter()
        .filter(|(_, role)| role.as_str() == "Pooka")
        .map(|(&position, _)| position)
        .max()
    {
        for target in adjacent_positions(pooka_position, n_cards) {
            if context.real_villagers_before_puppet.contains(&target) {
                if !context.corruption_resistant_at_init.contains(&target) {
                    corrupted.insert(target);
                }
                // Pooka attempts this independently after Corrupted. The
                // represented Init resistance is exact-Corrupted only.
                if !context.messed_up_resistant_at_init.contains(&target) {
                    messed_up_by_evil.insert(target);
                }
            }
        }
    }

    // Poisoner is an explicit all-matches exception. Each later Poisoner sees
    // the mutations left by the earlier, higher-ID actor.
    let mut poisoner_positions: Vec<u8> = full_evil
        .iter()
        .filter(|(_, role)| role.as_str() == "Poisoner")
        .map(|(&position, _)| position)
        .collect();
    poisoner_positions.sort_unstable_by(|a, b| b.cmp(a));

    let mut corruption_branches = vec![(corrupted, messed_up_by_evil)];
    for poisoner_position in poisoner_positions {
        let mut next_branches = Vec::new();
        for (branch, affected) in corruption_branches {
            let mut candidates: Vec<u8> = adjacent_positions(poisoner_position, n_cards)
                .into_iter()
                .filter(|target| {
                    context.real_villagers_before_puppet.contains(target)
                        && !branch.contains(target)
                        && !context.corruption_resistant_at_init.contains(target)
                })
                .collect();
            candidates.sort_unstable();
            candidates.dedup();

            if candidates.is_empty() {
                next_branches.push((branch, affected));
            } else {
                // Native selection is mandatory when the filtered pool is not
                // empty. Duplicate occurrences only change probability weight,
                // not the set of logical outcomes.
                for target in candidates {
                    let mut selected = branch.clone();
                    selected.insert(target);
                    let mut selected_affected = affected.clone();
                    if !context.messed_up_resistant_at_init.contains(&target) {
                        selected_affected.insert(target);
                    }
                    next_branches.push((selected, selected_affected));
                }
            }
        }
        corruption_branches = dedup_status_sets(next_branches);
    }

    let mut pd_branches: Vec<(HashSet<u8>, HashSet<u8>, Option<u8>)> = Vec::new();
    for (mut branch, mut affected) in corruption_branches {
        // Drunk acts after Poisoner and writes a self-targeted Corrupted status.
        if let Some(drunk_position) = context.drunk_position {
            if !context.corruption_resistant_at_init.contains(&drunk_position) {
                branch.insert(drunk_position);
            }
        }

        // Puppeteer conversion calls Character.Init again, clearing active
        // statuses before Plague Doctor and Alchemist act.
        if let Some(puppet_position) = context.puppet_position {
            branch.remove(&puppet_position);
            affected.remove(&puppet_position);
            // Puppeteer initializes the replacement Puppet, which clears the
            // former role's statuses and immediately marks the new Puppet as
            // MessedUpByEvil for Witness.
            if !context.messed_up_resistant_at_init.contains(&puppet_position) {
                affected.insert(puppet_position);
            }
        }

        if !context.plague_doctor_acts {
            if known_pd_target.is_none() {
                pd_branches.push((branch, affected, None));
            }
            continue;
        }

        let mut candidates: Vec<u8> = context
            .registered_villagers_at_pd_call
            .iter()
            .copied()
            .filter(|target| {
                !branch.contains(target) && !context.corruption_resistant_at_init.contains(target)
            })
            .collect();
        candidates.sort_unstable();

        if let Some(known_target) = known_pd_target {
            if candidates.binary_search(&known_target).is_ok() {
                branch.insert(known_target);
                pd_branches.push((branch, affected, Some(known_target)));
            }
        } else if candidates.is_empty() {
            pd_branches.push((branch, affected, None));
        } else {
            for target in candidates {
                let mut selected = branch.clone();
                selected.insert(target);
                pd_branches.push((selected, affected.clone(), Some(target)));
            }
        }
    }

    let mut outcomes = Vec::new();
    for (mut branch, affected, pd_target) in pd_branches {
        let alchemist_counts = apply_alchemists(
            n_cards,
            &context.true_alchemist_positions,
            context.drunk_position,
            &mut branch,
        );
        outcomes.push(StartCorruptionOutcome {
            corrupted: branch,
            pd_target,
            alchemist_counts,
            messed_up_by_evil: affected,
        });
    }
    dedup_outcomes(outcomes)
}

fn apply_alchemists(
    n_cards: u8,
    positions: &[u8],
    drunk_position: Option<u8>,
    corrupted: &mut HashSet<u8>,
) -> HashMap<u8, u8> {
    let mut actors = positions.to_vec();
    actors.sort_unstable_by(|a, b| b.cmp(a));
    actors.dedup();

    let mut counts = HashMap::new();
    for actor in actors {
        // The native helper builds this list from live status at call time and
        // preserves the overlap duplicate found on three- and four-card boards.
        let poisoned_scan: Vec<u8> = alchemist_scan_positions(actor, n_cards)
            .into_iter()
            .filter(|position| corrupted.contains(position))
            .collect();
        let count = u8::try_from(poisoned_scan.len()).unwrap_or(u8::MAX);
        counts.insert(actor, count);

        // CurePoisons increments before every attempt and ignores the return
        // value. Drunk's role veto leaves its self-targeted status in place.
        for target in poisoned_scan {
            if Some(target) != drunk_position {
                corrupted.remove(&target);
            }
        }
    }
    counts
}

/// Reproduce the shipped helper's asymmetric list scan. CurrentCharacters is
/// published in descending displayed-ID order. After rotating the actor first
/// and removing it, Alchemist scans the first two entries and then up to two
/// from the end while deliberately stopping before index zero.
fn alchemist_scan_positions(actor: u8, n_cards: u8) -> Vec<u8> {
    if n_cards <= 1 || actor == 0 || actor > n_cards {
        return Vec::new();
    }

    let after_actor: Vec<u8> = (1..n_cards)
        .map(|step| ((actor as i16 - 1 - step as i16).rem_euclid(n_cards as i16) + 1) as u8)
        .collect();

    let mut result: Vec<u8> = after_actor.iter().take(2).copied().collect();
    for index in (1..after_actor.len()).rev().take(2) {
        result.push(after_actor[index]);
    }
    result
}

fn sorted_set_key(values: &HashSet<u8>) -> Vec<u8> {
    let mut key: Vec<u8> = values.iter().copied().collect();
    key.sort_unstable();
    key
}

fn sorted_count_key(values: &HashMap<u8, u8>) -> Vec<(u8, u8)> {
    let mut key: Vec<(u8, u8)> = values
        .iter()
        .map(|(&position, &count)| (position, count))
        .collect();
    key.sort_unstable();
    key
}

fn dedup_status_sets(
    values: Vec<(HashSet<u8>, HashSet<u8>)>,
) -> Vec<(HashSet<u8>, HashSet<u8>)> {
    let mut seen = HashSet::new();
    values
        .into_iter()
        .filter(|(corrupted, affected)| {
            seen.insert((sorted_set_key(corrupted), sorted_set_key(affected)))
        })
        .collect()
}

fn dedup_outcomes(values: Vec<StartCorruptionOutcome>) -> Vec<StartCorruptionOutcome> {
    let mut seen = HashSet::new();
    values
        .into_iter()
        .filter(|value| {
            seen.insert((
                sorted_set_key(&value.corrupted),
                value.pd_target,
                sorted_count_key(&value.alchemist_counts),
                sorted_set_key(&value.messed_up_by_evil),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roles(entries: &[(u8, &str)]) -> HashMap<u8, String> {
        entries
            .iter()
            .map(|&(position, role)| (position, role.to_string()))
            .collect()
    }

    fn context(real_villagers: &[u8]) -> StartCorruptionContext {
        let real: HashSet<u8> = real_villagers.iter().copied().collect();
        StartCorruptionContext {
            real_villagers_before_puppet: real.clone(),
            registered_villagers_at_pd_call: real,
            ..StartCorruptionContext::default()
        }
    }

    #[test]
    fn pooka_mutation_changes_later_poisoner_candidates() {
        let ctx = context(&[2, 4, 5]);
        let outcomes =
            enumerate_start_corruption(5, &roles(&[(1, "Pooka"), (3, "Poisoner")]), &ctx, None);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].corrupted, HashSet::from([2, 4, 5]));
    }

    #[test]
    fn only_highest_id_pooka_acts() {
        let ctx = context(&[2, 3, 5, 6]);
        let outcomes =
            enumerate_start_corruption(6, &roles(&[(1, "Pooka"), (4, "Pooka")]), &ctx, None);
        assert_eq!(outcomes[0].corrupted, HashSet::from([3, 5]));
    }

    #[test]
    fn all_poisoners_act_highest_id_first() {
        let ctx = context(&[2, 4]);
        let outcomes =
            enumerate_start_corruption(6, &roles(&[(3, "Poisoner"), (5, "Poisoner")]), &ctx, None);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].corrupted, HashSet::from([2, 4]));
    }

    #[test]
    fn alchemist_init_resistance_blocks_corruption() {
        let mut ctx = context(&[2, 5]);
        ctx.corruption_resistant_at_init.insert(2);
        let outcomes = enumerate_start_corruption(5, &roles(&[(1, "Pooka")]), &ctx, None);
        assert_eq!(outcomes[0].corrupted, HashSet::from([5]));
        assert_eq!(outcomes[0].messed_up_by_evil, HashSet::from([2, 5]));
    }

    #[test]
    fn alchemist_cure_does_not_remove_messed_up_by_evil() {
        let mut ctx = context(&[2, 5]);
        ctx.true_alchemist_positions = vec![3];
        ctx.corruption_resistant_at_init.insert(3);

        let outcomes = enumerate_start_corruption(5, &roles(&[(1, "Pooka")]), &ctx, None);

        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].corrupted.is_empty());
        assert_eq!(outcomes[0].alchemist_counts.get(&3), Some(&2));
        assert_eq!(outcomes[0].messed_up_by_evil, HashSet::from([2, 5]));
    }

    #[test]
    fn exact_corrupted_resistance_blocks_drunk_self_corruption() {
        let mut ctx = context(&[]);
        ctx.drunk_position = Some(2);
        ctx.corruption_resistant_at_init.insert(2);

        let outcomes = enumerate_start_corruption(5, &HashMap::new(), &ctx, None);

        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].corrupted.is_empty());
    }

    #[test]
    fn drunk_acts_after_poisoner_and_cannot_be_cured() {
        let mut ctx = context(&[2]);
        ctx.drunk_position = Some(5);
        ctx.true_alchemist_positions = vec![3];
        ctx.corruption_resistant_at_init.insert(3);
        let outcomes = enumerate_start_corruption(5, &roles(&[(1, "Poisoner")]), &ctx, None);
        assert_eq!(outcomes[0].alchemist_counts.get(&3), Some(&2));
        assert_eq!(outcomes[0].corrupted, HashSet::from([5]));
    }

    #[test]
    fn future_puppet_can_be_poisoned_then_conversion_clears_it() {
        let mut ctx = context(&[2]);
        ctx.puppet_position = Some(2);
        ctx.registered_villagers_at_pd_call.remove(&2);
        let outcomes =
            enumerate_start_corruption(5, &roles(&[(1, "Poisoner"), (2, "Puppet")]), &ctx, None);
        assert_eq!(outcomes[0].corrupted, HashSet::new());
        assert_eq!(outcomes[0].messed_up_by_evil, HashSet::from([2]));
    }

    #[test]
    fn plague_doctor_observes_prior_statuses_and_records_a_later_cured_target() {
        let mut ctx = context(&[2, 3, 5]);
        ctx.plague_doctor_acts = true;
        ctx.true_alchemist_positions = vec![5];
        ctx.corruption_resistant_at_init.insert(5);
        let outcomes = enumerate_start_corruption(5, &roles(&[(1, "Pooka")]), &ctx, Some(3));
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].pd_target, Some(3));
        assert_eq!(outcomes[0].alchemist_counts.get(&5), Some(&2));
        assert!(outcomes[0].corrupted.is_empty());
    }

    #[test]
    fn invalid_known_plague_doctor_target_eliminates_the_world() {
        let mut ctx = context(&[2]);
        ctx.plague_doctor_acts = true;
        ctx.corruption_resistant_at_init.insert(2);
        let outcomes = enumerate_start_corruption(5, &HashMap::new(), &ctx, Some(2));
        assert!(outcomes.is_empty());
    }

    #[test]
    fn sequential_alchemists_read_the_live_mutated_set() {
        let mut ctx = context(&[2, 4, 6]);
        ctx.plague_doctor_acts = true;
        ctx.true_alchemist_positions = vec![2, 6];
        ctx.corruption_resistant_at_init.extend([2, 6]);
        let outcomes = enumerate_start_corruption(7, &HashMap::new(), &ctx, Some(4));
        assert_eq!(outcomes[0].alchemist_counts.get(&6), Some(&1));
        assert_eq!(outcomes[0].alchemist_counts.get(&2), Some(&0));
        assert!(outcomes[0].corrupted.is_empty());
    }

    #[test]
    fn small_board_alchemist_scan_preserves_overlap_duplicates() {
        let mut ctx = context(&[1, 2]);
        ctx.plague_doctor_acts = true;
        ctx.true_alchemist_positions = vec![1];
        ctx.corruption_resistant_at_init.insert(1);
        let outcomes = enumerate_start_corruption(3, &HashMap::new(), &ctx, Some(2));
        assert_eq!(outcomes[0].alchemist_counts.get(&1), Some(&2));
        assert!(outcomes[0].corrupted.is_empty());
    }

    #[test]
    fn no_eligible_plague_doctor_target_is_a_single_no_target_outcome() {
        let mut ctx = context(&[]);
        ctx.plague_doctor_acts = true;
        let outcomes = enumerate_start_corruption(5, &HashMap::new(), &ctx, None);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].pd_target, None);
    }
}
