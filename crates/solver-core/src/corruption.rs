//! Native-ordered Start-phase corruption simulation.
//!
//! The shipped build does not compute one immutable corruption snapshot. Its
//! serialized Start order mutates the live status lists in this order:
//! Pooka, every Poisoner, Drunk, then (after the Witch/Twin boundary)
//! Puppeteer conversion, every Plague Doctor, Shaman, and every Alchemist.
//! Witch has no status mutation represented here; Twin mutates current role
//! data between the two phases. This module keeps status mutations ordered so
//! target eligibility and later Alchemist clues observe the correct prior state.

use std::collections::{HashMap, HashSet};

use crate::geometry::adjacent_positions;
use crate::knowledge_base::normalize_role;
use crate::types::ShamanTrace;

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

    /// Caller-enumerated native Shaman source/target history. Placement and
    /// final-role consistency are validated outside this status simulator.
    pub shaman_trace: Option<ShamanTrace>,
}

/// Facts consumed by the serialized Start slots before Twin Minion acts.
///
/// This is deliberately independent from [`PostTwinCorruptionContext`]. A
/// later checkpoint can branch current role data at the Twin slot without
/// replaying Pooka, Poisoner, or Drunk.
#[derive(Debug, Clone, Default)]
pub(crate) struct PreTwinCorruptionContext {
    /// Real Villager data visible to Pooka and Poisoner before Twin and
    /// Puppeteer mutate current role data.
    pub real_villagers_at_pre_twin: HashSet<u8>,

    /// Positions whose Init hook installed exact Corrupted resistance.
    pub corruption_resistant_at_init: HashSet<u8>,

    /// Positions whose Init hook installed exact `MessedUpByEvil`
    /// resistance.
    pub messed_up_resistant_at_init: HashSet<u8>,

    /// Markers already present when Pooka begins, normally from Chancellor.
    pub initial_messed_up_by_evil: HashSet<u8>,

    /// The physical Drunk actor dispatched in the pre-Twin Start slot.
    pub drunk_actor_position: Option<u8>,
}

/// Live status state at the boundary immediately after the pre-Twin Start
/// producers have run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreTwinCorruptionOutcome {
    pub corrupted: HashSet<u8>,
    pub messed_up_by_evil: HashSet<u8>,
}

/// Facts consumed by Start slots after Twin Minion acts.
#[derive(Debug, Clone, Default)]
pub(crate) struct PostTwinCorruptionContext {
    /// Positions satisfying Plague Doctor's registered/current Villager
    /// predicate at its ordered Start slot.
    pub registered_villagers_at_pd_call: HashSet<u8>,

    pub corruption_resistant_at_init: HashSet<u8>,
    pub messed_up_resistant_at_init: HashSet<u8>,

    /// Alchemist actors that own current Alchemist data at the global
    /// Alchemist Start slot.
    pub true_alchemist_positions: Vec<u8>,

    /// Compatibility-only cure veto inherited from the original monolithic
    /// model. Exact Twin modeling must derive this from post-Twin current data
    /// before relying on it.
    pub legacy_drunk_cure_veto_position: Option<u8>,

    pub puppet_position: Option<u8>,
    pub plague_doctor_acts: bool,
    pub shaman_trace: Option<ShamanTrace>,
}

impl StartCorruptionContext {
    pub(crate) fn pre_twin_context(&self) -> PreTwinCorruptionContext {
        PreTwinCorruptionContext {
            real_villagers_at_pre_twin: self.real_villagers_before_puppet.clone(),
            corruption_resistant_at_init: self.corruption_resistant_at_init.clone(),
            messed_up_resistant_at_init: self.messed_up_resistant_at_init.clone(),
            initial_messed_up_by_evil: self.initial_messed_up_by_evil.clone(),
            drunk_actor_position: self.drunk_position,
        }
    }

    pub(crate) fn post_twin_context(&self) -> PostTwinCorruptionContext {
        PostTwinCorruptionContext {
            registered_villagers_at_pd_call: self.registered_villagers_at_pd_call.clone(),
            corruption_resistant_at_init: self.corruption_resistant_at_init.clone(),
            messed_up_resistant_at_init: self.messed_up_resistant_at_init.clone(),
            true_alchemist_positions: self.true_alchemist_positions.clone(),
            legacy_drunk_cure_veto_position: self.drunk_position,
            puppet_position: self.puppet_position,
            plague_doctor_acts: self.plague_doctor_acts,
            shaman_trace: self.shaman_trace.clone(),
        }
    }
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
    /// The ordered trace representation that produced Shaman's status/action
    /// effects; solver-equivalent overwritten roles share one candidate class.
    pub shaman_trace: Option<ShamanTrace>,
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
    let pre_twin_context = context.pre_twin_context();
    let post_twin_context = context.post_twin_context();
    let mut outcomes = Vec::new();

    for pre_twin_outcome in enumerate_pre_twin_corruption(n_cards, full_evil, &pre_twin_context) {
        outcomes.extend(enumerate_post_twin_corruption(
            n_cards,
            &pre_twin_outcome,
            &post_twin_context,
            known_pd_target,
        ));
    }

    // Preserve the monolithic API's global logical-outcome deduplication,
    // including duplicates that converge from distinct pre-Twin branches.
    dedup_outcomes(outcomes)
}

/// Enumerate the Start corruption producers serialized before Twin Minion:
/// Pooka, every Poisoner, and Drunk.
pub(crate) fn enumerate_pre_twin_corruption(
    n_cards: u8,
    pre_twin_current_roles: &HashMap<u8, String>,
    context: &PreTwinCorruptionContext,
) -> Vec<PreTwinCorruptionOutcome> {
    let mut corrupted = HashSet::new();
    let mut messed_up_by_evil: HashSet<u8> = context
        .initial_messed_up_by_evil
        .iter()
        .copied()
        .filter(|position| !context.messed_up_resistant_at_init.contains(position))
        .collect();

    // Ordinary Start roles stop after the first match in CurrentCharacters,
    // whose construction order is highest displayed ID first.
    if let Some(pooka_position) = pre_twin_current_roles
        .iter()
        .filter(|(_, role)| role.as_str() == "Pooka")
        .map(|(&position, _)| position)
        .max()
    {
        for target in adjacent_positions(pooka_position, n_cards) {
            if context.real_villagers_at_pre_twin.contains(&target) {
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
    let mut poisoner_positions: Vec<u8> = pre_twin_current_roles
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
                    context.real_villagers_at_pre_twin.contains(target)
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

    corruption_branches
        .into_iter()
        .map(|(mut corrupted, messed_up_by_evil)| {
            // Drunk acts after Poisoner and writes a self-targeted Corrupted status.
            if let Some(drunk_position) = context.drunk_actor_position {
                if !context
                    .corruption_resistant_at_init
                    .contains(&drunk_position)
                {
                    corrupted.insert(drunk_position);
                }
            }

            PreTwinCorruptionOutcome {
                corrupted,
                messed_up_by_evil,
            }
        })
        .collect()
}

/// Continue one pre-Twin status branch through Puppeteer conversion, Plague
/// Doctor, Shaman, and the global Alchemist pass.
pub(crate) fn enumerate_post_twin_corruption(
    n_cards: u8,
    pre_twin_outcome: &PreTwinCorruptionOutcome,
    context: &PostTwinCorruptionContext,
    known_pd_target: Option<u8>,
) -> Vec<StartCorruptionOutcome> {
    let mut branch = pre_twin_outcome.corrupted.clone();
    let mut affected = pre_twin_outcome.messed_up_by_evil.clone();

    // Puppeteer conversion calls Character.Init again, clearing active
    // statuses before Plague Doctor and Alchemist act.
    if let Some(puppet_position) = context.puppet_position {
        branch.remove(&puppet_position);
        affected.remove(&puppet_position);
        // Puppeteer initializes the replacement Puppet, which clears the
        // former role's statuses and immediately marks the new Puppet as
        // MessedUpByEvil for Witness.
        if !context
            .messed_up_resistant_at_init
            .contains(&puppet_position)
        {
            affected.insert(puppet_position);
        }
    }

    let mut pd_branches: Vec<(HashSet<u8>, HashSet<u8>, Option<u8>)> = Vec::new();
    if !context.plague_doctor_acts {
        if known_pd_target.is_none() {
            pd_branches.push((branch, affected, None));
        }
    } else {
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
    for (mut branch, mut affected, pd_target) in pd_branches {
        let shaman_trace = context.shaman_trace.clone();
        let mut alchemist_counts = HashMap::new();

        // Shaman acts after Plague Doctor and before the global Alchemist pass.
        // Its source marker precedes the target reinitialization and copied
        // Start action; its target marker follows that action.
        if let Some(trace) = shaman_trace.as_ref() {
            if !context
                .messed_up_resistant_at_init
                .contains(&trace.source_position)
            {
                affected.insert(trace.source_position);
            }

            if normalize_role(&trace.copied_role) == "alchemist" {
                let target = trace.target_position;
                if branch.contains(&target) {
                    // A pre-Shaman Corrupted Good target dispatches copied
                    // Alchemist.BluffAct(Start). Without WorkingAbility it
                    // records/reset state at zero and does not cure.
                    alchemist_counts.insert(target, 0);
                } else {
                    let count = apply_alchemist(
                        target,
                        n_cards,
                        context.legacy_drunk_cure_veto_position,
                        &mut branch,
                    );
                    alchemist_counts.insert(target, count);
                }
            }

            if !context
                .messed_up_resistant_at_init
                .contains(&trace.target_position)
            {
                affected.insert(trace.target_position);
            }
        }

        // The caller removes any copied target whose immediate Start guard
        // suppresses its later global dispatch. Preserve an immediate count if
        // a transitional caller accidentally supplies the same position here.
        let global_alchemist_counts = apply_alchemists(
            n_cards,
            &context.true_alchemist_positions,
            context.legacy_drunk_cure_veto_position,
            &mut branch,
        );
        for (position, count) in global_alchemist_counts {
            alchemist_counts.entry(position).or_insert(count);
        }
        outcomes.push(StartCorruptionOutcome {
            corrupted: branch,
            pd_target,
            alchemist_counts,
            messed_up_by_evil: affected,
            shaman_trace,
        });
    }
    dedup_outcomes(outcomes)
}

fn apply_alchemists(
    n_cards: u8,
    positions: &[u8],
    legacy_drunk_cure_veto_position: Option<u8>,
    corrupted: &mut HashSet<u8>,
) -> HashMap<u8, u8> {
    let mut actors = positions.to_vec();
    actors.sort_unstable_by(|a, b| b.cmp(a));
    actors.dedup();

    let mut counts = HashMap::new();
    for actor in actors {
        let count = apply_alchemist(actor, n_cards, legacy_drunk_cure_veto_position, corrupted);
        counts.insert(actor, count);
    }
    counts
}

fn apply_alchemist(
    actor: u8,
    n_cards: u8,
    legacy_drunk_cure_veto_position: Option<u8>,
    corrupted: &mut HashSet<u8>,
) -> u8 {
    // The native helper builds this list from live status at call time and
    // preserves the overlap duplicate found on three- and four-card boards.
    let poisoned_scan: Vec<u8> = alchemist_scan_positions(actor, n_cards)
        .into_iter()
        .filter(|position| corrupted.contains(position))
        .collect();
    let count = u8::try_from(poisoned_scan.len()).unwrap_or(u8::MAX);

    // CurePoisons increments before every attempt and ignores the return
    // value. Drunk's role veto leaves its self-targeted status in place.
    for target in poisoned_scan {
        if Some(target) != legacy_drunk_cure_veto_position {
            corrupted.remove(&target);
        }
    }
    count
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

fn dedup_status_sets(values: Vec<(HashSet<u8>, HashSet<u8>)>) -> Vec<(HashSet<u8>, HashSet<u8>)> {
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
                value.shaman_trace.clone(),
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

    fn shaman_trace(source: u8, target: u8, copied_role: &str) -> ShamanTrace {
        ShamanTrace {
            source_position: source,
            target_position: target,
            copied_role: copied_role.to_string(),
            target_previous_roles: vec!["Witness".to_string()],
        }
    }

    #[test]
    fn shaman_marks_both_distinct_endpoints_and_preserves_trace() {
        let mut ctx = context(&[2, 4]);
        ctx.shaman_trace = Some(shaman_trace(2, 4, "Scout"));

        let outcomes = enumerate_start_corruption(5, &HashMap::new(), &ctx, None);

        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].messed_up_by_evil, HashSet::from([2, 4]));
        assert_eq!(outcomes[0].shaman_trace, ctx.shaman_trace);
    }

    #[test]
    fn shaman_endpoint_markers_honor_exact_resistance() {
        let mut ctx = context(&[2, 4]);
        ctx.shaman_trace = Some(shaman_trace(2, 4, "Scout"));
        ctx.messed_up_resistant_at_init.insert(2);

        let outcomes = enumerate_start_corruption(5, &HashMap::new(), &ctx, None);

        assert_eq!(outcomes[0].messed_up_by_evil, HashSet::from([4]));
    }

    #[test]
    fn clean_copied_alchemist_acts_before_global_alchemists() {
        let mut ctx = context(&[2, 3, 4, 5]);
        ctx.shaman_trace = Some(shaman_trace(4, 3, "Alchemist"));
        ctx.true_alchemist_positions = vec![4];
        ctx.corruption_resistant_at_init.insert(4);

        let outcomes = enumerate_start_corruption(5, &roles(&[(1, "Pooka")]), &ctx, None);

        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].alchemist_counts.get(&3), Some(&2));
        assert_eq!(outcomes[0].alchemist_counts.get(&4), Some(&0));
        assert!(outcomes[0].corrupted.is_empty());
    }

    #[test]
    fn corrupted_copied_alchemist_records_zero_and_does_not_cure() {
        let mut ctx = context(&[1, 4]);
        ctx.plague_doctor_acts = true;
        ctx.shaman_trace = Some(shaman_trace(1, 4, "alchemist"));
        ctx.true_alchemist_positions = vec![1];
        ctx.corruption_resistant_at_init.insert(1);

        let outcomes = enumerate_start_corruption(7, &HashMap::new(), &ctx, Some(4));

        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].pd_target, Some(4));
        assert_eq!(outcomes[0].alchemist_counts.get(&4), Some(&0));
        assert_eq!(outcomes[0].alchemist_counts.get(&1), Some(&0));
        assert_eq!(outcomes[0].corrupted, HashSet::from([4]));
    }

    #[test]
    fn wrapper_matches_explicit_pre_and_post_twin_split() {
        let full_evil = roles(&[(1, "Pooka"), (4, "Poisoner")]);
        let legacy_context = StartCorruptionContext {
            real_villagers_before_puppet: HashSet::from([2, 3, 5, 6, 7, 8, 9]),
            registered_villagers_at_pd_call: HashSet::from([3, 5, 6, 7, 8, 9]),
            corruption_resistant_at_init: HashSet::from([5]),
            messed_up_resistant_at_init: HashSet::new(),
            true_alchemist_positions: vec![5],
            initial_messed_up_by_evil: HashSet::new(),
            drunk_position: Some(3),
            puppet_position: Some(2),
            plague_doctor_acts: true,
            shaman_trace: Some(shaman_trace(5, 7, "Alchemist")),
        };
        let pre_twin_context = PreTwinCorruptionContext {
            real_villagers_at_pre_twin: legacy_context.real_villagers_before_puppet.clone(),
            corruption_resistant_at_init: legacy_context.corruption_resistant_at_init.clone(),
            messed_up_resistant_at_init: legacy_context.messed_up_resistant_at_init.clone(),
            initial_messed_up_by_evil: legacy_context.initial_messed_up_by_evil.clone(),
            drunk_actor_position: legacy_context.drunk_position,
        };
        let post_twin_context = PostTwinCorruptionContext {
            registered_villagers_at_pd_call: legacy_context.registered_villagers_at_pd_call.clone(),
            corruption_resistant_at_init: legacy_context.corruption_resistant_at_init.clone(),
            messed_up_resistant_at_init: legacy_context.messed_up_resistant_at_init.clone(),
            true_alchemist_positions: legacy_context.true_alchemist_positions.clone(),
            legacy_drunk_cure_veto_position: legacy_context.drunk_position,
            puppet_position: legacy_context.puppet_position,
            plague_doctor_acts: legacy_context.plague_doctor_acts,
            shaman_trace: legacy_context.shaman_trace.clone(),
        };

        let wrapped = enumerate_start_corruption(9, &full_evil, &legacy_context, Some(6));
        let pre_twin_outcomes = enumerate_pre_twin_corruption(9, &full_evil, &pre_twin_context);
        assert_eq!(pre_twin_outcomes.len(), 1);
        assert_eq!(pre_twin_outcomes[0].corrupted, HashSet::from([2, 3, 9]));

        let explicit_split = dedup_outcomes(
            pre_twin_outcomes
                .iter()
                .flat_map(|outcome| {
                    enumerate_post_twin_corruption(9, outcome, &post_twin_context, Some(6))
                })
                .collect(),
        );

        assert_eq!(wrapped, explicit_split);
        assert_eq!(wrapped.len(), 1);
        assert_eq!(wrapped[0].pd_target, Some(6));
        assert_eq!(wrapped[0].alchemist_counts.get(&7), Some(&2));
        assert_eq!(wrapped[0].alchemist_counts.get(&5), Some(&1));
        assert_eq!(wrapped[0].corrupted, HashSet::from([3]));
        assert_eq!(wrapped[0].messed_up_by_evil, HashSet::from([2, 3, 5, 7, 9]));
    }

    #[test]
    fn outcome_dedup_keeps_distinct_shaman_traces() {
        let outcome = |trace| StartCorruptionOutcome {
            corrupted: HashSet::new(),
            pd_target: None,
            alchemist_counts: HashMap::new(),
            messed_up_by_evil: HashSet::from([2, 4]),
            shaman_trace: Some(trace),
        };

        let outcomes = dedup_outcomes(vec![
            outcome(shaman_trace(2, 4, "Scout")),
            outcome(shaman_trace(4, 2, "Scout")),
        ]);

        assert_eq!(outcomes.len(), 2);
    }
}
