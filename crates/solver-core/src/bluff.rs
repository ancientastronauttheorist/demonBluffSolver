//! Exact weighted Minion bluff draw at one moved Twin recipient.
//!
//! This boundary starts from the occurrence-preserving native pools as they
//! existed at the recipient's first successful delayed-Reveal acquisition.
//! It deliberately does not reconstruct round-pool selection or coroutine
//! ordering. Callers must supply that hidden provenance through the guarded
//! offline context and fall back atomically when it is unavailable.

pub mod ledger;
pub mod reveal;
pub mod twin_writer;

use crate::knowledge_base::{get_card, Faction};
use crate::types::{
    BluffAcquisitionSource, RevealBluffAcquisitionTrace, TwinRecipientBluffContext,
    TwinRecipientBluffPrefixContext, TwinRecipientBluffTrace,
};

/// Exact provenance marker accepted by this native-current enumerator.
pub const TWIN_RECIPIENT_BLUFF_NATIVE_V1: &str = "twin_recipient_bluff_native_v1";

/// Exact bounded delayed-Reveal prefix: one Lilis acquisition, then the moved
/// Twin recipient, then the stable Shaman acquisition.
pub const TWIN_RECIPIENT_BLUFF_ONE_LILIS_PREFIX_NATIVE_V1: &str =
    "twin_recipient_bluff_one_lilis_prefix_native_v1";

/// Keep equal-ticket expansion bounded before scenario integration clones it.
const MAX_TOTAL_TICKETS: u64 = 65_536;

/// One occurrence-sensitive bluff result and its reduced integer mass.
///
/// Every returned ticket is equiprobable relative to every other ticket from
/// the same call. Repeated role names remain separate outcomes because native
/// selection is uniform over list occurrences, not distinct identities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightedTwinRecipientBluffOutcome {
    pub trace: TwinRecipientBluffTrace,
    pub tickets: u64,
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn checked_lcm(left: u64, right: u64) -> Option<u64> {
    let divisor = gcd(left, right);
    left.checked_div(divisor)?.checked_mul(right)
}

fn is_canonical_supported_bluff(role: &str) -> bool {
    if role.is_empty() || role.trim() != role {
        return false;
    }
    let Some(card) = get_card(role) else {
        return false;
    };
    if role != card.name {
        return false;
    }

    match card.faction {
        Faction::Villager => true,
        Faction::Outcast => !matches!(card.name, "Doppelganger" | "Drunk" | "Wretch"),
        Faction::Minion | Faction::Demon => false,
    }
}

fn pool_is_valid(pool: &[String]) -> bool {
    // A u16 index can represent the inclusive range 0..=u16::MAX.
    pool.len() <= usize::from(u16::MAX) + 1
        && pool.iter().all(|role| is_canonical_supported_bluff(role))
}

fn is_supported_villager_bluff(role: &str) -> bool {
    is_canonical_supported_bluff(role)
        && get_card(role).is_some_and(|card| card.faction == Faction::Villager)
}

/// Enumerate the native 40% duplicate / 60% unique Minion draw.
///
/// For duplicate width `D`, each occurrence has mass `2/(5D)`. For the active
/// unique width `U`, each occurrence has mass `3/(5U)`. A non-empty
/// must-include snapshot replaces the ordinary unique pool exactly at this
/// event. The returned integer tickets divide both fractions by their common
/// gcd, retaining exact relative mass without a floating-point weight.
///
/// `None` is a fail-closed malformed or over-cap context. Native empty-pool
/// failures are not converted into a no-bluff outcome.
pub fn enumerate_twin_recipient_bluffs(
    context: &TwinRecipientBluffContext,
) -> Option<Vec<WeightedTwinRecipientBluffOutcome>> {
    if context.rule_version != TWIN_RECIPIENT_BLUFF_NATIVE_V1
        || context.recipient_position == 0
        || context.duplicate_pool.is_empty()
        || context.unique_pool.is_empty()
        || !pool_is_valid(&context.duplicate_pool)
        || !pool_is_valid(&context.unique_pool)
        || !pool_is_valid(&context.bluff_must_include_at_recipient)
    {
        return None;
    }

    let (unique_source, active_unique_pool) = if context.bluff_must_include_at_recipient.is_empty()
    {
        (false, context.unique_pool.as_slice())
    } else {
        (true, context.bluff_must_include_at_recipient.as_slice())
    };

    let duplicate_width = u64::try_from(context.duplicate_pool.len()).ok()?;
    let unique_width = u64::try_from(active_unique_pool.len()).ok()?;
    let common_width = checked_lcm(duplicate_width, unique_width)?;

    let duplicate_tickets = 2u64
        .checked_mul(common_width)?
        .checked_div(duplicate_width)?;
    let unique_tickets = 3u64.checked_mul(common_width)?.checked_div(unique_width)?;
    if duplicate_tickets == 0 || unique_tickets == 0 {
        return None;
    }

    // Reduce the equal-ticket sample space when the two branch numerators
    // share a factor (for example D=4, U=6 makes every path equally likely).
    let ticket_divisor = gcd(duplicate_tickets, unique_tickets);
    let duplicate_tickets = duplicate_tickets.checked_div(ticket_divisor)?;
    let unique_tickets = unique_tickets.checked_div(ticket_divisor)?;
    let total_tickets = duplicate_width
        .checked_mul(duplicate_tickets)?
        .checked_add(unique_width.checked_mul(unique_tickets)?)?;
    if total_tickets == 0 || total_tickets > MAX_TOTAL_TICKETS {
        return None;
    }

    let outcome_capacity = context
        .duplicate_pool
        .len()
        .checked_add(active_unique_pool.len())?;
    let mut outcomes = Vec::with_capacity(outcome_capacity);

    for (index, role) in context.duplicate_pool.iter().enumerate() {
        outcomes.push(WeightedTwinRecipientBluffOutcome {
            trace: TwinRecipientBluffTrace {
                recipient_position: context.recipient_position,
                acquisition_ordinal: context.acquisition_ordinal,
                bluff_role: role.clone(),
                source: BluffAcquisitionSource::DuplicatePool {
                    occurrence_index: u16::try_from(index).ok()?,
                },
                prior_acquisitions: Vec::new(),
            },
            tickets: duplicate_tickets,
        });
    }

    for (index, role) in active_unique_pool.iter().enumerate() {
        let occurrence_index = u16::try_from(index).ok()?;
        let source = if unique_source {
            BluffAcquisitionSource::BluffMustInclude { occurrence_index }
        } else {
            BluffAcquisitionSource::UniquePool { occurrence_index }
        };
        outcomes.push(WeightedTwinRecipientBluffOutcome {
            trace: TwinRecipientBluffTrace {
                recipient_position: context.recipient_position,
                acquisition_ordinal: context.acquisition_ordinal,
                bluff_role: role.clone(),
                source,
                prior_acquisitions: Vec::new(),
            },
            tickets: unique_tickets,
        });
    }

    Some(outcomes)
}

/// Replay the smallest exact shared-state prefix before one moved Twin
/// recipient's Minion bluff acquisition.
///
/// Lilis uses the typed unique-Villager selector. A Villager occurrence in the
/// pre-prefix must-include list is chosen uniformly and removed; otherwise a
/// Villager occurrence is chosen uniformly from the immutable unique pool and
/// the must-include list is unchanged. Only branches producing the recipient's
/// independently captured post-Lilis snapshot survive. An empty result is an
/// exact contradiction; malformed provenance or an oversized ticket space is
/// `None` so the ordered caller can fall back atomically.
pub fn enumerate_twin_recipient_bluffs_after_one_lilis(
    context: &TwinRecipientBluffContext,
    prefix: &TwinRecipientBluffPrefixContext,
    lilis_position: u8,
    shaman_position: u8,
) -> Option<Vec<WeightedTwinRecipientBluffOutcome>> {
    let recipient_outcomes = enumerate_twin_recipient_bluffs(context)?;
    if prefix.rule_version != TWIN_RECIPIENT_BLUFF_ONE_LILIS_PREFIX_NATIVE_V1
        || lilis_position == 0
        || shaman_position == 0
        || lilis_position == shaman_position
        || lilis_position == context.recipient_position
        || shaman_position == context.recipient_position
        || prefix.acquisition_order.len() != 3
        || !pool_is_valid(&prefix.bluff_must_include_before_prefix)
    {
        return None;
    }

    let order = &prefix.acquisition_order;
    if order.iter().any(|event| event.position == 0)
        || order[0].position != lilis_position
        || order[1].position != context.recipient_position
        || order[2].position != shaman_position
        || order[1].acquisition_ordinal != context.acquisition_ordinal
        || order[0].acquisition_ordinal >= order[1].acquisition_ordinal
        || order[1].acquisition_ordinal >= order[2].acquisition_ordinal
    {
        return None;
    }

    let villager_must_include: Vec<(usize, &String)> = prefix
        .bluff_must_include_before_prefix
        .iter()
        .enumerate()
        .filter(|(_, role)| is_supported_villager_bluff(role))
        .collect();

    let mut lilis_outcomes = Vec::new();
    if villager_must_include.is_empty() {
        if prefix.bluff_must_include_before_prefix != context.bluff_must_include_at_recipient {
            return Some(Vec::new());
        }
        for (index, role) in context.unique_pool.iter().enumerate() {
            if !is_supported_villager_bluff(role) {
                continue;
            }
            lilis_outcomes.push(RevealBluffAcquisitionTrace {
                position: lilis_position,
                acquisition_ordinal: order[0].acquisition_ordinal,
                current_role: "Lilis".to_string(),
                bluff_role: role.clone(),
                source: BluffAcquisitionSource::UniquePool {
                    occurrence_index: u16::try_from(index).ok()?,
                },
            });
        }
        // Native Demon selection fails rather than yielding a no-bluff branch.
        if lilis_outcomes.is_empty() {
            return None;
        }
    } else {
        for (index, role) in villager_must_include {
            let mut remaining = prefix.bluff_must_include_before_prefix.clone();
            // The typed helper selects from a filtered copy, then calls
            // List.Remove(selected) on the original list. Canonical repeated
            // role occurrences represent the same CharacterData object here,
            // so native removal deletes the first equal occurrence rather
            // than necessarily the occurrence selected in the filtered copy.
            let removed_index = remaining.iter().position(|candidate| candidate == role)?;
            remaining.remove(removed_index);
            if remaining != context.bluff_must_include_at_recipient {
                continue;
            }
            lilis_outcomes.push(RevealBluffAcquisitionTrace {
                position: lilis_position,
                acquisition_ordinal: order[0].acquisition_ordinal,
                current_role: "Lilis".to_string(),
                bluff_role: role.clone(),
                source: BluffAcquisitionSource::BluffMustInclude {
                    occurrence_index: u16::try_from(index).ok()?,
                },
            });
        }
    }

    if lilis_outcomes.is_empty() {
        return Some(Vec::new());
    }

    let capacity = lilis_outcomes.len().checked_mul(recipient_outcomes.len())?;
    if capacity > usize::try_from(MAX_TOTAL_TICKETS).ok()? {
        return None;
    }
    let mut outcomes = Vec::with_capacity(capacity);
    let mut total_tickets = 0u64;
    for lilis_trace in lilis_outcomes {
        for recipient_outcome in &recipient_outcomes {
            total_tickets = total_tickets.checked_add(recipient_outcome.tickets)?;
            if total_tickets > MAX_TOTAL_TICKETS {
                return None;
            }
            let mut weighted = recipient_outcome.clone();
            weighted.trace.prior_acquisitions = vec![lilis_trace.clone()];
            outcomes.push(weighted);
        }
    }
    Some(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DelayedRevealAcquisitionEvent;
    use std::collections::HashMap;

    fn context(duplicate: &[&str], unique: &[&str]) -> TwinRecipientBluffContext {
        TwinRecipientBluffContext {
            rule_version: TWIN_RECIPIENT_BLUFF_NATIVE_V1.to_string(),
            recipient_position: 4,
            acquisition_ordinal: 7,
            duplicate_pool: duplicate.iter().map(|role| (*role).to_string()).collect(),
            unique_pool: unique.iter().map(|role| (*role).to_string()).collect(),
            bluff_must_include_at_recipient: Vec::new(),
        }
    }

    fn one_lilis_prefix(before: &[&str]) -> TwinRecipientBluffPrefixContext {
        TwinRecipientBluffPrefixContext {
            rule_version: TWIN_RECIPIENT_BLUFF_ONE_LILIS_PREFIX_NATIVE_V1.to_string(),
            acquisition_order: vec![
                DelayedRevealAcquisitionEvent {
                    position: 2,
                    acquisition_ordinal: 3,
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
            bluff_must_include_before_prefix: before
                .iter()
                .map(|role| (*role).to_string())
                .collect(),
        }
    }

    #[test]
    fn preserves_duplicate_occurrences_and_exact_reduced_mass() {
        let outcomes = enumerate_twin_recipient_bluffs(&context(
            &["Scout", "Scout", "Confessor"],
            &["Witness", "Confessor"],
        ))
        .expect("the exact pools are supported");

        assert_eq!(outcomes.len(), 5);
        assert_eq!(
            outcomes.iter().map(|outcome| outcome.tickets).sum::<u64>(),
            30
        );
        let mut role_tickets: HashMap<&str, u64> = HashMap::new();
        for outcome in &outcomes {
            *role_tickets
                .entry(outcome.trace.bluff_role.as_str())
                .or_default() += outcome.tickets;
        }
        assert_eq!(role_tickets.get("Scout"), Some(&8));
        assert_eq!(role_tickets.get("Witness"), Some(&9));
        assert_eq!(role_tickets.get("Confessor"), Some(&13));
        assert!(matches!(
            outcomes[0].trace.source,
            BluffAcquisitionSource::DuplicatePool {
                occurrence_index: 0
            }
        ));
        assert!(matches!(
            outcomes[1].trace.source,
            BluffAcquisitionSource::DuplicatePool {
                occurrence_index: 1
            }
        ));
        assert!(matches!(
            outcomes[4].trace.source,
            BluffAcquisitionSource::UniquePool {
                occurrence_index: 1
            }
        ));
        assert!(outcomes.iter().all(|outcome| {
            outcome.trace.recipient_position == 4
                && outcome.trace.acquisition_ordinal == 7
                && outcome.tickets > 0
        }));
    }

    #[test]
    fn nonempty_must_include_replaces_the_unique_pool() {
        let mut input = context(&["Scout"], &["Witness", "Bard"]);
        input.bluff_must_include_at_recipient =
            vec!["Confessor".to_string(), "Confessor".to_string()];

        let outcomes = enumerate_twin_recipient_bluffs(&input)
            .expect("the must-include snapshot is supported");
        assert_eq!(outcomes.len(), 3);
        assert_eq!(
            outcomes.iter().map(|outcome| outcome.tickets).sum::<u64>(),
            10
        );
        assert_eq!(outcomes[0].tickets, 4);
        assert!(outcomes[1..].iter().all(|outcome| {
            outcome.tickets == 3
                && outcome.trace.bluff_role == "Confessor"
                && matches!(
                    outcome.trace.source,
                    BluffAcquisitionSource::BluffMustInclude { .. }
                )
        }));
        assert!(!outcomes.iter().any(|outcome| matches!(
            outcome.trace.source,
            BluffAcquisitionSource::UniquePool { .. }
        )));
    }

    #[test]
    fn malformed_contexts_fail_closed() {
        let base = context(&["Scout"], &["Witness"]);

        let mut wrong_rule = base.clone();
        wrong_rule.rule_version = "future_rule".to_string();
        assert!(enumerate_twin_recipient_bluffs(&wrong_rule).is_none());

        let mut zero_recipient = base.clone();
        zero_recipient.recipient_position = 0;
        assert!(enumerate_twin_recipient_bluffs(&zero_recipient).is_none());

        let mut empty_duplicate = base.clone();
        empty_duplicate.duplicate_pool.clear();
        assert!(enumerate_twin_recipient_bluffs(&empty_duplicate).is_none());

        let mut empty_unique = base.clone();
        empty_unique.unique_pool.clear();
        assert!(enumerate_twin_recipient_bluffs(&empty_unique).is_none());

        for unsupported in [
            "",
            " Scout",
            "scout",
            "Not A Role",
            "Twin Minion",
            "Doppelganger",
            "Drunk",
            "Wretch",
        ] {
            let mut malformed = base.clone();
            malformed.unique_pool = vec![unsupported.to_string()];
            assert!(
                enumerate_twin_recipient_bluffs(&malformed).is_none(),
                "accepted unsupported bluff role {unsupported:?}"
            );
        }
    }

    #[test]
    fn oversized_occurrence_or_ticket_spaces_fail_closed() {
        let mut occurrence_overflow = context(&["Scout"], &["Witness"]);
        occurrence_overflow.duplicate_pool = vec!["Scout".to_string(); usize::from(u16::MAX) + 2];
        assert!(enumerate_twin_recipient_bluffs(&occurrence_overflow).is_none());

        // Coprime widths make the reduced common ticket space 5 * 257 * 263,
        // comfortably beyond the conservative integration cap.
        let capped = context(&vec!["Scout"; 257], &vec!["Witness"; 263]);
        assert!(enumerate_twin_recipient_bluffs(&capped).is_none());
    }

    #[test]
    fn common_ticket_factor_is_removed() {
        let outcomes = enumerate_twin_recipient_bluffs(&context(
            &["Scout", "Scout", "Scout", "Scout"],
            &[
                "Witness", "Witness", "Witness", "Witness", "Witness", "Witness",
            ],
        ))
        .expect("the reduced ticket space is small");

        assert_eq!(outcomes.len(), 10);
        assert_eq!(
            outcomes.iter().map(|outcome| outcome.tickets).sum::<u64>(),
            10
        );
        assert!(outcomes.iter().all(|outcome| outcome.tickets == 1));
    }

    #[test]
    fn one_lilis_prefix_preserves_duplicate_must_include_occurrences() {
        let mut input = context(&["Scout"], &["Witness"]);
        input.bluff_must_include_at_recipient = vec!["Scout".to_string(), "Witness".to_string()];
        let prefix = one_lilis_prefix(&["Scout", "Scout", "Witness"]);

        let outcomes = enumerate_twin_recipient_bluffs_after_one_lilis(&input, &prefix, 2, 6)
            .expect("the one-Lilis prefix is exact");

        assert_eq!(
            outcomes.iter().map(|outcome| outcome.tickets).sum::<u64>(),
            20
        );
        let source_totals: HashMap<u16, u64> =
            outcomes.iter().fold(HashMap::new(), |mut totals, outcome| {
                let prior = &outcome.trace.prior_acquisitions[0];
                let BluffAcquisitionSource::BluffMustInclude { occurrence_index } = prior.source
                else {
                    panic!("Lilis must use the typed must-include source");
                };
                *totals.entry(occurrence_index).or_default() += outcome.tickets;
                totals
            });
        assert_eq!(source_totals, HashMap::from([(0, 10), (1, 10)]));
        assert!(outcomes.iter().all(|outcome| {
            let prior = &outcome.trace.prior_acquisitions[0];
            prior.position == 2
                && prior.acquisition_ordinal == 3
                && prior.current_role == "Lilis"
                && prior.bluff_role == "Scout"
        }));
    }

    #[test]
    fn one_lilis_prefix_removes_first_equal_separated_duplicate() {
        let mut input = context(&["Confessor"], &["Witness"]);
        input.bluff_must_include_at_recipient = vec!["Witness".to_string(), "Scout".to_string()];
        let prefix = one_lilis_prefix(&["Scout", "Witness", "Scout"]);

        let outcomes = enumerate_twin_recipient_bluffs_after_one_lilis(&input, &prefix, 2, 6)
            .expect("both selected Scout occurrences remove the first equal asset");

        let source_totals: HashMap<u16, u64> =
            outcomes.iter().fold(HashMap::new(), |mut totals, outcome| {
                let prior = &outcome.trace.prior_acquisitions[0];
                let BluffAcquisitionSource::BluffMustInclude { occurrence_index } = prior.source
                else {
                    panic!("Lilis must use the typed must-include source");
                };
                *totals.entry(occurrence_index).or_default() += outcome.tickets;
                totals
            });
        assert_eq!(source_totals, HashMap::from([(0, 10), (2, 10)]));
        assert_eq!(
            outcomes.iter().map(|outcome| outcome.tickets).sum::<u64>(),
            20
        );
        assert!(outcomes
            .iter()
            .all(|outcome| { outcome.trace.prior_acquisitions[0].bluff_role == "Scout" }));
    }

    #[test]
    fn lilis_typed_selector_removes_only_villager_must_include() {
        let mut input = context(&["Scout"], &["Witness"]);
        input.bluff_must_include_at_recipient = vec!["Bombardier".to_string()];
        let prefix = one_lilis_prefix(&["Bombardier", "Scout"]);

        let outcomes = enumerate_twin_recipient_bluffs_after_one_lilis(&input, &prefix, 2, 6)
            .expect("the typed must-include transition is exact");

        assert_eq!(
            outcomes.iter().map(|outcome| outcome.tickets).sum::<u64>(),
            5
        );
        assert!(outcomes.iter().all(|outcome| {
            let prior = &outcome.trace.prior_acquisitions[0];
            prior.bluff_role == "Scout"
                && matches!(
                    prior.source,
                    BluffAcquisitionSource::BluffMustInclude {
                        occurrence_index: 1
                    }
                )
        }));
        assert!(outcomes.iter().any(|outcome| {
            outcome.trace.bluff_role == "Bombardier"
                && matches!(
                    outcome.trace.source,
                    BluffAcquisitionSource::BluffMustInclude {
                        occurrence_index: 0
                    }
                )
        }));
    }

    #[test]
    fn lilis_unique_fallback_leaves_outcast_must_include_untouched() {
        let mut input = context(&["Scout"], &["Witness", "Confessor", "Bombardier"]);
        input.bluff_must_include_at_recipient = vec!["Bombardier".to_string()];
        let prefix = one_lilis_prefix(&["Bombardier"]);

        let outcomes = enumerate_twin_recipient_bluffs_after_one_lilis(&input, &prefix, 2, 6)
            .expect("the unique-Villager fallback is exact");

        assert_eq!(
            outcomes.iter().map(|outcome| outcome.tickets).sum::<u64>(),
            10
        );
        let lilis_sources: std::collections::HashSet<(u16, String)> = outcomes
            .iter()
            .map(|outcome| {
                let prior = &outcome.trace.prior_acquisitions[0];
                let BluffAcquisitionSource::UniquePool { occurrence_index } = prior.source else {
                    panic!("Lilis must use the unique pool fallback");
                };
                (occurrence_index, prior.bluff_role.clone())
            })
            .collect();
        assert_eq!(
            lilis_sources,
            std::collections::HashSet::from([
                (0, "Witness".to_string()),
                (1, "Confessor".to_string()),
            ])
        );
        assert!(!lilis_sources.iter().any(|(index, _)| *index == 2));

        let mut unsupported = input.clone();
        unsupported.unique_pool = vec!["Bombardier".to_string()];
        assert!(
            enumerate_twin_recipient_bluffs_after_one_lilis(&unsupported, &prefix, 2, 6,).is_none()
        );
    }

    #[test]
    fn malformed_one_lilis_order_fails_and_snapshot_mismatch_is_empty() {
        let mut input = context(&["Scout"], &["Witness"]);
        input.bluff_must_include_at_recipient = Vec::new();
        let prefix = one_lilis_prefix(&["Scout", "Witness"]);
        assert_eq!(
            enumerate_twin_recipient_bluffs_after_one_lilis(&input, &prefix, 2, 6),
            Some(Vec::new())
        );

        let mut malformed = prefix;
        malformed.acquisition_order.swap(0, 1);
        assert!(
            enumerate_twin_recipient_bluffs_after_one_lilis(&input, &malformed, 2, 6,).is_none()
        );
    }

    #[test]
    fn zero_based_global_prefix_ordinal_is_accepted() {
        let mut input = context(&["Scout"], &["Witness"]);
        input.bluff_must_include_at_recipient = vec!["Witness".to_string()];
        let mut prefix = one_lilis_prefix(&["Scout", "Witness"]);
        prefix.acquisition_order[0].acquisition_ordinal = 0;

        let outcomes = enumerate_twin_recipient_bluffs_after_one_lilis(&input, &prefix, 2, 6)
            .expect("zero is valid when strict global order is retained");
        assert!(!outcomes.is_empty());
        assert!(outcomes
            .iter()
            .all(|outcome| { outcome.trace.prior_acquisitions[0].acquisition_ordinal == 0 }));
    }

    #[test]
    fn one_lilis_cross_product_cap_fails_closed() {
        let mut input = context(&["Scout"], &["Witness"]);
        input.bluff_must_include_at_recipient = vec!["Scout".to_string(); 256];
        let prefix = TwinRecipientBluffPrefixContext {
            bluff_must_include_before_prefix: vec!["Scout".to_string(); 257],
            ..one_lilis_prefix(&[])
        };

        assert!(enumerate_twin_recipient_bluffs_after_one_lilis(&input, &prefix, 2, 6,).is_none());
    }
}
