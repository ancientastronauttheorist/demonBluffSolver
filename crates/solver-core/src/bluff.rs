//! Exact weighted Minion bluff draw at one moved Twin recipient.
//!
//! This boundary starts from the occurrence-preserving native pools as they
//! existed at the recipient's first successful delayed-Reveal acquisition.
//! It deliberately does not reconstruct round-pool selection or coroutine
//! ordering. Callers must supply that hidden provenance through the guarded
//! offline context and fall back atomically when it is unavailable.

use crate::knowledge_base::{get_card, Faction};
use crate::types::{BluffAcquisitionSource, TwinRecipientBluffContext, TwinRecipientBluffTrace};

/// Exact provenance marker accepted by this native-current enumerator.
pub const TWIN_RECIPIENT_BLUFF_NATIVE_V1: &str = "twin_recipient_bluff_native_v1";

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
            },
            tickets: unique_tickets,
        });
    }

    Some(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
