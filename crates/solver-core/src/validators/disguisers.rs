//! Versioned native source-pool checks for disguising Good Outcasts.

use super::baker::baker_history_supports_pre_day_role;
use super::known_evil_role;
use crate::knowledge_base::{get_card, normalize_role, Faction};
use crate::types::{BoardCountProvenance, GameState, Scenario};

pub(super) const DOPPEL_DRUNK_CURRENT_RULE: &str = "doppel_drunk_reveal_v1";

fn role_is_state_villager(role: &str, state: &GameState) -> bool {
    let normalized = normalize_role(role);
    if state
        .deck
        .villagers
        .iter()
        .any(|candidate| normalize_role(candidate) == normalized)
    {
        return true;
    }
    if state
        .deck
        .outcasts
        .iter()
        .chain(state.deck.minions.iter())
        .chain(state.deck.demons.iter())
        .any(|candidate| normalize_role(candidate) == normalized)
    {
        return false;
    }
    get_card(role).is_some_and(|card| card.faction == Faction::Villager)
}

fn unknown_role_label(role: &str) -> bool {
    matches!(
        normalize_role(role).as_str(),
        "" | "unknown" | "none" | "hidden"
    )
}

fn observed_final_role<'a>(position: u8, state: &'a GameState) -> Option<&'a str> {
    state
        .executed_good_roles
        .get(&position)
        .map(String::as_str)
        .or_else(|| {
            state
                .card_at(position)
                .map(|card| card.apparent_role.as_str())
        })
        .filter(|role| !unknown_role_label(role))
}

fn source_can_supply_clean_doppel_role(
    source: u8,
    actor: u8,
    copied_role: &str,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if source == actor
        || known_evil_role(source, scenario, state).is_some()
        || scenario.puppet_position == Some(source)
        || scenario.doppelganger_position == Some(source)
        || scenario.drunk_position == Some(source)
        || scenario.chancellor_added_outcast_position() == Some(source)
    {
        return false;
    }

    // Shaman has already completed before delayed Doppelganger Reveal. Both
    // ordered endpoints therefore carry the copied real Villager identity at
    // this boundary, even if a later Baker Day reveal changes appearance.
    if let Some(trace) = scenario.shaman_trace.as_ref() {
        if trace.source_position == source || trace.target_position == source {
            return role_is_state_villager(&trace.copied_role, state)
                && normalize_role(&trace.copied_role) == normalize_role(copied_role);
        }
    }

    match observed_final_role(source, state) {
        Some(final_role) if normalize_role(final_role) == "baker" => {
            baker_history_supports_pre_day_role(scenario, state, source, copied_role)
        }
        Some(final_role) if normalize_role(final_role) == normalize_role(copied_role) => {
            // The current data model has no per-asset `bluffable` bit. A
            // physically Good Villager is therefore retained unless another
            // represented fact proves it ineligible.
            role_is_state_villager(final_role, state)
        }
        Some(_) => false,
        // With a trusted pre-Start header, the shared history search can prove
        // whether this concrete hidden seat fits the exact post-Start physical
        // Villager occupancy and residual role multiset after Puppet,
        // Chancellor, Shaman, and Baker transformations. Legacy counts retain
        // their deliberately conservative unknown-seat behavior.
        None if state.board_count_provenance == BoardCountProvenance::TrustedPreStart => {
            baker_history_supports_pre_day_role(scenario, state, source, copied_role)
        }
        None => true,
    }
}

/// Reject only clean Doppelganger worlds whose audited physical source list is
/// provably empty. Corrupted Doppelganger output needs a separate source/bluff
/// model and intentionally remains conservative in this slice.
pub(super) fn validate_clean_doppel_source_support(scenario: &Scenario, state: &GameState) -> bool {
    if state.doppel_drunk_rule_version.as_deref() != Some(DOPPEL_DRUNK_CURRENT_RULE) {
        return true;
    }
    let Some(actor) = scenario.doppelganger_position else {
        return true;
    };
    if scenario.corrupted.contains(&actor) || actor == 0 || actor > state.n_cards {
        return true;
    }
    let Some(copied_role) = state
        .card_at(actor)
        .map(|card| card.apparent_role.as_str())
        .filter(|role| !unknown_role_label(role))
    else {
        return true;
    };
    if !role_is_state_villager(copied_role, state) {
        return false;
    }

    (1..=state.n_cards).any(|source| {
        source_can_supply_clean_doppel_role(source, actor, copied_role, scenario, state)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CardInfo, DeckComposition};
    use serde_json::json;

    fn card(position: u8, role: &str, info: serde_json::Value) -> CardInfo {
        CardInfo {
            position,
            apparent_role: role.to_string(),
            info_text: String::new(),
            info_parsed: info.as_object().unwrap().clone(),
        }
    }

    fn current_state(n_cards: u8, villagers: &[&str], cards: Vec<CardInfo>) -> GameState {
        GameState {
            n_cards,
            deck: DeckComposition {
                villagers: villagers.iter().map(|role| (*role).to_string()).collect(),
                outcasts: vec!["Doppelganger".to_string()],
                ..DeckComposition::default()
            },
            cards,
            doppel_drunk_rule_version: Some(DOPPEL_DRUNK_CURRENT_RULE.to_string()),
            ..GameState::default()
        }
    }

    fn clean_doppel(position: u8) -> Scenario {
        Scenario {
            doppelganger_position: Some(position),
            ..Scenario::default()
        }
    }

    #[test]
    fn one_card_clean_doppel_rejects_the_native_empty_source_pool() {
        let state = current_state(1, &["Knight"], vec![card(1, "Knight", json!({}))]);
        let scenario = clean_doppel(1);

        assert!(!validate_clean_doppel_source_support(&scenario, &state));
        assert!(!crate::validators::check_scenario(&scenario, &state));
    }

    #[test]
    fn puppet_erasure_is_not_a_surviving_physical_source() {
        let mut state = current_state(
            3,
            &["Knight"],
            vec![
                card(1, "Knight", json!({})),
                card(2, "Knight", json!({})),
                card(3, "Puppeteer", json!({})),
            ],
        );
        state.deck.minions = vec!["Puppeteer".to_string()];
        let mut scenario = clean_doppel(1);
        scenario.puppet_position = Some(2);
        scenario.evil_positions.insert(3, "Puppeteer".to_string());

        assert!(!validate_clean_doppel_source_support(&scenario, &state));
    }

    #[test]
    fn separate_surviving_role_copy_remains_eligible_after_puppet_erasure() {
        let mut state = current_state(
            4,
            &["Knight", "Knight"],
            vec![
                card(1, "Knight", json!({})),
                card(2, "Knight", json!({})),
                card(3, "Puppeteer", json!({})),
                card(4, "Knight", json!({})),
            ],
        );
        state.deck.minions = vec!["Puppeteer".to_string()];
        let mut scenario = clean_doppel(1);
        scenario.puppet_position = Some(2);
        scenario.evil_positions.insert(3, "Puppeteer".to_string());

        assert!(validate_clean_doppel_source_support(&scenario, &state));
    }

    #[test]
    fn hidden_and_dead_sources_remain_eligible() {
        let hidden = current_state(2, &["Knight"], vec![card(1, "Knight", json!({}))]);
        let scenario = clean_doppel(1);
        assert!(validate_clean_doppel_source_support(&scenario, &hidden));

        let mut dead = current_state(
            2,
            &["Knight"],
            vec![card(1, "Knight", json!({})), card(2, "Knight", json!({}))],
        );
        dead.executed = vec![2];
        dead.executed_good_roles.insert(2, "Knight".to_string());
        assert!(validate_clean_doppel_source_support(&scenario, &dead));
    }

    #[test]
    fn trusted_header_resolves_hidden_source_role_capacity_jointly() {
        let mut state = current_state(
            4,
            &["Knight"],
            vec![
                card(1, "Knight", json!({})),
                card(2, "Knight", json!({})),
                card(3, "Puppeteer", json!({})),
            ],
        );
        state.deck.minions = vec!["Puppeteer".to_string()];
        state.board_villager_count = Some(2);
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        let mut scenario = clean_doppel(1);
        scenario.puppet_position = Some(2);
        scenario.evil_positions.insert(3, "Puppeteer".to_string());

        // The header's two Villager slots were the Doppelganger and the sole
        // Knight erased into Puppet. Hidden #4 cannot invent another source.
        assert!(!validate_clean_doppel_source_support(&scenario, &state));
        assert!(!crate::validators::check_scenario(&scenario, &state));

        // A third header slot plus a second physical Knight asset lets hidden
        // #4 carry a surviving source jointly with Puppet's erased identity.
        state.board_villager_count = Some(3);
        state.deck.villagers.push("Knight".to_string());
        assert!(validate_clean_doppel_source_support(&scenario, &state));
        assert!(crate::validators::check_scenario(&scenario, &state));
    }

    #[test]
    fn baker_history_can_prove_a_pre_conversion_source_identity() {
        let mut state = current_state(
            3,
            &["Baker", "Scout"],
            vec![
                card(1, "Scout", json!({})),
                card(2, "Baker", json!({"original_role": "Scout"})),
                card(3, "Baker", json!({"original_role": "original"})),
            ],
        );
        state.baker_rule_version = Some("baker_day_reveal_v1".to_string());
        state.reveal_order = vec![3, 2, 1];
        let scenario = clean_doppel(1);

        assert!(baker_history_supports_pre_day_role(
            &scenario, &state, 2, "Scout",
        ));
        assert!(!baker_history_supports_pre_day_role(
            &scenario, &state, 2, "Baker",
        ));
        assert!(validate_clean_doppel_source_support(&scenario, &state));
    }

    #[test]
    fn unmarked_legacy_state_keeps_the_conservative_model() {
        let mut state = current_state(1, &["Knight"], vec![card(1, "Knight", json!({}))]);
        state.doppel_drunk_rule_version = None;
        let scenario = clean_doppel(1);

        assert!(validate_clean_doppel_source_support(&scenario, &state));
        assert!(crate::validators::check_scenario(&scenario, &state));
    }
}
