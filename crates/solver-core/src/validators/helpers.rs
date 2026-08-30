/// Shared helper functions used by all validators.

use crate::knowledge_base::{get_card, normalize_role};
use crate::types::{GameState, Scenario};

/// Compare two role names ignoring case, spaces, and underscores.
/// "Twin Minion" == "Twin_Minion" == "twinminion"
pub fn roles_equal(a: &str, b: &str) -> bool {
    normalize_role(a) == normalize_role(b)
}

/// Truth status of a card in a given scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthStatus {
    Truthful,
    Lying,
}

/// Get the evil role name at a position, or None if the position is good.
/// Checks: scenario.evil_positions, puppet_position, executed_evil_roles, confirmed_evil.
pub fn known_evil_role<'a>(pos: u8, scenario: &'a Scenario, state: &'a GameState) -> Option<&'a str> {
    // Check scenario evil positions
    if let Some(role) = scenario.evil_positions.get(&pos) {
        return Some(role.as_str());
    }
    // Check puppet
    if scenario.puppet_position == Some(pos) {
        return Some("Puppet");
    }
    // Check executed evil roles
    if let Some(role) = state.executed_evil_roles.get(&pos) {
        return Some(role.as_str());
    }
    // Check confirmed evil that have been executed (role unknown)
    if state.confirmed_evil.contains(&pos) && state.executed.contains(&pos) {
        return Some("Unknown");
    }
    None
}

/// Is this position evil in the current board state?
pub fn is_evil_in_board_state(pos: u8, scenario: &Scenario, state: &GameState) -> bool {
    known_evil_role(pos, scenario, state).is_some()
}

/// True underlying role at a position, considering evil placements,
/// outcast placements, and apparent_role for undisguised good cards.
///
/// Used by public Dreamer role-pair validation, which can name any role.
pub fn effective_role_at(pos: u8, scenario: &Scenario, state: &GameState) -> Option<String> {
    // Shaman's later InitWithNoReset changes the current role data at both
    // endpoints without clearing the destination's runtime alignment. Current
    // role consumers must therefore prefer the trace over the original Evil,
    // Drunk, or Doppelganger identity; alignment consumers still use
    // `known_evil_role` / `effective_alignment` below.
    if let Some(trace) = scenario.shaman_trace.as_ref() {
        if trace.source_position == pos || trace.target_position == pos {
            if state
                .executed_good_roles
                .get(&pos)
                .map(String::as_str)
                .or_else(|| state.card_at(pos).map(|card| card.apparent_role.as_str()))
                .is_some_and(|role| normalize_role(role) == "baker")
            {
                return Some("Baker".to_string());
            }
            return Some(trace.copied_role.clone());
        }
    }
    if let Some(role) = known_evil_role(pos, scenario, state) {
        return Some(role.to_string());
    }
    if scenario.chancellor_added_outcast_position() == Some(pos) {
        if let Some(role) = scenario.chancellor_added_outcast_role() {
            return Some(role.to_string());
        }
    }
    if scenario.doppelganger_position == Some(pos) {
        return Some("Doppelganger".to_string());
    }
    if scenario.drunk_position == Some(pos) {
        return Some("Drunk".to_string());
    }
    if let Some(card) = state.card_at(pos) {
        return Some(card.apparent_role.clone());
    }
    None
}

/// Effective alignment as seen by abilities.
/// Wretch registers as Evil to abilities (but is actually Good).
pub fn effective_alignment(pos: u8, scenario: &Scenario, state: &GameState) -> EffectiveAlignment {
    if is_evil_in_board_state(pos, scenario, state) {
        return EffectiveAlignment::Evil;
    }
    // Wretch registers as Evil, including a Chancellor-generated Wretch whose
    // apparent Villager identity does not expose that role.
    if effective_role_at(pos, scenario, state)
        .is_some_and(|role| roles_equal(&role, "Wretch"))
    {
        return EffectiveAlignment::Evil;
    }
    EffectiveAlignment::Good
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveAlignment {
    Good,
    Evil,
}

/// Determine truth status using the native `CharacterHelper.CheckLying` order:
/// Corrupted, HealthyBluff, runtime Evil/non-null bluff, then truthful.
///
/// The scenario model has no general bluff-data pointer or status collection;
/// consequently there is no `Lying`/`Appear` status to consult (native code does
/// not consult either one here anyway).
/// It can represent the native cases relevant to generated scenarios as follows:
/// clean Puppet and clean Doppelganger model HealthyBluff, while Drunk models a
/// non-null bluff without HealthyBluff. Arbitrary good characters with bluff data
/// cannot be expressed until the scenario model gains such a field.
pub fn truth_status(pos: u8, scenario: &Scenario, state: &GameState) -> TruthStatus {
    // Native precedence: corruption overrides HealthyBluff and cant_lie roles.
    // Drunk is normally also present in this set; the explicit bluff mapping
    // below keeps hand-built scenarios faithful even when it is omitted.
    if scenario.corrupted.contains(&pos) {
        return TruthStatus::Lying;
    }

    let evil_role = known_evil_role(pos, scenario, state);
    let effective_role = effective_role_at(pos, scenario, state);

    // Puppet applies HealthyBluff during Start. A clean Doppelganger applies it
    // while acquiring its good bluff. Corrupted variants were handled above.
    let modeled_healthy_bluff = evil_role
        .map(|role| roles_equal(role, "Puppet"))
        .unwrap_or(false)
        || scenario.doppelganger_position == Some(pos)
        || effective_role
            .as_deref()
            .is_some_and(|role| roles_equal(role, "Doppelganger"));
    if modeled_healthy_bluff {
        return TruthStatus::Truthful;
    }

    // Runtime Evil is sufficient to lie even without bluff data. Drunk and
    // Doppelganger are the model's explicit non-null-bluff positions; the clean
    // Doppelganger case already returned via HealthyBluff.
    let modeled_non_null_bluff = scenario.drunk_position == Some(pos)
        || scenario.doppelganger_position == Some(pos)
        || effective_role.as_deref().is_some_and(|role| {
            roles_equal(role, "Drunk") || roles_equal(role, "Doppelganger")
        });
    if evil_role.is_some() || modeled_non_null_bluff {
        return TruthStatus::Lying;
    }

    TruthStatus::Truthful
}

/// Determine the truth condition exposed by native
/// `CharacterHelper.CheckLyingAppearance` for the statuses represented by the
/// scenario model.
///
/// Confessor applies `AppearTruthfull` during both its real and bluff-role Init
/// dispatch, so every card appearing as Confessor is perceived as truthful even
/// when actual action dispatch lies because it is corrupted or evil. The model
/// does not yet represent arbitrary appearance statuses; all other cards fall
/// back to the actual native lie predicate modeled by [`truth_status`].
pub fn truth_appearance_status(pos: u8, scenario: &Scenario, state: &GameState) -> TruthStatus {
    if state
        .card_at(pos)
        .map(|card| normalize_role(&card.apparent_role) == "confessor")
        .unwrap_or(false)
    {
        return TruthStatus::Truthful;
    }

    truth_status(pos, scenario, state)
}

#[cfg(test)]
mod truth_status_tests {
    use super::*;
    use crate::types::{CardInfo, ChancellorTrace, ShamanTrace};
    use std::collections::{HashMap, HashSet};

    fn state_with_apparent_role(role: &str) -> GameState {
        let mut state = GameState::default();
        state.n_cards = 1;
        state.cards.push(CardInfo {
            position: 1,
            apparent_role: role.to_string(),
            ..CardInfo::default()
        });
        state
    }

    fn scenario() -> Scenario {
        Scenario {
            evil_positions: HashMap::new(),
            puppet_position: None,
            corrupted: HashSet::new(),
            pd_corrupted: None,
            doppelganger_position: None,
            drunk_position: None,
            alchemist_cures: HashMap::new(),
            messed_up_by_evil: HashSet::new(),
            shaman_trace: None,
            chancellor_trace: None,
            chancellor_conversion: None,
        }
    }

    #[test]
    fn corrupted_confessor_lies_despite_cant_lie_role() {
        let state = state_with_apparent_role("Confessor");
        let mut scenario = scenario();
        scenario.corrupted.insert(1);

        assert_eq!(truth_status(1, &scenario, &state), TruthStatus::Lying);
    }

    #[test]
    fn corruption_overrides_puppet_healthy_bluff() {
        let state = state_with_apparent_role("Baker");
        let mut scenario = scenario();
        scenario.puppet_position = Some(1);
        scenario.corrupted.insert(1);

        assert_eq!(truth_status(1, &scenario, &state), TruthStatus::Lying);
    }

    #[test]
    fn corruption_overrides_doppelganger_healthy_bluff() {
        let state = state_with_apparent_role("Baker");
        let mut scenario = scenario();
        scenario.doppelganger_position = Some(1);
        scenario.corrupted.insert(1);

        assert_eq!(truth_status(1, &scenario, &state), TruthStatus::Lying);
    }

    #[test]
    fn clean_puppet_and_doppelganger_model_healthy_bluff() {
        let state = state_with_apparent_role("Baker");

        let mut puppet = scenario();
        puppet.puppet_position = Some(1);
        assert_eq!(truth_status(1, &puppet, &state), TruthStatus::Truthful);

        let mut doppelganger = scenario();
        doppelganger.doppelganger_position = Some(1);
        assert_eq!(
            truth_status(1, &doppelganger, &state),
            TruthStatus::Truthful
        );
    }

    #[test]
    fn drunk_models_non_null_bluff_without_healthy_bluff() {
        let state = state_with_apparent_role("Baker");
        let mut scenario = scenario();
        scenario.drunk_position = Some(1);

        assert_eq!(truth_status(1, &scenario, &state), TruthStatus::Lying);
    }

    #[test]
    fn shaman_role_overwrite_preserves_existing_bluff_truth_statuses() {
        let state = state_with_apparent_role("Knight");
        let trace = ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Knight".to_string(),
            target_previous_roles: vec!["Drunk".to_string(), "Doppelganger".to_string()],
        };

        let mut doppelganger = scenario();
        doppelganger.doppelganger_position = Some(1);
        doppelganger.shaman_trace = Some(trace.clone());
        assert_eq!(effective_role_at(1, &doppelganger, &state).as_deref(), Some("Knight"));
        assert_eq!(truth_status(1, &doppelganger, &state), TruthStatus::Truthful);

        let mut drunk = scenario();
        drunk.drunk_position = Some(1);
        drunk.shaman_trace = Some(trace);
        assert_eq!(effective_role_at(1, &drunk, &state).as_deref(), Some("Knight"));
        assert_eq!(truth_status(1, &drunk, &state), TruthStatus::Lying);
    }

    #[test]
    fn trace_only_resistant_drunk_still_lies() {
        let state = state_with_apparent_role("Baker");
        let mut scenario = scenario();
        scenario.chancellor_trace = Some(ChancellorTrace {
            original_positions: vec![1],
            added_outcast_position: 1,
            added_outcast_role: "Drunk".to_string(),
            affected_anchor_positions: vec![],
        });

        assert!(scenario.corrupted.is_empty());
        assert_eq!(truth_status(1, &scenario, &state), TruthStatus::Lying);
    }

    #[test]
    fn ordinary_evil_lies_even_when_appearing_as_confessor() {
        let state = state_with_apparent_role("Confessor");
        let mut scenario = scenario();
        scenario.evil_positions.insert(1, "Pooka".to_string());

        assert_eq!(truth_status(1, &scenario, &state), TruthStatus::Lying);
        assert_eq!(
            truth_appearance_status(1, &scenario, &state),
            TruthStatus::Truthful
        );
    }

    #[test]
    fn corrupted_confessor_lies_but_appears_truthful() {
        let state = state_with_apparent_role("Confessor");
        let mut scenario = scenario();
        scenario.corrupted.insert(1);

        assert_eq!(truth_status(1, &scenario, &state), TruthStatus::Lying);
        assert_eq!(
            truth_appearance_status(1, &scenario, &state),
            TruthStatus::Truthful
        );
    }

    #[test]
    fn clean_good_character_is_truthful() {
        let state = state_with_apparent_role("Baker");

        assert_eq!(truth_status(1, &scenario(), &state), TruthStatus::Truthful);
    }
}

/// Get the "real" role at a position, accounting for Doppelganger/Drunk disguise.
pub fn get_real_role<'a>(pos: u8, scenario: &'a Scenario, state: &'a GameState) -> &'a str {
    if let Some(trace) = scenario.shaman_trace.as_ref() {
        if trace.source_position == pos || trace.target_position == pos {
            if state
                .executed_good_roles
                .get(&pos)
                .map(String::as_str)
                .or_else(|| state.card_at(pos).map(|card| card.apparent_role.as_str()))
                .is_some_and(|role| normalize_role(role) == "baker")
            {
                return "Baker";
            }
            return &trace.copied_role;
        }
    }
    // Check evil role first
    if let Some(role) = known_evil_role(pos, scenario, state) {
        return role;
    }
    if scenario.chancellor_added_outcast_position() == Some(pos) {
        if let Some(role) = scenario.chancellor_added_outcast_role() {
            return role;
        }
    }
    // Hidden outcasts
    if scenario.doppelganger_position == Some(pos) {
        return "Doppelganger";
    }
    if scenario.drunk_position == Some(pos) {
        return "Drunk";
    }
    // Card's apparent role
    if let Some(card) = state.card_at(pos) {
        return &card.apparent_role;
    }
    "Unknown"
}

/// Get the type category of a position for Bishop validator.
/// Returns "Villager", "Outcast", "Minion", "Demon", or None if unknown.
///
/// When `include_chancellor_conv` is true (default), the final home of
/// Chancellor's added identity is reported by its real Outcast type. When
/// false, its revealed/register-as role surface is used instead. Bishop
/// fixtures currently require both native timing views.
pub fn get_position_type(pos: u8, scenario: &Scenario, state: &GameState) -> Option<&'static str> {
    get_position_type_ex(pos, scenario, state, true)
}

pub fn get_position_type_ex(
    pos: u8,
    scenario: &Scenario,
    state: &GameState,
    include_chancellor_conv: bool,
) -> Option<&'static str> {
    // Check evil role
    if let Some(role) = known_evil_role(pos, scenario, state) {
        if role == "Unknown" {
            return None;
        }
        // Check if demon
        if state.deck.demons.iter().any(|d| d == role) {
            return Some("Demon");
        }
        return Some("Minion");
    }
    // Hidden outcasts
    if scenario.doppelganger_position == Some(pos) || scenario.drunk_position == Some(pos) {
        return Some("Outcast");
    }
    // Chancellor's added identity has real Outcast data at its final home.
    if include_chancellor_conv && scenario.chancellor_added_outcast_position() == Some(pos) {
        if scenario
            .chancellor_added_outcast_role()
            .is_some_and(|role| roles_equal(role, "Wretch"))
        {
            return Some("Minion");
        }
        return Some("Outcast");
    }
    // Revealed card
    if let Some(card) = state.card_at(pos) {
        if card.apparent_role == "Wretch" {
            return Some("Minion"); // Wretch registers as Minion to Bishop
        }
        if let Some(kb_card) = get_card(&card.apparent_role) {
            return Some(kb_card.faction.as_str());
        }
    }
    None // Unrevealed
}

/// Helper to read an i64 from info_parsed.
pub fn info_i64(info: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<i64> {
    info.get(key)?.as_i64()
}

/// Helper to read a u8 position from info_parsed.
pub fn info_pos(info: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<u8> {
    info_i64(info, key).map(|v| v as u8)
}

/// Helper to read a bool from info_parsed.
pub fn info_bool(info: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<bool> {
    info.get(key)?.as_bool()
}

/// Helper to read a string from info_parsed.
pub fn info_str<'a>(
    info: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a str> {
    info.get(key)?.as_str()
}

/// Helper to read a list of positions from info_parsed.
pub fn info_targets(info: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<Vec<u8>> {
    let arr = info.get(key)?.as_array()?;
    Some(arr.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect())
}

/// Helper to read a list of strings from info_parsed (for example role options).
pub fn info_str_array<'a>(
    info: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<Vec<&'a str>> {
    let arr = info.get(key)?.as_array()?;
    Some(arr.iter().filter_map(|v| v.as_str()).collect())
}
