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

/// Effective alignment as seen by abilities.
/// Wretch registers as Evil to abilities (but is actually Good).
pub fn effective_alignment(pos: u8, scenario: &Scenario, state: &GameState) -> EffectiveAlignment {
    if is_evil_in_board_state(pos, scenario, state) {
        return EffectiveAlignment::Evil;
    }
    // Wretch registers as Evil
    if let Some(card) = state.card_at(pos) {
        if card.apparent_role == "Wretch" {
            return EffectiveAlignment::Evil;
        }
    }
    EffectiveAlignment::Good
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveAlignment {
    Good,
    Evil,
}

/// Determine truth status of a card at a position in a scenario.
/// - Confessor: always Truthful (can't lie)
/// - Puppet: always Truthful (evil but can't lie)
/// - Evil (non-Puppet, non-Confessor): Lying
/// - Corrupted: Lying
/// - Otherwise: Truthful
pub fn truth_status(pos: u8, scenario: &Scenario, state: &GameState) -> TruthStatus {
    // Confessor always truthful
    if let Some(card) = state.card_at(pos) {
        if card.apparent_role == "Confessor" {
            return TruthStatus::Truthful;
        }
    }
    // Evil characters
    if let Some(role) = known_evil_role(pos, scenario, state) {
        if role == "Puppet" {
            return TruthStatus::Truthful;
        }
        return TruthStatus::Lying;
    }
    // Corrupted
    if scenario.corrupted.contains(&pos) {
        return TruthStatus::Lying;
    }
    TruthStatus::Truthful
}

/// Get the "real" role at a position, accounting for Doppelganger/Drunk disguise.
pub fn get_real_role<'a>(pos: u8, scenario: &'a Scenario, state: &'a GameState) -> &'a str {
    // Check evil role first
    if let Some(role) = known_evil_role(pos, scenario, state) {
        return role;
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
/// When `include_chancellor_conv` is true (default), Chancellor's conversion
/// target is reported as Outcast. When false, the conversion is ignored — used
/// to model the "pre-conversion view" for Bishop's if-possible semantics.
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
    // Chancellor conversion target becomes Outcast (only if caller wants it)
    if include_chancellor_conv && scenario.chancellor_conversion == Some(pos) {
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
