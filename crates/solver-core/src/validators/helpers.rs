/// Shared helper functions used by all validators.

use crate::knowledge_base::{get_card, normalize_role};
use crate::puppeteer::current_data_after_puppeteer_at;
use crate::twin::current_data_after_twin_at;
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

/// Stable runtime-Evil origin role at one physical position, before the later
/// generated-Puppet overlay.
///
/// Twin and Shaman current-data writes never change this layer. Puppeteer's
/// later full Init is modeled separately because it changes both runtime
/// alignment and current data after Twin.
pub fn stable_evil_origin_role_at<'a>(
    pos: u8,
    scenario: &'a Scenario,
    state: &'a GameState,
) -> Option<&'a str> {
    let exact_generated_puppet = scenario.puppet_position == Some(pos)
        && scenario
            .evil_positions
            .get(&pos)
            .is_some_and(|role| normalize_role(role) == "puppet");
    // Check scenario evil positions
    if let Some(role) = scenario.evil_positions.get(&pos) {
        if scenario.puppet_position != Some(pos) || normalize_role(role) != "puppet" {
            return Some(role.as_str());
        }
    }
    if exact_generated_puppet {
        // A public `Unknown` execution record must not overwrite the exact
        // generated-Puppet branch with a fictitious stable Evil origin.
        return None;
    }
    // Check executed evil roles
    if let Some(role) = state.executed_evil_roles.get(&pos) {
        if scenario.puppet_position != Some(pos) || normalize_role(role) != "puppet" {
            return Some(role.as_str());
        }
    }
    // Check confirmed evil that have been executed (role unknown)
    if scenario.puppet_position != Some(pos)
        && state.confirmed_evil.contains(&pos)
        && state.executed.contains(&pos)
    {
        return Some("Unknown");
    }
    None
}

/// Compatibility name preserving the historical stable-role-plus-Puppet view.
pub fn known_evil_role<'a>(
    pos: u8,
    scenario: &'a Scenario,
    state: &'a GameState,
) -> Option<&'a str> {
    stable_evil_origin_role_at(pos, scenario, state)
        .or_else(|| (scenario.puppet_position == Some(pos)).then_some("Puppet"))
}

/// Whether the physical Character remains runtime Evil.
pub fn is_runtime_evil_at(pos: u8, scenario: &Scenario, state: &GameState) -> bool {
    stable_evil_origin_role_at(pos, scenario, state).is_some()
        || scenario.puppet_position == Some(pos)
}

/// Compatibility name for physical runtime alignment.
pub fn is_evil_in_board_state(pos: u8, scenario: &Scenario, state: &GameState) -> bool {
    is_runtime_evil_at(pos, scenario, state)
}

/// Modeled current CharacterData before Twin's ordered Start dispatch.
///
/// Generated Puppet is deliberately absent: Puppeteer performs its full Init
/// after Twin and therefore belongs to the later writer below.
fn pre_twin_current_data_role_at(
    pos: u8,
    scenario: &Scenario,
    state: &GameState,
) -> Option<String> {
    if let Some(role) = scenario.pre_twin_current_roles.get(&pos) {
        return Some(role.clone());
    }
    if scenario.puppet_position != Some(pos) {
        if let Some(role) = stable_evil_origin_role_at(pos, scenario, state) {
            return Some(role.to_string());
        }
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
    state.card_at(pos).map(|card| card.apparent_role.clone())
}

/// Current CharacterData role at a physical position after modeled Start
/// writers, independently of physical runtime alignment.
///
/// Native writer order is the pre-Twin baseline, exact Twin swap, generated
/// Puppet full Init, then Shaman's later copied-role InitWithNoReset. CardInfo
/// presentation is only a baseline fallback and never overrides an exact Twin,
/// Puppet, or Shaman writer.
pub fn current_data_role_at(
    pos: u8,
    scenario: &Scenario,
    state: &GameState,
) -> Option<String> {
    let mut current = pre_twin_current_data_role_at(pos, scenario, state);

    if let Some(trace) = scenario.twin_trace.as_ref() {
        current = current_data_after_twin_at(pos, current.as_deref(), trace);
    }

    if let Some(trace) = scenario.puppeteer_trace.as_ref() {
        current = current_data_after_puppeteer_at(pos, current.as_deref(), trace);
    } else if scenario.puppet_position == Some(pos) {
        current = Some("Puppet".to_string());
    }

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

    current
}

/// Villager identity saved by Puppeteer before replacing `pos` with Puppet
/// current data. This is copied/bluff provenance, not the current role.
pub fn puppeteer_erased_villager_role_at<'a>(
    pos: u8,
    scenario: &'a Scenario,
) -> Option<&'a str> {
    let trace = scenario.puppeteer_trace.as_ref()?;
    match &trace.outcome {
        crate::types::PuppeteerStartOutcome::Converted {
            target_position,
            erased_villager_role,
            ..
        } if *target_position == pos => Some(erased_villager_role.as_str()),
        _ => None,
    }
}

/// Compatibility name for the modeled current CharacterData role.
pub fn effective_role_at(pos: u8, scenario: &Scenario, state: &GameState) -> Option<String> {
    current_data_role_at(pos, scenario, state)
}

/// Registered alignment as seen by abilities.
/// Wretch current data registers as Evil even on a runtime-Good body.
pub fn registered_alignment_at(
    pos: u8,
    scenario: &Scenario,
    state: &GameState,
) -> EffectiveAlignment {
    if is_runtime_evil_at(pos, scenario, state) {
        return EffectiveAlignment::Evil;
    }
    // Wretch registers as Evil, including a Chancellor-generated Wretch whose
    // apparent Villager identity does not expose that role.
    if current_data_role_at(pos, scenario, state)
        .is_some_and(|role| roles_equal(&role, "Wretch"))
    {
        return EffectiveAlignment::Evil;
    }
    EffectiveAlignment::Good
}

/// Compatibility name for registered alignment.
pub fn effective_alignment(pos: u8, scenario: &Scenario, state: &GameState) -> EffectiveAlignment {
    registered_alignment_at(pos, scenario, state)
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
    let modeled_healthy_bluff = scenario.puppet_position == Some(pos)
        || evil_role
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

/// Whether the exact shipped Shaman history guarantees that one physical card
/// received Confessor's `AppearTruthfull` status.
///
/// A copied Confessor's immediate Start dispatch is a no-op. Both ordered
/// endpoints nevertheless own the copied identity when their delayed internal
/// Reveal dispatches Init, and both the real and bluff Confessor Init paths add
/// the status. No shipped producer grants resistance to this status. Later
/// `InitWithNoReset` writers preserve it. Merely listing Confessor among the
/// target's possible erased roles is not sufficient evidence.
pub fn shaman_copied_confessor_status_at(pos: u8, scenario: &Scenario) -> bool {
    scenario.shaman_trace.as_ref().is_some_and(|trace| {
        roles_equal(&trace.copied_role, "Confessor")
            && (trace.source_position == pos || trace.target_position == pos)
    })
}

/// Determine the truth condition exposed by native
/// `CharacterHelper.CheckLyingAppearance` for the statuses represented by the
/// scenario model.
///
/// Confessor applies `AppearTruthfull` during both its real and bluff-role Init
/// dispatch, so every card appearing as Confessor is perceived as truthful even
/// when actual action dispatch lies because it is corrupted or evil. The exact
/// Shaman copied-Confessor endpoints retain the same physical status even after
/// a later presentation change. The model does not yet represent arbitrary
/// appearance statuses; all other cards fall back to the actual native lie
/// predicate modeled by [`truth_status`].
pub fn truth_appearance_status(pos: u8, scenario: &Scenario, state: &GameState) -> TruthStatus {
    if shaman_copied_confessor_status_at(pos, scenario)
        || state
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
    use crate::types::{
        CardInfo, ChancellorTrace, ShamanTrace, TwinNeighborSide, TwinStartOutcome, TwinTrace,
    };
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
            twin_trace: None,
            pre_twin_current_roles: HashMap::new(),
            puppeteer_trace: None,
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
    fn generated_puppet_over_stable_twin_uses_puppet_healthy_bluff() {
        let state = state_with_apparent_role("Scout");
        let mut scenario = scenario();
        scenario
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        scenario.puppet_position = Some(1);

        assert_eq!(known_evil_role(1, &scenario, &state), Some("Twin Minion"));
        assert_eq!(
            current_data_role_at(1, &scenario, &state).as_deref(),
            Some("Puppet")
        );
        assert_eq!(truth_status(1, &scenario, &state), TruthStatus::Truthful);

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
    fn shaman_copied_confessor_endpoints_keep_truthful_appearance_after_presentation_changes() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.cards = vec![
            CardInfo {
                position: 1,
                apparent_role: "Baker".to_string(),
                ..CardInfo::default()
            },
            CardInfo {
                position: 2,
                apparent_role: "Scout".to_string(),
                ..CardInfo::default()
            },
            CardInfo {
                position: 3,
                apparent_role: "Witness".to_string(),
                ..CardInfo::default()
            },
        ];
        let mut scenario = scenario();
        scenario.corrupted = HashSet::from([1, 2, 3]);
        scenario.shaman_trace = Some(ShamanTrace {
            source_position: 1,
            target_position: 2,
            copied_role: "Confessor".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });

        for pos in 1..=3 {
            assert_eq!(truth_status(pos, &scenario, &state), TruthStatus::Lying);
        }
        assert_eq!(
            truth_appearance_status(1, &scenario, &state),
            TruthStatus::Truthful,
        );
        assert_eq!(
            truth_appearance_status(2, &scenario, &state),
            TruthStatus::Truthful,
        );
        assert_eq!(
            truth_appearance_status(3, &scenario, &state),
            TruthStatus::Lying,
        );
    }

    #[test]
    fn clean_good_character_is_truthful() {
        let state = state_with_apparent_role("Baker");

        assert_eq!(truth_status(1, &scenario(), &state), TruthStatus::Truthful);
    }

    #[test]
    fn twin_current_data_moves_without_runtime_alignment_or_presentation() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.cards.push(CardInfo {
            position: 2,
            apparent_role: "Knight".to_string(),
            ..CardInfo::default()
        });
        let mut scenario = scenario();
        scenario.evil_positions = HashMap::from([
            (1, "Twin Minion".to_string()),
            (3, "Pooka".to_string()),
        ]);
        scenario.twin_trace = Some(TwinTrace {
            actor_position: 1,
            outcome: TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Scout".to_string(),
            },
        });

        assert_eq!(stable_evil_origin_role_at(1, &scenario, &state), Some("Twin Minion"));
        assert_eq!(stable_evil_origin_role_at(2, &scenario, &state), None);
        assert!(is_runtime_evil_at(1, &scenario, &state));
        assert!(!is_runtime_evil_at(2, &scenario, &state));
        assert_eq!(
            current_data_role_at(1, &scenario, &state).as_deref(),
            Some("Scout")
        );
        assert_eq!(
            current_data_role_at(2, &scenario, &state).as_deref(),
            Some("Twin Minion")
        );
        assert_eq!(
            registered_alignment_at(1, &scenario, &state),
            EffectiveAlignment::Evil
        );
        assert_eq!(
            registered_alignment_at(2, &scenario, &state),
            EffectiveAlignment::Good
        );
    }

    #[test]
    fn current_data_writers_apply_twin_then_puppet_then_shaman_baker() {
        let mut state = state_with_apparent_role("Scout");
        state.n_cards = 3;
        let mut scenario = scenario();
        scenario.evil_positions = HashMap::from([
            (1, "Twin Minion".to_string()),
            (3, "Pooka".to_string()),
        ]);
        scenario.twin_trace = Some(TwinTrace {
            actor_position: 1,
            outcome: TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Scout".to_string(),
            },
        });
        scenario.puppet_position = Some(1);

        assert_eq!(stable_evil_origin_role_at(1, &scenario, &state), Some("Twin Minion"));
        assert_eq!(
            current_data_role_at(1, &scenario, &state).as_deref(),
            Some("Puppet")
        );

        scenario.shaman_trace = Some(ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Knight".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });
        assert_eq!(
            current_data_role_at(1, &scenario, &state).as_deref(),
            Some("Knight")
        );
        assert!(is_runtime_evil_at(1, &scenario, &state));

        state.cards[0].apparent_role = "Baker".to_string();
        assert_eq!(
            current_data_role_at(1, &scenario, &state).as_deref(),
            Some("Baker")
        );
    }

    #[test]
    fn generated_puppet_is_not_part_of_the_pre_twin_baseline() {
        let mut state = GameState::default();
        state.n_cards = 3;
        let mut scenario = scenario();
        scenario
            .evil_positions
            .insert(1, "Puppet".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());
        scenario.puppet_position = Some(1);
        scenario.twin_trace = Some(TwinTrace {
            actor_position: 1,
            outcome: TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Scout".to_string(),
            },
        });

        assert_eq!(stable_evil_origin_role_at(1, &scenario, &state), None);
        assert_eq!(known_evil_role(1, &scenario, &state), Some("Puppet"));
        assert!(is_runtime_evil_at(1, &scenario, &state));
        assert_eq!(pre_twin_current_data_role_at(1, &scenario, &state), None);
        assert_eq!(
            current_data_role_at(1, &scenario, &state).as_deref(),
            Some("Puppet")
        );

        let mut unmarked_legacy = scenario.clone();
        unmarked_legacy.puppet_position = None;
        unmarked_legacy.twin_trace = None;
        assert_eq!(
            stable_evil_origin_role_at(1, &unmarked_legacy, &state),
            Some("Puppet")
        );
        assert_eq!(
            current_data_role_at(1, &unmarked_legacy, &state).as_deref(),
            Some("Puppet")
        );

        let mut executed_state = state.clone();
        executed_state.executed.push(1);
        executed_state.confirmed_evil.push(1);
        executed_state
            .executed_evil_roles
            .insert(1, "Puppet".to_string());
        let mut executed_only = Scenario::default();
        executed_only.puppet_position = Some(1);
        assert_eq!(
            stable_evil_origin_role_at(1, &executed_only, &executed_state),
            None
        );
        assert_eq!(
            known_evil_role(1, &executed_only, &executed_state),
            Some("Puppet")
        );
        assert_eq!(
            current_data_role_at(1, &executed_only, &executed_state).as_deref(),
            Some("Puppet")
        );
        assert_eq!(
            truth_status(1, &executed_only, &executed_state),
            TruthStatus::Truthful
        );
    }

    #[test]
    fn current_wretch_registration_does_not_change_runtime_alignment() {
        let mut state = state_with_apparent_role("Wretch");
        state.n_cards = 2;
        let scenario = scenario();

        assert!(!is_runtime_evil_at(1, &scenario, &state));
        assert_eq!(
            registered_alignment_at(1, &scenario, &state),
            EffectiveAlignment::Evil
        );
        assert_eq!(known_evil_role(1, &scenario, &state), None);
        assert_eq!(
            effective_role_at(1, &scenario, &state),
            current_data_role_at(1, &scenario, &state)
        );
        assert_eq!(
            is_evil_in_board_state(1, &scenario, &state),
            is_runtime_evil_at(1, &scenario, &state)
        );
        assert_eq!(
            effective_alignment(1, &scenario, &state),
            registered_alignment_at(1, &scenario, &state)
        );
    }

    #[test]
    fn trace_none_current_data_preserves_legacy_role_precedence() {
        let mut state = GameState::default();
        state.n_cards = 7;
        state.cards.push(CardInfo {
            position: 6,
            apparent_role: "Scout".to_string(),
            ..CardInfo::default()
        });
        let mut scenario = scenario();
        scenario.evil_positions.insert(1, "Pooka".to_string());
        scenario.puppet_position = Some(2);
        scenario.chancellor_trace = Some(ChancellorTrace {
            original_positions: vec![1],
            added_outcast_position: 3,
            added_outcast_role: "Bombardier".to_string(),
            affected_anchor_positions: vec![],
        });
        scenario.doppelganger_position = Some(4);
        scenario.drunk_position = Some(5);

        let expected = [
            (1, Some("Pooka")),
            (2, Some("Puppet")),
            (3, Some("Bombardier")),
            (4, Some("Doppelganger")),
            (5, Some("Drunk")),
            (6, Some("Scout")),
            (7, None),
        ];
        for (position, role) in expected {
            assert_eq!(
                current_data_role_at(position, &scenario, &state).as_deref(),
                role
            );
            assert_eq!(
                effective_role_at(position, &scenario, &state).as_deref(),
                role
            );
        }
    }
}

/// Compatibility helper for validators that require an owned fallback role.
pub fn get_real_role(pos: u8, scenario: &Scenario, state: &GameState) -> String {
    current_data_role_at(pos, scenario, state).unwrap_or_else(|| "Unknown".to_string())
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
