/// Core data types for the Demon Bluff solver.
/// Matches the Python GameState/Scenario/SolverResult exactly for JSON compat.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{HashMap, HashSet};

// ── CardInfo ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardInfo {
    pub position: u8,
    pub apparent_role: String,
    #[serde(default)]
    pub info_text: String,
    #[serde(default)]
    pub info_parsed: serde_json::Map<String, serde_json::Value>,
}

// ── DeckComposition ──

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeckComposition {
    #[serde(default)]
    pub villagers: Vec<String>,
    #[serde(default)]
    pub outcasts: Vec<String>,
    #[serde(default)]
    pub minions: Vec<String>,
    #[serde(default)]
    pub demons: Vec<String>,
}

impl DeckComposition {
    pub fn evil_roles(&self) -> Vec<String> {
        let mut r = self.minions.clone();
        r.extend(self.demons.iter().cloned());
        r
    }

    pub fn all_roles(&self) -> Vec<String> {
        let mut r = self.villagers.clone();
        r.extend(self.outcasts.iter().cloned());
        r.extend(self.minions.iter().cloned());
        r.extend(self.demons.iter().cloned());
        r
    }
}

/// Whether serialized board V/O counts are normalized current-build header
/// evidence or an older value whose UI source is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BoardCountProvenance {
    TrustedPreStart,
    #[default]
    LegacyUnknown,
}

// ── Serde helpers for dict[int, T] stored as dict[str, T] in JSON ──

fn serialize_int_key_map<S: Serializer, V: Serialize>(
    map: &HashMap<u8, V>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;
    let mut m = serializer.serialize_map(Some(map.len()))?;
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_unstable_by_key(|(key, _)| **key);
    for (k, v) in entries {
        m.serialize_entry(&k.to_string(), v)?;
    }
    m.end()
}

fn deserialize_int_key_map_str<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<HashMap<u8, String>, D::Error> {
    let raw: HashMap<String, String> = HashMap::deserialize(deserializer)?;
    raw.into_iter()
        .map(|(k, v)| {
            k.parse::<u8>()
                .map(|k| (k, v))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

fn deserialize_int_key_map_bool<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<HashMap<u8, bool>, D::Error> {
    let raw: HashMap<String, bool> = HashMap::deserialize(deserializer)?;
    raw.into_iter()
        .map(|(k, v)| {
            k.parse::<u8>()
                .map(|k| (k, v))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

fn deserialize_int_key_map_u8<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<HashMap<u8, u8>, D::Error> {
    let raw: HashMap<String, u8> = HashMap::deserialize(deserializer)?;
    raw.into_iter()
        .map(|(k, v)| {
            k.parse::<u8>()
                .map(|k| (k, v))
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

// ── SlayerResult / PdAbilityResult ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlayerResult {
    pub slayer_pos: u8,
    pub target_pos: u8,
    pub killed: bool,
    /// Public true role revealed by the native kill path. Historical saves
    /// called this `evil_role`, before Wretch kills were modeled correctly.
    #[serde(default, alias = "evil_role")]
    pub revealed_role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdAbilityResult {
    pub pd_pos: u8,
    pub target: u8,
    pub is_corrupted: bool,
    /// Public evil-position result. Early v2 fixtures used `evil_pos` before
    /// the live bridge standardized on `evil_revealed`.
    #[serde(default, alias = "evil_pos")]
    pub evil_revealed: Option<u8>,
}

/// One publicly observed Rambler2 replacement in chronological capture order.
///
/// `speaker_position` is the character whose clue was replaced;
/// `shut_up_target` is the Rambler source named by the public text. The
/// top-level history survives later actions that replace a card's latest
/// `info_parsed.shut_up_target` alias.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamblerShutUpObservation {
    pub speaker_position: u8,
    pub shut_up_target: u8,
}

// ── GameState ──

/// Full state of a game in progress. Matches Python's GameState exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub n_cards: u8,
    #[serde(default)]
    pub n_evil: u8,

    /// Deck composition — supports both nested {"deck": {...}} and flat format.
    #[serde(
        default,
        deserialize_with = "deserialize_deck",
        skip_serializing // we serialize it manually
    )]
    pub deck: DeckComposition,

    // For serialization, we handle deck nesting in a custom impl if needed.
    // For now, the flat fields are captured by serde(flatten) in RawGameState.

    #[serde(default)]
    pub cards: Vec<CardInfo>,
    #[serde(default)]
    pub executed: Vec<u8>,
    #[serde(default)]
    pub confirmed_evil: Vec<u8>,
    #[serde(default)]
    pub confirmed_good: Vec<u8>,
    /// Hidden native Start history. Live play must leave this unknown; an
    /// explicit value is reserved for offline fixtures/post-mortem checks.
    #[serde(default)]
    pub pd_corruption_target: Option<u8>,

    #[serde(
        default,
        serialize_with = "serialize_int_key_map",
        deserialize_with = "deserialize_int_key_map_str"
    )]
    pub executed_evil_roles: HashMap<u8, String>,

    #[serde(default)]
    pub slayer_results: Vec<SlayerResult>,
    #[serde(default)]
    pub pd_ability_results: Vec<PdAbilityResult>,
    #[serde(default)]
    pub blocked_positions: Vec<u8>,
    #[serde(default)]
    pub night_kills: Vec<u8>,
    #[serde(default)]
    pub night_kill_evil_count: u8,

    #[serde(default = "default_hp")]
    pub hp: i32,
    #[serde(default = "default_wrong_exec_cost")]
    pub wrong_exec_cost: i32,

    #[serde(default)]
    pub board_villager_count: Option<u8>,
    #[serde(default)]
    pub board_outcast_count: Option<u8>,
    #[serde(default)]
    pub board_minion_count: Option<u8>,
    #[serde(default)]
    pub board_demon_count: Option<u8>,
    #[serde(default)]
    pub board_count_provenance: BoardCountProvenance,

    #[serde(default)]
    pub reveal_order: Vec<u8>,

    /// Public capture provenance for shipped Baker Day/reveal chronology.
    /// Missing/null preserves archived fixtures whose `reveal_order` recorded
    /// click attempts or card-entry order rather than verified native reveals.
    #[serde(default)]
    pub baker_rule_version: Option<String>,

    /// Native clean-Doppelganger physical-source filtering provenance.
    /// Missing/null keeps archived fixtures on the pre-audit conservative
    /// model; fresh current sessions opt into the audited reveal lifecycle.
    #[serde(default)]
    pub doppel_drunk_rule_version: Option<String>,

    /// Exact current-build Fortune Teller picker/output/history provenance.
    /// Missing/null preserves archived scalar fixtures whose references or
    /// speech were not captured from one coherent native event.
    #[serde(default)]
    pub fortune_teller_rule_version: Option<String>,

    #[serde(
        default,
        serialize_with = "serialize_int_key_map",
        deserialize_with = "deserialize_int_key_map_bool"
    )]
    pub executed_good_corrupted: HashMap<u8, bool>,

    /// Revealed true roles of executed good cards. This preserves observable
    /// generated-Outcast identity across solver calls without exposing it
    /// before execution.
    #[serde(
        default,
        serialize_with = "serialize_int_key_map",
        deserialize_with = "deserialize_int_key_map_str"
    )]
    pub executed_good_roles: HashMap<u8, String>,

    // ── Test-only / session fields (silently ignored if absent) ──
    #[serde(default)]
    pub used_abilities: Vec<u8>,
    /// Public clue-capture provenance for Rambler's current replacement rule.
    /// Missing/null preserves frozen fixtures whose absent `shut_up_target`
    /// values cannot be interpreted as negative evidence.
    #[serde(default)]
    pub rambler_rule_version: Option<String>,
    /// Append-only public Rambler interruption history. Per-card
    /// `info_parsed.shut_up_target` remains a backward-compatible latest-value
    /// alias and is merged with this history during validation.
    #[serde(default)]
    pub rambler_shut_up_observations: Vec<RamblerShutUpObservation>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

fn default_hp() -> i32 {
    10
}
fn default_wrong_exec_cost() -> i32 {
    2
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            n_cards: 0,
            n_evil: 0,
            deck: DeckComposition::default(),
            cards: vec![],
            executed: vec![],
            confirmed_evil: vec![],
            confirmed_good: vec![],
            pd_corruption_target: None,
            executed_evil_roles: HashMap::new(),
            slayer_results: vec![],
            pd_ability_results: vec![],
            blocked_positions: vec![],
            night_kills: vec![],
            night_kill_evil_count: 0,
            hp: default_hp(),
            wrong_exec_cost: default_wrong_exec_cost(),
            board_villager_count: None,
            board_outcast_count: None,
            board_minion_count: None,
            board_demon_count: None,
            board_count_provenance: BoardCountProvenance::LegacyUnknown,
            reveal_order: vec![],
            baker_rule_version: None,
            doppel_drunk_rule_version: None,
            fortune_teller_rule_version: None,
            executed_good_corrupted: HashMap::new(),
            executed_good_roles: HashMap::new(),
            used_abilities: vec![],
            rambler_rule_version: None,
            rambler_shut_up_observations: vec![],
            name: None,
            notes: None,
        }
    }
}

impl Default for CardInfo {
    fn default() -> Self {
        Self {
            position: 0,
            apparent_role: String::new(),
            info_text: String::new(),
            info_parsed: serde_json::Map::new(),
        }
    }
}

/// Custom deserializer that handles both nested {"deck": {...}} and flat deck fields.
fn deserialize_deck<'de, D: Deserializer<'de>>(
    _deserializer: D,
) -> Result<DeckComposition, D::Error> {
    // This is a placeholder — we actually handle deck in from_json below
    // because serde can't easily do "check for nested key, fallback to flat".
    Ok(DeckComposition::default())
}

impl GameState {
    /// Deserialize from a JSON value, handling the deck format duality.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let obj = value.as_object().ok_or("Expected JSON object")?;

        // Parse deck: try nested first, then flat
        let deck = if let Some(deck_val) = obj.get("deck") {
            serde_json::from_value::<DeckComposition>(deck_val.clone())
                .map_err(|e| format!("deck parse error: {e}"))?
        } else {
            DeckComposition {
                villagers: parse_string_array(obj, "villagers"),
                outcasts: parse_string_array(obj, "outcasts"),
                minions: parse_string_array(obj, "minions"),
                demons: parse_string_array(obj, "demons"),
            }
        };

        // Parse everything else via serde, then fix up the deck
        let mut state: GameState = serde_json::from_value(value.clone())
            .map_err(|e| format!("GameState parse error: {e}"))?;
        state.deck = deck;
        Ok(state)
    }

    /// Get a card by position from the revealed cards.
    pub fn card_at(&self, pos: u8) -> Option<&CardInfo> {
        self.cards.iter().find(|c| c.position == pos)
    }
}

fn parse_string_array(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Vec<String> {
    obj.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

// ── Scenario ──

/// Native Chancellor/Baron Start history projected onto the final board.
///
/// `original_positions` is a sorted equivalence class rather than a source of
/// scenario multiplicity: every listed original Chancellor seat produces the
/// same represented runtime outcome. The final Chancellor position is derived
/// from `Scenario::evil_positions`.
///
/// `affected_anchor_positions` likewise groups native histories that converge
/// to the same complete represented state. These are the real-Outcast anchors
/// Chancellor attempted to mark; `Scenario::messed_up_by_evil` is the
/// authoritative set of statuses that survived all later Start actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChancellorTrace {
    #[serde(default)]
    pub original_positions: Vec<u8>,
    pub added_outcast_position: u8,
    pub added_outcast_role: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub affected_anchor_positions: Vec<u8>,
}

/// Native Shaman/Illuzionist Start history for one copied Villager identity.
///
/// The ordered source/target distinction is observable: Shaman independently
/// attempts `MessedUpByEvil` on both endpoints, while only the target is
/// reinitialized and immediately dispatches the copied role's Start action.
/// `target_previous_roles` preserves the viable identities of the overwritten
/// physical card. Roles with identical remaining solver state share one
/// probability-safe trace; identities with a distinguishable Init-time effect
/// occupy separate traces.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShamanTrace {
    pub source_position: u8,
    pub target_position: u8,
    pub copied_role: String,
    pub target_previous_roles: Vec<String>,
}

/// A hypothetical assignment of evil roles to positions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Scenario {
    #[serde(
        serialize_with = "serialize_int_key_map",
        deserialize_with = "deserialize_int_key_map_str"
    )]
    pub evil_positions: HashMap<u8, String>,
    pub puppet_position: Option<u8>,
    pub corrupted: HashSet<u8>,
    pub pd_corrupted: Option<u8>,
    pub doppelganger_position: Option<u8>,
    pub drunk_position: Option<u8>,
    #[serde(
        serialize_with = "serialize_int_key_map",
        deserialize_with = "deserialize_int_key_map_u8"
    )]
    pub alchemist_cures: HashMap<u8, u8>,
    /// Positions retaining native `MessedUpByEvil` after the ordered Start
    /// pass. This is distinct from Corrupted: Alchemist only cures Corrupted.
    #[serde(default)]
    pub messed_up_by_evil: HashSet<u8>,
    /// Ordered native Shaman copy history. This remains distinct from the
    /// final role multiset because copied Start timing depends on the target;
    /// solver-equivalent overwritten roles share one candidate-class trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shaman_trace: Option<ShamanTrace>,
    /// Probability-safe Chancellor history projection. Original physical-seat
    /// alternatives are grouped inside this value instead of duplicating
    /// otherwise identical solver scenarios.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chancellor_trace: Option<ChancellorTrace>,
    /// Final physical card holding the Outcast identity Chancellor added.
    ///
    /// The historical field name is retained for JSON/Python compatibility;
    /// this is not necessarily the first Villager target when Chancellor's
    /// later role swap passes through that card.
    #[serde(default)]
    pub chancellor_conversion: Option<u8>,
}

impl Scenario {
    pub fn is_evil(&self, pos: u8) -> bool {
        self.evil_positions.contains_key(&pos) || self.puppet_position == Some(pos)
    }

    /// Final home of Chancellor's generated Outcast identity. New traces take
    /// precedence; the historical scalar remains a deserialization fallback.
    pub fn chancellor_added_outcast_position(&self) -> Option<u8> {
        self.chancellor_trace
            .as_ref()
            .map(|trace| trace.added_outcast_position)
            .or(self.chancellor_conversion)
    }

    pub fn chancellor_added_outcast_role(&self) -> Option<&str> {
        self.chancellor_trace
            .as_ref()
            .map(|trace| trace.added_outcast_role.as_str())
    }

    /// Candidate physical cards selected as Chancellor's first, real-Villager
    /// target. The target's former role is erased, while the generated Outcast
    /// identity can move during Chancellor's later neighbour swap. For each
    /// grouped original Chancellor seat `c`, native identity flow gives
    /// `v = f` when `c == a`, otherwise `v = a`.
    pub fn chancellor_original_villager_positions(&self) -> Vec<u8> {
        let Some(trace) = self.chancellor_trace.as_ref() else {
            return Vec::new();
        };
        let mut final_chancellors = self
            .evil_positions
            .iter()
            .filter(|(_, role)| role.eq_ignore_ascii_case("Chancellor"))
            .map(|(&position, _)| position);
        let Some(final_chancellor_position) = final_chancellors.next() else {
            return Vec::new();
        };
        if final_chancellors.next().is_some() {
            return Vec::new();
        }

        let mut positions: Vec<u8> = trace
            .original_positions
            .iter()
            .map(|&original_position| {
                if original_position == trace.added_outcast_position {
                    final_chancellor_position
                } else {
                    trace.added_outcast_position
                }
            })
            .collect();
        positions.sort_unstable();
        positions.dedup();
        positions
    }
}

// ── SolverResult ──

/// Output of the solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverResult {
    pub definite_evil: Vec<u8>,
    pub definite_good: Vec<u8>,
    pub bombardier_positions: Vec<u8>,
    pub n_scenarios: usize,
    pub n_surviving: usize,
    pub surviving_scenarios: Vec<Scenario>,
    pub reasoning: Vec<String>,
}

// ── TestCase ── for loading test JSON files

/// A test case includes GameState fields plus ground truth.
#[derive(Debug, Clone)]
pub struct TestCase {
    pub state: GameState,
    pub true_evil_positions: HashMap<u8, String>,
}

impl TestCase {
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let state = GameState::from_json(value)?;
        let obj = value.as_object().ok_or("Expected JSON object")?;

        let true_evil = obj
            .get("true_evil_positions")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        let pos = k.parse::<u8>().ok()?;
                        let role = v.as_str()?.to_string();
                        Some((pos, role))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(TestCase {
            state,
            true_evil_positions: true_evil,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let json = serde_json::json!({
            "n_cards": 7,
            "deck": {
                "villagers": ["Enlightened"],
                "outcasts": [],
                "minions": [],
                "demons": ["Pooka"]
            },
            "cards": []
        });
        let state = GameState::from_json(&json).unwrap();
        assert_eq!(state.n_cards, 7);
        assert_eq!(state.hp, 10);
        assert_eq!(state.wrong_exec_cost, 2);
        assert_eq!(state.deck.villagers, vec!["Enlightened"]);
        assert!(state.executed.is_empty());
        assert!(state.pd_corruption_target.is_none());
        assert_eq!(
            state.board_count_provenance,
            BoardCountProvenance::LegacyUnknown,
        );
        assert!(state.rambler_rule_version.is_none());
        assert!(state.baker_rule_version.is_none());
        assert!(state.doppel_drunk_rule_version.is_none());
        assert!(state.fortune_teller_rule_version.is_none());
        assert!(state.rambler_shut_up_observations.is_empty());
    }

    #[test]
    fn fortune_teller_rule_version_defaults_legacy_and_round_trips_current_marker() {
        let legacy = GameState::from_json(&serde_json::json!({
            "n_cards": 2,
            "deck": {"villagers": ["Fortune Teller"], "outcasts": [], "minions": [], "demons": []}
        }))
        .unwrap();
        assert!(legacy.fortune_teller_rule_version.is_none());

        let current = GameState::from_json(&serde_json::json!({
            "n_cards": 2,
            "deck": {"villagers": ["Fortune Teller"], "outcasts": [], "minions": [], "demons": []},
            "fortune_teller_rule_version": "fortune_teller_native_v1"
        }))
        .unwrap();
        assert_eq!(
            current.fortune_teller_rule_version.as_deref(),
            Some("fortune_teller_native_v1"),
        );
        assert_eq!(
            serde_json::to_value(current).unwrap()["fortune_teller_rule_version"],
            serde_json::json!("fortune_teller_native_v1"),
        );
    }

    #[test]
    fn doppel_drunk_rule_version_defaults_for_legacy_and_round_trips_current_marker() {
        let legacy = GameState::from_json(&serde_json::json!({
            "n_cards": 1,
            "deck": {"villagers": ["Knight"], "outcasts": ["Doppelganger"], "minions": [], "demons": []}
        }))
        .unwrap();
        assert!(legacy.doppel_drunk_rule_version.is_none());

        let current = GameState::from_json(&serde_json::json!({
            "n_cards": 1,
            "deck": {"villagers": ["Knight"], "outcasts": ["Doppelganger"], "minions": [], "demons": []},
            "doppel_drunk_rule_version": "doppel_drunk_reveal_v1"
        }))
        .unwrap();
        assert_eq!(
            current.doppel_drunk_rule_version.as_deref(),
            Some("doppel_drunk_reveal_v1"),
        );
        assert_eq!(
            serde_json::to_value(current).unwrap()["doppel_drunk_rule_version"],
            serde_json::json!("doppel_drunk_reveal_v1"),
        );
    }

    #[test]
    fn baker_rule_version_defaults_for_legacy_and_round_trips_current_marker() {
        let legacy = GameState::from_json(&serde_json::json!({
            "n_cards": 1,
            "deck": {"villagers": ["Baker"], "outcasts": [], "minions": [], "demons": []}
        }))
        .unwrap();
        assert!(legacy.baker_rule_version.is_none());

        let current = GameState::from_json(&serde_json::json!({
            "n_cards": 1,
            "deck": {"villagers": ["Baker"], "outcasts": [], "minions": [], "demons": []},
            "baker_rule_version": "baker_day_reveal_v1"
        }))
        .unwrap();
        assert_eq!(
            current.baker_rule_version.as_deref(),
            Some("baker_day_reveal_v1"),
        );
        assert_eq!(
            serde_json::to_value(current).unwrap()["baker_rule_version"],
            serde_json::json!("baker_day_reveal_v1"),
        );
    }

    #[test]
    fn rambler_history_deserializes_with_legacy_default_and_round_trips() {
        let legacy = GameState::from_json(&serde_json::json!({
            "n_cards": 2,
            "deck": {"villagers": [], "outcasts": ["Rambler"], "minions": [], "demons": []}
        }))
        .unwrap();
        assert!(legacy.rambler_rule_version.is_none());
        assert!(legacy.rambler_shut_up_observations.is_empty());

        let current = GameState::from_json(&serde_json::json!({
            "n_cards": 3,
            "deck": {"villagers": [], "outcasts": ["Rambler"], "minions": [], "demons": []},
            "rambler_rule_version": "rambler2_shut_up",
            "rambler_shut_up_observations": [
                {"speaker_position": 2, "shut_up_target": 1},
                {"speaker_position": 2, "shut_up_target": 3}
            ]
        }))
        .unwrap();
        assert_eq!(
            current.rambler_rule_version.as_deref(),
            Some("rambler2_shut_up"),
        );
        assert_eq!(
            current.rambler_shut_up_observations,
            vec![
                RamblerShutUpObservation {
                    speaker_position: 2,
                    shut_up_target: 1,
                },
                RamblerShutUpObservation {
                    speaker_position: 2,
                    shut_up_target: 3,
                },
            ],
        );

        let serialized = serde_json::to_value(&current).unwrap();
        assert_eq!(
            serialized["rambler_shut_up_observations"],
            serde_json::json!([
                {"speaker_position": 2, "shut_up_target": 1},
                {"speaker_position": 2, "shut_up_target": 3}
            ]),
        );
    }

    #[test]
    fn rambler_history_rejects_non_integer_and_non_u8_positions() {
        for malformed in [
            serde_json::json!(-1),
            serde_json::json!(256),
            serde_json::json!("1"),
            serde_json::json!(1.0),
            serde_json::json!(1.5),
            serde_json::json!(true),
            serde_json::json!([]),
            serde_json::json!({}),
            serde_json::Value::Null,
        ] {
            for field in ["speaker_position", "shut_up_target"] {
                let mut observation = serde_json::json!({
                    "speaker_position": 2,
                    "shut_up_target": 1
                });
                observation[field] = malformed.clone();
                let value = serde_json::json!({
                    "n_cards": 3,
                    "deck": {
                        "villagers": [],
                        "outcasts": ["Rambler"],
                        "minions": [],
                        "demons": []
                    },
                    "rambler_shut_up_observations": [observation]
                });
                assert!(
                    GameState::from_json(&value).is_err(),
                    "{field}={malformed} must not deserialize as a position",
                );
            }
        }
    }

    #[test]
    fn test_string_key_deserialization() {
        let json = serde_json::json!({
            "n_cards": 7,
            "n_evil": 1,
            "deck": { "villagers": [], "outcasts": [], "minions": [], "demons": ["Pooka"] },
            "cards": [],
            "executed_evil_roles": {"7": "Pooka"},
            "executed_good_corrupted": {"3": true},
            "executed_good_roles": {"3": "Plague_Doctor"}
        });
        let state = GameState::from_json(&json).unwrap();
        assert_eq!(state.executed_evil_roles.get(&7), Some(&"Pooka".to_string()));
        assert_eq!(state.executed_good_corrupted.get(&3), Some(&true));
        assert_eq!(
            state.executed_good_roles.get(&3),
            Some(&"Plague_Doctor".to_string()),
        );
        let legacy = GameState::from_json(&serde_json::json!({
            "n_cards": 1,
            "deck": {"villagers": [], "outcasts": [], "minions": [], "demons": []}
        })).unwrap();
        assert!(legacy.executed_good_roles.is_empty());
    }

    #[test]
    fn slayer_revealed_role_accepts_legacy_key_and_serializes_neutrally() {
        let legacy: SlayerResult = serde_json::from_value(serde_json::json!({
            "slayer_pos": 1,
            "target_pos": 2,
            "killed": true,
            "evil_role": "Shaman"
        }))
        .unwrap();
        assert_eq!(legacy.revealed_role.as_deref(), Some("Shaman"));

        let value = serde_json::to_value(&legacy).unwrap();
        assert_eq!(value["revealed_role"], "Shaman");
        assert!(value.get("evil_role").is_none());
    }

    #[test]
    fn pd_revealed_position_accepts_legacy_key_and_serializes_neutrally() {
        let legacy: PdAbilityResult = serde_json::from_value(serde_json::json!({
            "pd_pos": 6,
            "target": 1,
            "is_corrupted": true,
            "evil_pos": 5
        }))
        .unwrap();
        assert_eq!(legacy.evil_revealed, Some(5));

        let value = serde_json::to_value(&legacy).unwrap();
        assert_eq!(value["evil_revealed"], 5);
        assert!(value.get("evil_pos").is_none());
    }

    #[test]
    fn test_flat_deck_format() {
        let json = serde_json::json!({
            "n_cards": 7,
            "villagers": ["Enlightened", "Knitter"],
            "outcasts": ["Wretch"],
            "minions": [],
            "demons": ["Pooka"],
            "cards": []
        });
        let state = GameState::from_json(&json).unwrap();
        assert_eq!(state.deck.villagers, vec!["Enlightened", "Knitter"]);
        assert_eq!(state.deck.outcasts, vec!["Wretch"]);
    }

    #[test]
    fn test_scenario_serialization_roundtrip() {
        let scenario = Scenario {
            evil_positions: HashMap::from([(3, "Pooka".to_string()), (7, "Chancellor".to_string())]),
            puppet_position: None,
            corrupted: HashSet::from([2, 4]),
            pd_corrupted: Some(2),
            doppelganger_position: None,
            drunk_position: None,
            alchemist_cures: HashMap::from([(5, 2)]),
            messed_up_by_evil: HashSet::from([4]),
            shaman_trace: Some(ShamanTrace {
                source_position: 5,
                target_position: 4,
                copied_role: "Alchemist".to_string(),
                target_previous_roles: vec!["Witness".to_string(), "Scout".to_string()],
            }),
            chancellor_trace: Some(ChancellorTrace {
                original_positions: vec![1, 6],
                added_outcast_position: 2,
                added_outcast_role: "Plague Doctor".to_string(),
                affected_anchor_positions: vec![4],
            }),
            chancellor_conversion: Some(2),
        };
        let json = serde_json::to_value(&scenario).unwrap();
        // Keys must be strings in JSON
        assert!(json["evil_positions"]["3"].is_string());
        assert_eq!(json["evil_positions"]["3"], "Pooka");
        assert!(json["alchemist_cures"]["5"].is_number());
        assert!(json["corrupted"].is_array());
        assert_eq!(json["shaman_trace"]["source_position"], 5);
        assert_eq!(json["shaman_trace"]["target_position"], 4);
        assert_eq!(json["shaman_trace"]["copied_role"], "Alchemist");
        assert_eq!(
            json["shaman_trace"]["target_previous_roles"],
            serde_json::json!(["Witness", "Scout"])
        );
        assert_eq!(json["chancellor_trace"]["original_positions"], serde_json::json!([1, 6]));
        assert_eq!(json["chancellor_trace"]["added_outcast_position"], 2);
        assert_eq!(json["chancellor_trace"]["added_outcast_role"], "Plague Doctor");
        assert_eq!(
            json["chancellor_trace"]["affected_anchor_positions"],
            serde_json::json!([4])
        );
        // Round-trip
        let back: Scenario = serde_json::from_value(json).unwrap();
        assert_eq!(back.evil_positions.get(&3), Some(&"Pooka".to_string()));
        assert_eq!(back.alchemist_cures.get(&5), Some(&2));
        assert!(back.corrupted.contains(&2));
        assert!(back.messed_up_by_evil.contains(&4));
        assert_eq!(
            back.shaman_trace,
            Some(ShamanTrace {
                source_position: 5,
                target_position: 4,
                copied_role: "Alchemist".to_string(),
                target_previous_roles: vec!["Witness".to_string(), "Scout".to_string()],
            })
        );
        assert_eq!(back.chancellor_added_outcast_position(), Some(2));
        assert_eq!(back.chancellor_added_outcast_role(), Some("Plague Doctor"));
        assert_eq!(
            back.chancellor_trace
                .as_ref()
                .unwrap()
                .affected_anchor_positions,
            vec![4]
        );
    }

    #[test]
    fn chancellor_first_villager_candidates_follow_native_identity_flow() {
        let mut scenario = Scenario::default();
        scenario
            .evil_positions
            .insert(4, "Chancellor".to_string());
        scenario.chancellor_trace = Some(ChancellorTrace {
            // c=2 equals a, so v=f=4. c=1 does not, so v=a=2.
            original_positions: vec![1, 2],
            added_outcast_position: 2,
            added_outcast_role: "Bombardier".to_string(),
            affected_anchor_positions: vec![3],
        });

        assert_eq!(
            scenario.chancellor_original_villager_positions(),
            vec![2, 4]
        );
    }

    #[test]
    fn pre_anchor_chancellor_trace_deserializes_with_empty_history_candidates() {
        let json = serde_json::json!({
            "evil_positions": {"4": "Chancellor"},
            "puppet_position": null,
            "corrupted": [],
            "pd_corrupted": null,
            "doppelganger_position": null,
            "drunk_position": null,
            "alchemist_cures": {},
            "chancellor_trace": {
                "original_positions": [1],
                "added_outcast_position": 2,
                "added_outcast_role": "Bombardier"
            },
            "chancellor_conversion": 2
        });

        let scenario: Scenario = serde_json::from_value(json).unwrap();

        assert!(scenario
            .chancellor_trace
            .as_ref()
            .unwrap()
            .affected_anchor_positions
            .is_empty());
        assert_eq!(scenario.chancellor_original_villager_positions(), vec![2]);
    }

    #[test]
    fn int_key_maps_serialize_in_numeric_position_order() {
        let mut scenario = Scenario::default();
        scenario.evil_positions.insert(2, "Witch".to_string());
        scenario.evil_positions.insert(10, "Shaman".to_string());
        scenario.evil_positions.insert(1, "Pooka".to_string());

        let json = serde_json::to_string(&scenario).unwrap();

        assert!(json.contains(
            "\"evil_positions\":{\"1\":\"Pooka\",\"2\":\"Witch\",\"10\":\"Shaman\"}"
        ));
    }

    #[test]
    fn legacy_chancellor_conversion_deserializes_as_position_fallback() {
        let legacy = serde_json::json!({
            "evil_positions": {"1": "Chancellor"},
            "puppet_position": null,
            "corrupted": [],
            "pd_corrupted": null,
            "doppelganger_position": null,
            "drunk_position": null,
            "alchemist_cures": {},
            "chancellor_conversion": 3
        });

        let scenario: Scenario = serde_json::from_value(legacy).unwrap();

        assert!(scenario.chancellor_trace.is_none());
        assert!(scenario.shaman_trace.is_none());
        assert!(scenario.messed_up_by_evil.is_empty());
        assert_eq!(scenario.chancellor_added_outcast_position(), Some(3));
        assert_eq!(scenario.chancellor_added_outcast_role(), None);
    }

    #[test]
    fn test_solver_result_serialization() {
        let result = SolverResult {
            definite_evil: vec![3],
            definite_good: vec![1, 2],
            bombardier_positions: vec![],
            n_scenarios: 100,
            n_surviving: 5,
            surviving_scenarios: vec![],
            reasoning: vec!["test".to_string()],
        };
        let json_str = serde_json::to_string(&result).unwrap();
        let back: SolverResult = serde_json::from_str(&json_str).unwrap();
        assert_eq!(back.definite_evil, vec![3]);
        assert_eq!(back.n_surviving, 5);
    }
}
