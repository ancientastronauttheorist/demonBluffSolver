//! Strict player-information boundary, implementing the S0 history-v1 envelope.
//!
//! Raw JSON is not evidence. A trusted capture reviewer must separately bind
//! each exact event to a reviewed UI capture before admission. This module does
//! not verify image pixels, native scheduling, world completeness, or policy
//! quality. The legacy snapshot projection is deliberately narrower than the
//! lossless history: one Day, a fully exposed deck and empty passive Judge
//! reveals plus exact Judge results. Other transitions remain unsupported.

use crate::types::{CardInfo, DeckComposition, GameState};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA_VERSION: &str = "player_history_v1";
pub const BUILD_ID: &str = "f530404b0f3f_807de4a83df4";
pub const PROJECTION_DOMAIN: &str = "public_judge_single_day_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerHistory {
    pub schema_version: String,
    pub build_id: String,
    pub solver_commit: String,
    pub parser_version: String,
    pub corpus_version: String,
    pub information_mode: InformationMode,
    pub domain_id: String,
    pub events: Vec<PlayerEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InformationMode {
    Player,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Setup,
    Day,
    Night,
    Terminal,
}

/// Serialization retains the contract's sibling `kind` and typed `payload`.
/// Deserialization uses a strict envelope first; flatten alone would silently
/// admit unknown fields on an event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlayerEvent {
    pub ordinal: u64,
    pub phase: Phase,
    pub action_ordinal: Option<u64>,
    pub evidence_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub captured_at_ms: Option<u64>,
    #[serde(flatten)]
    pub observation: Observation,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEvent {
    ordinal: u64,
    phase: Phase,
    action_ordinal: Option<u64>,
    evidence_id: String,
    #[serde(default)]
    captured_at_ms: Option<u64>,
    kind: String,
    payload: serde_json::Value,
}

impl<'de> Deserialize<'de> for PlayerEvent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawEvent::deserialize(deserializer)?;
        let observation = serde_json::from_value(serde_json::json!({
            "kind": raw.kind, "payload": raw.payload,
        }))
        .map_err(serde::de::Error::custom)?;
        Ok(Self {
            ordinal: raw.ordinal,
            phase: raw.phase,
            action_ordinal: raw.action_ordinal,
            evidence_id: raw.evidence_id,
            captured_at_ms: raw.captured_at_ms,
            observation,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "payload",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Observation {
    DeckObserved(DeckObserved),
    CardRevealed(CardRevealed),
    ActionRequested(ActionRequested),
    AbilityObserved(AbilityObserved),
    ExecutionObserved(ExecutionObserved),
    StatusObserved(StatusObserved),
    PhaseObserved(PhaseObserved),
    TerminalObserved(TerminalObserved),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeckObserved {
    pub n_cards: u8,
    pub n_evil: u8,
    pub slots: Vec<DeckSlot>,
    pub header_counts: HeaderCounts,
}

/// An obscured strip identity has no role-name field, even if the oracle knows
/// which identity occupies it. Slot order and duplicate exposed names survive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "visibility", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeckSlot {
    Exposed {
        role: String,
        faction: PublicFaction,
    },
    Obscured {},
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicFaction {
    Villager,
    Outcast,
    Minion,
    Demon,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderCounts {
    pub villagers: Option<u8>,
    pub outcasts: Option<u8>,
    pub minions: Option<u8>,
    pub demons: Option<u8>,
    pub source: HeaderSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeaderSource {
    VisibleHud,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CardRevealed {
    pub position: u8,
    pub apparent_role: String,
    /// Null is missing capture; an empty string is an observed empty bubble.
    pub speech: Option<String>,
    pub targets: Vec<u8>,
    pub parser_version: String,
    pub rule_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    Reveal,
    Ability,
    Execution,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionRequested {
    pub action: ActionKind,
    pub actor: Option<u8>,
    pub targets: Vec<u8>,
    pub public_cost: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbilityObserved {
    pub actor: u8,
    pub speech: Option<String>,
    pub targets: Vec<u8>,
    pub parser_version: String,
    pub rule_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionOutcome {
    Killed,
    Protected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionObserved {
    pub target: u8,
    pub outcome: ExecutionOutcome,
    pub exposed_role: Option<String>,
    pub hp: Option<i32>,
    pub remaining_evil: Option<u8>,
    /// Exact public feedback; this is not a true alignment/corruption flag.
    pub feedback: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisibleStatus {
    Blocked,
    Silenced,
    NightKilled,
    Protected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusObserved {
    pub status: VisibleStatus,
    pub positions: Vec<u8>,
    pub speech: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseObserved {
    pub hp: Option<i32>,
    pub remaining_evil: Option<u8>,
    pub wrong_execution_cost: Option<i32>,
    pub ability_resets: Vec<u8>,
    pub reset_rule_version: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalOutcome {
    Win,
    Loss,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TerminalObserved {
    pub outcome: TerminalOutcome,
    pub score: Option<i64>,
    pub progression_text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryError {
    UnsupportedIdentity(String),
    InvalidEnvelope(String),
    InvalidChronology { ordinal: u64, reason: String },
    InvalidPayload { ordinal: u64, reason: String },
    EvidenceUnadmitted { ordinal: u64, reason: String },
}

fn payload_error(event: &PlayerEvent, reason: &str) -> HistoryError {
    HistoryError::InvalidPayload {
        ordinal: event.ordinal,
        reason: reason.into(),
    }
}

/// Deliberately neither Deserialize nor Serialize. This is a trusted reviewer
/// capability, not a field callers may include in a player-history JSON blob.
/// The caller is responsible for actually reviewing the referenced UI capture.
/// The boundary below checks the binding, not that external review procedure.
#[derive(Debug, Default)]
pub struct ReviewedEvidenceRegistry {
    reviewed: BTreeMap<String, ReviewedCapture>,
}

#[derive(Debug)]
struct ReviewedCapture {
    event: PlayerEvent,
    prefix: PlayerHistory,
    capture_reference: String,
    memory_visibility_gate: Option<u64>,
}

impl ReviewedEvidenceRegistry {
    /// Admit an exact event after external trusted review of UI pixels. An ID
    /// alone, a raw memory row, or a self-certified visibility bit is not review.
    pub fn record_trusted_ui_review(
        &mut self,
        history: &PlayerHistory,
        ordinal: u64,
        capture_reference: &str,
    ) -> Result<(), HistoryError> {
        validate_history_shape(history)?;
        let index = history
            .events
            .iter()
            .position(|event| event.ordinal == ordinal)
            .ok_or_else(|| HistoryError::EvidenceUnadmitted {
                ordinal,
                reason: "reviewed event absent from history".into(),
            })?;
        let event = &history.events[index];
        if event.evidence_id.trim().is_empty() || capture_reference.trim().is_empty() {
            return Err(HistoryError::EvidenceUnadmitted {
                ordinal: event.ordinal,
                reason: "empty evidence/capture reference".into(),
            });
        }
        if self.reviewed.contains_key(&event.evidence_id) {
            return Err(HistoryError::EvidenceUnadmitted {
                ordinal: event.ordinal,
                reason: "evidence ID already bound".into(),
            });
        }
        let mut prefix = history.clone();
        prefix.events.truncate(index + 1);
        self.reviewed.insert(
            event.evidence_id.clone(),
            ReviewedCapture {
                event: event.clone(),
                prefix,
                capture_reference: capture_reference.into(),
                memory_visibility_gate: None,
            },
        );
        Ok(())
    }

    /// A memory transcription cannot register itself. First review the exact
    /// public event against UI, then bind it to a prior reviewed reveal of that
    /// actor. The gate must also occur in the history being admitted.
    pub fn bind_reviewed_memory_transcription(
        &mut self,
        evidence_id: &str,
        prior_reveal_evidence_id: &str,
    ) -> Result<(), HistoryError> {
        let gate = self.reviewed.get(prior_reveal_evidence_id).ok_or_else(|| {
            HistoryError::EvidenceUnadmitted {
                ordinal: 0,
                reason: "visibility gate has no UI review".into(),
            }
        })?;
        let position = match &gate.event.observation {
            Observation::CardRevealed(card) => card.position,
            _ => {
                return Err(payload_error(
                    &gate.event,
                    "visibility gate must be a verified reveal",
                ))
            }
        };
        let gate_ordinal = gate.event.ordinal;
        let gate_prefix = gate.prefix.clone();
        let capture =
            self.reviewed
                .get_mut(evidence_id)
                .ok_or_else(|| HistoryError::EvidenceUnadmitted {
                    ordinal: 0,
                    reason: "memory transcription lacks exact UI cross-check".into(),
                })?;
        let actor = match &capture.event.observation {
            Observation::AbilityObserved(result) => result.actor,
            Observation::CardRevealed(card) => card.position,
            _ => {
                return Err(payload_error(
                    &capture.event,
                    "memory lane supports only visible card speech",
                ))
            }
        };
        if actor != position || gate_ordinal >= capture.event.ordinal {
            return Err(payload_error(
                &capture.event,
                "memory transcription needs a prior reveal of its actor",
            ));
        }
        let gate_index = capture
            .prefix
            .events
            .iter()
            .position(|event| event.ordinal == gate_ordinal)
            .ok_or_else(|| {
                payload_error(
                    &capture.event,
                    "visibility gate absent from transcription prefix",
                )
            })?;
        let mut transcription_gate_prefix = capture.prefix.clone();
        transcription_gate_prefix.events.truncate(gate_index + 1);
        if transcription_gate_prefix != gate_prefix {
            return Err(payload_error(
                &capture.event,
                "visibility gate must match the exact reviewed transcription prefix",
            ));
        }
        capture.memory_visibility_gate = Some(gate_ordinal);
        Ok(())
    }
}

/// Validate structural/public chronology without asserting gameplay legality.
/// Picker filters, cancellations, role changes and transition mechanics need a
/// named rules-domain validator; this function is not that certificate.
pub fn validate_history_shape(history: &PlayerHistory) -> Result<(), HistoryError> {
    if history.schema_version != SCHEMA_VERSION || history.build_id != BUILD_ID {
        return Err(HistoryError::UnsupportedIdentity(
            "schema or build identity".into(),
        ));
    }
    if history.solver_commit.len() != 40
        || !history.solver_commit.bytes().all(|c| c.is_ascii_hexdigit())
        || [
            &history.parser_version,
            &history.corpus_version,
            &history.domain_id,
        ]
        .iter()
        .any(|value| value.trim().is_empty())
    {
        return Err(HistoryError::InvalidEnvelope(
            "missing/versionless identity".into(),
        ));
    }
    let mut previous = None;
    let mut current_phase = Phase::Setup;
    let mut request: Option<(u64, &ActionRequested)> = None;
    let mut completed = BTreeSet::new();
    let mut n_cards = None;
    let mut terminal = false;
    for event in &history.events {
        let chronology = |reason: &str| HistoryError::InvalidChronology {
            ordinal: event.ordinal,
            reason: reason.into(),
        };
        if previous.is_some_and(|ordinal| event.ordinal <= ordinal) || terminal {
            return Err(chronology("ordinals must increase; terminal must be last"));
        }
        previous = Some(event.ordinal);
        if event.evidence_id.trim().is_empty() {
            return Err(payload_error(event, "missing evidence ID"));
        }
        if matches!(event.observation, Observation::PhaseObserved(_)) {
            current_phase = event.phase;
        } else if event.phase != current_phase {
            return Err(chronology("phase change requires a phase observation"));
        }
        if let Observation::ActionRequested(action) = &event.observation {
            if event.action_ordinal != Some(event.ordinal) {
                return Err(chronology("request action_ordinal must name itself"));
            }
            if request.is_some_and(|(ordinal, _)| !completed.contains(&ordinal)) {
                return Err(chronology(
                    "new action before prior completion; cancellation unsupported",
                ));
            }
            if action.targets.is_empty()
                || (action.action == ActionKind::Ability && action.actor.is_none())
            {
                return Err(payload_error(
                    event,
                    "action needs targets and ability needs actor",
                ));
            }
            request = Some((event.ordinal, action));
        } else if event.action_ordinal != request.map(|(ordinal, _)| ordinal) {
            return Err(chronology(
                "action reference must identify latest prior request or be null before actions",
            ));
        }
        let positions: Vec<u8> = match &event.observation {
            Observation::DeckObserved(deck) => {
                if deck.n_cards == 0 || deck.n_evil > deck.n_cards {
                    return Err(payload_error(event, "invalid board/header size"));
                }
                if n_cards.is_some_and(|size| size != deck.n_cards) {
                    return Err(payload_error(
                        event,
                        "board resizing requires a new history",
                    ));
                }
                if deck.header_counts.source == HeaderSource::Missing
                    && [
                        deck.header_counts.villagers,
                        deck.header_counts.outcasts,
                        deck.header_counts.minions,
                        deck.header_counts.demons,
                    ]
                    .iter()
                    .any(Option::is_some)
                {
                    return Err(payload_error(event, "missing header cannot carry counts"));
                }
                if deck.slots.iter().any(
                    |slot| matches!(slot, DeckSlot::Exposed { role, .. } if role.trim().is_empty()),
                ) {
                    return Err(payload_error(event, "empty exposed role"));
                }
                n_cards = Some(deck.n_cards);
                vec![]
            }
            Observation::CardRevealed(card) => {
                if card.apparent_role.trim().is_empty()
                    || card.parser_version != history.parser_version
                    || card.rule_version.trim().is_empty()
                {
                    return Err(payload_error(event, "missing role/parser/rule identity"));
                }
                if let Some((ordinal, action)) = request {
                    if action.action != ActionKind::Reveal
                        || !action.targets.contains(&card.position)
                    {
                        return Err(chronology("reveal does not match current reveal request"));
                    }
                    // Only single-card reveal requests are admitted in v1.
                    if action.targets.len() != 1 || !completed.insert(ordinal) {
                        return Err(chronology(
                            "duplicate completion or unsupported multi-card reveal request",
                        ));
                    }
                }
                std::iter::once(card.position)
                    .chain(card.targets.iter().copied())
                    .collect()
            }
            Observation::ActionRequested(action) => action
                .actor
                .iter()
                .copied()
                .chain(action.targets.iter().copied())
                .collect(),
            Observation::AbilityObserved(result) => {
                let (ordinal, action) =
                    request.ok_or_else(|| chronology("ability result requires a request"))?;
                if action.action != ActionKind::Ability
                    || action.actor != Some(result.actor)
                    || action.targets != result.targets
                    || !completed.insert(ordinal)
                {
                    return Err(chronology(
                        "ability actor/ordered targets/completion mismatch",
                    ));
                }
                if result.parser_version != history.parser_version
                    || result.rule_version.trim().is_empty()
                {
                    return Err(payload_error(event, "missing ability parser/rule identity"));
                }
                std::iter::once(result.actor)
                    .chain(result.targets.iter().copied())
                    .collect()
            }
            Observation::ExecutionObserved(result) => {
                let (ordinal, action) =
                    request.ok_or_else(|| chronology("execution result requires a request"))?;
                if action.action != ActionKind::Execution
                    || action.targets != vec![result.target]
                    || !completed.insert(ordinal)
                {
                    return Err(chronology("execution target/completion mismatch"));
                }
                vec![result.target]
            }
            Observation::StatusObserved(status) => status.positions.clone(),
            Observation::PhaseObserved(phase) => {
                if !phase.ability_resets.is_empty()
                    && phase
                        .reset_rule_version
                        .as_ref()
                        .is_none_or(|v| v.trim().is_empty())
                {
                    return Err(payload_error(
                        event,
                        "resets require a reviewed rule identity",
                    ));
                }
                phase.ability_resets.clone()
            }
            Observation::TerminalObserved(_) => {
                if event.phase != Phase::Terminal {
                    return Err(chronology("terminal outcome requires terminal phase"));
                }
                terminal = true;
                vec![]
            }
        };
        if positions
            .iter()
            .any(|position| *position == 0 || n_cards.is_none_or(|size| *position > size))
        {
            return Err(payload_error(
                event,
                "position unavailable/outside observed board",
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlannerEvent {
    pub ordinal: u64,
    pub phase: Phase,
    pub action_ordinal: Option<u64>,
    #[serde(flatten)]
    pub observation: Observation,
}

/// The fair planner view intentionally excludes evidence IDs, timestamps,
/// capture references, solver commits and all oracle state. Domain/parser/build
/// versions remain because they select the rules interpreting observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlannerHistory {
    pub schema_version: String,
    pub build_id: String,
    pub parser_version: String,
    pub domain_id: String,
    pub events: Vec<PlannerEvent>,
}

#[derive(Debug, Clone)]
pub struct AdmittedPlayerHistory {
    history: PlayerHistory,
}

pub fn admit_history(
    history: &PlayerHistory,
    registry: &ReviewedEvidenceRegistry,
) -> Result<AdmittedPlayerHistory, HistoryError> {
    validate_history_shape(history)?;
    let ordinals: BTreeSet<_> = history.events.iter().map(|event| event.ordinal).collect();
    for (index, event) in history.events.iter().enumerate() {
        let invalid = |reason: &str| HistoryError::EvidenceUnadmitted {
            ordinal: event.ordinal,
            reason: reason.into(),
        };
        let capture = registry
            .reviewed
            .get(&event.evidence_id)
            .ok_or_else(|| invalid("no trusted UI review"))?;
        let mut prefix = history.clone();
        prefix.events.truncate(index + 1);
        if capture.prefix != prefix
            || capture.event != *event
            || capture.capture_reference.is_empty()
        {
            return Err(invalid(
                "review does not bind this exact payload and observation prefix",
            ));
        }
        if capture
            .memory_visibility_gate
            .is_some_and(|gate| !ordinals.contains(&gate) || gate >= event.ordinal)
        {
            return Err(invalid(
                "memory visibility gate absent from this history prefix",
            ));
        }
    }
    Ok(AdmittedPlayerHistory {
        history: history.clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionError {
    Incomplete {
        ordinal: Option<u64>,
        reason: String,
    },
    Unsupported {
        ordinal: Option<u64>,
        reason: String,
    },
}

impl AdmittedPlayerHistory {
    pub fn planner_history(&self) -> PlannerHistory {
        PlannerHistory {
            schema_version: self.history.schema_version.clone(),
            build_id: self.history.build_id.clone(),
            parser_version: self.history.parser_version.clone(),
            domain_id: self.history.domain_id.clone(),
            events: self
                .history
                .events
                .iter()
                .map(|event| PlannerEvent {
                    ordinal: event.ordinal,
                    phase: event.phase,
                    action_ordinal: event.action_ordinal,
                    observation: event.observation.clone(),
                })
                .collect(),
        }
    }

    /// Mechanical compatibility projection, NOT a certified solver domain or
    /// planner recommendation. No clue maps are accepted from callers. Judge
    /// claims are parsed from the exact native public string, not truth flags.
    /// Unsupported transitions fail closed rather than erase chronology.
    pub fn project_legacy_snapshot(&self) -> Result<GameState, ProjectionError> {
        let unsupported = |ordinal, reason: &str| ProjectionError::Unsupported {
            ordinal,
            reason: reason.into(),
        };
        let incomplete = |ordinal, reason: &str| ProjectionError::Incomplete {
            ordinal,
            reason: reason.into(),
        };
        if self.history.domain_id != PROJECTION_DOMAIN {
            return Err(unsupported(None, "no legacy adapter for this named domain"));
        }
        let mut state = GameState::default();
        let mut deck_seen = false;
        let mut hp_seen = false;
        let mut cost_seen = false;
        let mut phase_checkpoint = None;
        let mut pending = None;
        for event in &self.history.events {
            let ordinal = Some(event.ordinal);
            if matches!(event.phase, Phase::Night | Phase::Terminal) {
                return Err(unsupported(
                    ordinal,
                    "temporal Night/terminal transition cannot be flattened",
                ));
            }
            match &event.observation {
                Observation::DeckObserved(deck) => {
                    if deck_seen {
                        return Err(unsupported(
                            ordinal,
                            "deck refresh/history projection not established",
                        ));
                    }
                    let mut composition = DeckComposition::default();
                    for slot in &deck.slots {
                        let DeckSlot::Exposed { role, faction } = slot else {
                            return Err(unsupported(
                                ordinal,
                                "obscured deck support not established",
                            ));
                        };
                        let roles = match faction {
                            PublicFaction::Villager => &mut composition.villagers,
                            PublicFaction::Outcast => &mut composition.outcasts,
                            PublicFaction::Minion => &mut composition.minions,
                            PublicFaction::Demon => &mut composition.demons,
                        };
                        roles.push(role.clone());
                    }
                    state.n_cards = deck.n_cards;
                    state.n_evil = deck.n_evil;
                    state.deck = composition;
                    // A visible current HUD does not establish pre-Start count
                    // provenance. Keep legacy unknown rather than invent it.
                    state.board_villager_count = deck.header_counts.villagers;
                    state.board_outcast_count = deck.header_counts.outcasts;
                    state.board_minion_count = deck.header_counts.minions;
                    state.board_demon_count = deck.header_counts.demons;
                    deck_seen = true;
                }
                Observation::PhaseObserved(phase) => {
                    if phase_checkpoint == Some(Phase::Day) || phase_checkpoint == Some(event.phase)
                    {
                        return Err(unsupported(
                            ordinal,
                            "repeated phase checkpoint or return to Setup requires temporal model",
                        ));
                    }
                    phase_checkpoint = Some(event.phase);
                    if !phase.ability_resets.is_empty() {
                        return Err(unsupported(
                            ordinal,
                            "reset history requires temporal model",
                        ));
                    }
                    if let Some(hp) = phase.hp {
                        if hp_seen && hp != state.hp {
                            return Err(unsupported(
                                ordinal,
                                "HP change requires an established transition model",
                            ));
                        }
                        state.hp = hp;
                        hp_seen = true;
                    }
                    if let Some(cost) = phase.wrong_execution_cost {
                        if cost < 0 {
                            return Err(unsupported(ordinal, "negative execution cost"));
                        }
                        if cost_seen && cost != state.wrong_exec_cost {
                            return Err(unsupported(
                                ordinal,
                                "execution-cost change requires an established transition model",
                            ));
                        }
                        state.wrong_exec_cost = cost;
                        cost_seen = true;
                    }
                    if phase
                        .remaining_evil
                        .is_some_and(|remaining| remaining != state.n_evil)
                    {
                        return Err(unsupported(
                            ordinal,
                            "objective change requires transition model",
                        ));
                    }
                }
                Observation::CardRevealed(card) => {
                    if event.phase != Phase::Day {
                        return Err(unsupported(ordinal, "only Day reveals are projected"));
                    }
                    let speech = card
                        .speech
                        .as_ref()
                        .ok_or_else(|| incomplete(ordinal, "missing revealed speech"))?;
                    if card.apparent_role != "Judge"
                        || card.rule_version != "public_current"
                        || !speech.is_empty()
                        || !card.targets.is_empty()
                    {
                        return Err(unsupported(
                            ordinal,
                            "only audited empty passive Judge reveal is projected",
                        ));
                    }
                    if state.card_at(card.position).is_some() {
                        return Err(unsupported(
                            ordinal,
                            "repeated reveal/correction requires history model",
                        ));
                    }
                    state.cards.push(CardInfo {
                        position: card.position,
                        apparent_role: card.apparent_role.clone(),
                        info_text: speech.clone(),
                        info_parsed: serde_json::Map::new(),
                    });
                    state.reveal_order.push(card.position);
                    pending = None;
                }
                Observation::ActionRequested(action) => {
                    if action.public_cost.is_some_and(|cost| cost != 0) {
                        return Err(unsupported(
                            ordinal,
                            "action cost effects require an established transition model",
                        ));
                    }
                    if action.action == ActionKind::Execution {
                        return Err(unsupported(ordinal, "execution transition adapter absent"));
                    }
                    if event.phase != Phase::Day {
                        return Err(unsupported(ordinal, "only Day actions are projected"));
                    }
                    if let Some(actor) = action.actor {
                        if action.action == ActionKind::Ability && state.card_at(actor).is_none() {
                            return Err(incomplete(ordinal, "ability actor has no public reveal"));
                        }
                    }
                    pending = ordinal;
                }
                Observation::AbilityObserved(result) => {
                    let speech = result
                        .speech
                        .as_ref()
                        .ok_or_else(|| incomplete(ordinal, "missing ability result"))?;
                    if result.rule_version != "public_current" || result.targets.len() != 1 {
                        return Err(unsupported(
                            ordinal,
                            "only exact current public Judge results are projected",
                        ));
                    }
                    let target = result.targets[0];
                    let claimed_lying = if speech == &format!("#{target} is\nLying") {
                        true
                    } else if speech == &format!("#{target} is\nsaying Truth") {
                        false
                    } else {
                        return Err(unsupported(
                            ordinal,
                            "unrecognized exact public Judge speech",
                        ));
                    };
                    if state.used_abilities.contains(&result.actor) {
                        return Err(unsupported(
                            ordinal,
                            "repeated Judge use needs established reset transition",
                        ));
                    }
                    let card = state
                        .cards
                        .iter_mut()
                        .find(|card| card.position == result.actor)
                        .ok_or_else(|| incomplete(ordinal, "ability actor has no public reveal"))?;
                    if card.apparent_role != "Judge" {
                        return Err(unsupported(ordinal, "unsupported apparent ability role"));
                    }
                    card.info_text = speech.clone();
                    card.info_parsed.insert(
                        "observations".into(),
                        serde_json::json!([{"target": target, "is_lying": claimed_lying}]),
                    );
                    state.used_abilities.push(result.actor);
                    pending = None;
                }
                _ => {
                    return Err(unsupported(
                        ordinal,
                        "execution/status/terminal projection absent",
                    ))
                }
            }
        }
        if pending.is_some() {
            return Err(incomplete(
                pending,
                "requested action has no observed completion",
            ));
        }
        if !deck_seen || !hp_seen || !cost_seen {
            return Err(incomplete(
                None,
                "missing public deck, HP or wrong-execution cost",
            ));
        }
        Ok(state)
    }
}

#[cfg(test)]
#[path = "player_history_tests.rs"]
mod tests;
