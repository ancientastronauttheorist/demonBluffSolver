//! Offline RoguelikeStandard progression with explicit service boundaries.
//! Native reset/abandon/save calls compose here; no live saves or solver state change.

use serde::{Deserialize, Serialize};

pub const ROGUELIKE_PROGRESSION_NATIVE_V1: &str = "roguelike_progression_native_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RoguelikeStandardState {
    pub best_ascension: i32,
    pub current_ascension: i32,
    pub current_died_times: i32,
    pub best_village: i32,
    pub current_village: i32,
    pub best_score: i32,
    pub round_score: i32,
    pub ascension_score: i32,
    pub prev_ascension_score: i32,
    /// Preserve the native byte; progression does not rewrite this field.
    pub showed_new_characters: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoguelikeGateway {
    ClassInit,
    IncreaseVillage,
    AbandonDispatch,
    Json,
    SetString,
}

impl RoguelikeGateway {
    fn index(self) -> usize {
        match self {
            Self::ClassInit => 0,
            Self::IncreaseVillage => 1,
            Self::AbandonDispatch => 2,
            Self::Json => 3,
            Self::SetString => 4,
        }
    }

    fn maximum_occurrences(self) -> u32 {
        match self {
            Self::Json | Self::SetString => 2,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoguelikeFailurePoint {
    pub gateway: RoguelikeGateway,
    /// One-based occurrence of this gateway in the current replay.
    pub occurrence: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoguelikeProgressionAction {
    StageCompleted,
    AscensionComplete,
    Failed,
    ResetScores,
    AbandonRun,
    Save,
    /// Immediate SavesGame.set_RoguelikeStandard receiver serialization.
    SaveSetter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoguelikeProgressionContext {
    pub rule_version: String,
    pub initial: RoguelikeStandardState,
    pub initial_global_village: i32,
    pub game_data_initialized: bool,
    pub operation: RoguelikeProgressionAction,
    pub metadata_initialized: bool,
    /// External services preserve the modeled mode object throughout the call.
    pub services_preserve_mode: bool,
    /// External services preserve CurrentVillage; native reset writes still apply.
    /// In particular, this does not infer IncreaseVillage's own global effects.
    pub services_preserve_global_village: bool,
    /// Slot 15 resolves to audited RoguelikeStandard.AbandonRun -> ResetScores.
    pub abandon_uses_audited_override: bool,
    /// A valid but unreached failure point has no effect.
    pub fail_at: Option<RoguelikeFailurePoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoguelikeProgressionEvent {
    pub gateway: RoguelikeGateway,
    pub occurrence: u32,
    /// One-based save attempt for JSON/SetString; absent on other services.
    pub save_index: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoguelikeSaveAttempt {
    pub index: u32,
    /// Mode fields supplied to JSON, including when JSON itself fails.
    pub state: RoguelikeStandardState,
    pub json_completed: bool,
    pub set_string_completed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoguelikeProgressionResult {
    pub state: RoguelikeStandardState,
    pub global_village: i32,
    pub game_data_initialized: bool,
    pub events: Vec<RoguelikeProgressionEvent>,
    pub save_attempts: Vec<RoguelikeSaveAttempt>,
    /// Native partial writes remain visible at an explicit failed gateway.
    pub failure: Option<RoguelikeFailurePoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidRoguelikeProgressionContext;

/// Native stage writes shared by the gateway replay and the opt-in village bridge.
pub(crate) fn apply_stage_writes(state: &mut RoguelikeStandardState) {
    state.current_village = state.current_village.wrapping_add(1);
    let amount = state
        .round_score
        .wrapping_add(50)
        .wrapping_add(state.current_ascension.wrapping_mul(50));
    state.round_score = 0;
    state.ascension_score = state.ascension_score.wrapping_add(amount);
    if state.current_ascension >= state.best_ascension
        && state.current_village >= state.best_village
    {
        state.best_village = state.current_village;
    }
    state.best_score = state.best_score.max(state.ascension_score);
}

struct Replay {
    output: RoguelikeProgressionResult,
    fail_at: Option<RoguelikeFailurePoint>,
    occurrences: [u32; 5],
}

impl Replay {
    fn gateway(
        &mut self,
        gateway: RoguelikeGateway,
        save_index: Option<u32>,
    ) -> Result<(), RoguelikeFailurePoint> {
        self.occurrences[gateway.index()] += 1;
        let occurrence = self.occurrences[gateway.index()];
        self.output.events.push(RoguelikeProgressionEvent {
            gateway,
            occurrence,
            save_index,
        });
        let point = RoguelikeFailurePoint {
            gateway,
            occurrence,
        };
        if self.fail_at == Some(point) {
            Err(point)
        } else {
            Ok(())
        }
    }

    fn initialize_game_data(&mut self) -> Result<(), RoguelikeFailurePoint> {
        if !self.output.game_data_initialized {
            self.gateway(RoguelikeGateway::ClassInit, None)?;
            // Successful initialization satisfies the following native recheck.
            self.output.game_data_initialized = true;
        }
        Ok(())
    }

    fn save(&mut self) -> Result<(), RoguelikeFailurePoint> {
        let index = self.output.save_attempts.len() as u32 + 1;
        self.output.save_attempts.push(RoguelikeSaveAttempt {
            index,
            state: self.output.state.clone(),
            json_completed: false,
            set_string_completed: false,
        });
        self.gateway(RoguelikeGateway::Json, Some(index))?;
        self.output.save_attempts[index as usize - 1].json_completed = true;
        self.gateway(RoguelikeGateway::SetString, Some(index))?;
        self.output.save_attempts[index as usize - 1].set_string_completed = true;
        Ok(())
    }

    fn reset_scores(&mut self) -> Result<(), RoguelikeFailurePoint> {
        self.initialize_game_data()?;
        self.output.global_village = 0;
        let state = &mut self.output.state;
        state.current_died_times = 0;
        state.current_village = 0;
        state.round_score = 0;
        state.ascension_score = 0;
        self.save()
    }

    fn apply(&mut self, action: RoguelikeProgressionAction) -> Result<(), RoguelikeFailurePoint> {
        match action {
            RoguelikeProgressionAction::StageCompleted => {
                apply_stage_writes(&mut self.output.state);
                self.initialize_game_data()?;
                self.gateway(RoguelikeGateway::IncreaseVillage, None)?;
                self.save()
            }
            RoguelikeProgressionAction::AscensionComplete => {
                let state = &mut self.output.state;
                state.current_ascension = state.current_ascension.wrapping_add(1);
                state.best_village = 0;
                state.best_ascension = state.best_ascension.max(state.current_ascension);
                state.prev_ascension_score = state.ascension_score.wrapping_add(state.round_score);
                self.initialize_game_data()?;
                self.output.global_village = 0;
                self.reset_scores()?;
                // ResetScores already saved; the enclosing caller saves again.
                self.save()
            }
            RoguelikeProgressionAction::Failed => {
                self.output.state.current_died_times =
                    self.output.state.current_died_times.wrapping_add(1);
                if self.output.state.current_died_times >= 4 {
                    self.gateway(RoguelikeGateway::AbandonDispatch, None)?;
                    self.reset_scores()?;
                }
                self.save()
            }
            RoguelikeProgressionAction::ResetScores | RoguelikeProgressionAction::AbandonRun => {
                self.reset_scores()
            }
            RoguelikeProgressionAction::Save | RoguelikeProgressionAction::SaveSetter => {
                self.save()
            }
        }
    }
}

/// Replay one bounded native caller. Unsupported provenance is rejected before
/// any replay; native service failures instead return their exact partial state.
/// Services are synchronous and successful class initialization marks GameData
/// initialized. External serialization, preferences and village effects remain
/// outside the contract; this API never writes live state.
pub fn replay_roguelike_progression(
    context: &RoguelikeProgressionContext,
) -> Result<RoguelikeProgressionResult, InvalidRoguelikeProgressionContext> {
    if context.rule_version != ROGUELIKE_PROGRESSION_NATIVE_V1
        || !context.metadata_initialized
        || !context.services_preserve_mode
        || !context.services_preserve_global_village
        || !context.abandon_uses_audited_override
        || context.fail_at.is_some_and(|point| {
            point.occurrence == 0 || point.occurrence > point.gateway.maximum_occurrences()
        })
    {
        return Err(InvalidRoguelikeProgressionContext);
    }
    let mut replay = Replay {
        output: RoguelikeProgressionResult {
            state: context.initial.clone(),
            global_village: context.initial_global_village,
            game_data_initialized: context.game_data_initialized,
            events: Vec::new(),
            save_attempts: Vec::new(),
            failure: None,
        },
        fail_at: context.fail_at,
        occurrences: [0; 5],
    };
    replay.output.failure = replay.apply(context.operation).err();
    Ok(replay.output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture_report() -> Value {
        serde_json::from_str(include_str!(
            "../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_standard_progression.json"
        ))
        .unwrap()
    }

    fn fixture_context(case: &Value) -> Option<RoguelikeProgressionContext> {
        let operation = match case["method"].as_str().unwrap() {
            "OnStageCompleted" => RoguelikeProgressionAction::StageCompleted,
            "AscensionComplete" => RoguelikeProgressionAction::AscensionComplete,
            "OnFailed" => RoguelikeProgressionAction::Failed,
            "ResetScores" => RoguelikeProgressionAction::ResetScores,
            "AbandonRun" => RoguelikeProgressionAction::AbandonRun,
            "Save" => RoguelikeProgressionAction::Save,
            "set_RoguelikeStandard" => RoguelikeProgressionAction::SaveSetter,
            _ => return None,
        };
        let fail_at = case["failure"]
            .as_array()
            .map(|point| RoguelikeFailurePoint {
                gateway: serde_json::from_value(point[0].clone()).unwrap(),
                occurrence: point[1].as_u64().unwrap() as u32,
            });
        Some(RoguelikeProgressionContext {
            rule_version: ROGUELIKE_PROGRESSION_NATIVE_V1.into(),
            initial: serde_json::from_value(case["initial"].clone()).unwrap(),
            initial_global_village: 123,
            game_data_initialized: !case["cold_class"].as_bool().unwrap(),
            operation,
            metadata_initialized: true,
            services_preserve_mode: true,
            services_preserve_global_village: true,
            abandon_uses_audited_override: true,
            fail_at,
        })
    }

    fn basic_context(action: RoguelikeProgressionAction) -> RoguelikeProgressionContext {
        RoguelikeProgressionContext {
            rule_version: ROGUELIKE_PROGRESSION_NATIVE_V1.into(),
            initial: RoguelikeStandardState {
                best_ascension: 7,
                current_ascension: 6,
                current_died_times: 2,
                best_village: 4,
                current_village: 3,
                best_score: 1000,
                round_score: 80,
                ascension_score: 500,
                prev_ascension_score: 200,
                showed_new_characters: 255,
            },
            initial_global_village: 123,
            game_data_initialized: false,
            operation: action,
            metadata_initialized: true,
            services_preserve_mode: true,
            services_preserve_global_village: true,
            abandon_uses_audited_override: true,
            fail_at: None,
        }
    }

    #[test]
    fn all_scoped_native_fixtures_match_state_events_saves_and_failures() {
        let report = fixture_report();
        let mut checked = 0;
        for case in report["cases"].as_array().unwrap() {
            let Some(context) = fixture_context(case) else {
                continue;
            };
            let original = context.clone();
            let result = replay_roguelike_progression(&context).unwrap();
            assert_eq!(context, original);
            assert_eq!(
                serde_json::to_value(&result.state).unwrap(),
                case["result_fields"]
            );
            assert_eq!(
                result.global_village,
                case["result_global_village"].as_i64().unwrap() as i32
            );
            assert_eq!(
                result.game_data_initialized,
                case["result_game_data_initialized"].as_bool().unwrap()
            );
            let expected_gateways: Vec<RoguelikeGateway> =
                serde_json::from_value(case["events"].clone()).unwrap();
            assert_eq!(
                result
                    .events
                    .iter()
                    .map(|event| event.gateway)
                    .collect::<Vec<_>>(),
                expected_gateways
            );
            let expected_snapshots: Vec<RoguelikeStandardState> =
                serde_json::from_value(case["saved_snapshots"].clone()).unwrap();
            assert_eq!(
                result
                    .save_attempts
                    .iter()
                    .map(|attempt| attempt.state.clone())
                    .collect::<Vec<_>>(),
                expected_snapshots
            );
            let failed = case["stopped_at_gateway"].as_bool().unwrap();
            assert_eq!(result.failure, if failed { context.fail_at } else { None });
            let mut counts = [0; 5];
            let mut save_index = 0;
            for (index, event) in result.events.iter().enumerate() {
                counts[event.gateway.index()] += 1;
                assert_eq!(event.occurrence, counts[event.gateway.index()]);
                if event.gateway == RoguelikeGateway::Json {
                    save_index += 1;
                }
                let saving = matches!(
                    event.gateway,
                    RoguelikeGateway::Json | RoguelikeGateway::SetString
                );
                assert_eq!(event.save_index, saving.then_some(save_index));
                if saving {
                    let attempt = &result.save_attempts[save_index as usize - 1];
                    assert_eq!(attempt.index, save_index);
                    let completed = !failed || index + 1 != result.events.len();
                    match event.gateway {
                        RoguelikeGateway::Json => assert_eq!(attempt.json_completed, completed),
                        RoguelikeGateway::SetString => {
                            assert_eq!(attempt.set_string_completed, completed)
                        }
                        _ => unreachable!(),
                    }
                }
            }
            for attempt in &result.save_attempts {
                let completed = |gateway| {
                    result.events.iter().any(|event| {
                        event.gateway == gateway
                            && event.save_index == Some(attempt.index)
                            && result.failure
                                != Some(RoguelikeFailurePoint {
                                    gateway,
                                    occurrence: event.occurrence,
                                })
                    })
                };
                assert_eq!(attempt.json_completed, completed(RoguelikeGateway::Json));
                assert_eq!(
                    attempt.set_string_completed,
                    completed(RoguelikeGateway::SetString)
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 1138);
    }

    #[test]
    fn second_save_failure_preserves_first_save_and_native_reset() {
        for action in [
            RoguelikeProgressionAction::AscensionComplete,
            RoguelikeProgressionAction::Failed,
        ] {
            let mut context = basic_context(action);
            context.initial.current_died_times = 3;
            context.fail_at = Some(RoguelikeFailurePoint {
                gateway: RoguelikeGateway::SetString,
                occurrence: 2,
            });
            let result = replay_roguelike_progression(&context).unwrap();
            assert_eq!(result.failure, context.fail_at);
            assert_eq!(result.save_attempts.len(), 2);
            assert!(result.save_attempts[0].set_string_completed);
            assert!(result.save_attempts[1].json_completed);
            assert!(!result.save_attempts[1].set_string_completed);
            assert_eq!(result.state.current_died_times, 0);
            assert_eq!(result.state.current_village, 0);
            assert_eq!(result.state.round_score, 0);
            assert_eq!(result.state.ascension_score, 0);
            assert_eq!(result.state.showed_new_characters, 255);
            assert_eq!(result.global_village, 0);
        }
    }

    #[test]
    fn wrapped_death_counter_bypasses_abandonment_and_preserves_scores() {
        let mut context = basic_context(RoguelikeProgressionAction::Failed);
        context.initial.current_died_times = i32::MAX;
        let result = replay_roguelike_progression(&context).unwrap();
        assert_eq!(result.state.current_died_times, i32::MIN);
        assert_eq!(result.state.round_score, context.initial.round_score);
        assert_eq!(
            result.state.ascension_score,
            context.initial.ascension_score
        );
        assert_eq!(result.global_village, 123);
        assert_eq!(result.save_attempts.len(), 1);
        assert_eq!(
            result
                .events
                .iter()
                .map(|event| event.gateway)
                .collect::<Vec<_>>(),
            vec![RoguelikeGateway::Json, RoguelikeGateway::SetString]
        );
    }

    #[test]
    fn cold_class_failure_occurs_after_ascension_writes_but_before_reset() {
        let mut context = basic_context(RoguelikeProgressionAction::AscensionComplete);
        context.initial.current_ascension = i32::MAX;
        context.initial.ascension_score = i32::MAX;
        context.initial.round_score = 1;
        context.fail_at = Some(RoguelikeFailurePoint {
            gateway: RoguelikeGateway::ClassInit,
            occurrence: 1,
        });
        let result = replay_roguelike_progression(&context).unwrap();
        assert_eq!(result.state.current_ascension, i32::MIN);
        assert_eq!(result.state.best_ascension, 7);
        assert_eq!(result.state.prev_ascension_score, i32::MIN);
        assert_eq!(result.state.best_village, 0);
        assert_eq!(result.state.current_village, 3);
        assert_eq!(result.state.ascension_score, i32::MAX);
        assert_eq!(result.global_village, 123);
        assert!(!result.game_data_initialized);
        assert!(result.save_attempts.is_empty());
    }

    #[test]
    fn unsupported_contexts_reject_atomically_and_schema_requires_provenance() {
        let valid = basic_context(RoguelikeProgressionAction::StageCompleted);
        for index in 0..8 {
            let mut context = valid.clone();
            match index {
                0 => context.rule_version.clear(),
                1 => context.metadata_initialized = false,
                2 => context.services_preserve_mode = false,
                3 => context.services_preserve_global_village = false,
                4 => context.abandon_uses_audited_override = false,
                5 => {
                    context.fail_at = Some(RoguelikeFailurePoint {
                        gateway: RoguelikeGateway::Json,
                        occurrence: 0,
                    })
                }
                6 => {
                    context.fail_at = Some(RoguelikeFailurePoint {
                        gateway: RoguelikeGateway::Json,
                        occurrence: 3,
                    })
                }
                _ => {
                    context.fail_at = Some(RoguelikeFailurePoint {
                        gateway: RoguelikeGateway::ClassInit,
                        occurrence: 2,
                    })
                }
            }
            let original = context.clone();
            assert_eq!(
                replay_roguelike_progression(&context),
                Err(InvalidRoguelikeProgressionContext)
            );
            assert_eq!(context, original);
        }
        let encoded = serde_json::to_value(&valid).unwrap();
        assert_eq!(
            serde_json::from_value::<RoguelikeProgressionContext>(encoded.clone()).unwrap(),
            valid
        );
        for field in [
            "metadata_initialized",
            "services_preserve_mode",
            "services_preserve_global_village",
            "abandon_uses_audited_override",
            "game_data_initialized",
            "initial_global_village",
        ] {
            let mut missing = encoded.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<RoguelikeProgressionContext>(missing).is_err());
        }
        let mut extra = encoded;
        extra["infer_services"] = true.into();
        assert!(serde_json::from_value::<RoguelikeProgressionContext>(extra).is_err());
    }
}
