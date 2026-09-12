//! Offline StandardMode caller replay. Services and their failures are explicit.
//! This does not execute delegates, serialize saves, or change live solver state.

use serde::{Deserialize, Serialize};

pub const STANDARD_PROGRESSION_NATIVE_V1: &str = "standard_progression_native_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StandardModeState {
    pub score: i32,
    pub current_level: i32,
    pub saved_villages: i32,
    pub current_died_times: i32,
    pub best_died_times: i32,
    /// Native nonzero byte; distinct from completion of the current run.
    pub completed: u8,
    pub round_score: i32,
    pub current_score: i32,
    pub best_score: i32,
    pub current_completed: u8,
    pub fail_score_decrease: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgressionGateway {
    Unrevealed,
    Ui,
    ClassInit,
    IncreaseVillage,
    GetInt,
    SetInt,
    Json,
    SetString,
    VirtualDeinit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum StandardProgressionAction {
    Failed,
    CharacterKilled {
        character_present: bool,
        /// None is an actual null dataRef, not unknown identity.
        data_type: Option<i32>,
        unrevealed_count: i32,
        ui_subscribed: bool,
    },
    StageCompleted {
        /// None denotes a null project/game/profile-array chain.
        profile_count: Option<i32>,
        game_data_initialized: bool,
    },
    UpdateScore {
        score: i32,
        level: i32,
        profile_count: Option<i32>,
        saved_max_first: i32,
        saved_max_second: i32,
        /// Actual global value consumed after any class-initialization call.
        global_village: i32,
        game_data_initialized: bool,
    },
    AbandonRun {
        ui_subscribed: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StandardProgressionContext {
    pub rule_version: String,
    pub initial: StandardModeState,
    pub operation: StandardProgressionAction,
    /// Required: native metadata slots/guards were initialized before entry.
    pub metadata_initialized: bool,
    /// Required scope: external services do not mutate the modeled mode fields.
    /// Their own globals, delegates, save data and UI are outside this replay.
    pub services_preserve_mode: bool,
    /// Stop at the first invocation of this service, if reached.
    pub fail_at: Option<ProgressionGateway>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "gateway", rename_all = "snake_case")]
pub enum ProgressionFailure {
    Null,
    Gateway(ProgressionGateway),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StandardProgressionResult {
    pub state: StandardModeState,
    pub events: Vec<ProgressionGateway>,
    /// Value passed to SetInt, including when that service subsequently fails.
    pub requested_saved_max: Option<i32>,
    pub failure: Option<ProgressionFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidProgressionContext;

struct Replay {
    output: StandardProgressionResult,
    fail_at: Option<ProgressionGateway>,
}

impl Replay {
    fn gateway(&mut self, gateway: ProgressionGateway) -> Result<(), ProgressionFailure> {
        self.output.events.push(gateway);
        if self.fail_at == Some(gateway) {
            Err(ProgressionFailure::Gateway(gateway))
        } else {
            Ok(())
        }
    }

    fn save(&mut self) -> Result<(), ProgressionFailure> {
        self.gateway(ProgressionGateway::Json)?;
        self.gateway(ProgressionGateway::SetString)
    }

    fn apply(&mut self, operation: &StandardProgressionAction) -> Result<(), ProgressionFailure> {
        use ProgressionGateway as G;
        match *operation {
            StandardProgressionAction::Failed => {
                let state = &mut self.output.state;
                state.current_died_times = state.current_died_times.wrapping_add(1);
                state.current_score = state
                    .current_score
                    .wrapping_sub(state.fail_score_decrease.wrapping_add(state.round_score));
                state.round_score = 0;
                if state.completed == 0 {
                    state.best_died_times = state.current_died_times;
                }
                self.save()?;
            }
            StandardProgressionAction::CharacterKilled {
                character_present,
                data_type,
                unrevealed_count,
                ui_subscribed,
            } => {
                if !character_present || data_type.is_none() {
                    return Err(ProgressionFailure::Null);
                }
                if matches!(data_type, Some(30 | 100)) {
                    self.gateway(G::Unrevealed)?;
                    let amount = unrevealed_count.wrapping_add(5).wrapping_mul(10);
                    let state = &mut self.output.state;
                    state.round_score = state.round_score.wrapping_add(amount);
                    state.current_score = state.current_score.wrapping_add(amount);
                    if ui_subscribed {
                        self.gateway(G::Ui)?;
                    }
                }
            }
            StandardProgressionAction::StageCompleted {
                profile_count,
                game_data_initialized,
            } => {
                self.output.state.saved_villages = self.output.state.saved_villages.wrapping_add(1);
                let last = profile_count
                    .ok_or(ProgressionFailure::Null)?
                    .wrapping_sub(1);
                let state = &mut self.output.state;
                state.round_score = 0;
                if state.current_level < last {
                    if state.completed == 0 {
                        state.best_score = state.current_score;
                    }
                    if !game_data_initialized {
                        self.gateway(G::ClassInit)?;
                    }
                    self.gateway(G::IncreaseVillage)?;
                } else {
                    state.current_completed = 1;
                    // Native retains an already nonzero completed byte.
                    if state.completed == 0 {
                        state.completed = 1;
                    }
                    state.best_died_times = state.best_died_times.max(state.current_died_times);
                    state.best_score = state.best_score.max(state.current_score);
                }
                self.save()?;
            }
            StandardProgressionAction::UpdateScore {
                score,
                level,
                profile_count,
                saved_max_first,
                saved_max_second,
                global_village,
                game_data_initialized,
            } => {
                let last = profile_count
                    .ok_or(ProgressionFailure::Null)?
                    .wrapping_sub(1);
                let state = &mut self.output.state;
                if state.current_level < last {
                    state.current_level = level;
                }
                state.score = state.score.max(score);
                let current = state.current_level;
                self.gateway(G::GetInt)?;
                if current > saved_max_first {
                    if !game_data_initialized {
                        self.gateway(G::ClassInit)?;
                    }
                    self.gateway(G::GetInt)?;
                    if global_village > saved_max_second {
                        self.output.requested_saved_max = Some(global_village);
                        self.gateway(G::SetInt)?;
                    }
                }
                self.save()?;
            }
            StandardProgressionAction::AbandonRun { ui_subscribed } => {
                self.gateway(G::VirtualDeinit)?;
                let state = &mut self.output.state;
                state.current_completed = 0;
                state.score = 0;
                state.current_level = 0;
                state.round_score = 0;
                state.current_score = 0;
                state.current_died_times = 0;
                self.save()?;
                if ui_subscribed {
                    self.gateway(G::Ui)?;
                }
            }
        }
        Ok(())
    }
}

/// Return exact partial caller writes at a native failure boundary. Unsupported
/// provenance rejects the contract; it is distinct from an observed null/failure.
pub fn replay_standard_progression(
    context: &StandardProgressionContext,
) -> Result<StandardProgressionResult, InvalidProgressionContext> {
    if context.rule_version != STANDARD_PROGRESSION_NATIVE_V1
        || !context.services_preserve_mode
        || !context.metadata_initialized
    {
        return Err(InvalidProgressionContext);
    }
    let mut replay = Replay {
        output: StandardProgressionResult {
            state: context.initial.clone(),
            events: Vec::new(),
            requested_saved_max: None,
            failure: None,
        },
        fail_at: context.fail_at,
    };
    replay.output.failure = replay.apply(&context.operation).err();
    Ok(replay.output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn report() -> Value {
        serde_json::from_str(include_str!(
            "../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_standard_mode_progression.json"
        ))
        .unwrap()
    }

    fn context(case: &Value) -> Option<StandardProgressionContext> {
        let options = &case["options"];
        let number =
            |name: &str, default: i32| options[name].as_i64().map_or(default, |v| v as i32);
        let flag = |name: &str| options[name].as_bool().unwrap_or(false);
        let missing = options["missing"].as_str();
        let count = if matches!(missing, Some("project" | "game" | "array")) {
            None
        } else {
            Some(number("count", 7))
        };
        let operation = match case["method"].as_str().unwrap() {
            "StandardMode.OnFailed" => StandardProgressionAction::Failed,
            "StandardMode.OnCharacterKilled" => StandardProgressionAction::CharacterKilled {
                character_present: missing != Some("character"),
                data_type: if missing == Some("data") {
                    None
                } else {
                    Some(number("type", 100))
                },
                unrevealed_count: number("hidden", 3),
                ui_subscribed: flag("ui"),
            },
            "StandardMode.OnStageCompleted" => StandardProgressionAction::StageCompleted {
                profile_count: count,
                game_data_initialized: !flag("cold"),
            },
            "StandardMode.UpdateScore" => {
                let prefs: Vec<i32> = options["prefs"].as_array().map_or(vec![0], |values| {
                    values
                        .iter()
                        .map(|value| value.as_i64().unwrap() as i32)
                        .collect()
                });
                StandardProgressionAction::UpdateScore {
                    score: case["args"][0].as_i64().unwrap() as i32,
                    level: case["args"][1].as_i64().unwrap() as i32,
                    profile_count: count,
                    saved_max_first: prefs[0],
                    saved_max_second: *prefs.get(1).unwrap_or(&prefs[0]),
                    global_village: number("village", 4),
                    game_data_initialized: !flag("cold"),
                }
            }
            "StandardMode.AbandonRun" => StandardProgressionAction::AbandonRun {
                ui_subscribed: flag("ui"),
            },
            _ => return None,
        };
        Some(StandardProgressionContext {
            rule_version: STANDARD_PROGRESSION_NATIVE_V1.into(),
            initial: serde_json::from_value(case["initial"].clone()).unwrap(),
            operation,
            metadata_initialized: true,
            services_preserve_mode: true,
            fail_at: serde_json::from_value(options["fail"].clone()).unwrap(),
        })
    }

    #[test]
    fn matches_native_progression_fields_events_and_partial_failures() {
        let data = report();
        let mut checked = 0;
        for case in data["cases"].as_array().unwrap() {
            let Some(context) = context(case) else {
                continue;
            };
            let result = replay_standard_progression(&context).unwrap();
            assert_eq!(
                serde_json::to_value(&result.state).unwrap(),
                case["final"],
                "{case}"
            );
            assert_eq!(
                serde_json::to_value(&result.events).unwrap(),
                case["events"],
                "{case}"
            );
            let expected: Vec<i32> = serde_json::from_value(case["set_int"].clone()).unwrap();
            assert_eq!(
                result.requested_saved_max.into_iter().collect::<Vec<_>>(),
                expected,
                "{case}"
            );
            let failure = match result.failure {
                None => Value::Null,
                Some(ProgressionFailure::Null) => Value::String("null".into()),
                Some(ProgressionFailure::Gateway(g)) => serde_json::to_value(g).unwrap(),
            };
            assert_eq!(failure, case["error"], "{case}");
            checked += 1;
        }
        assert_eq!(checked, 304, "native fixture selection changed");
    }

    #[test]
    fn unsupported_contract_is_atomic_and_fields_are_strict() {
        let data = report();
        let mut ctx = context(&data["cases"][0]).unwrap();
        let initial = ctx.initial.clone();
        ctx.rule_version = "unknown".into();
        assert_eq!(
            replay_standard_progression(&ctx),
            Err(InvalidProgressionContext)
        );
        ctx.rule_version = STANDARD_PROGRESSION_NATIVE_V1.into();
        ctx.services_preserve_mode = false;
        assert_eq!(
            replay_standard_progression(&ctx),
            Err(InvalidProgressionContext)
        );
        assert_eq!(ctx.initial, initial);
        ctx.services_preserve_mode = true;
        ctx.metadata_initialized = false;
        assert_eq!(
            replay_standard_progression(&ctx),
            Err(InvalidProgressionContext)
        );
        let mut value = serde_json::to_value(&ctx).unwrap();
        value["unknown_truth"] = true.into();
        assert!(serde_json::from_value::<StandardProgressionContext>(value).is_err());
    }

    #[test]
    fn noncanonical_completed_byte_survives_terminal_success() {
        let data = report();
        let mut ctx = context(&data["cases"][0]).unwrap();
        ctx.initial.completed = 7;
        ctx.operation = StandardProgressionAction::StageCompleted {
            profile_count: Some(0),
            game_data_initialized: true,
        };
        ctx.fail_at = None;
        let result = replay_standard_progression(&ctx).unwrap();
        assert_eq!(result.state.completed, 7);
        assert_eq!(result.state.current_completed, 1);
    }
}
