//! Opt-in native RoguelikeStandard stage -> GameData.IncreaseVillage bridge.
//! The stage receiver and the static current mode are distinct explicit inputs.
//! Existing roguelike_progression replay retains its preserve-global contract.

use crate::roguelike_progression::{
    apply_stage_writes, RoguelikeSaveAttempt, RoguelikeStandardState,
};
use serde::{Deserialize, Serialize};

pub const ROGUELIKE_VILLAGE_BRIDGE_NATIVE_V1: &str = "roguelike_village_bridge_native_v1";
const MAX_PROFILES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "receiver", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoguelikeVillageReceiver {
    Caller,
    Distinct { current_ascension: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GlobalVillageMode {
    Null,
    Other,
    /// Runtime assignability to StandardMode, including verified subtypes.
    Standard,
    /// Concrete MaxLevel uses this global receiver, not virtual GetMaxLevel.
    Roguelike {
        receiver: RoguelikeVillageReceiver,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoguelikeVillageProfile {
    /// None is a null inner array. Some retains the native low int32 length bits.
    pub inner_array_length: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VillageConfiguration {
    pub project_present: bool,
    pub game_data_present: bool,
    /// None is a null standard array; this is not an unknown count.
    pub standard_array_length: Option<i32>,
    /// None is a null outer array; None entries are null AscensionsList objects.
    pub roguelike_profiles: Option<Vec<Option<RoguelikeVillageProfile>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VillageBridgeGateway {
    ClassInit,
    Json,
    SetString,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VillageBridgeEvent {
    ClassInit,
    IncreaseVillage,
    MaxLevel,
    Json,
    SetString,
    Null,
    Bounds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "gateway", rename_all = "snake_case")]
pub enum VillageBridgeFailure {
    Gateway(VillageBridgeGateway),
    Null,
    Bounds,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoguelikeVillageBridgeContext {
    pub rule_version: String,
    pub initial: RoguelikeStandardState,
    pub initial_global_village: i32,
    pub game_data_initialized: bool,
    pub global_mode: GlobalVillageMode,
    pub configuration: VillageConfiguration,
    pub metadata_initialized: bool,
    /// The mode kind comes from native runtime assignability, not GetGameMode.
    pub mode_kind_verified: bool,
    /// Global mode identity, class hierarchy and configuration stay stable.
    pub global_mode_and_configuration_stable: bool,
    /// Only external class-init/JSON/preferences services preserve mode/village.
    /// Native IncreaseVillage is modeled and may update the static village.
    pub external_services_preserve_mode_and_village: bool,
    /// Stop at this gateway if reached. This single-stage bridge saves at most once.
    pub fail_at: Option<VillageBridgeGateway>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VillageBridgeSnapshot {
    pub caller_fields: RoguelikeStandardState,
    pub global_village: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoguelikeVillageBridgeResult {
    pub state: RoguelikeStandardState,
    pub global_village: i32,
    pub game_data_initialized: bool,
    pub events: Vec<VillageBridgeEvent>,
    pub helper_entry_snapshot: Option<VillageBridgeSnapshot>,
    pub save_snapshot: Option<VillageBridgeSnapshot>,
    pub save_attempt: Option<RoguelikeSaveAttempt>,
    /// Includes a negative selected index when the native unsigned check fails.
    pub selected_outer_index: Option<i32>,
    pub failure: Option<VillageBridgeFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidVillageBridgeContext;

struct Replay {
    output: RoguelikeVillageBridgeResult,
    fail_at: Option<VillageBridgeGateway>,
}

impl Replay {
    fn gateway(&mut self, gateway: VillageBridgeGateway) -> Result<(), VillageBridgeFailure> {
        self.output.events.push(match gateway {
            VillageBridgeGateway::ClassInit => VillageBridgeEvent::ClassInit,
            VillageBridgeGateway::Json => VillageBridgeEvent::Json,
            VillageBridgeGateway::SetString => VillageBridgeEvent::SetString,
        });
        if self.fail_at == Some(gateway) {
            Err(VillageBridgeFailure::Gateway(gateway))
        } else {
            Ok(())
        }
    }

    fn snapshot(&self) -> VillageBridgeSnapshot {
        VillageBridgeSnapshot {
            caller_fields: self.output.state.clone(),
            global_village: self.output.global_village,
        }
    }

    fn save(&mut self) -> Result<(), VillageBridgeFailure> {
        self.output.save_snapshot = Some(self.snapshot());
        self.output.save_attempt = Some(RoguelikeSaveAttempt {
            index: 1,
            state: self.output.state.clone(),
            json_completed: false,
            set_string_completed: false,
        });
        self.gateway(VillageBridgeGateway::Json)?;
        if let Some(attempt) = &mut self.output.save_attempt {
            attempt.json_completed = true;
        }
        self.gateway(VillageBridgeGateway::SetString)?;
        if let Some(attempt) = &mut self.output.save_attempt {
            attempt.set_string_completed = true;
        }
        Ok(())
    }

    fn maximum_level(
        &mut self,
        receiver: RoguelikeVillageReceiver,
        configuration: &VillageConfiguration,
    ) -> Result<i32, VillageBridgeFailure> {
        self.output.events.push(VillageBridgeEvent::MaxLevel);
        if !configuration.project_present || !configuration.game_data_present {
            return Err(VillageBridgeFailure::Null);
        }
        let profiles = configuration
            .roguelike_profiles
            .as_ref()
            .ok_or(VillageBridgeFailure::Null)?;
        let ascension = match receiver {
            RoguelikeVillageReceiver::Caller => self.output.state.current_ascension,
            RoguelikeVillageReceiver::Distinct { current_ascension } => current_ascension,
        };
        // Context validation bounds this cast. Native upper-clamps first, then
        // rejects negative/empty indices through an unsigned array-bounds gate.
        let index = ascension.min(profiles.len() as i32 - 1);
        self.output.selected_outer_index = Some(index);
        if index < 0 {
            return Err(VillageBridgeFailure::Bounds);
        }
        let profile = profiles[index as usize]
            .as_ref()
            .ok_or(VillageBridgeFailure::Null)?;
        let count = profile
            .inner_array_length
            .ok_or(VillageBridgeFailure::Null)?;
        Ok(count.wrapping_sub(1))
    }

    fn increase_village(
        &mut self,
        mode: GlobalVillageMode,
        configuration: &VillageConfiguration,
    ) -> Result<(), VillageBridgeFailure> {
        self.output.events.push(VillageBridgeEvent::IncreaseVillage);
        self.output.helper_entry_snapshot = Some(self.snapshot());
        let maximum = match mode {
            GlobalVillageMode::Null | GlobalVillageMode::Other => None,
            GlobalVillageMode::Standard => {
                if !configuration.project_present || !configuration.game_data_present {
                    return Err(VillageBridgeFailure::Null);
                }
                Some(
                    configuration
                        .standard_array_length
                        .ok_or(VillageBridgeFailure::Null)?
                        .wrapping_sub(1),
                )
            }
            GlobalVillageMode::Roguelike { receiver } => {
                Some(self.maximum_level(receiver, configuration)?)
            }
        };
        if maximum.is_some_and(|maximum| self.output.global_village < maximum) {
            self.output.global_village = self.output.global_village.wrapping_add(1);
        }
        Ok(())
    }

    fn apply(
        &mut self,
        context: &RoguelikeVillageBridgeContext,
    ) -> Result<(), VillageBridgeFailure> {
        apply_stage_writes(&mut self.output.state);
        if !self.output.game_data_initialized {
            self.gateway(VillageBridgeGateway::ClassInit)?;
            self.output.game_data_initialized = true;
        }
        self.increase_village(context.global_mode, &context.configuration)?;
        self.save()
    }
}

/// Opt into the audited stage/village composition. This does not loosen the
/// original progression API's gateway contract. Null configuration and bounds
/// errors return native partial caller writes; unsupported provenance rejects
/// before replay. Successful external class initialization marks GameData ready.
pub fn replay_roguelike_village_bridge(
    context: &RoguelikeVillageBridgeContext,
) -> Result<RoguelikeVillageBridgeResult, InvalidVillageBridgeContext> {
    if context.rule_version != ROGUELIKE_VILLAGE_BRIDGE_NATIVE_V1
        || !context.metadata_initialized
        || !context.mode_kind_verified
        || !context.global_mode_and_configuration_stable
        || !context.external_services_preserve_mode_and_village
        || context
            .configuration
            .roguelike_profiles
            .as_ref()
            .is_some_and(|profiles| profiles.len() > MAX_PROFILES)
    {
        return Err(InvalidVillageBridgeContext);
    }
    let mut replay = Replay {
        output: RoguelikeVillageBridgeResult {
            state: context.initial.clone(),
            global_village: context.initial_global_village,
            game_data_initialized: context.game_data_initialized,
            events: Vec::new(),
            helper_entry_snapshot: None,
            save_snapshot: None,
            save_attempt: None,
            selected_outer_index: None,
            failure: None,
        },
        fail_at: context.fail_at,
    };
    let failure = replay.apply(context).err();
    match failure {
        Some(VillageBridgeFailure::Null) => replay.output.events.push(VillageBridgeEvent::Null),
        Some(VillageBridgeFailure::Bounds) => replay.output.events.push(VillageBridgeEvent::Bounds),
        _ => {}
    }
    replay.output.failure = failure;
    Ok(replay.output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roguelike_progression::{
        replay_roguelike_progression, RoguelikeProgressionAction, RoguelikeProgressionContext,
        ROGUELIKE_PROGRESSION_NATIVE_V1,
    };
    use serde_json::Value;

    fn report() -> Value {
        serde_json::from_str(include_str!(
            "../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_village_bridge.json"
        ))
        .unwrap()
    }

    fn fixture_context(case: &Value) -> RoguelikeVillageBridgeContext {
        let missing = case["missing"].as_str();
        let global_mode = match case["mode_kind"].as_str().unwrap() {
            "null" => GlobalVillageMode::Null,
            "other" | "short_other" => GlobalVillageMode::Other,
            "standard" | "standard_subclass" => GlobalVillageMode::Standard,
            "rogue_same" => GlobalVillageMode::Roguelike {
                receiver: RoguelikeVillageReceiver::Caller,
            },
            "rogue" | "rogue_subclass" => GlobalVillageMode::Roguelike {
                receiver: RoguelikeVillageReceiver::Distinct {
                    current_ascension: case["global_mode_ascension"].as_i64().unwrap() as i32,
                },
            },
            _ => panic!("Unknown native fixture mode kind"),
        };
        let profiles = case["profile_lengths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| {
                (missing != Some("rogue_entry")).then_some(RoguelikeVillageProfile {
                    inner_array_length: (missing != Some("rogue_inner"))
                        .then_some(value.as_i64().unwrap() as i32),
                })
            })
            .collect();
        RoguelikeVillageBridgeContext {
            rule_version: ROGUELIKE_VILLAGE_BRIDGE_NATIVE_V1.into(),
            initial: serde_json::from_value(case["caller_initial"].clone()).unwrap(),
            initial_global_village: case["initial_global_village"].as_i64().unwrap() as i32,
            game_data_initialized: !case["cold_class"].as_bool().unwrap(),
            global_mode,
            configuration: VillageConfiguration {
                project_present: missing != Some("project"),
                game_data_present: missing != Some("game"),
                standard_array_length: (missing != Some("standard_array"))
                    .then_some(case["standard_count"].as_i64().unwrap() as i32),
                roguelike_profiles: (missing != Some("rogue_outer")).then_some(profiles),
            },
            metadata_initialized: true,
            mode_kind_verified: true,
            global_mode_and_configuration_stable: true,
            external_services_preserve_mode_and_village: true,
            fail_at: serde_json::from_value(case["gateway_failure"].clone()).unwrap(),
        }
    }

    fn basic_context() -> RoguelikeVillageBridgeContext {
        fixture_context(&report()["cases"][0])
    }

    #[test]
    fn all_660_native_cases_match_fields_identity_branch_order_and_partial_failures() {
        let report = report();
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 660);
        for case in cases {
            let context = fixture_context(case);
            let before = context.clone();
            let result = replay_roguelike_village_bridge(&context).unwrap();
            assert_eq!(context, before);
            assert_eq!(
                serde_json::to_value(&result.state).unwrap(),
                case["caller_result"]
            );
            assert_eq!(
                result.global_village,
                case["result_global_village"].as_i64().unwrap() as i32
            );
            assert_eq!(
                serde_json::to_value(&result.events).unwrap(),
                case["events"]
            );
            assert_eq!(
                serde_json::to_value(&result.helper_entry_snapshot).unwrap(),
                case["helper_entry_snapshot"]
            );
            assert_eq!(
                serde_json::to_value(&result.save_snapshot).unwrap(),
                case["save_snapshot"]
            );
            assert_eq!(
                serde_json::to_value(result.selected_outer_index).unwrap(),
                case["selected_outer_index"]
            );
            let expected_failure = match case["error"].as_str() {
                None => None,
                Some("null") => Some(VillageBridgeFailure::Null),
                Some("bounds") => Some(VillageBridgeFailure::Bounds),
                Some("class_init") => Some(VillageBridgeFailure::Gateway(
                    VillageBridgeGateway::ClassInit,
                )),
                Some("json") => Some(VillageBridgeFailure::Gateway(VillageBridgeGateway::Json)),
                Some("set_string") => Some(VillageBridgeFailure::Gateway(
                    VillageBridgeGateway::SetString,
                )),
                _ => panic!("Unknown native fixture failure"),
            };
            assert_eq!(result.failure, expected_failure);
            let expected_initialized = context.game_data_initialized
                || (result.events.contains(&VillageBridgeEvent::ClassInit)
                    && result.failure
                        != Some(VillageBridgeFailure::Gateway(
                            VillageBridgeGateway::ClassInit,
                        )));
            assert_eq!(result.game_data_initialized, expected_initialized);
            if let Some(save) = &result.save_attempt {
                assert_eq!(save.index, 1);
                assert_eq!(save.state, result.state);
                assert_eq!(
                    save.json_completed,
                    result.failure
                        != Some(VillageBridgeFailure::Gateway(VillageBridgeGateway::Json))
                );
                assert_eq!(save.set_string_completed, result.failure.is_none());
            } else {
                assert!(result.save_snapshot.is_none());
            }
        }
    }

    #[test]
    fn global_mode_receiver_selects_profile_independently_from_stage_caller() {
        let mut context = basic_context();
        context.initial.current_ascension = 2;
        context.initial.current_village = 10;
        context.initial_global_village = 1;
        context.configuration.roguelike_profiles = Some(
            [1, 3, 7]
                .into_iter()
                .map(|count| {
                    Some(RoguelikeVillageProfile {
                        inner_array_length: Some(count),
                    })
                })
                .collect(),
        );
        context.global_mode = GlobalVillageMode::Roguelike {
            receiver: RoguelikeVillageReceiver::Distinct {
                current_ascension: 0,
            },
        };
        let distinct = replay_roguelike_village_bridge(&context).unwrap();
        assert_eq!(distinct.selected_outer_index, Some(0));
        assert_eq!(distinct.global_village, 1);
        assert_eq!(distinct.state.current_village, 11);
        context.global_mode = GlobalVillageMode::Roguelike {
            receiver: RoguelikeVillageReceiver::Caller,
        };
        let same = replay_roguelike_village_bridge(&context).unwrap();
        assert_eq!(same.selected_outer_index, Some(2));
        assert_eq!(same.global_village, 2);
        assert_eq!(same.state, distinct.state);
    }

    #[test]
    fn helper_bounds_failure_retains_wrapped_caller_writes_without_a_save() {
        let mut context = basic_context();
        context.initial.current_village = i32::MAX;
        context.initial.current_ascension = -1;
        context.global_mode = GlobalVillageMode::Roguelike {
            receiver: RoguelikeVillageReceiver::Caller,
        };
        context.configuration.roguelike_profiles = Some(vec![Some(RoguelikeVillageProfile {
            inner_array_length: Some(7),
        })]);
        let result = replay_roguelike_village_bridge(&context).unwrap();
        assert_eq!(result.failure, Some(VillageBridgeFailure::Bounds));
        assert_eq!(result.state.current_village, i32::MIN);
        assert_eq!(result.global_village, context.initial_global_village);
        assert!(result.helper_entry_snapshot.is_some());
        assert!(result.save_attempt.is_none());
    }

    #[test]
    fn configuration_nulls_are_unused_for_other_modes_and_late_save_failure_keeps_increment() {
        let mut context = basic_context();
        context.global_mode = GlobalVillageMode::Null;
        context.configuration.project_present = false;
        context.configuration.game_data_present = false;
        context.configuration.standard_array_length = None;
        context.configuration.roguelike_profiles = None;
        let skipped = replay_roguelike_village_bridge(&context).unwrap();
        assert_eq!(skipped.failure, None);
        assert_eq!(skipped.global_village, context.initial_global_village);
        context.global_mode = GlobalVillageMode::Standard;
        context.configuration.project_present = true;
        context.configuration.game_data_present = true;
        context.configuration.standard_array_length = Some(7);
        context.initial_global_village = 0;
        context.fail_at = Some(VillageBridgeGateway::SetString);
        let failed = replay_roguelike_village_bridge(&context).unwrap();
        assert_eq!(failed.global_village, 1);
        assert_eq!(
            failed.failure,
            Some(VillageBridgeFailure::Gateway(
                VillageBridgeGateway::SetString
            ))
        );
        let save = failed.save_attempt.unwrap();
        assert!(save.json_completed);
        assert!(!save.set_string_completed);
    }

    #[test]
    fn original_gateway_replay_keeps_its_existing_global_preservation_semantics() {
        let bridge = basic_context();
        let context = RoguelikeProgressionContext {
            rule_version: ROGUELIKE_PROGRESSION_NATIVE_V1.into(),
            initial: bridge.initial.clone(),
            initial_global_village: 42,
            game_data_initialized: true,
            operation: RoguelikeProgressionAction::StageCompleted,
            metadata_initialized: true,
            services_preserve_mode: true,
            services_preserve_global_village: true,
            abandon_uses_audited_override: true,
            fail_at: None,
        };
        let result = replay_roguelike_progression(&context).unwrap();
        assert_eq!(result.global_village, 42);
        assert_eq!(
            result.state.current_village,
            context.initial.current_village.wrapping_add(1)
        );
        let mut invalid = context;
        invalid.services_preserve_global_village = false;
        assert!(replay_roguelike_progression(&invalid).is_err());
    }

    #[test]
    fn unsupported_contracts_reject_atomically_and_schema_requires_opt_in() {
        let valid = basic_context();
        for index in 0..6 {
            let mut context = valid.clone();
            match index {
                0 => context.rule_version.clear(),
                1 => context.metadata_initialized = false,
                2 => context.mode_kind_verified = false,
                3 => context.global_mode_and_configuration_stable = false,
                4 => context.external_services_preserve_mode_and_village = false,
                _ => context.configuration.roguelike_profiles = Some(vec![None; MAX_PROFILES + 1]),
            }
            let before = context.clone();
            assert_eq!(
                replay_roguelike_village_bridge(&context),
                Err(InvalidVillageBridgeContext)
            );
            assert_eq!(context, before);
        }
        let encoded = serde_json::to_value(valid).unwrap();
        for field in [
            "metadata_initialized",
            "mode_kind_verified",
            "global_mode_and_configuration_stable",
            "external_services_preserve_mode_and_village",
            "global_mode",
            "configuration",
        ] {
            let mut missing = encoded.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<RoguelikeVillageBridgeContext>(missing).is_err());
        }
        let mut extra = encoded;
        extra["assume_same_mode"] = true.into();
        assert!(serde_json::from_value::<RoguelikeVillageBridgeContext>(extra).is_err());
    }
}
