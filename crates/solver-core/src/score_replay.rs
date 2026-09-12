//! Opt-in offline replay of the pinned Score lifecycle and Gameplay.UpdateScore.
//! Collection content and external services are explicit, stable inputs.
use serde::{Deserialize, Serialize};

pub const SCORE_REPLAY_NATIVE_V1: &str = "score_replay_native_v1";
pub const MAX_CONTRIBUTIONS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ScoreState {
    pub completed_stages: i32,
    pub points_for_completing: i32,
    pub completed_days: i32,
    pub multiplier: i32,
    pub round_points: i32,
    pub overall_points: i32,
    /// Old-specific fields remain opaque preserved bytes for Base/New fixtures.
    pub killed_goods: i32,
    pub temp_unrevealed_cards: i32,
    pub unrevealed_cards: i32,
    pub killed_evils: i32,
    pub temp_killed_evils: i32,
    pub point_per_kill: i32,
    pub points_per_unrevealed: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreKind {
    Base,
    New,
    Old,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreOperation {
    GetBaseScore,
    GetMultiplier,
    AddPointsOnEvilKill,
    UpdateFullScore,
    GetFullPoints,
    Constructor,
    GameplayUpdateScore,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Contribution {
    Null,
    MissingInfo,
    Value { point: i32, multiplier_bits: u32 },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreMissing {
    Instance,
    Characters,
    Relics,
    Character,
    Icon,
    Transform,
    Vfx,
    Mode,
    Score,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreGateway {
    ClassInit,
    Characters,
    Enumerator,
    MoveNext,
    Dispose,
    Unrevealed,
    BaseScore,
    Multiplier,
    FullPoints,
    Transform,
    Position,
    FloatingScore,
    ModeUpdate,
    ObjectCtor,
}
impl ScoreGateway {
    fn index(self) -> usize {
        self as usize
    }
    fn bound(self) -> u32 {
        match self {
            Self::ClassInit | Self::Characters => 2,
            Self::Enumerator | Self::Dispose => 4,
            Self::MoveNext => 4 * (MAX_CONTRIBUTIONS as u32 + 1),
            _ => 1,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreFailurePoint {
    pub gateway: ScoreGateway,
    pub occurrence: u32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreFailure {
    NullReference,
    Gateway(ScoreFailurePoint),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreSource {
    Characters,
    Relics,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreRuntimeClass {
    Gameplay,
    GameData,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreReplayContext {
    pub rule_version: String,
    pub kind: ScoreKind,
    pub operation: ScoreOperation,
    pub initial: ScoreState,
    pub metadata_initialized: bool,
    /// Native fixture FP environment. Exception status bits (0..5) are ignored;
    /// controls must mask exceptions, round nearest-even and preserve subnormals.
    pub native_mxcsr: u32,
    pub services_preserve_state_and_globals: bool,
    pub uses_audited_virtual_slots: bool,
    /// Required for the composed Gameplay entry: its receiver is the singleton.
    pub gameplay_receiver_is_instance: bool,
    pub gameplay_initialized: bool,
    pub game_data_initialized: bool,
    pub current_level: i32,
    pub completed_days_argument: i32,
    pub unrevealed_count: i32,
    pub characters: Vec<Contribution>,
    pub relics: Vec<Contribution>,
    pub missing: Option<ScoreMissing>,
    pub position_bits: [u32; 3],
    pub fail_at: Option<ScoreFailurePoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScoreEvent {
    pub gateway: ScoreGateway,
    pub occurrence: u32,
    pub source: Option<ScoreSource>,
    pub runtime_class: Option<ScoreRuntimeClass>,
    pub score: Option<i32>,
    pub level: Option<i32>,
    pub position_bits: Option<[u32; 3]>,
    pub state: ScoreState,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreValue {
    Integer(i32),
    FloatBits(u32),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScoreReplayResult {
    pub state: ScoreState,
    pub value: Option<ScoreValue>,
    pub failure: Option<ScoreFailure>,
    pub events: Vec<ScoreEvent>,
    pub gameplay_initialized: bool,
    pub game_data_initialized: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidScoreReplayContext;

// Read-only host check: floating arithmetic below must use the fixture controls.
// Production replay never changes the caller's floating-point environment.
fn host_mxcsr() -> Option<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("sse") {
            return None;
        }
        let mut value = 0_u32;
        // SAFETY: SSE is available and the writable destination is a valid u32.
        unsafe {
            core::arch::asm!("stmxcsr [{address}]", address = in(reg) &mut value, options(nostack, preserves_flags));
        }
        Some(value)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}
fn supported_mxcsr(value: u32) -> bool {
    value & !0x3f == 0x1f80
}

// Explicit SSE NaN priority and invalid-operation NaN; finite multiplication
// rounds once to binary32. No Rust saturating float-to-int conversion is used.
fn native_mul(left: u32, right: u32) -> u32 {
    if left & 0x7fff_ffff > 0x7f80_0000 {
        return left | 0x0040_0000;
    }
    if right & 0x7fff_ffff > 0x7f80_0000 {
        return right | 0x0040_0000;
    }
    let result = f32::from_bits(left) * f32::from_bits(right);
    if result.is_nan() {
        0xffc0_0000
    } else {
        result.to_bits()
    }
}
fn native_cvtt(bits: u32) -> i32 {
    let value = f32::from_bits(bits);
    if !value.is_finite() || !(-2147483648.0..2147483648.0).contains(&value) {
        i32::MIN
    } else {
        value as i32
    }
}
fn full_points(kind: ScoreKind, s: &ScoreState) -> i32 {
    let completion = s.completed_days.wrapping_mul(s.points_for_completing);
    if kind == ScoreKind::Old {
        s.point_per_kill
            .wrapping_mul(s.killed_evils)
            .wrapping_add(s.points_per_unrevealed.wrapping_mul(s.unrevealed_cards))
            .wrapping_add(completion)
    } else {
        completion.wrapping_add(s.overall_points)
    }
}
struct Replay<'a> {
    context: &'a ScoreReplayContext,
    result: ScoreReplayResult,
    counts: [u32; 14],
}
impl Replay<'_> {
    fn event(
        &mut self,
        gateway: ScoreGateway,
        source: Option<ScoreSource>,
        class: Option<ScoreRuntimeClass>,
        score: Option<i32>,
        level: Option<i32>,
    ) -> Result<(), ScoreFailure> {
        let count = &mut self.counts[gateway.index()];
        *count += 1;
        let point = ScoreFailurePoint {
            gateway,
            occurrence: *count,
        };
        self.result.events.push(ScoreEvent {
            gateway,
            occurrence: *count,
            source,
            runtime_class: class,
            score,
            level,
            position_bits: if gateway == ScoreGateway::FloatingScore {
                Some(self.context.position_bits)
            } else {
                None
            },
            state: self.result.state.clone(),
        });
        if self.context.fail_at == Some(point) {
            Err(ScoreFailure::Gateway(point))
        } else {
            Ok(())
        }
    }
    fn simple(&mut self, gateway: ScoreGateway) -> Result<(), ScoreFailure> {
        self.event(gateway, None, None, None, None)
    }
    fn require(&self, missing: ScoreMissing) -> Result<(), ScoreFailure> {
        if self.context.missing == Some(missing) {
            Err(ScoreFailure::NullReference)
        } else {
            Ok(())
        }
    }
    fn initialize(&mut self, class: ScoreRuntimeClass) -> Result<(), ScoreFailure> {
        let ready = match class {
            ScoreRuntimeClass::Gameplay => self.result.gameplay_initialized,
            ScoreRuntimeClass::GameData => self.result.game_data_initialized,
        };
        if !ready {
            self.event(ScoreGateway::ClassInit, None, Some(class), None, None)?;
            match class {
                ScoreRuntimeClass::Gameplay => self.result.gameplay_initialized = true,
                ScoreRuntimeClass::GameData => self.result.game_data_initialized = true,
            }
        }
        Ok(())
    }
    fn fold(&mut self, multiply: bool) -> Result<u32, ScoreFailure> {
        self.initialize(ScoreRuntimeClass::Gameplay)?;
        self.require(ScoreMissing::Instance)?;
        self.simple(ScoreGateway::Characters)?;
        self.require(ScoreMissing::Characters)?;
        let mut accumulator = if multiply { 1.0_f32.to_bits() } else { 0 };
        for source in [ScoreSource::Characters, ScoreSource::Relics] {
            if source == ScoreSource::Relics {
                self.require(ScoreMissing::Relics)?;
            }
            self.event(ScoreGateway::Enumerator, Some(source), None, None, None)?;
            let items = match source {
                ScoreSource::Characters => &self.context.characters,
                ScoreSource::Relics => &self.context.relics,
            };
            for item in items {
                self.event(ScoreGateway::MoveNext, Some(source), None, None, None)?;
                let Contribution::Value {
                    point,
                    multiplier_bits,
                } = item
                else {
                    return Err(ScoreFailure::NullReference);
                };
                accumulator = if multiply {
                    native_mul(accumulator, *multiplier_bits)
                } else {
                    accumulator.wrapping_add(*point as u32)
                };
            }
            self.event(ScoreGateway::MoveNext, Some(source), None, None, None)?;
            self.event(ScoreGateway::Dispose, Some(source), None, None, None)?;
        }
        if multiply {
            self.simple(ScoreGateway::Unrevealed)?;
            let count = self.context.unrevealed_count.min(3) as f32;
            accumulator = native_mul(native_mul(count.to_bits(), 1.25_f32.to_bits()), accumulator);
        }
        Ok(accumulator)
    }
    fn visual(&mut self, amount: i32) -> Result<(), ScoreFailure> {
        self.require(ScoreMissing::Character)?;
        self.require(ScoreMissing::Icon)?;
        self.simple(ScoreGateway::Transform)?;
        self.require(ScoreMissing::Transform)?;
        self.simple(ScoreGateway::Position)?;
        self.require(ScoreMissing::Vfx)?;
        self.event(ScoreGateway::FloatingScore, None, None, Some(amount), None)
    }
    fn run(&mut self) -> Result<(), ScoreFailure> {
        use ScoreOperation::*;
        match self.context.operation {
            GetBaseScore => self.result.value = Some(ScoreValue::Integer(self.fold(false)? as i32)),
            GetMultiplier => self.result.value = Some(ScoreValue::FloatBits(self.fold(true)?)),
            GetFullPoints => {
                self.result.value = Some(ScoreValue::Integer(full_points(
                    self.context.kind,
                    &self.result.state,
                )))
            }
            Constructor => {
                self.result.state.points_for_completing = 100;
                if self.context.kind == ScoreKind::Old {
                    self.result.state.point_per_kill = 50;
                    self.result.state.points_per_unrevealed = 10;
                }
                self.simple(ScoreGateway::ObjectCtor)?;
            }
            AddPointsOnEvilKill => {
                let amount = if self.context.kind == ScoreKind::Old {
                    self.simple(ScoreGateway::Unrevealed)?;
                    let s = &mut self.result.state;
                    s.temp_unrevealed_cards = s
                        .temp_unrevealed_cards
                        .wrapping_add(self.context.unrevealed_count);
                    s.temp_killed_evils = s.temp_killed_evils.wrapping_add(1);
                    self.context
                        .unrevealed_count
                        .wrapping_mul(s.points_per_unrevealed)
                        .wrapping_add(s.point_per_kill)
                } else {
                    self.simple(ScoreGateway::BaseScore)?;
                    let base = self.fold(false)? as i32;
                    self.simple(ScoreGateway::Multiplier)?;
                    let multiplier = self.fold(true)?;
                    let amount = native_cvtt(native_mul(multiplier, (base as f32).to_bits()));
                    self.result.state.round_points =
                        self.result.state.round_points.wrapping_add(amount);
                    amount
                };
                self.visual(amount)?;
            }
            UpdateFullScore | GameplayUpdateScore => {
                self.initialize(ScoreRuntimeClass::Gameplay)?;
                if self.context.operation == GameplayUpdateScore {
                    self.require(ScoreMissing::Score)?;
                }
                self.require(ScoreMissing::Instance)?;
                let s = &mut self.result.state;
                s.completed_stages = self.context.current_level.wrapping_add(1);
                if self.context.kind == ScoreKind::Old {
                    s.killed_evils = s.killed_evils.wrapping_add(s.temp_killed_evils);
                    s.unrevealed_cards = s.unrevealed_cards.wrapping_add(s.temp_unrevealed_cards);
                    s.temp_killed_evils = 0;
                    s.temp_unrevealed_cards = 0;
                } else {
                    s.overall_points = s.overall_points.wrapping_add(s.round_points);
                    s.round_points = 0;
                }
                s.completed_days = self.context.completed_days_argument.wrapping_add(1);
                self.initialize(ScoreRuntimeClass::GameData)?;
                self.simple(ScoreGateway::FullPoints)?;
                let total = full_points(self.context.kind, &self.result.state);
                self.require(ScoreMissing::Mode)?;
                self.event(
                    ScoreGateway::ModeUpdate,
                    None,
                    None,
                    Some(total),
                    Some(self.context.current_level.wrapping_add(1)),
                )?;
            }
        }
        Ok(())
    }
}

/// Unsupported contexts return Err without mutation; native gateway/null failures
/// return completed state writes and their ordered event snapshots.
pub fn replay_score(
    context: &ScoreReplayContext,
) -> Result<ScoreReplayResult, InvalidScoreReplayContext> {
    if context.rule_version != SCORE_REPLAY_NATIVE_V1
        || !context.metadata_initialized
        || !supported_mxcsr(context.native_mxcsr)
        || !host_mxcsr().is_some_and(supported_mxcsr)
        || !context.services_preserve_state_and_globals
        || !context.uses_audited_virtual_slots
        || (context.operation == ScoreOperation::GameplayUpdateScore
            && !context.gameplay_receiver_is_instance)
        || context.characters.len() > MAX_CONTRIBUTIONS
        || context.relics.len() > MAX_CONTRIBUTIONS
        || context
            .fail_at
            .is_some_and(|p| p.occurrence == 0 || p.occurrence > p.gateway.bound())
    {
        return Err(InvalidScoreReplayContext);
    }
    let mut replay = Replay {
        context,
        counts: [0; 14],
        result: ScoreReplayResult {
            state: context.initial.clone(),
            value: None,
            failure: None,
            events: Vec::new(),
            gameplay_initialized: context.gameplay_initialized,
            game_data_initialized: context.game_data_initialized,
        },
    };
    replay.result.failure = replay.run().err();
    Ok(replay.result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    fn gateway(name: &str) -> ScoreGateway {
        match name {
            "class_init" => ScoreGateway::ClassInit,
            "characters" => ScoreGateway::Characters,
            "enumerator" => ScoreGateway::Enumerator,
            "move_next" => ScoreGateway::MoveNext,
            "dispose" => ScoreGateway::Dispose,
            "unrevealed" => ScoreGateway::Unrevealed,
            "base_score" => ScoreGateway::BaseScore,
            "multiplier" => ScoreGateway::Multiplier,
            "full_points" => ScoreGateway::FullPoints,
            "transform" => ScoreGateway::Transform,
            "position" => ScoreGateway::Position,
            "floating_score" => ScoreGateway::FloatingScore,
            "mode_update" => ScoreGateway::ModeUpdate,
            "object_ctor" => ScoreGateway::ObjectCtor,
            _ => panic!("unknown native gateway {name}"),
        }
    }
    fn operation(name: &str) -> ScoreOperation {
        match name {
            "GetBaseScore" => ScoreOperation::GetBaseScore,
            "GetMultiplier" => ScoreOperation::GetMultiplier,
            "AddPointsOnEvilKill" => ScoreOperation::AddPointsOnEvilKill,
            "UpdateFullScore" => ScoreOperation::UpdateFullScore,
            "GetFullPoints" => ScoreOperation::GetFullPoints,
            ".ctor" => ScoreOperation::Constructor,
            "Gameplay.UpdateScore" => ScoreOperation::GameplayUpdateScore,
            _ => panic!(),
        }
    }
    fn contribution(value: &Value) -> Contribution {
        if value.is_null() {
            Contribution::Null
        } else if value.as_str() == Some("null_info") {
            Contribution::MissingInfo
        } else {
            Contribution::Value {
                point: value[0].as_i64().unwrap() as i32,
                multiplier_bits: value[1].as_u64().unwrap() as u32,
            }
        }
    }
    fn context(case: &Value) -> ScoreReplayContext {
        let input = &case["input"];
        let cold = |name| {
            input["cold"]
                .as_array()
                .is_some_and(|values| values.contains(&json!(name)))
        };
        ScoreReplayContext {
            rule_version: SCORE_REPLAY_NATIVE_V1.into(),
            kind: match case["type"].as_str().unwrap() {
                "Score" => ScoreKind::Base,
                "ScoreNew" => ScoreKind::New,
                "ScoreOld" => ScoreKind::Old,
                _ => panic!(),
            },
            operation: operation(case["method"].as_str().unwrap()),
            initial: serde_json::from_value(case["initial"].clone()).unwrap(),
            metadata_initialized: true,
            native_mxcsr: 0x1f80,
            services_preserve_state_and_globals: true,
            uses_audited_virtual_slots: true,
            gameplay_receiver_is_instance: true,
            gameplay_initialized: !cold("Gameplay"),
            game_data_initialized: !cold("GameData"),
            current_level: input["level"].as_i64().unwrap_or(6) as i32,
            completed_days_argument: input["day"].as_i64().unwrap_or(3) as i32,
            unrevealed_count: input["unrevealed"].as_i64().unwrap_or(2) as i32,
            characters: input["characters"]
                .as_array()
                .map(|v| v.iter().map(contribution).collect())
                .unwrap_or_default(),
            relics: input["relics"]
                .as_array()
                .map(|v| v.iter().map(contribution).collect())
                .unwrap_or_default(),
            missing: input["null"].as_str().map(|s| match s {
                "instance" => ScoreMissing::Instance,
                "characters" => ScoreMissing::Characters,
                "relics" => ScoreMissing::Relics,
                "character" => ScoreMissing::Character,
                "view" => ScoreMissing::Icon,
                "transform" => ScoreMissing::Transform,
                "vfx" => ScoreMissing::Vfx,
                "mode" => ScoreMissing::Mode,
                "score" => ScoreMissing::Score,
                _ => panic!(),
            }),
            position_bits: [0x3f800000, 0xc0000000, 0x40400000],
            fail_at: input["fail"].as_array().map(|values| ScoreFailurePoint {
                gateway: gateway(values[0].as_str().unwrap()),
                occurrence: values[1].as_u64().unwrap() as u32,
            }),
        }
    }
    fn native_event(event: &ScoreEvent) -> Value {
        let mut value = json!({"kind":event.gateway,"state":event.state});
        if let Some(source) = event.source {
            value["source"] = json!(source);
        }
        if let Some(class) = event.runtime_class {
            value["type"] = json!(match class {
                ScoreRuntimeClass::Gameplay => "Gameplay",
                ScoreRuntimeClass::GameData => "GameData",
            });
        }
        if let Some(score) = event.score {
            value[if event.gateway == ScoreGateway::FloatingScore {
                "amount"
            } else {
                "score"
            }] = json!(score);
        }
        if let Some(level) = event.level {
            value["level"] = json!(level);
        }
        if let Some(position) = event.position_bits {
            value["position_bits"] = json!(position);
        }
        value
    }
    fn sample() -> ScoreReplayContext {
        ScoreReplayContext {
            rule_version: SCORE_REPLAY_NATIVE_V1.into(),
            kind: ScoreKind::New,
            operation: ScoreOperation::AddPointsOnEvilKill,
            initial: ScoreState {
                completed_stages: 1,
                points_for_completing: 100,
                completed_days: 2,
                multiplier: 17,
                round_points: 3,
                overall_points: 4,
                killed_goods: 5,
                temp_unrevealed_cards: 6,
                unrevealed_cards: 7,
                killed_evils: 8,
                temp_killed_evils: 9,
                point_per_kill: 50,
                points_per_unrevealed: 10,
            },
            metadata_initialized: true,
            native_mxcsr: 0x1f80,
            services_preserve_state_and_globals: true,
            uses_audited_virtual_slots: true,
            gameplay_receiver_is_instance: true,
            gameplay_initialized: true,
            game_data_initialized: true,
            current_level: 6,
            completed_days_argument: 3,
            unrevealed_count: 2,
            characters: vec![Contribution::Value {
                point: 3,
                multiplier_bits: 1.0_f32.to_bits(),
            }],
            relics: vec![],
            missing: None,
            position_bits: [0; 3],
            fail_at: None,
        }
    }
    #[test]
    fn all_4703_native_score_fixtures() {
        let report: Value = serde_json::from_str(include_str!(
            "../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_score_lifecycle.json"
        ))
        .unwrap();
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 4703);
        for (index, case) in cases.iter().enumerate() {
            let ctx = context(case);
            let output = replay_score(&ctx).unwrap();
            assert_eq!(json!(output.state), case["state"], "case {index}");
            let expected_value = match output.value {
                None => Value::Null,
                Some(ScoreValue::Integer(v)) => json!(v),
                Some(ScoreValue::FloatBits(v)) => json!(v),
            };
            assert_eq!(expected_value, case["value"], "case {index}");
            let error = match output.failure {
                None => Value::Null,
                Some(ScoreFailure::NullReference) => json!("null"),
                Some(ScoreFailure::Gateway(point)) => json!(point.gateway),
            };
            assert_eq!(error, case["error"], "case {index}");
            let events = case["events"]
                .as_array()
                .unwrap()
                .iter()
                .map(|event| {
                    let mut expanded = event.clone();
                    let table_index = expanded
                        .as_object_mut()
                        .unwrap()
                        .remove("state_index")
                        .unwrap()
                        .as_u64()
                        .unwrap() as usize;
                    expanded["state"] = report["event_state_table"][table_index].clone();
                    expanded
                })
                .collect::<Vec<_>>();
            assert_eq!(
                output.events.iter().map(native_event).collect::<Vec<_>>(),
                events,
                "case {index}"
            );
            let mut initialized = (ctx.gameplay_initialized, ctx.game_data_initialized);
            for event in &output.events {
                let point = ScoreFailurePoint {
                    gateway: event.gateway,
                    occurrence: event.occurrence,
                };
                if output.failure == Some(ScoreFailure::Gateway(point)) {
                    continue;
                }
                match event.runtime_class {
                    Some(ScoreRuntimeClass::Gameplay) => initialized.0 = true,
                    Some(ScoreRuntimeClass::GameData) => initialized.1 = true,
                    None => {}
                }
            }
            assert_eq!(
                (output.gameplay_initialized, output.game_data_initialized),
                initialized,
                "case {index}"
            );
        }
    }
    #[test]
    fn invalid_contexts_are_atomic() {
        for which in 0..9 {
            let mut ctx = sample();
            match which {
                0 => ctx.rule_version = "unknown".into(),
                1 => ctx.metadata_initialized = false,
                2 => ctx.services_preserve_state_and_globals = false,
                3 => ctx.uses_audited_virtual_slots = false,
                4 => {
                    ctx.operation = ScoreOperation::GameplayUpdateScore;
                    ctx.gameplay_receiver_is_instance = false;
                }
                5 => ctx.characters = vec![Contribution::Null; MAX_CONTRIBUTIONS + 1],
                6 => {
                    ctx.fail_at = Some(ScoreFailurePoint {
                        gateway: ScoreGateway::MoveNext,
                        occurrence: 261,
                    })
                }
                7 => {
                    ctx.fail_at = Some(ScoreFailurePoint {
                        gateway: ScoreGateway::Dispose,
                        occurrence: 0,
                    })
                }
                _ => ctx.native_mxcsr = 0x9f80,
            }
            let before = ctx.clone();
            assert_eq!(replay_score(&ctx), Err(InvalidScoreReplayContext));
            assert_eq!(ctx, before);
        }
    }
    #[test]
    fn score_write_survives_presentation_failure_and_nonfinite_conversion() {
        let mut ctx = sample();
        ctx.missing = Some(ScoreMissing::Vfx);
        let out = replay_score(&ctx).unwrap();
        assert_eq!(out.state.round_points, 10);
        assert_eq!(out.failure, Some(ScoreFailure::NullReference));
        assert_eq!(out.events.last().unwrap().gateway, ScoreGateway::Position);
        ctx.characters = vec![Contribution::Value {
            point: 1,
            multiplier_bits: 0x7f800000,
        }];
        let out = replay_score(&ctx).unwrap();
        assert_eq!(out.state.round_points, 3_i32.wrapping_add(i32::MIN));
        assert_eq!(native_mul(0x7f812345, 0x7fc54321), 0x7fc12345);
        assert_eq!(native_mul(0, 0x7f800000), 0xffc00000);
    }
    #[test]
    fn maximum_inputs_and_unreached_failure_are_bounded() {
        let mut ctx = sample();
        ctx.characters = vec![
            Contribution::Value {
                point: 1,
                multiplier_bits: 1.0_f32.to_bits()
            };
            MAX_CONTRIBUTIONS
        ];
        ctx.relics = ctx.characters.clone();
        let out = replay_score(&ctx).unwrap();
        assert_eq!(out.failure, None);
        assert!(out.events.len() < 300);
        ctx.operation = ScoreOperation::GetFullPoints;
        ctx.fail_at = Some(ScoreFailurePoint {
            gateway: ScoreGateway::MoveNext,
            occurrence: 260,
        });
        let out = replay_score(&ctx).unwrap();
        assert_eq!(out.failure, None);
        assert!(out.events.is_empty());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn host_control_mismatch_rejected_and_status_flags_ignored() {
        // MXCSR is thread-local. A separate thread and Drop guard restore it even
        // if an assertion panics; other concurrently running tests are untouched.
        std::thread::spawn(|| {
            fn set(value: u32) {
                // SAFETY: only supported control/status bits from existing MXCSR
                // or the known valid fixture value are supplied.
                unsafe { core::arch::asm!("ldmxcsr [{address}]", address = in(reg) &value, options(nostack, preserves_flags, readonly)); }
            }
            struct Restore(u32);
            impl Drop for Restore { fn drop(&mut self) { set(self.0); } }
            let _restore = Restore(host_mxcsr().unwrap());
            let ctx = sample();
            for controls in [0x3f80, 0x1fc0, 0x9f80, 0x1f00] {
                set(controls);
                assert_eq!(replay_score(&ctx), Err(InvalidScoreReplayContext));
            }
            set(0x1fbf);
            assert!(replay_score(&ctx).is_ok());
        }).join().unwrap();
    }
}
