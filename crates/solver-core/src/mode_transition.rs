//! Bounded, opt-in replay of initialized native GameData.ChangeGameMode.
//! Preferences, JSON identity and stable external services are explicit inputs.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MODE_TRANSITION_NATIVE_V1: &str = "mode_transition_native_v1";
pub const MAX_OBJECTS: usize = 16;
pub const MAX_HANDLERS_PER_EVENT: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeKind {
    Standard,
    RoguelikeStandard,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeObject {
    pub kind: ModeKind,
    /// Native +0x14: Standard currentLevel or RoguelikeStandard currentAscension.
    pub level_or_ascension: i32,
    pub round_score: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSlot {
    Kill,
    Won,
    Died,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Handler {
    pub target: u64,
    /// Together with slot, identifies the exact audited callback (no overrides).
    pub mode: ModeKind,
    pub slot: EventSlot,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventLists {
    pub kill: Vec<Handler>,
    pub won: Vec<Handler>,
    pub died: Vec<Handler>,
}
impl EventLists {
    fn list_mut(&mut self, slot: EventSlot) -> &mut Vec<Handler> {
        match slot {
            EventSlot::Kill => &mut self.kill,
            EventSlot::Won => &mut self.won,
            EventSlot::Died => &mut self.died,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionState {
    pub objects: BTreeMap<u64, ModeObject>,
    pub current_mode: Option<u64>,
    pub current_village: i32,
    pub events: EventLists,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Failure {
    Json,
    Delegate(u8),
    NullOrCast,
    ModeChanged,
    GameInit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeTransitionContext {
    pub rule_version: String,
    pub initial: TransitionState,
    pub input: Option<u64>,
    /// Explicit identity returned by JSON, including a null result.
    pub json_loaded: Option<u64>,
    pub metadata_initialized: bool,
    pub runtime_classes_initialized: bool,
    /// Both preference reads in the saved-mode getter return nonempty strings.
    pub preferences_nonempty: bool,
    /// JSON, delegate gateways and notifications preserve modeled state except
    /// the operations explicitly replayed here; no callback bodies run.
    pub services_preserve_state: bool,
    pub uses_audited_lifecycle_and_callbacks: bool,
    pub notify_mode_changed: bool,
    pub notify_game_init: bool,
    /// Delegate occurrences are one-based across teardown and initialization.
    pub fail_at: Option<Failure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionEventKind {
    Load { target: u64, mode: ModeKind },
    Json,
    Deinit { target: u64, mode: ModeKind },
    Init { target: u64, mode: ModeKind },
    Store(EventSlot),
    Publish,
    ModeChanged,
    GameInit,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransitionEvent {
    pub kind: TransitionEventKind,
    pub current_mode: Option<u64>,
    pub current_village: i32,
    pub events: EventLists,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModeTransitionResult {
    pub state: TransitionState,
    pub trace: Vec<TransitionEvent>,
    pub failure: Option<Failure>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidModeTransitionContext;

impl ModeTransitionResult {
    fn emit(&mut self, kind: TransitionEventKind) {
        self.trace.push(TransitionEvent {
            kind,
            current_mode: self.state.current_mode,
            current_village: self.state.current_village,
            events: self.state.events.clone(),
        });
    }
    fn delegate(
        &mut self,
        target: u64,
        slot: EventSlot,
        append: bool,
        count: &mut u8,
        fail_at: Option<Failure>,
    ) -> Result<bool, InvalidModeTransitionContext> {
        *count += 1;
        if fail_at == Some(Failure::Delegate(*count)) {
            self.failure = fail_at;
            return Ok(false);
        }
        let handler = Handler {
            target,
            mode: self.state.objects[&target].kind,
            slot,
        };
        let list = self.state.events.list_mut(slot);
        if append {
            if list.len() == MAX_HANDLERS_PER_EVENT {
                return Err(InvalidModeTransitionContext);
            }
            list.push(handler);
        } else if let Some(index) = list.iter().rposition(|entry| *entry == handler) {
            list.remove(index);
        }
        self.emit(TransitionEventKind::Store(slot));
        Ok(true)
    }
}

/// Unsupported contracts/capacities return Err atomically; native failures return
/// the exact completed write prefix. Input state is never modified.
pub fn replay_mode_transition(
    context: &ModeTransitionContext,
) -> Result<ModeTransitionResult, InvalidModeTransitionContext> {
    let initial = &context.initial;
    let known = |id: Option<u64>| id.is_none_or(|id| id != 0 && initial.objects.contains_key(&id));
    if context.rule_version != MODE_TRANSITION_NATIVE_V1
        || !context.metadata_initialized
        || !context.runtime_classes_initialized
        || !context.preferences_nonempty
        || !context.services_preserve_state
        || !context.uses_audited_lifecycle_and_callbacks
        || initial.objects.len() > MAX_OBJECTS
        || initial.objects.contains_key(&0)
        || !known(initial.current_mode)
        || !known(context.input)
        || !known(context.json_loaded)
        || matches!(
            context.fail_at,
            Some(Failure::NullOrCast | Failure::Delegate(0 | 7..=255))
        )
    {
        return Err(InvalidModeTransitionContext);
    }
    for (slot, list) in [
        (EventSlot::Kill, &initial.events.kill),
        (EventSlot::Won, &initial.events.won),
        (EventSlot::Died, &initial.events.died),
    ] {
        if list.len() > MAX_HANDLERS_PER_EVENT
            || list.iter().any(|h| {
                h.slot != slot
                    || initial
                        .objects
                        .get(&h.target)
                        .is_none_or(|o| o.kind != h.mode)
            })
        {
            return Err(InvalidModeTransitionContext);
        }
    }
    let mut out = ModeTransitionResult {
        state: initial.clone(),
        trace: Vec::new(),
        failure: None,
    };
    let Some(input) = context.input else {
        out.failure = Some(Failure::NullOrCast);
        return Ok(out);
    };
    out.emit(TransitionEventKind::Load {
        target: input,
        mode: initial.objects[&input].kind,
    });
    out.emit(TransitionEventKind::Json);
    if context.fail_at == Some(Failure::Json) {
        out.failure = context.fail_at;
        return Ok(out);
    }
    let mut count = 0;
    if let Some(old) = initial.current_mode {
        let old_kind = initial.objects[&old].kind;
        out.emit(TransitionEventKind::Deinit {
            target: old,
            mode: old_kind,
        });
        for slot in [EventSlot::Kill, EventSlot::Won, EventSlot::Died] {
            if !out.delegate(
                old,
                slot,
                old_kind == ModeKind::RoguelikeStandard && slot == EventSlot::Kill,
                &mut count,
                context.fail_at,
            )? {
                return Ok(out);
            }
        }
        let Some(loaded) = context.json_loaded else {
            out.failure = Some(Failure::NullOrCast);
            return Ok(out);
        };
        let kind = initial.objects[&loaded].kind;
        out.emit(TransitionEventKind::Init {
            target: loaded,
            mode: kind,
        });
        let order = if kind == ModeKind::Standard {
            [EventSlot::Won, EventSlot::Died, EventSlot::Kill]
        } else {
            [EventSlot::Kill, EventSlot::Won, EventSlot::Died]
        };
        for slot in order {
            if !out.delegate(loaded, slot, true, &mut count, context.fail_at)? {
                return Ok(out);
            }
        }
        if kind == ModeKind::Standard {
            let object = out.state.objects.get_mut(&loaded).unwrap();
            object.round_score = 0;
            out.state.current_village = object.level_or_ascension;
        }
    }
    out.state.current_mode = context.json_loaded;
    out.emit(TransitionEventKind::Publish);
    for (present, kind, failure) in [
        (
            context.notify_mode_changed,
            TransitionEventKind::ModeChanged,
            Failure::ModeChanged,
        ),
        (
            context.notify_game_init,
            TransitionEventKind::GameInit,
            Failure::GameInit,
        ),
    ] {
        if present {
            out.emit(kind);
            if context.fail_at == Some(failure) {
                out.failure = Some(failure);
                return Ok(out);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn mode(value: &str) -> ModeKind {
        match value {
            "StandardMode" => ModeKind::Standard,
            "RoguelikeStandard" => ModeKind::RoguelikeStandard,
            _ => panic!("unknown mode"),
        }
    }
    fn label(id: Option<u64>) -> &'static str {
        match id {
            None => "null",
            Some(1) => "old",
            Some(2) => "input",
            Some(3) => "loaded",
            _ => panic!("unknown fixture id"),
        }
    }
    fn native_token(handler: &Handler) -> u64 {
        // Exact synthetic method identities in the pinned composition harness.
        match (handler.mode, handler.slot) {
            (ModeKind::Standard, EventSlot::Kill) => 8589951488,
            (ModeKind::Standard, EventSlot::Won) => 8589975552,
            (ModeKind::Standard, EventSlot::Died) => 8589951744,
            (ModeKind::RoguelikeStandard, EventSlot::Kill) => 8589950976,
            (ModeKind::RoguelikeStandard, EventSlot::Won) => 8589975808,
            (ModeKind::RoguelikeStandard, EventSlot::Died) => 8589951232,
        }
    }
    fn lists(lists: &EventLists) -> Value {
        let convert = |list: &[Handler]| {
            list.iter()
                .map(|h| json!([label(Some(h.target)), native_token(h)]))
                .collect::<Vec<_>>()
        };
        json!({"kill":convert(&lists.kill),"won":convert(&lists.won),"died":convert(&lists.died)})
    }
    fn snapshot(global: Option<u64>, village: i32, events: &EventLists) -> Value {
        json!({"global":label(global),"village":village,"events":lists(events)})
    }
    fn trace(event: &TransitionEvent) -> Value {
        let mut value = snapshot(event.current_mode, event.current_village, &event.events);
        let kind = match event.kind {
            TransitionEventKind::Load { target, mode }
            | TransitionEventKind::Deinit { target, mode }
            | TransitionEventKind::Init { target, mode } => {
                value["target"] = json!(label(Some(target)));
                value["mode"] = json!(if mode == ModeKind::Standard {
                    "StandardMode"
                } else {
                    "RoguelikeStandard"
                });
                match event.kind {
                    TransitionEventKind::Load { .. } => "load",
                    TransitionEventKind::Deinit { .. } => "deinit",
                    _ => "init",
                }
            }
            TransitionEventKind::Json => "json",
            TransitionEventKind::Store(slot) => {
                value["slot"] = json!(match slot {
                    EventSlot::Kill => "kill",
                    EventSlot::Won => "won",
                    EventSlot::Died => "died",
                });
                "store"
            }
            TransitionEventKind::Publish => {
                value["slot"] = json!("mode");
                "store"
            }
            TransitionEventKind::ModeChanged => "mode_changed",
            TransitionEventKind::GameInit => "game_init",
        };
        value["kind"] = json!(kind);
        value
    }
    fn context(old: ModeKind, requested: ModeKind, relation: &str) -> ModeTransitionContext {
        let objects = [(1, old), (2, requested), (3, requested)]
            .into_iter()
            .map(|(id, kind)| {
                (
                    id,
                    ModeObject {
                        kind,
                        level_or_ascension: 7,
                        round_score: 123,
                    },
                )
            })
            .collect();
        let mut events = EventLists::default();
        if relation != "old_null" {
            for slot in [EventSlot::Kill, EventSlot::Won, EventSlot::Died] {
                events.list_mut(slot).push(Handler {
                    target: 1,
                    mode: old,
                    slot,
                });
            }
        }
        ModeTransitionContext {
            rule_version: MODE_TRANSITION_NATIVE_V1.into(),
            initial: TransitionState {
                objects,
                current_mode: if relation == "old_null" {
                    None
                } else {
                    Some(1)
                },
                current_village: 99,
                events,
            },
            input: Some(2),
            json_loaded: match relation {
                "same" => Some(1),
                "loaded_null" => None,
                _ => Some(3),
            },
            metadata_initialized: true,
            runtime_classes_initialized: true,
            preferences_nonempty: true,
            services_preserve_state: true,
            uses_audited_lifecycle_and_callbacks: true,
            notify_mode_changed: true,
            notify_game_init: true,
            fail_at: None,
        }
    }
    #[test]
    fn all_161_native_composition_cases() {
        let report: Value = serde_json::from_str(include_str!("../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_mode_transition_composition.json")).unwrap();
        let cases = report["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 161);
        for case in &cases[..160] {
            let mut ctx = context(
                mode(case["old_type"].as_str().unwrap()),
                mode(case["requested_type"].as_str().unwrap()),
                case["identity"].as_str().unwrap(),
            );
            ctx.fail_at = match &case["failure"] {
                Value::Null => None,
                Value::Number(n) => Some(Failure::Delegate(n.as_u64().unwrap() as u8)),
                Value::String(s) => Some(match s.as_str() {
                    "json" => Failure::Json,
                    "mode_changed" => Failure::ModeChanged,
                    "game_init" => Failure::GameInit,
                    _ => panic!(),
                }),
                _ => panic!(),
            };
            let out = replay_mode_transition(&ctx).unwrap();
            let error = out.failure.map(|f| match f {
                Failure::Json => "json",
                Failure::Delegate(_) => "delegate",
                Failure::NullOrCast => "null_or_cast",
                Failure::ModeChanged => "mode_changed",
                Failure::GameInit => "game_init",
            });
            assert_eq!(json!(error), case["error"], "{case}");
            assert_eq!(
                json!(out.trace.iter().map(trace).collect::<Vec<_>>()),
                case["trace"],
                "{case}"
            );
            assert_eq!(
                snapshot(
                    out.state.current_mode,
                    out.state.current_village,
                    &out.state.events
                ),
                case["final"],
                "{case}"
            );
            let initialized_standard = out
                .trace
                .iter()
                .any(|e| e.kind == TransitionEventKind::Publish)
                && ctx.initial.current_mode.is_some()
                && ctx
                    .json_loaded
                    .is_some_and(|id| ctx.initial.objects[&id].kind == ModeKind::Standard);
            let mut expected_objects = ctx.initial.objects.clone();
            if initialized_standard {
                expected_objects
                    .get_mut(&ctx.json_loaded.unwrap())
                    .unwrap()
                    .round_score = 0;
            }
            assert_eq!(out.state.objects, expected_objects);
        }
        let mut ctx = context(ModeKind::RoguelikeStandard, ModeKind::Standard, "distinct");
        ctx.input = None;
        let out = replay_mode_transition(&ctx).unwrap();
        assert_eq!(out.state, ctx.initial);
        assert!(out.trace.is_empty());
        assert_eq!(out.failure, Some(Failure::NullOrCast));
    }
    #[test]
    fn optional_notifications_and_last_matching_removal() {
        let mut ctx = context(ModeKind::Standard, ModeKind::Standard, "distinct");
        ctx.notify_mode_changed = false;
        ctx.notify_game_init = false;
        ctx.fail_at = Some(Failure::ModeChanged);
        let old = ctx.initial.events.kill[0];
        let other = Handler { target: 2, ..old };
        ctx.initial.events.kill = vec![old, other, old, other];
        let out = replay_mode_transition(&ctx).unwrap();
        assert_eq!(out.failure, None);
        assert_eq!(&out.state.events.kill[..3], &[old, other, other]);
        assert_eq!(out.trace.last().unwrap().kind, TransitionEventKind::Publish);
        ctx.notify_game_init = true;
        ctx.fail_at = Some(Failure::GameInit);
        let out = replay_mode_transition(&ctx).unwrap();
        assert_eq!(out.failure, Some(Failure::GameInit));
        assert!(!out
            .trace
            .iter()
            .any(|e| e.kind == TransitionEventKind::ModeChanged));
    }
    #[test]
    fn unsupported_contracts_and_capacity_are_atomic() {
        let base = context(ModeKind::RoguelikeStandard, ModeKind::Standard, "distinct");
        for which in 0..10 {
            let mut ctx = base.clone();
            match which {
                0 => ctx.metadata_initialized = false,
                1 => ctx.runtime_classes_initialized = false,
                2 => ctx.preferences_nonempty = false,
                3 => ctx.services_preserve_state = false,
                4 => ctx.uses_audited_lifecycle_and_callbacks = false,
                5 => ctx.rule_version = "unknown".into(),
                6 => ctx.json_loaded = Some(99),
                7 => ctx.fail_at = Some(Failure::Delegate(7)),
                8 => ctx.initial.events.kill[0].mode = ModeKind::Standard,
                _ => {
                    ctx.initial.events.kill =
                        vec![ctx.initial.events.kill[0]; MAX_HANDLERS_PER_EVENT]
                }
            }
            let before = ctx.clone();
            assert_eq!(
                replay_mode_transition(&ctx),
                Err(InvalidModeTransitionContext)
            );
            assert_eq!(ctx, before);
        }
        let mut ctx = base;
        for id in 4..=17 {
            ctx.initial
                .objects
                .insert(id, ctx.initial.objects[&1].clone());
        }
        assert_eq!(
            replay_mode_transition(&ctx),
            Err(InvalidModeTransitionContext)
        );
    }
}
