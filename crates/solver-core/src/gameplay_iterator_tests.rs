use super::*;
use serde_json::Value;
fn report() -> Value {
    serde_json::from_str(include_str!("../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_gameplay_iterator_factories.json")).unwrap()
}
fn initial() -> GameplayIteratorState {
    GameplayIteratorState {
        state: 0,
        current: 8589967360,
        captured_receiver: None,
    }
}
fn ctx(operation: GameplayIteratorOperation) -> GameplayIteratorContext {
    GameplayIteratorContext {
        rule_version: GAMEPLAY_ITERATOR_NATIVE_V1.into(),
        metadata_initialized: true,
        services_preserve_iterator: true,
        initial: initial(),
        operation,
        allocation_id: 8590000384,
        blind_deck: 0,
        gameplay_initialized: true,
        fail_at: None,
    }
}
#[test]
fn all_64_native_runs_match_fields_events_and_failures() {
    let data = report();
    let mut previous = initial();
    let mut count = 0;
    for case in data["cases"].as_array().unwrap() {
        let entry = case["entry"].as_str().unwrap();
        let options = &case["options"];
        let kind = match entry {
            "0x37bde0" | "0x38fa90" => GameplayIteratorKind::DelayedDeckIntro,
            "0x37dea0" | "0x38fe50" => GameplayIteratorKind::InitCoroutine,
            _ => GameplayIteratorKind::SetupDelay,
        };
        let factory = matches!(entry, "0x37bde0" | "0x37dea0" | "0x380c70");
        let operation = if factory {
            GameplayIteratorOperation::Factory {
                kind,
                receiver: if options["null_receiver"] == true {
                    0
                } else {
                    8589963264
                },
            }
        } else if entry == "0x38f9c0" {
            GameplayIteratorOperation::DelayedDeckMoveNext
        } else {
            GameplayIteratorOperation::Reset { kind }
        };
        let mut c = ctx(operation);
        // Only continuation runs without an explicit state preserve prior native fields.
        c.initial = if entry == "0x38f9c0" && options["state"].is_null() {
            previous.clone()
        } else {
            GameplayIteratorState {
                state: options["state"].as_i64().unwrap_or(0) as i32,
                ..initial()
            }
        };
        c.blind_deck = options["blind"].as_i64().unwrap_or(0) as i32;
        c.gameplay_initialized = options["cold"] != true;
        c.fail_at = serde_json::from_value(options["fail"].clone()).unwrap();
        let before = c.clone();
        let r = replay_gameplay_iterator(&c).unwrap();
        assert_eq!(c, before);
        assert_eq!(
            r.iterator.state as i64,
            case["state"].as_i64().unwrap(),
            "{case}"
        );
        assert_eq!(
            r.iterator.current,
            case["current"].as_u64().unwrap(),
            "{case}"
        );
        let names: Vec<_> = r
            .events
            .iter()
            .map(|e| match e {
                IteratorEvent::Allocate => "allocate",
                IteratorEvent::WaitCtor { .. } => "wait_ctor",
                IteratorEvent::Barrier => "barrier",
                IteratorEvent::BlindDeck => "blind_deck",
                IteratorEvent::ClassInit => "class_init",
                IteratorEvent::ChangeState { .. } => "change_state",
                IteratorEvent::ExceptionCtor => "exception_ctor",
                IteratorEvent::Throw { .. } => "throw",
            })
            .collect();
        let native: Vec<_> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["event"].as_str().unwrap())
            .collect();
        assert_eq!(names, native, "{case}");
        let error = match r.failure {
            None => Value::Null,
            Some(IteratorFailure::NotSupported) => "not_supported".into(),
            Some(IteratorFailure::Gateway(g)) => serde_json::to_value(g).unwrap(),
        };
        assert_eq!(error, case["error"], "{case}");
        if let Some(b) = r.move_next_result {
            assert_eq!(b, case["result_bool"].as_bool().unwrap());
        }
        if let Some(created) = &r.created_iterator {
            assert_eq!(created.state, 0);
            assert_eq!(created.current, 0);
            if let GameplayIteratorOperation::Factory { kind, receiver } = operation {
                assert_eq!(
                    created.captured_receiver,
                    (kind == GameplayIteratorKind::SetupDelay).then_some(receiver)
                );
            }
        }
        if r.move_next_result == Some(true) {
            assert_eq!(r.yielded_wait_f32_bits, Some(0x3f800000));
        }
        previous = r.iterator;
        count += 1;
    }
    assert_eq!(count, 64);
}
#[test]
fn rejects_unsupported_guards_and_aliases_atomically() {
    let original = ctx(GameplayIteratorOperation::DelayedDeckMoveNext);
    for field in 0..6 {
        let mut c = original.clone();
        match field {
            0 => c.rule_version = "future".into(),
            1 => c.metadata_initialized = false,
            2 => c.services_preserve_iterator = false,
            3 => c.allocation_id = 0,
            4 => c.allocation_id = c.initial.current,
            _ => c.initial.captured_receiver = Some(c.allocation_id),
        }
        let before = c.clone();
        assert_eq!(replay_gameplay_iterator(&c), Err(InvalidIteratorContext));
        assert_eq!(c, before);
    }
    let mut value = serde_json::to_value(original).unwrap();
    value["scheduler_time"] = 99.into();
    assert!(serde_json::from_value::<GameplayIteratorContext>(value).is_err());
}
#[test]
fn resumption_failure_preserves_current_and_request_is_attempted() {
    let mut c = ctx(GameplayIteratorOperation::DelayedDeckMoveNext);
    c.initial.state = 1;
    c.blind_deck = 2;
    c.fail_at = Some(IteratorGateway::ChangeState);
    c.allocation_id = 0;
    let r = replay_gameplay_iterator(&c).unwrap();
    assert_eq!(r.iterator.state, -1);
    assert_eq!(r.iterator.current, c.initial.current);
    assert_eq!(r.requested_gameplay_state, Some(8));
    assert_eq!(r.move_next_result, None);
    assert_eq!(r.yielded_wait_f32_bits, None);
    c.blind_deck = 1;
    let r = replay_gameplay_iterator(&c).unwrap();
    assert_eq!(r.failure, None);
    assert_eq!(r.move_next_result, Some(false));
    assert_eq!(r.requested_gameplay_state, None);
}
