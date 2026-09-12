use super::*;
use serde_json::{json, Value};

fn context(case: &Value) -> Context {
    let input = &case["input"];
    let operation = match case["method"].as_str().unwrap() {
        "UpdateCharacterPositions" => Operation::Rotate,
        "HighlightCharacters" => Operation::Highlight,
        "DisableHighlightAll" => Operation::Disable,
        _ => panic!("unrecognized native method"),
    };
    let values = |name: &str| -> Option<Vec<Option<u16>>> {
        if input[format!("null_{name}")].as_bool() == Some(true) {
            None
        } else {
            Some(
                input[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().map(|v| v as u16))
                    .collect(),
            )
        }
    };
    let characters = (0..8)
        .map(|n| Character {
            identity: n,
            transform: (input["missing_transform"] != format!("character{n}")).then_some(100 + n),
            icon: (input["null_icon"].as_u64() != Some(n as u64)).then_some(200 + n),
            icon_transform: (input["missing_transform"] != format!("icon{n}")).then_some(300 + n),
            highlight: (input["null_highlight"].as_u64() != Some(n as u64)).then_some(400 + n),
            unity_nonnull: !input["destroyed"]
                .as_array()
                .is_some_and(|a| a.contains(&json!(format!("character{n}")))),
        })
        .collect();
    let failure = input["fail"].as_array().map(|a| FailurePoint {
        gateway: serde_json::from_value(a[0].clone()).unwrap(),
        occurrence: a[1].as_u64().unwrap() as u16,
    });
    Context {
        rule_version: CHARACTER_VISUALS_NATIVE_V1.into(),
        operation,
        stable_services: true,
        native_mxcsr: 0x1f80,
        board: values("board"),
        selected: values("selected"),
        characters,
        board_count: input["size"]
            .as_i64()
            .unwrap_or(input["board"].as_array().unwrap().len() as i64) as i32,
        allow_adversarial_count: !input["size"].is_null(),
        zero_vector_bits: input["zero_bits"]
            .as_array()
            .map(|a| {
                [
                    a[0].as_u64().unwrap() as u32,
                    a[1].as_u64().unwrap() as u32,
                    a[2].as_u64().unwrap() as u32,
                ]
            })
            .unwrap_or([0; 3]),
        object_initialized: input["cold"].as_bool() != Some(true),
        null_unity_nonnull: input["nonnull_override"].as_bool() == Some(true),
        failure,
    }
}
fn target(id: u16) -> String {
    match id {
        0..=7 => format!("character{id}"),
        100..=107 => format!("character{}", id - 100),
        200..=207 => format!("icon{}", id - 200),
        300..=307 => format!("icon{}", id - 300),
        400..=407 => format!("highlight{}", id - 400),
        _ => panic!("unknown fixture reference"),
    }
}
fn fixture_context() -> Context {
    context(
        &json!({"method":"UpdateCharacterPositions","input":{"board":[0,1,2],"selected":[2,0]}}),
    )
}

#[test]
fn replay_matches_all_96_native_visual_cases() {
    let report: Value = serde_json::from_str(include_str!("../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_characters_layout_highlight.json")).unwrap();
    assert_eq!(report["cases_passed"], 96);
    for (index, case) in report["cases"].as_array().unwrap().iter().enumerate() {
        let c = context(case);
        let original = c.clone();
        let out = replay(&c).unwrap();
        let events: Vec<Value> = out
            .events
            .iter()
            .map(|e| {
                let mut value = serde_json::to_value(e).unwrap();
                if let Some(id) = value["target"].as_u64() {
                    value["target"] = json!(target(id as u16));
                }
                value
            })
            .collect();
        let mut rotations = serde_json::Map::new();
        for (id, bits) in out.local_rotations.iter().chain(&out.world_rotations) {
            rotations.insert(target(*id), json!(bits));
        }
        let calls: Vec<Value> = out
            .highlights
            .iter()
            .map(|(g, id)| json!([g, target(*id)]))
            .collect();
        let error = match out.error {
            None => Value::Null,
            Some(Failure::Null) => json!("null"),
            Some(Failure::Gateway(f)) => json!(f.gateway),
        };
        assert_eq!(json!(events), case["events"], "event fixture {index}");
        assert_eq!(
            json!(rotations),
            case["rotations"],
            "rotation fixture {index}"
        );
        assert_eq!(
            json!(calls),
            case["highlight_calls"],
            "highlight fixture {index}"
        );
        assert_eq!(error, case["error"], "failure fixture {index}");
        assert_eq!(c, original);
    }
}

#[test]
fn strict_provenance_capacity_and_reference_validation() {
    let c = fixture_context();
    for variant in 0..7 {
        let mut changed = c.clone();
        match variant {
            0 => changed.rule_version = "unknown".into(),
            1 => changed.stable_services = false,
            2 => {
                changed.failure = Some(FailurePoint {
                    gateway: Gateway::Show,
                    occurrence: 0,
                })
            }
            3 => changed.characters.push(changed.characters[0].clone()),
            4 => changed.board.as_mut().unwrap().push(Some(999)),
            5 => changed.board_count = -1,
            _ => changed.characters[1].icon = changed.characters[0].icon,
        }
        assert_eq!(
            replay(&changed),
            Err(Unsupported::Context),
            "variant {variant}"
        );
    }
    let mut large = c.clone();
    large.selected = Some(vec![None; MAX_ENTRIES + 1]);
    assert_eq!(replay(&large), Err(Unsupported::Capacity));
    let mut fp = c;
    fp.native_mxcsr = 0x9f80;
    assert_eq!(replay(&fp), Err(Unsupported::FloatingEnvironment));
}

#[test]
fn duplicate_calls_preserve_rotation_and_failure_prefixes() {
    let mut c = fixture_context();
    c.board = Some(vec![Some(2), Some(2), Some(0)]);
    let out = replay(&c).unwrap();
    assert_eq!(out.local_rotations[&102], [0, 0, 120f32.to_bits()]);
    c.failure = Some(FailurePoint {
        gateway: Gateway::WorldEuler,
        occurrence: 2,
    });
    let out = replay(&c).unwrap();
    assert_eq!(out.local_rotations[&102], [0, 0, 120f32.to_bits()]);
    assert_eq!(out.world_rotations.len(), 1);
    assert!(!out.local_rotations.contains_key(&100));
    c.operation = Operation::Highlight;
    c.failure = None;
    c.selected = Some(vec![Some(2), Some(2), None, Some(0)]);
    c.characters[0].unity_nonnull = false;
    assert_eq!(
        replay(&c).unwrap().highlights,
        vec![(Gateway::Show, 402), (Gateway::Show, 402)]
    );
}

#[test]
fn stable_shared_components_and_transform_self_identity() {
    let mut c = fixture_context();
    c.characters[0].icon_transform = c.characters[0].icon;
    c.characters[1].icon = c.characters[0].icon;
    c.characters[1].icon_transform = c.characters[0].icon_transform;
    let out = replay(&c).unwrap();
    assert_eq!(out.world_rotations.len(), 2);
    assert_eq!(
        out.events
            .iter()
            .filter(|e| matches!(e, Event::WorldEuler { target: 200, .. }))
            .count(),
        2
    );
    c.characters[1].icon_transform = None;
    assert_eq!(replay(&c), Err(Unsupported::Context));
    c.characters[1].icon = Some(0);
    c.characters[1].icon_transform = c.characters[0].transform;
    assert!(replay(&c).is_ok());
    c.characters[1].icon_transform = Some(999);
    assert_eq!(replay(&c), Err(Unsupported::Context));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn rejects_host_control_changes_but_accepts_status_bits() {
    std::thread::spawn(|| {
        struct Restore(u32);
        fn set(value: u32) { unsafe { core::arch::asm!("ldmxcsr [{p}]", p=in(reg) &value, options(nostack, preserves_flags, readonly)); } }
        impl Drop for Restore { fn drop(&mut self) { set(self.0); } }
        let _restore = Restore(mxcsr().unwrap());
        let c = fixture_context();
        for control in [0x3f80,0x1fc0,0x9f80,0] {
            set(control); assert_eq!(replay(&c),Err(Unsupported::FloatingEnvironment));
        }
        set(0x1fbf); assert!(replay(&c).is_ok());
        assert_eq!(mxcsr().unwrap() & !0x3f,0x1f80);
    }).join().unwrap();
}
