use super::*;
use serde_json::Value as Json;
use std::sync::OnceLock;
const NAMES: [&str; 10] = [
    "hint",
    "other_hint",
    "text",
    "title",
    "hints",
    "flavor",
    "image",
    "replacement",
    "color",
    "other_color",
];
fn report() -> &'static Json {
    static R: OnceLock<Json> = OnceLock::new();
    R.get_or_init(||serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_hint_info_constructor.json")).unwrap())
}
fn snapshot(s: &Json) -> &Json {
    if let Some(h) = s["snapshot_sha256"].as_str() {
        &report()["snapshot_blobs"][h]
    } else {
        s
    }
}
fn id(n: &str) -> u64 {
    0x3_0000_0000 + 0xB0000 + NAMES.iter().position(|&v| v == n).unwrap() as u64 * 0x1000
}
fn field(n: &str) -> Field {
    match n {
        "text" => Field::Text,
        "title" => Field::Title,
        "image" => Field::Image,
        "hints" => Field::Hints,
        "flavor" => Field::Flavor,
        _ => panic!("unexpected field"),
    }
}
fn barrier(v: &Json) -> Barrier {
    Barrier {
        owner: id(v["receiver"].as_str().unwrap()),
        field: field(v["field"].as_str().unwrap()),
        value_bits: v["value_bits"].as_u64().unwrap(),
    }
}
fn arguments(s: &Json) -> Arguments {
    Arguments {
        owner: id("hint"),
        text: s["register_argument_bits"]["text"].as_u64().unwrap(),
        image: s["register_argument_bits"]["image"].as_u64().unwrap(),
        hints: s["register_argument_bits"]["hints"].as_u64().unwrap(),
        flavor: s["stack_argument_bits"]["flavor"].as_u64().unwrap(),
        title: s["stack_argument_bits"]["title"].as_u64().unwrap(),
        color: s["stack_argument_bits"]["color"].as_u64().unwrap(),
        method_info: s["stack_argument_bits"]["method_info"].as_u64().unwrap(),
    }
}
fn parse(s: &Json) -> State {
    let s = snapshot(s);
    let state = State {
        records: NAMES
            .into_iter()
            .map(|n| Record {
                identity: id(n),
                kind: match n {
                    "hint" | "other_hint" => Kind::Hint,
                    "image" => Kind::Sprite,
                    "color" | "other_color" => Kind::Color,
                    _ => Kind::String,
                },
                bytes: s["memory"][n]
                    .as_str()
                    .unwrap()
                    .as_bytes()
                    .chunks_exact(2)
                    .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
                    .collect(),
            })
            .collect(),
        arguments: arguments(s),
        completed_barriers: s["completed_barriers"]
            .as_array()
            .unwrap()
            .iter()
            .map(barrier)
            .collect(),
        native_entries: s["native_entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| match v.as_str().unwrap() {
                "HintInfo.ctor" => NativeEntry::HintConstructor,
                "folded_Object_ret0" => NativeEntry::FoldedObjectReturn,
                _ => panic!("unexpected native entry"),
            })
            .collect(),
    };
    let hint = record(&state, id("hint"));
    for name in ["text", "title", "image", "hints", "flavor"] {
        let off = field(name).offset();
        assert_eq!(
            u64::from_le_bytes(hint.bytes[off..off + 8].try_into().unwrap()),
            s["hint_field_bits"][name].as_u64().unwrap()
        );
    }
    for i in 0..4 {
        assert_eq!(
            u32::from_le_bytes(hint.bytes[0x38 + i * 4..0x3C + i * 4].try_into().unwrap()) as u64,
            s["hint_color_bits"][i].as_u64().unwrap()
        );
    }
    state
}
fn context(r: &Json) -> Context {
    let initial = parse(&r["initial"]);
    Context {
        version: HINT_INFO_CONSTRUCTOR_NATIVE_V1.into(),
        calls: vec![initial.arguments.clone()],
        state: initial,
        storage_verified: true,
        inert_barriers_verified: true,
        normal_completion_verified: true,
        barrier_volatile_r8_bits: 0xFACE123456789002,
        barrier_volatile_r9_bits: 0xFACE123456789003,
    }
}
fn check(r: &Json) {
    let c = context(r);
    let before = c.clone();
    let actual = replay(&c).unwrap();
    assert_eq!(before, c);
    let expected = r["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| Step {
            barrier: barrier(&e["args"]),
            raw_args: std::array::from_fn(|i| e["raw_args"][i].as_u64().unwrap()),
            caller_return_rva: u32::from_str_radix(
                e["caller_return_rva"]
                    .as_str()
                    .unwrap()
                    .strip_prefix("0x")
                    .unwrap(),
                16,
            )
            .unwrap(),
            state: parse(&e["snapshot"]),
        })
        .collect::<Vec<_>>();
    assert_eq!(actual.steps, expected);
    assert_eq!(actual.final_state, parse(&r["final"]));
}
fn supported(r: &Json) -> bool {
    r["returned"] == true && r["options"]["mutations"].is_null()
}
#[test]
fn matches_all_supported_native_masks_colors_aliases_and_baselines() {
    let mut n = 0;
    for r in report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report()["baselines"].as_array().unwrap())
    {
        if supported(r) {
            check(r);
            n += 1;
        }
    }
    assert_eq!(n, 258);
}
#[test]
fn preserves_native_prior_storage_and_history_in_retained_inert_calls() {
    let mut n = 0;
    for seq in report()["sequences"].as_array().unwrap() {
        for r in seq.as_array().unwrap() {
            if supported(r) {
                check(r);
                n += 1;
            }
        }
    }
    assert_eq!(n, 4);
}
fn normal() -> Context {
    context(&report()["cases"][0])
}
#[test]
fn rejects_invalid_storage_reference_kinds_arguments_and_provenance_atomically() {
    let c = normal();
    let mut cases = Vec::new();
    let mut b = c.clone();
    b.version = "unknown".into();
    cases.push(b);
    for i in 0..3 {
        let mut b = c.clone();
        match i {
            0 => b.storage_verified = false,
            1 => b.inert_barriers_verified = false,
            _ => b.normal_completion_verified = false,
        };
        cases.push(b);
    }
    let mut b = c.clone();
    b.state.records[0].bytes.pop();
    cases.push(b);
    let mut b = c.clone();
    b.state.records.push(b.state.records[0].clone());
    cases.push(b);
    let mut b = c.clone();
    b.calls[0].owner = 0;
    cases.push(b);
    let mut b = c.clone();
    b.calls[0].image = id("text");
    cases.push(b);
    let mut b = c.clone();
    b.calls[0].color = 0;
    cases.push(b);
    let mut b = c.clone();
    b.calls[0].method_info ^= 1;
    cases.push(b);
    let mut b = c.clone();
    b.state.records[1].identity = id("hint") + 8;
    cases.push(b);
    let mut b = c.clone();
    b.state.records[1].identity = u64::MAX - 64;
    cases.push(b);
    let mut b = c.clone();
    b.state.completed_barriers.push(Barrier {
        owner: id("hint"),
        field: Field::Image,
        value_bits: id("text"),
    });
    cases.push(b);
    for b in cases {
        let before = b.clone();
        assert_eq!(replay(&b), Err(LedgerError::InvalidContext));
        assert_eq!(b, before);
    }
}
#[test]
fn reserves_complete_future_history_and_state_clone_work() {
    let mut c = normal();
    c.calls = vec![c.calls[0].clone(); 17];
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = normal();
    c.calls = vec![c.calls[0].clone(); 16];
    assert!(replay(&c).is_ok());
    for i in 0..20 {
        c.state.records.push(Record {
            identity: 0x8_0000_0000 + i * 0x1000,
            kind: Kind::String,
            bytes: vec![0xA5; 128],
        });
    }
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
}
#[test]
fn repeats_constructor_with_new_arguments_and_independent_volatile_service_bits() {
    let mut c = normal();
    c.barrier_volatile_r8_bits = 0xABCD000000000080;
    c.barrier_volatile_r9_bits = 0x12340000000000FF;
    let mut a = c.calls[0].clone();
    a.text = 0;
    a.title = id("replacement");
    a.flavor = 0;
    a.color = id("other_color");
    a.method_info = 0;
    c.calls.push(a.clone());
    let r = replay(&c).unwrap();
    assert_eq!(r.steps.len(), 10);
    assert_eq!(r.final_state.completed_barriers.len(), 10);
    assert_eq!(r.final_state.native_entries.len(), 4);
    assert_eq!(r.steps[5].raw_args[2..], [a.image, a.hints]);
    assert_eq!(
        r.steps[6].raw_args[2..],
        [c.barrier_volatile_r8_bits, c.barrier_volatile_r9_bits]
    );
    assert_eq!(
        record(&r.final_state, id("hint")).bytes[0x38..0x48],
        record(&c.state, id("other_color")).bytes[..16]
    );
    assert_eq!(
        record(&r.final_state, id("other_hint")),
        record(&c.state, id("other_hint"))
    );
    assert_eq!(r.final_state.arguments, a);
    // Context preserves MethodInfo bits without consuming their value.
    let mut same = c.clone();
    same.state.arguments.method_info = 0;
    same.calls[0].method_info = 0;
    let r2 = replay(&same).unwrap();
    assert_eq!(r.final_state.records, r2.final_state.records);
}
