use super::*;
use serde_json::{json, Value as Json};
const NAMES: [&str; 10] = [
    "reveal",
    "text",
    "other_text",
    "text_class",
    "other_class",
    "game",
    "formatted",
    "other_formatted",
    "method",
    "other_method",
];
fn id(name: &str) -> u64 {
    0x3_0000_0000 + 0x50000 + NAMES.iter().position(|&n| n == name).unwrap() as u64 * 0x1000
}
fn label(p: Option<u64>) -> Json {
    p.map_or(Json::Null, |p| {
        json!(NAMES.iter().find(|&&n| id(n) == p).unwrap())
    })
}
fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}
fn method(row: &Json) -> Method {
    if row["method"] == "Init" {
        Method::Init
    } else {
        Method::Hide
    }
}
fn fixture(rows: &[&Json]) -> Context {
    let initial = &rows[0]["initial"];
    let records = NAMES
        .into_iter()
        .map(|n| Record {
            identity: id(n),
            kind: match n {
                "reveal" => Kind::Owner,
                "text" | "other_text" => Kind::Text,
                "text_class" | "other_class" => Kind::Class,
                "game" => Kind::GameObject,
                "formatted" | "other_formatted" => Kind::String,
                _ => Kind::MethodInfo,
            },
            bytes: bytes(initial["memory"][n].as_str().unwrap()),
        })
        .collect::<Vec<_>>();
    let callables = records
        .iter()
        .filter(|r| r.kind == Kind::Class)
        .map(|r| word(r, 0x558))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Context {
        version: REVEAL_ORDER_PRESENTATION_NATIVE_V1.into(),
        owner: id("reveal"),
        game_object: id("game"),
        callables,
        state: State {
            records,
            caller_order_slot_bits: initial["caller_order_slot_bits"].as_u64().unwrap(),
            game_active: initial["game_active"].as_bool().unwrap(),
            text_values: ["text", "other_text"]
                .into_iter()
                .map(|n| Value {
                    identity: id(n),
                    text: initial["text_values"][n].as_str().map(str::to_owned),
                })
                .collect(),
            formatted_values: initial["formatted_values"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(n, v)| Value {
                    identity: id(n),
                    text: Some(v.as_str().unwrap().into()),
                })
                .collect(),
        },
        calls: rows
            .iter()
            .map(|r| {
                let method = method(r);
                let bits = r["options"]["order_bits"].as_u64().unwrap_or(0x80000003) as u32;
                let formatted = if method == Method::Hide || r["options"]["null_formatted"] == true
                {
                    None
                } else {
                    Some(id(if r["options"]["other_formatted"] == true {
                        "other_formatted"
                    } else {
                        "formatted"
                    }))
                };
                Call {
                    method,
                    order_register_bits: 0xabcdef1200000000 | u64::from(bits),
                    initial_order_slot_bits: r["initial"]["caller_order_slot_bits"]
                        .as_u64()
                        .unwrap(),
                    formatted,
                    formatted_text: formatted.map(|_| (bits as i32).to_string()),
                }
            })
            .collect(),
        services: Services {
            storage_verified: true,
            unity_verified_inert: true,
            formatter_verified_inert: true,
            tmp_verified_inert: true,
            normal_completion_verified: true,
            game_object_return_rdx_bits: 0xFACE123456789090,
        },
    }
}
fn snapshot(s: &State) -> Json {
    let owner = s
        .records
        .iter()
        .find(|r| r.identity == id("reveal"))
        .unwrap();
    let mut memory = serde_json::Map::new();
    for r in &s.records {
        memory.insert(
            label(Some(r.identity)).as_str().unwrap().into(),
            json!(hex(&r.bytes)),
        );
    }
    let values = |values: &[Value]| {
        let mut m = serde_json::Map::new();
        for v in values {
            m.insert(
                label(Some(v.identity)).as_str().unwrap().into(),
                json!(v.text),
            );
        }
        Json::Object(m)
    };
    json!({"text_ref":label((word(owner,0x20)!=0).then(||word(owner,0x20))),"caller_order_slot_bits":s.caller_order_slot_bits,"game_active":s.game_active,"text_values":values(&s.text_values),"formatted_values":values(&s.formatted_values),"memory":memory})
}
fn event(e: &Event) -> Json {
    let (kind, args) = match e {
        Event::ComponentGameObjectService {
            owner,
            result,
            method_bits,
        } => (
            "component_game_object_service",
            json!([label(Some(*owner)), label(Some(*result)), method_bits]),
        ),
        Event::SetActiveService {
            game,
            rdx_bits,
            value,
            method_bits,
        } => (
            "set_active_service",
            json!([label(Some(*game)), rdx_bits, value, method_bits]),
        ),
        Event::Int32ToStringService {
            slot_offset,
            bits,
            signed,
            method_bits,
            result,
        } => (
            "int32_to_string_service",
            json!([slot_offset, bits, signed, method_bits, label(*result)]),
        ),
        Event::TmpTextSetterService {
            text,
            value,
            method,
        } => (
            "tmp_text_setter_service",
            json!([label(Some(*text)), label(*value), label(Some(*method))]),
        ),
    };
    json!({"kind":kind,"args":args})
}
fn report() -> Json {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_reveal_order_presentation.json")).unwrap()
}
fn verify(rows: &[&Json]) {
    let c = fixture(rows);
    let out = replay(&c).unwrap();
    assert_eq!(out.state.records, c.state.records);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(snapshot(&out.completed[index]), row["final"]);
        let actual = out
            .steps
            .iter()
            .filter(|s| s.call == index)
            .collect::<Vec<_>>();
        let expected = row["events"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (step, e) in actual.iter().zip(expected) {
            assert_eq!(
                event(&step.event),
                json!({"kind":e["kind"],"args":e["args"]})
            );
            assert_eq!(snapshot(&step.state), e["snapshot"]);
        }
    }
}
#[test]
fn stable_native_corpus_and_retained_sequences() {
    let r = report();
    let mut n = 0;
    for row in r["normal_and_edge_cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(r["baselines"].as_array().unwrap())
    {
        if row["returned"] == true
            && row["options"].get("mutation_phase").is_none()
            && row["options"]["null_owner"] != true
        {
            verify(&[row]);
            n += 1;
        }
    }
    assert_eq!(n, 23); // One stable native Hide accepts null owner; conservative replay excludes it.
    for seq in r["retained_sequences"].as_array().unwrap() {
        verify(&seq.as_array().unwrap().iter().collect::<Vec<_>>());
    }
}
fn basic() -> Context {
    let r = report();
    fixture(&[&r["baselines"][0]])
}
fn replace(c: &mut Context, name: &str, off: usize, v: u64) {
    c.state
        .records
        .iter_mut()
        .find(|r| r.identity == id(name))
        .unwrap()
        .bytes[off..off + 8]
        .copy_from_slice(&v.to_le_bytes());
}
#[test]
fn raw_abi_slots_and_nullable_formatter_are_retained() {
    let mut c = basic();
    c.calls[0].order_register_bits = 0xdeadbeef80000003;
    c.calls[0].initial_order_slot_bits = 0xabcddcba12345678;
    c.calls[0].formatted = None;
    c.calls[0].formatted_text = None;
    let out = replay(&c).unwrap();
    assert_eq!(out.state.caller_order_slot_bits, 0xabcddcba80000003);
    assert!(matches!(
        out.steps[1].event,
        Event::SetActiveService {
            rdx_bits: 0xFACE123456789001,
            ..
        }
    ));
    assert!(matches!(
        out.steps[2].event,
        Event::Int32ToStringService {
            bits: 0x80000003,
            signed: -2147483645,
            ..
        }
    ));
    assert_eq!(out.state.text_values[0].text, None);
    assert!(out.state.formatted_values.is_empty());
    c.calls[0].method = Method::Hide;
    c.calls[0].initial_order_slot_bits = 0x123456789abcdef0;
    let out = replay(&c).unwrap();
    assert_eq!(out.state.caller_order_slot_bits, 0x123456789abcdef0);
    assert!(matches!(
        out.steps[1].event,
        Event::SetActiveService {
            rdx_bits: 0,
            value: 0,
            ..
        }
    ));
}
#[test]
fn physical_tmp_headers_and_incompatible_types_are_guarded() {
    let c = basic();
    for (name, offset, value) in [
        ("reveal", 0x20, id("game")),
        ("text", 0, id("formatted")),
        ("text_class", 0x560, id("text")),
        ("text_class", 0x558, id("formatted")),
    ] {
        let mut v = c.clone();
        replace(&mut v, name, offset, value);
        assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
    }
    let mut v = c.clone();
    replace(&mut v, "reveal", 0x20, 0);
    assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
    v.calls[0].method = Method::Hide;
    v.calls[0].formatted = None;
    v.calls[0].formatted_text = None;
    assert!(replay(&v).is_ok());
    let mut v = c.clone();
    replace(&mut v, "other_text", 0, id("text_class"));
    assert!(replay(&v).is_ok());
    let mut v = c.clone();
    v.owner = 0;
    assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
    for flag in 0..5 {
        let mut v = c.clone();
        match flag {
            0 => v.services.storage_verified = false,
            1 => v.services.unity_verified_inert = false,
            2 => v.services.formatter_verified_inert = false,
            3 => v.services.tmp_verified_inert = false,
            _ => v.services.normal_completion_verified = false,
        };
        assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
    }
    let mut v = c.clone();
    v.state.records.push(v.state.records[0].clone());
    assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
    let mut v = c;
    v.calls[0].formatted = Some(id("game"));
    assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
}
#[test]
fn future_text_and_snapshot_work_is_bounded() {
    let mut c = basic();
    c.calls[0].formatted_text = Some("x".repeat(3000));
    assert!(replay(&c).is_ok());
    c.calls = vec![c.calls[0].clone(); 5];
    assert_eq!(replay(&c).unwrap_err(), LedgerError::Capacity);
    let mut c = basic();
    c.calls[0].formatted_text = Some("x".repeat(500));
    c.calls = vec![c.calls[0].clone(); 10];
    assert_eq!(replay(&c).unwrap_err(), LedgerError::Capacity);
    let mut c = basic();
    c.calls = vec![c.calls[0].clone(); 17];
    assert_eq!(replay(&c).unwrap_err(), LedgerError::Capacity);
}
