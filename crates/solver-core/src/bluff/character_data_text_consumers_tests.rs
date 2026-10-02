use super::*;
use serde_json::{json, Value as Json};
use std::sync::OnceLock;
const NAMES: [&str; 21] = [
    "data",
    "other_data",
    "data_class",
    "array",
    "other_array",
    "array_class",
    "flavor",
    "hints",
    "if_lies",
    "element0",
    "element1",
    "element2",
    "translation",
    "other_translation",
    "locale",
    "other_locale",
    "translated",
    "converted",
    "object_name",
    "other_name",
    "replacement",
];
fn report() -> &'static Json {
    static R: OnceLock<Json> = OnceLock::new();
    R.get_or_init(||serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_text_consumers.json")).unwrap())
}
fn id(n: &str) -> Identity {
    0x3_0000_0000 + 0x180000 + NAMES.iter().position(|&v| v == n).unwrap() as u64 * 0x1000
}
fn reference(v: &Json) -> Identity {
    v.as_str().map_or(0, id)
}
fn method(v: &Json) -> Method {
    serde_json::from_value(v.clone()).unwrap()
}
fn args(v: &Json) -> [u64; 4] {
    std::array::from_fn(|i| v[i].as_u64().unwrap())
}
fn snap(v: &Json) -> &Json {
    v["snapshot_sha256"]
        .as_str()
        .map_or(v, |h| &report()["snapshot_blobs"][h])
}
fn event(kind: &str, a: &Json) -> Completed {
    match kind {
        "UnityEngine.Random$$Range" => {
            assert_eq!(a[0], 0);
            assert_eq!(a[2], 0);
            Completed::Random {
                maximum: a[1].as_u64().unwrap() as u32,
                result_bits: a[3].as_u64().unwrap(),
            }
        }
        "CharacterLoc$$GetTranslatedName" | "CharacterLoc$$GetIWasTranslated" => {
            assert_eq!(a[2], 0);
            Completed::Translation {
                method: if kind.ends_with("GetTranslatedName") {
                    Method::GetTranslatedName
                } else {
                    Method::GetIWasTranslated
                },
                receiver: reference(&a[0]),
                locale: reference(&a[1]),
                result: reference(&a[3]),
            }
        }
        "UnityEngine.Object$$get_name" => {
            assert_eq!(a[1], 0);
            Completed::Name {
                owner: reference(&a[0]),
                result: reference(&a[2]),
            }
        }
        "StringHelper$$ConvertTextToTextWithTooltips" => {
            assert_eq!(a[1], 0);
            Completed::Converter {
                input: reference(&a[0]),
                result: reference(&a[2]),
            }
        }
        "reference_barrier" => {
            assert_eq!(a[1], 0x28);
            Completed::Barrier {
                owner: reference(&a[0]),
                value: reference(&a[2]),
            }
        }
        _ => panic!("unexpected {kind}"),
    }
}
fn parse(v: &Json) -> State {
    let v = snap(v);
    State {
        records: NAMES
            .into_iter()
            .map(|n| {
                let m = &v["memory"][n];
                let h = m.as_str().unwrap_or_else(|| {
                    report()["memory_blobs"][m["memory_sha256"].as_str().unwrap()]
                        .as_str()
                        .unwrap()
                });
                Record {
                    identity: id(n),
                    kind: match n {
                        "data" | "other_data" => Kind::Data,
                        "array" | "other_array" => Kind::Array,
                        "data_class" | "array_class" => Kind::Class,
                        "translation" | "other_translation" => Kind::Translation,
                        _ => Kind::String,
                    },
                    bytes: h
                        .as_bytes()
                        .chunks_exact(2)
                        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
                        .collect(),
                }
            })
            .collect(),
        native_entries: v["native_entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| Entry {
                method: method(&e["method"]),
                raw_args: args(&e["raw_args"]),
            })
            .collect(),
        service_history: v["service_history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| event(e["kind"].as_str().unwrap(), &e["args"]))
            .collect(),
    }
}
fn context(rows: &[&Json]) -> Context {
    Context {
        version: CHARACTER_DATA_TEXT_NATIVE_V1.into(),
        state: parse(&rows[0]["initial"]),
        calls: rows
            .iter()
            .map(|r| {
                let mut call = Call {
                    method: method(&r["method"]),
                    entry_raw_args: args(&r["entry_raw_args"]),
                    random_result_bits: 0,
                    translation_result: 0,
                    name_result: 0,
                    converter_result: 0,
                };
                for e in r["events"].as_array().unwrap() {
                    match event(e["kind"].as_str().unwrap(), &e["args"]) {
                        Completed::Random { result_bits, .. } => {
                            call.random_result_bits = result_bits
                        }
                        Completed::Translation { result, .. } => call.translation_result = result,
                        Completed::Name { result, .. } => call.name_result = result,
                        Completed::Converter { result, .. } => call.converter_result = result,
                        _ => {}
                    }
                }
                call
            })
            .collect(),
        native_base: 0x180000000,
        caller_return_bits: 0x500000000,
        volatile_return_bits: [0xFACE123456789090; 4],
        storage_verified: true,
        inert_services_verified: true,
        normal_completion_verified: true,
    }
}
fn eligible(r: &Json) -> bool {
    r["returned"] == true
        && r["options"]["mutation_phase"].is_null()
        && r["options"]["failure"].is_null()
}
fn matches(rows: &[&Json]) {
    let c = context(rows);
    let before = c.clone();
    let got = replay(&c).unwrap();
    assert_eq!(c, before);
    let mut steps = Vec::new();
    let mut states = Vec::new();
    let mut returns = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        assert!(eligible(r));
        for e in r["events"].as_array().unwrap() {
            steps.push(Step {
                call: i,
                event: event(e["kind"].as_str().unwrap(), &e["args"]),
                raw_args: args(&e["raw_args"]),
                native_site_rva: u32::from_str_radix(
                    e["native_site"].as_str().unwrap().trim_start_matches("0x"),
                    16,
                )
                .unwrap(),
                caller_return_bits: e["caller_return_bits"].as_u64().unwrap(),
                state: parse(&e["snapshot"]),
            });
        }
        states.push(parse(&r["final"]));
        returns.push(r["result_bits"].as_u64().unwrap());
    }
    assert_eq!(got.steps.len(), steps.len());
    for (actual, expected) in got.steps.iter().zip(&steps) {
        assert_eq!(actual.call, expected.call);
        assert_eq!(actual.event, expected.event);
        assert_eq!(actual.raw_args, expected.raw_args);
        assert_eq!(actual.native_site_rva, expected.native_site_rva);
        assert_eq!(actual.caller_return_bits, expected.caller_return_bits);
        assert!(
            actual.state == expected.state,
            "complete entry state differs"
        );
    }
    assert_eq!(got.completed, states);
    assert_eq!(got.return_bits, returns);
    assert_eq!(got.final_state, *states.last().unwrap());
}
#[test]
fn exact_native_normal_fixtures() {
    let mut counts = BTreeMap::new();
    let mut count = 0;
    for r in report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| eligible(r))
    {
        matches(&[r]);
        *counts.entry(r["method"].as_str().unwrap()).or_insert(0) += 1;
        count += 1;
    }
    assert_eq!(counts.len(), 6);
    assert_eq!(count, 100);
}
#[test]
fn retained_native_suffix_preserves_prior_history_and_array() {
    let seq = &report()["retained_sequences"][3];
    let rows = [&seq[1], &seq[2]];
    assert_eq!(snap(&rows[0]["initial"])["fields"]["array"], "other_array");
    assert_eq!(snap(&rows[0]["final"]), snap(&rows[1]["initial"]));
    matches(&rows);
}
#[test]
fn invalid_graphs_and_unsupported_paths_fail_atomically() {
    let row = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| {
            eligible(r)
                && r["method"] == "GetFlavorText"
                && snap(&r["initial"])["arrays"]["array"]["length_bits"] == 3
        })
        .unwrap();
    let c = context(&[row]);
    for variant in 0..9 {
        let mut bad = c.clone();
        match variant {
            0 => bad.inert_services_verified = false,
            1 => bad.calls[0].entry_raw_args[0] = 0,
            2 => bad.calls[0].random_result_bits = 0xFFFFFFFF,
            3 => bad.state.records[0].bytes[0x70..0x78].copy_from_slice(&0u64.to_le_bytes()),
            4 => {
                bad.state.records[0].bytes[0x70..0x78].copy_from_slice(&id("flavor").to_le_bytes())
            }
            5 => bad.state.records[1].identity = bad.state.records[0].identity + 8,
            6 => bad.native_base = u64::MAX,
            7 => bad.state.records[0].identity = u64::MAX - 1,
            _ => bad.calls[0].name_result = id("array"),
        }
        let unchanged = bad.clone();
        assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
        assert_eq!(bad, unchanged);
    }
}
#[test]
fn capacity_bounds_complete_storage_and_future_states() {
    let row = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| eligible(r))
        .unwrap();
    let c = context(&[row]);
    let mut bad = c.clone();
    bad.calls = vec![c.calls[0].clone(); 17];
    assert_eq!(replay(&bad), Err(LedgerError::Capacity));
    let mut bad = c.clone();
    bad.state.native_entries = vec![
        Entry {
            method: Method::GetHints,
            raw_args: [0; 4]
        };
        129
    ];
    assert_eq!(replay(&bad), Err(LedgerError::Capacity));
    let mut bad = c.clone();
    bad.state.records[0].bytes = vec![0; 9000];
    assert_eq!(replay(&bad), Err(LedgerError::Capacity));
    let mut admitted = 0;
    for n in 1..=16 {
        let mut candidate = c.clone();
        candidate.calls = vec![c.calls[0].clone(); n];
        match replay(&candidate) {
            Ok(_) => admitted = n,
            Err(e) => assert_eq!(e, LedgerError::Capacity),
        }
    }
    assert!(admitted >= 8 && admitted < 16);
}
#[test]
fn strict_serialization_and_typed_nullable_results() {
    let row = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| {
            eligible(r)
                && r["method"] == "UpdateCharacterName"
                && r["options"]["null_name_result"] == true
        })
        .unwrap();
    matches(&[row]);
    let c = context(&[row]);
    let v = serde_json::to_value(&c).unwrap();
    assert_eq!(serde_json::from_value::<Context>(v.clone()).unwrap(), c);
    let mut v = v;
    v["callback"] = json!(true);
    assert!(serde_json::from_value::<Context>(v).is_err());
}
