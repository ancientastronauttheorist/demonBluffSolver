use super::*;
use serde_json::{json, Value as Json};
use std::sync::OnceLock;

const NAMES: [&str; 17] = [
    "data",
    "skin",
    "other_skin",
    "object_class",
    "name",
    "other_name",
    "i_was",
    "translation",
    "default_art",
    "default_animated",
    "skin_art",
    "skin_animated",
    "other_art",
    "other_animated",
    "artist",
    "other_artist",
    "normandia",
];
const FLAGS: [Method; 4] = [
    Method::GetArt,
    Method::GetAnimatedArt,
    Method::GetArtType,
    Method::GetArtistName,
];
fn report() -> &'static Json {
    static REPORT: OnceLock<Json> = OnceLock::new();
    REPORT.get_or_init(||serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_consumers.json")).unwrap())
}
fn id(n: &str) -> u64 {
    0x3_0000_0000 + 0x90000 + NAMES.iter().position(|&v| v == n).unwrap() as u64 * 0x1000
}
fn identity(v: &Json) -> u64 {
    v.as_str().map(id).unwrap_or(0)
}
fn method(n: &str) -> Method {
    serde_json::from_value(json!(n)).unwrap()
}
fn parse(s: &Json) -> State {
    State {
        records: NAMES
            .into_iter()
            .map(|n| Record {
                identity: id(n),
                kind: match n {
                    "data" => Kind::Data,
                    "skin" | "other_skin" => Kind::Skin,
                    "object_class" => Kind::Class,
                    "translation" => Kind::Translation,
                    "default_art" | "default_animated" | "skin_art" | "skin_animated"
                    | "other_art" | "other_animated" => Kind::Sprite,
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
        metadata_flags: FLAGS
            .map(|m| s["metadata_flags_u8"][format!("{m:?}")].as_u64().unwrap() as u8),
        comparisons: s["comparisons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| Comparison {
                // All entries in a single native call use its equality kind. Retained
                // sequences need chronology to recover that kind from the event stream.
                inequality: false,
                skin: identity(&v[0]),
                method_info: v[2].as_u64().unwrap(),
                result_bits: v[3].as_u64().unwrap(),
            })
            .collect(),
    }
}
fn fixture(rows: &[&Json]) -> Context {
    let initial = &rows[0]["initial"];
    assert_eq!(initial["comparisons"], json!([]));
    Context {
        version: CHARACTER_DATA_CONSUMERS_NATIVE_V1.into(),
        data: id("data"),
        object_class: identity(&initial["metadata_slots"]["UnityEngine.Object_TypeInfo"]),
        default_artist: identity(&initial["metadata_slots"]["normandia"]),
        state: parse(initial),
        calls: rows
            .iter()
            .map(|r| Call {
                method: method(r["method"].as_str().unwrap()),
                comparison_return_bits: r["events"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|e| {
                        matches!(
                            e["kind"].as_str(),
                            Some("object_equality_service" | "object_inequality_service")
                        )
                    })
                    .map(|e| e["args"][3].as_u64().unwrap())
                    .unwrap_or(0),
            })
            .collect(),
        storage_verified: true,
        supplied_inert_verified: true,
        normal_completion_verified: true,
    }
}
fn native_state(s: &Json, history: &[bool]) -> State {
    let mut state = parse(s);
    assert_eq!(state.comparisons.len(), history.len());
    for (c, &kind) in state.comparisons.iter_mut().zip(history) {
        c.inequality = kind;
    }
    let data = record(&state, id("data"));
    assert_eq!(word(data, 0xC0), identity(&s["current_skin"]));
    assert_eq!(
        dword(record(&state, id("object_class")), 0xE0) as u64,
        s["object_class_word_u32"].as_u64().unwrap()
    );
    assert_eq!(
        s["metadata_slots"]["UnityEngine.Object_TypeInfo"],
        json!("object_class")
    );
    assert_eq!(s["metadata_slots"]["normandia"], json!("normandia"));
    state
}
fn matches_rows(rows: &[&Json]) {
    let c = fixture(rows);
    let before = c.clone();
    let actual = replay(&c).unwrap();
    assert_eq!(c, before);
    let mut steps = Vec::new();
    let mut history = Vec::new();
    let mut returns = Vec::new();
    for r in rows {
        assert!(r["returned"].as_bool().unwrap());
        for e in r["events"].as_array().unwrap() {
            let event = match e["kind"].as_str().unwrap() {
                "metadata_service" => {
                    if e["args"][0] == "normandia" {
                        Event::ArtistMetadata
                    } else {
                        assert_eq!(e["args"][0], "UnityEngine.Object_TypeInfo");
                        Event::ObjectMetadata
                    }
                }
                "object_class_initialize_service" => {
                    Event::InitializeObject(identity(&e["args"][0]))
                }
                kind @ ("object_equality_service" | "object_inequality_service") => {
                    assert!(e["args"][1].is_null());
                    Event::Compare(Comparison {
                        inequality: kind == "object_inequality_service",
                        skin: identity(&e["args"][0]),
                        method_info: e["args"][2].as_u64().unwrap(),
                        result_bits: e["args"][3].as_u64().unwrap(),
                    })
                }
                other => panic!("unexpected {other}"),
            };
            steps.push(Step {
                state: native_state(&e["snapshot"], &history),
                event: event.clone(),
            });
            if let Event::Compare(compare) = event {
                history.push(compare.inequality);
            }
        }
        returns.push(r["result_bits"].as_u64().unwrap());
        assert_eq!(
            native_state(&r["final"], &history).comparisons.len(),
            history.len()
        );
    }
    assert_eq!(actual.steps, steps);
    assert_eq!(actual.return_bits, returns);
    assert_eq!(
        actual.final_state,
        native_state(&rows.last().unwrap()["final"], &history)
    );
}
fn supported(r: &Json) -> bool {
    r["returned"] == true && r["options"]["mutation_phase"].is_null()
}
#[test]
fn matches_supported_native_profiles_and_baselines() {
    let mut n = 0;
    for r in report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report()["baselines"].as_array().unwrap())
    {
        if supported(r) {
            matches_rows(&[r]);
            n += 1;
        }
    }
    assert_eq!(n, 154);
}
#[test]
fn matches_retained_inert_skin_and_leaf_sequences() {
    let mut n = 0;
    for seq in report()["retained_sequences"].as_array().unwrap() {
        let rows = seq.as_array().unwrap();
        if rows.iter().all(supported) {
            matches_rows(&rows.iter().collect::<Vec<_>>());
            n += 1;
        }
    }
    assert_eq!(n, 3);
}
fn normal() -> Context {
    fixture(&[report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["method"] == "GetArt" && r["options"] == json!({}))
        .unwrap()])
}
#[test]
fn rejects_bad_storage_provenance_and_nullable_selected_skin_atomically() {
    let c = normal();
    let mut cases = Vec::new();
    let mut bad = c.clone();
    bad.version = "unknown".into();
    cases.push(bad);
    for i in 0..3 {
        let mut bad = c.clone();
        match i {
            0 => bad.storage_verified = false,
            1 => bad.supplied_inert_verified = false,
            _ => bad.normal_completion_verified = false,
        };
        cases.push(bad);
    }
    let mut bad = c.clone();
    bad.state.records[0].bytes.pop();
    cases.push(bad);
    let mut bad = c.clone();
    bad.state.records.push(bad.state.records[0].clone());
    cases.push(bad);
    let mut bad = c.clone();
    bad.data = id("skin");
    cases.push(bad);
    let mut bad = c.clone();
    bad.default_artist = id("skin_art");
    cases.push(bad);
    let mut bad = c.clone();
    bad.state.records[0].bytes[0xC0..0xC8].copy_from_slice(&0u64.to_le_bytes());
    bad.calls[0].comparison_return_bits = 0;
    cases.push(bad);
    let mut bad = c.clone();
    bad.state.records[0].bytes[0x98..0xA0].copy_from_slice(&id("skin").to_le_bytes());
    cases.push(bad);
    let mut bad = c.clone();
    bad.state.comparisons.push(Comparison {
        inequality: false,
        skin: id("skin"),
        method_info: 1,
        result_bits: 0,
    });
    cases.push(bad);
    for bad in cases {
        let before = bad.clone();
        assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
        assert_eq!(bad, before);
    }
}
#[test]
fn reserves_all_future_snapshot_work_before_cloning() {
    let mut c = normal();
    c.calls = vec![c.calls[0].clone(); 17];
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = normal();
    c.calls = vec![c.calls[0].clone(); 16];
    assert!(replay(&c).is_ok());
    let mut unused = c.state.records[0].clone();
    unused.identity = id("data") + 0x100000;
    c.state.records.push(unused);
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
}
#[test]
fn consumes_only_comparison_low_byte_and_zero_extends_raw_enum_values() {
    let mut c = normal();
    c.state.metadata_flags = [0xFE; 4];
    let class = c
        .state
        .records
        .iter_mut()
        .find(|r| r.kind == Kind::Class)
        .unwrap();
    class.bytes[0xE0..0xE4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let skin = c
        .state
        .records
        .iter_mut()
        .find(|r| r.identity == id("skin"))
        .unwrap();
    skin.bytes[0x50..0x54].copy_from_slice(&0x8000000Au32.to_le_bytes());
    for byte in [0, 1, 0x80, 0xFF] {
        c.calls = vec![Call {
            method: Method::GetArtType,
            comparison_return_bits: 0x123456789ABCDE00 | byte,
        }];
        let r = replay(&c).unwrap();
        assert_eq!(r.return_bits, vec![if byte == 0 { 0x8000000A } else { 0 }]);
        assert_eq!(r.steps.len(), 1);
        assert_eq!(r.final_state.records, c.state.records);
    }
}
