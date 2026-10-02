use super::*;
use serde_json::Value as Json;
use std::sync::OnceLock;

const NAMES: [&str; 18] = [
    "data",
    "other_data",
    "data_class",
    "project_class",
    "other_class",
    "statics",
    "other_statics",
    "context",
    "other_context",
    "game",
    "other_game",
    "description",
    "polish_description",
    "chinese_description",
    "replacement",
    "result0",
    "result1",
    "result2",
];
fn report() -> &'static Json {
    static R: OnceLock<Json> = OnceLock::new();
    R.get_or_init(|| serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_description.json")).unwrap())
}
fn id(n: &str) -> Identity {
    0x3_0000_0000 + 0x200000 + NAMES.iter().position(|&v| v == n).unwrap() as u64 * 0x1000
}
fn reference(v: &Json) -> Identity {
    v.as_str().map_or(0, id)
}
fn snapshot(v: &Json) -> &Json {
    v["snapshot_sha256"]
        .as_str()
        .map_or(v, |h| &report()["snapshot_blobs"][h])
}
fn parse(v: &Json) -> State {
    let v = snapshot(v);
    State {
        records: NAMES
            .into_iter()
            .map(|n| {
                let memory = &v["memory"][n];
                let hex = memory.as_str().unwrap_or_else(|| {
                    report()["memory_blobs"][memory["memory_sha256"].as_str().unwrap()]
                        .as_str()
                        .unwrap()
                });
                Record {
                    identity: id(n),
                    kind: match n {
                        "data" | "other_data" => Kind::Data,
                        "statics" | "other_statics" => Kind::Statics,
                        "context" | "other_context" => Kind::ProjectContext,
                        "game" | "other_game" => Kind::GameData,
                        n if n.ends_with("class") => Kind::Class,
                        _ => Kind::String,
                    },
                    bytes: hex
                        .as_bytes()
                        .chunks_exact(2)
                        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
                        .collect(),
                }
            })
            .collect(),
        metadata_flag: v["metadata_flag_byte"].as_u64().unwrap() as u8,
        project_class: reference(&v["metadata_slot"]),
        native_entries: v["native_entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(args)
            .collect(),
        service_history: v["service_history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| match h[0].as_str().unwrap() {
                "metadata" => Completed::Metadata {
                    class: reference(&h[1]),
                },
                "converter" => Completed::Converter {
                    input: reference(&h[1]),
                    result: reference(&h[2]),
                },
                _ => panic!("unexpected history"),
            })
            .collect(),
    }
}
fn args(v: &Json) -> [u64; 4] {
    std::array::from_fn(|i| v[i].as_u64().unwrap())
}
fn counts(v: &Json) -> BTreeMap<Service, u64> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(n, v)| {
            (
                match n.as_str() {
                    "metadata" => Service::Metadata,
                    "converter" => Service::Converter,
                    _ => panic!("unexpected count"),
                },
                v.as_u64().unwrap(),
            )
        })
        .collect()
}
fn eligible(row: &Json) -> bool {
    row["returned"] == true
        && row["options"]["mutations"].is_null()
        && row["options"]["failure"].is_null()
}
fn context(rows: &[&Json]) -> Context {
    Context {
        version: CHARACTER_DATA_DESCRIPTION_NATIVE_V1.into(),
        owner: id("data"),
        state: parse(&rows[0]["initial"]),
        calls: rows
            .iter()
            .map(|r| {
                let events = r["events"].as_array().unwrap();
                let converters: Vec<_> =
                    events.iter().filter(|e| e["kind"] == "converter").collect();
                let a = args(&r["entry_raw_args"]);
                Call {
                    converter_results: [
                        reference(&converters[0]["args"][2]),
                        converters.get(1).map_or(0, |e| reference(&e["args"][2])),
                    ],
                    entry_method_bits: a[1],
                    entry_r8_bits: a[2],
                    entry_r9_bits: a[3],
                }
            })
            .collect(),
        service_counts: counts(&rows[0]["service_counts_before"]),
        native_base: 0x180000000,
        volatile_return_bits: [0xFACE123456789090; 4],
        storage_verified: true,
        inert_services_verified: true,
        normal_completion_verified: true,
    }
}
fn check(rows: &[&Json]) {
    let c = context(rows);
    let replay = replay(&c).unwrap();
    let mut events = rows.iter().flat_map(|r| r["events"].as_array().unwrap());
    for step in &replay.steps {
        let event = events.next().unwrap();
        let a = &event["args"];
        let expected = if event["kind"] == "metadata" {
            Completed::Metadata {
                class: reference(&a[1]),
            }
        } else {
            Completed::Converter {
                input: reference(&a[0]),
                result: reference(&a[2]),
            }
        };
        assert_eq!(step.event, expected);
        assert_eq!(step.state, parse(&event["snapshot"]));
        assert_eq!(step.raw_args, args(&event["raw_args"]));
        assert_eq!(step.ordinal, event["ordinal"].as_u64().unwrap());
        assert_eq!(
            step.caller_return_bits,
            event["caller_return_bits"].as_u64().unwrap()
        );
        assert_eq!(
            format!("{:#x}", step.native_site_rva),
            event["native_site"].as_str().unwrap()
        );
    }
    assert!(events.next().is_none());
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(replay.completed[i], parse(&row["final"]));
        assert_eq!(replay.return_bits[i], row["result_bits"].as_u64().unwrap());
    }
    assert_eq!(replay.final_state, parse(&rows.last().unwrap()["final"]));
    assert_eq!(
        replay.service_counts,
        counts(&rows.last().unwrap()["service_counts_after"])
    );
}
#[test]
fn inert_native_complete_records_and_abi_match() {
    let mut matched = 0;
    for row in report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| eligible(r))
    {
        check(&[row]);
        matched += 1;
    }
    assert_eq!(matched, 145);
}
#[test]
fn complete_retained_sequences_match() {
    let mut matched = 0;
    for seq in report()["retained_sequences"].as_array().unwrap() {
        let rows: Vec<_> = seq.as_array().unwrap().iter().collect();
        if rows.iter().all(|r| eligible(r)) {
            check(&rows);
            matched += 1;
        }
    }
    assert_eq!(matched, 3);
}
#[test]
fn guards_and_wrong_nominal_storage_fall_back_atomically() {
    let row = &report()["cases"][0];
    let base = context(&[row]);
    for mutate in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
        let mut c = base.clone();
        match mutate {
            0 => c.storage_verified = false,
            1 => c.inert_services_verified = false,
            2 => c.normal_completion_verified = false,
            3 => c.version.push('x'),
            4 => c.state.project_class = 0,
            5 => c
                .state
                .records
                .iter_mut()
                .find(|r| r.identity == id("statics"))
                .unwrap()
                .bytes[..8]
                .fill(0),
            6 => c.calls[0].converter_results[0] = id("game"),
            7 => c.state.records[0].bytes.pop().map(|_| ()).unwrap(),
            8 => c.state.records[1].identity = c.state.records[0].identity + 1,
            _ => {
                c.service_counts.insert(Service::Converter, u64::MAX);
            }
        }
        let saved = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, saved);
    }
}
#[test]
fn capacity_includes_future_full_snapshots_and_history() {
    let mut c = context(&[&report()["cases"][0]]);
    let call = c.calls[0].clone();
    c.calls = vec![call.clone(); 15];
    assert!(replay(&c).is_ok());
    c.calls.push(call);
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    c.calls.pop();
    c.state.native_entries = vec![[0; 4]; 128];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    c.calls.clear();
    assert!(replay(&c).is_ok());
    c.state.records[0].bytes = vec![0; 8193];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}
#[test]
fn strict_schema_rejects_extra_fields() {
    let c = context(&[&report()["cases"][0]]);
    let mut v = serde_json::to_value(&c).unwrap();
    v["callback"] = serde_json::json!(true);
    assert!(serde_json::from_value::<Context>(v).is_err());
}
