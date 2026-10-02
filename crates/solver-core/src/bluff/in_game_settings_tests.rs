use super::*;
use serde_json::{json, Value as Json};
use std::sync::OnceLock;
const NAMES: [&str; 4] = ["owner", "settings", "other", "class"];
fn report() -> &'static Json {
    static REPORT: OnceLock<Json> = OnceLock::new();
    REPORT.get_or_init(||serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_in_game_settings.json")).unwrap())
}
fn id(name: &str) -> u64 {
    0x3_0000_0000 + 0x60000 + NAMES.iter().position(|&n| n == name).unwrap() as u64 * 0x1000
}
fn label(id_value: u64) -> Json {
    if id_value == 0 {
        Json::Null
    } else {
        json!(NAMES.iter().find(|&&n| id(n) == id_value).unwrap())
    }
}
fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect()
}
fn parse_state(snapshot: &Json) -> State {
    State {
        records: NAMES
            .into_iter()
            .map(|n| Record {
                identity: id(n),
                kind: match n {
                    "owner" => Kind::Owner,
                    "class" => Kind::Class,
                    _ => Kind::GameObject,
                },
                bytes: bytes(snapshot["memory"][n].as_str().unwrap()),
            })
            .collect(),
        games: ["settings", "other"]
            .into_iter()
            .map(|n| Game {
                identity: id(n),
                active: snapshot["games"][n].as_bool().unwrap(),
            })
            .collect(),
        keys: snapshot["key_requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| KeyRead {
                key: v[0].as_u64().unwrap() as u32,
                result_bits: v[1].as_u64().unwrap(),
            })
            .collect(),
        active_reads: snapshot["active_reads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| ActiveRead {
                game: id(v[0].as_str().unwrap()),
                result_bits: v[1].as_u64().unwrap(),
            })
            .collect(),
        active_writes: snapshot["active_writes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| ActiveWrite {
                game: id(v[0].as_str().unwrap()),
                rdx_bits: v[1].as_u64().unwrap(),
                value: v[2].as_u64().unwrap() as u8,
            })
            .collect(),
    }
}
fn fixture(rows: &[&Json]) -> Context {
    Context {
        version: IN_GAME_SETTINGS_NATIVE_V1.into(),
        owner: id("owner"),
        state: parse_state(&rows[0]["initial"]),
        calls: rows
            .iter()
            .map(|r| Call {
                method: match r["method"].as_str().unwrap() {
                    "Update" => Method::Update,
                    "OnEnable" => Method::OnEnable,
                    _ => Method::ManageShowSettings,
                },
                key_return_bits: 0xFACE123456789000
                    | if r["options"]["key_pressed"] == false {
                        0
                    } else {
                        r["options"]["key_true_byte"].as_u64().unwrap_or(0x80)
                    },
            })
            .collect(),
        services: Services {
            storage_verified: true,
            supplied_inert_verified: true,
            normal_completion_verified: true,
            active_false_return_bits: 0xFACE123456789000,
            active_true_return_bits: 0xFACE123456789000
                | rows[0]["options"]["active_true_byte"]
                    .as_u64()
                    .unwrap_or(0xFE),
            volatile_rdx_bits: 0xFACE123456789090,
        },
    }
}
fn event(event: &Event) -> Json {
    match event {
        Event::KeyDownService { key, result_bits } => {
            json!({"kind":"key_down_service","args":[key,result_bits,0]})
        }
        Event::ActiveSelfService { game, result_bits } => {
            json!({"kind":"active_self_service","args":[label(*game),result_bits,0]})
        }
        Event::SetActiveService {
            game,
            rdx_bits,
            value,
        } => json!({"kind":"set_active_service","args":[label(*game),rdx_bits,value,0]}),
    }
}
fn assert_snapshot(state: &State, native: &Json) {
    assert_eq!(*state, parse_state(native));
    let owner = state
        .records
        .iter()
        .find(|r| r.identity == id("owner"))
        .unwrap();
    assert_eq!(label(word(owner, 0x20)), native["settings_ref"]);
}
fn compare(rows: &[&Json]) {
    let c = fixture(rows);
    let result = replay(&c).unwrap();
    let mut cursor = 0;
    for (call, row) in rows.iter().enumerate() {
        for native in row["events"].as_array().unwrap() {
            let step = &result.steps[cursor];
            assert_eq!(step.call, call);
            assert_eq!(
                event(&step.event),
                json!({"kind":native["kind"],"args":native["args"]})
            );
            assert_snapshot(&step.state, &native["snapshot"]);
            cursor += 1;
        }
        assert_snapshot(&result.completed[call], &row["final"]);
    }
    assert_eq!(cursor, result.steps.len());
    assert_snapshot(&result.state, &rows.last().unwrap()["final"]);
    assert_eq!(c.state, parse_state(&rows[0]["initial"]));
}
#[test]
fn matches_all_supported_normal_native_profiles() {
    let mut count = 0;
    for row in report()["cases"].as_array().unwrap() {
        if row["returned"] != true
            || !row["options"]["mutation_phase"].is_null()
            || row["options"]["null_owner"] == true
        {
            continue;
        }
        compare(&[row]);
        count += 1;
    }
    assert_eq!(count, 92);
}
#[test]
fn matches_retained_reset_toggle_and_idle_sequences() {
    let mut count = 0;
    for seq in report()["retained_sequences"].as_array().unwrap() {
        let rows: Vec<_> = seq.as_array().unwrap().iter().collect();
        if rows
            .iter()
            .any(|r| !r["options"]["mutation_phase"].is_null())
        {
            continue;
        }
        assert_eq!(rows.len(), 7);
        compare(&rows);
        count += 1;
    }
    assert_eq!(count, 2);
}
fn baseline() -> Context {
    fixture(&[&report()["cases"][0]])
}
#[test]
fn rejects_invalid_provenance_identities_and_consumed_refs_atomically() {
    let base = baseline();
    let changes: Vec<Box<dyn Fn(&mut Context)>> = vec![
        Box::new(|c| c.version.push('x')),
        Box::new(|c| c.owner = 0),
        Box::new(|c| c.services.storage_verified = false),
        Box::new(|c| c.services.supplied_inert_verified = false),
        Box::new(|c| c.services.normal_completion_verified = false),
        Box::new(|c| c.state.records[0].kind = Kind::GameObject),
        Box::new(|c| c.state.records[1].kind = Kind::Character),
        Box::new(|c| c.state.records[1].bytes.push(0)),
        Box::new(|c| c.state.records.push(c.state.records[0].clone())),
        Box::new(|c| c.state.games.push(c.state.games[0].clone())),
        Box::new(|c| {
            c.state.games.remove(0);
        }),
        Box::new(|c| {
            c.state.records[0].bytes[0x20..0x28].copy_from_slice(&id("class").to_le_bytes())
        }),
        Box::new(|c| c.state.records[0].bytes[0x20..0x28].copy_from_slice(&0u64.to_le_bytes())),
        Box::new(|c| {
            c.state.keys.push(KeyRead {
                key: 53,
                result_bits: 0,
            })
        }),
        Box::new(|c| {
            c.state.active_reads.push(ActiveRead {
                game: id("class"),
                result_bits: 0,
            })
        }),
        Box::new(|c| {
            c.state.active_writes.push(ActiveWrite {
                game: id("settings"),
                rdx_bits: 0,
                value: 1,
            })
        }),
    ];
    for change in changes {
        let mut c = base.clone();
        change(&mut c);
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
    let mut idle = base;
    idle.state.records[0].bytes[0x20..0x28].copy_from_slice(&0u64.to_le_bytes());
    idle.calls[0].key_return_bits = 0xFF00000000000000;
    let result = replay(&idle).unwrap();
    assert_eq!(result.steps.len(), 1);
    assert!(result.state.active_writes.is_empty());
}
#[test]
fn reserves_aggregate_future_state_and_logs_before_cloning() {
    for kind in 0..4 {
        let mut c = baseline();
        match kind {
            0 => c.calls = vec![c.calls[0].clone(); 17],
            1 => c.state.records[0].bytes.resize(10_000, 0),
            2 => {
                c.state.keys = vec![
                    KeyRead {
                        key: 27,
                        result_bits: 0
                    };
                    129
                ]
            }
            _ => {
                c.state.records = vec![c.state.records[0].clone(); 32];
                c.calls = vec![c.calls[0].clone(); 16];
                c.state.keys = vec![
                    KeyRead {
                        key: 27,
                        result_bits: 0
                    };
                    128
                ];
                c.state.active_reads = vec![
                    ActiveRead {
                        game: id("settings"),
                        result_bits: 0
                    };
                    128
                ];
                c.state.active_writes = vec![
                    ActiveWrite {
                        game: id("settings"),
                        rdx_bits: 0,
                        value: 0
                    };
                    128
                ];
            }
        }
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::Capacity));
        assert_eq!(c, before);
    }
}
#[test]
fn honors_supplied_low_bytes_and_method_specific_register_writes() {
    let mut c = baseline();
    c.services.active_false_return_bits = 0xBA000000000000FF;
    c.services.active_true_return_bits = 0xBA000000000000FF;
    c.services.volatile_rdx_bits = 0xCA1234567890ABCD;
    c.calls = vec![
        Call {
            method: Method::Update,
            key_return_bits: 0xFF00000000000080,
        },
        Call {
            method: Method::ManageShowSettings,
            key_return_bits: 0,
        },
        Call {
            method: Method::OnEnable,
            key_return_bits: 0,
        },
    ];
    let result = replay(&c).unwrap();
    assert_eq!(
        result
            .state
            .active_writes
            .iter()
            .map(|r| r.rdx_bits)
            .collect::<Vec<_>>(),
        vec![0xCA1234567890AB00, 0, 0]
    );
    assert!(result.state.active_writes.iter().all(|r| r.value == 0));
    assert_eq!(result.state.records, c.state.records);
}
