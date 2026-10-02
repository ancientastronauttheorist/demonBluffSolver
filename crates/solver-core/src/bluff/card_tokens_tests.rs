use super::*;
use serde_json::{json, Value as Json};
use std::sync::OnceLock;

const NAMES: [&str; 8] = [
    "owner",
    "character",
    "good",
    "excl",
    "unsure",
    "bad",
    "other",
    "class",
];
const GAMES: [&str; 5] = ["good", "excl", "unsure", "bad", "other"];
fn report() -> &'static Json {
    static REPORT: OnceLock<Json> = OnceLock::new();
    REPORT.get_or_init(|| {
        serde_json::from_str(include_str!(
            "../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_card_tokens.json"
        ))
        .unwrap()
    })
}
fn id(name: &str) -> u64 {
    0x3_0000_0000 + 0x10000 + NAMES.iter().position(|&n| n == name).unwrap() as u64 * 0x1000
}
fn label(p: u64) -> Json {
    if p == 0 {
        Json::Null
    } else {
        json!(NAMES.iter().find(|&&n| id(n) == p).unwrap())
    }
}
fn raw(snapshot: &Json, name: &str) -> Vec<u8> {
    let digest = snapshot["memory"][name]["memory_sha256"].as_str().unwrap();
    report()["memory_blobs"][digest]
        .as_str()
        .unwrap()
        .as_bytes()
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
                    "character" => Kind::Character,
                    "class" => Kind::Class,
                    _ => Kind::GameObject,
                },
                bytes: raw(snapshot, n),
            })
            .collect(),
        games: GAMES
            .into_iter()
            .map(|n| Game {
                identity: id(n),
                active: snapshot["games"][n].as_bool().unwrap(),
            })
            .collect(),
        keys: snapshot["keys"]
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
    let options = &rows[0]["options"];
    Context {
        version: CARD_TOKENS_NATIVE_V1.into(),
        owner: id("owner"),
        state: parse_state(&rows[0]["initial"]),
        calls: rows
            .iter()
            .map(|r| {
                let mask = r["options"]["key_mask"].as_u64().unwrap_or(31);
                let low = r["options"]["key_true_byte"].as_u64().unwrap_or(0x80);
                Call {
                    method: if r["method"] == "Update" {
                        Method::Update
                    } else {
                        Method::OnEnable
                    },
                    key_return_bits: std::array::from_fn(|i| {
                        0xFACE123456789000 | if mask & (1 << i) != 0 { low } else { 0 }
                    }),
                }
            })
            .collect(),
        services: Services {
            storage_verified: true,
            supplied_inert_verified: true,
            normal_completion_verified: true,
            active_false_return_bits: 0xFACE123456789000,
            active_true_return_bits: 0xFACE123456789000
                | options["active_true_byte"].as_u64().unwrap_or(0xFE),
            volatile_rdx_bits: 0xFACE123456789090,
        },
    }
}
fn event(e: &Event) -> Json {
    match e {
        Event::KeyDownService { key, result_bits } => {
            json!({"kind":"key_down_service","args":[key,result_bits]})
        }
        Event::ActiveSelfService { game, result_bits } => {
            json!({"kind":"active_self_service","args":[label(*game),result_bits]})
        }
        Event::SetActiveService {
            game,
            rdx_bits,
            value,
        } => json!({"kind":"set_active_service","args":[label(*game),rdx_bits,value]}),
    }
}
fn assert_snapshot(state: &State, expected: &Json) {
    assert_eq!(*state, parse_state(expected));
    let owner = state
        .records
        .iter()
        .find(|r| r.identity == id("owner"))
        .unwrap();
    for (name, off) in [
        ("character", 0x20),
        ("good", 0x28),
        ("excl", 0x30),
        ("unsure", 0x38),
        ("bad", 0x40),
    ] {
        assert_eq!(label(word(owner, off)), expected["fields"][name]);
    }
    let ch = state
        .records
        .iter()
        .find(|r| r.identity == id("character"))
        .unwrap();
    assert_eq!(
        json!(u32::from_le_bytes(ch.bytes[0xe8..0xec].try_into().unwrap())),
        expected["placement_bits"]
    );
    assert_eq!(json!(ch.bytes[0x190]), expected["hover_bits"]);
}
fn compare(rows: &[&Json]) {
    let c = fixture(rows);
    let r = replay(&c).unwrap();
    let mut cursor = 0;
    for (call, row) in rows.iter().enumerate() {
        for native in row["events"].as_array().unwrap() {
            let step = &r.steps[cursor];
            assert_eq!(step.call, call);
            assert_eq!(
                event(&step.event),
                json!({"kind":native["kind"],"args":native["args"]})
            );
            assert_snapshot(&step.state, &native["snapshot"]);
            cursor += 1;
        }
        assert_snapshot(&r.completed[call], &row["final"]);
    }
    assert_eq!(cursor, r.steps.len());
    assert_snapshot(&r.state, &rows.last().unwrap()["final"]);
    assert_eq!(c.state, parse_state(&rows[0]["initial"]));
}

#[test]
fn normal_native_cases_match_complete_trace_and_storage() {
    let cases = report()["cases"].as_array().unwrap();
    let mut count = 0;
    for row in cases {
        if row["returned"] != true || !row["options"]["mutation_phase"].is_null() {
            continue;
        }
        compare(&[row]);
        count += 1;
    }
    assert_eq!(count, 267);
}
#[test]
fn retained_native_sequences_match_physical_aliases_and_ledgers() {
    let seqs = report()["retained_sequences"].as_array().unwrap();
    assert_eq!(seqs.len(), 2);
    for seq in seqs {
        let rows: Vec<_> = seq.as_array().unwrap().iter().collect();
        compare(&rows);
    }
}
fn baseline() -> Context {
    fixture(&[&report()["cases"][0]])
}
#[test]
fn rejects_unverified_and_invalid_contexts_atomically() {
    let base = baseline();
    let invalid: Vec<Box<dyn Fn(&mut Context)>> = vec![
        Box::new(|c| c.version.push('x')),
        Box::new(|c| c.services.storage_verified = false),
        Box::new(|c| c.services.supplied_inert_verified = false),
        Box::new(|c| c.services.normal_completion_verified = false),
        Box::new(|c| c.owner = 0),
        Box::new(|c| c.state.records[0].identity = 0),
        Box::new(|c| c.state.records[0].kind = Kind::GameObject),
        Box::new(|c| c.state.records[1].bytes.pop().map(|_| ()).unwrap()),
        Box::new(|c| c.state.records.push(c.state.records[0].clone())),
        Box::new(|c| c.state.games.push(c.state.games[0].clone())),
        Box::new(|c| c.state.games.remove(0).identity = 0),
        Box::new(|c| {
            c.state.records[0].bytes[0x28..0x30].copy_from_slice(&id("class").to_le_bytes())
        }),
        Box::new(|c| {
            c.state.records[0].bytes[0x20..0x28].copy_from_slice(&id("good").to_le_bytes())
        }),
        Box::new(|c| {
            c.state.keys.push(KeyRead {
                key: 99,
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
                game: id("good"),
                rdx_bits: 0,
                value: 1,
            })
        }),
    ];
    for alter in invalid {
        let mut c = base.clone();
        alter(&mut c);
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
    // A pressed branch with a missing tag must be rejected before any writes.
    let mut c = base.clone();
    c.calls[0].key_return_bits[1] = 1;
    c.state.records[0].bytes[0x38..0x40].copy_from_slice(&0u64.to_le_bytes());
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
}
#[test]
fn budgets_complete_future_snapshots_before_cloning() {
    let base = baseline();
    for alteration in 0..5 {
        let mut c = base.clone();
        match alteration {
            0 => c.calls = vec![c.calls[0].clone(); 17],
            1 => c.state.records[0].bytes.resize(20_000, 0),
            2 => {
                c.state.keys = vec![
                    KeyRead {
                        key: 53,
                        result_bits: 0
                    };
                    129
                ]
            }
            3 => c.state.games = vec![c.state.games[0].clone(); 33],
            _ => {
                c.calls = vec![c.calls[0].clone(); 16];
                c.state.active_writes = vec![
                    ActiveWrite {
                        game: id("good"),
                        rdx_bits: 0,
                        value: 0
                    };
                    128
                ];
                c.state.active_reads = vec![
                    ActiveRead {
                        game: id("good"),
                        result_bits: 0
                    };
                    128
                ];
                c.state.keys = vec![
                    KeyRead {
                        key: 53,
                        result_bits: 0
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
fn supplied_low_bytes_and_register_high_bits_are_independent() {
    let mut c = baseline();
    c.calls[0].key_return_bits = [0xAB00000000000000, 0xAB00000000000080, 0, 0, 0];
    c.services.active_false_return_bits = 0xBA000000000000FF;
    c.services.active_true_return_bits = 0xBA00000000000000;
    c.services.volatile_rdx_bits = 0xCA1234567890ABCD;
    let r = replay(&c).unwrap();
    let reads: Vec<_> = r
        .steps
        .iter()
        .filter_map(|s| {
            if let Event::ActiveSelfService { result_bits, .. } = s.event {
                Some(result_bits)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(reads, vec![0xBA000000000000FF]);
    let first = r.state.active_writes.first().unwrap();
    assert_eq!(first.value, 0);
    assert_eq!(first.rdx_bits, 0xCA1234567890AB00);
    assert_eq!(r.state.keys.len(), 5);
    let mut idle = c.clone();
    idle.state.records[1].bytes[0x190] = 0;
    assert!(replay(&idle).unwrap().steps.is_empty());
}
