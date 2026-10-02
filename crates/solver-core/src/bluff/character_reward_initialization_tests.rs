use super::*;
use serde_json::Value as Json;
use std::sync::OnceLock;
const NAMES: [&str; 21] = [
    "owner",
    "character_class",
    "data0",
    "data1",
    "data_class",
    "bluff",
    "register_as",
    "acteds0",
    "acteds1",
    "acteds_class",
    "gameobject0",
    "gameobject1",
    "gameobject_class",
    "state_action",
    "other_state_action",
    "action_class_token",
    "state_code",
    "other_code",
    "state_method",
    "other_method",
    "unused_managed_target",
];
fn report() -> &'static Json {
    static R: OnceLock<Json> = OnceLock::new();
    R.get_or_init(|| serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_init_reward.json")).unwrap())
}
fn snapshot(v: &Json) -> &Json {
    v["snapshot_sha256"]
        .as_str()
        .map_or(v, |h| &report()["snapshot_blobs"][h])
}
fn id(name: &str) -> Identity {
    0x3_0000_0000 + 0xD0000 + NAMES.iter().position(|&n| n == name).unwrap() as u64 * 0x1000
}
fn reference(v: &Json) -> Identity {
    v.as_str().map_or(0, id)
}
fn kind(name: &str) -> Kind {
    match name {
        "owner" => Kind::Character,
        "data0" | "data1" | "bluff" | "register_as" => Kind::Data,
        "acteds0" | "acteds1" => Kind::Acted,
        "gameobject0" | "gameobject1" => Kind::GameObject,
        "state_action" | "other_state_action" => Kind::Action,
        n if n.contains("class") => Kind::Class,
        _ => Kind::Token,
    }
}
fn raw_args(v: &Json) -> [u64; 4] {
    std::array::from_fn(|i| v[i].as_u64().unwrap_or_else(|| reference(&v[i])))
}
fn parse(v: &Json) -> State {
    let v = snapshot(v);
    let s = State {
        records: NAMES
            .into_iter()
            .map(|n| Record {
                identity: id(n),
                kind: kind(n),
                bytes: v["memory"][n]
                    .as_str()
                    .unwrap()
                    .as_bytes()
                    .chunks_exact(2)
                    .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
                    .collect(),
            })
            .collect(),
        games: ["gameobject0", "gameobject1"]
            .into_iter()
            .map(|n| GameObject {
                identity: id(n),
                active: v["gameobject_active"][n].as_bool().unwrap(),
            })
            .collect(),
        activation_requests: v["activation_requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(reference)
            .collect(),
        callbacks: v["callbacks"]
            .as_array()
            .unwrap()
            .iter()
            .map(raw_args)
            .collect(),
        reveal_requests: v["reveal_requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(raw_args)
            .collect(),
        native_phases: v["native_phases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| match p["phase"].as_str().unwrap() {
                "entry" => NativePhase::Entry {
                    input_data: reference(&p["captured_input"]),
                },
                "alignment_load" => NativePhase::AlignmentLoad {
                    input_data: reference(&p["input"]),
                    bits: p["bits"].as_u64().unwrap() as u32,
                },
                "current_state_load" => NativePhase::CurrentStateLoad {
                    bits: p["bits"].as_u64().unwrap() as u32,
                },
                _ => panic!("unexpected phase"),
            })
            .collect(),
    };
    for (n, off) in [
        ("data_ref", 0x50),
        ("bluff", 0x58),
        ("register_as", 0x60),
        ("acteds", 0xA8),
        ("state_action", 0x180),
    ] {
        assert_eq!(
            word(record(&s, id("owner")), off),
            reference(&v["pointers"][n])
        );
    }
    for (n, off) in [
        ("pickable_uses", 0xDC),
        ("prev_state", 0xE0),
        ("state", 0xE4),
        ("alignment", 0xF8),
    ] {
        assert_eq!(
            dword(record(&s, id("owner")), off) as u64,
            v["words"][n].as_u64().unwrap()
        );
    }
    for n in ["data0", "data1"] {
        assert_eq!(
            dword(record(&s, id(n)), 0x134) as u64,
            v["data_alignment_bits"][n].as_u64().unwrap()
        );
    }
    s
}
fn service(name: &str) -> Service {
    match name {
        "get_gameobject" => Service::GetGameObject,
        "set_active" => Service::SetActive,
        "barrier" => Service::Barrier,
        "callback" => Service::Callback,
        "supplied_RevealReal" => Service::RevealReal,
        _ => panic!("unexpected service"),
    }
}
fn event(e: &Json) -> Event {
    let a = &e["args"];
    match e["kind"].as_str().unwrap() {
        "get_gameobject" => Event::GetGameObject {
            acted: reference(&a[0]),
            result: reference(&a[2]),
        },
        "set_active" => Event::SetActive {
            game: reference(&a[0]),
        },
        "barrier" => Event::Barrier {
            field: match a[0].as_str().unwrap() {
                "bluff" => Field::Bluff,
                "data_ref" => Field::DataRef,
                "register_as" => Field::RegisterAs,
                _ => panic!("unexpected barrier"),
            },
            value: reference(&a[1]),
        },
        "callback" => Event::Callback,
        "supplied_RevealReal" => Event::RevealReal,
        _ => panic!("unsupported native stop"),
    }
}
fn context(r: &Json) -> Context {
    let e = &r["events"][0];
    let initial = parse(&r["initial"]);
    let input_data = if r["options"]["alias_input_data"] == true {
        id("data0")
    } else {
        id("data1")
    };
    let caller = &e["abi"]["caller_return"];
    let rva = u32::from_str_radix(
        caller["native_rva"]
            .as_str()
            .unwrap()
            .trim_start_matches("0x"),
        16,
    )
    .unwrap();
    Context {
        version: CHARACTER_REWARD_INITIALIZATION_NATIVE_V1.into(),
        owner: id("owner"),
        state: initial,
        calls: vec![Call {
            input_data,
            gameobject_result: reference(&e["args"][2]),
            entry_r8_bits: e["abi"]["r8_bits"].as_u64().unwrap(),
            entry_r9_bits: e["abi"]["r9_bits"].as_u64().unwrap(),
        }],
        service_counts: r["service_counts_before"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (service(k), v.as_u64().unwrap()))
            .collect(),
        native_base: caller["address_bits"].as_u64().unwrap() - u64::from(rva),
        caller_return_bits: 0x5_0000_0000,
        callback_entry_bits: 0x5_0000_0100,
        volatile_return_bits: std::array::from_fn(|i| 0xFACE123456789000 + i as u64),
        storage_verified: true,
        inert_services_verified: true,
        normal_completion_verified: true,
    }
}
fn expected_step(e: &Json, call: usize) -> Step {
    let abi = &e["abi"];
    Step {
        call,
        ordinal: e["ordinal"].as_u64().unwrap(),
        event: event(e),
        raw_args: ["rcx_bits", "rdx_bits", "r8_bits", "r9_bits"].map(|n| abi[n].as_u64().unwrap()),
        caller_return_rva: abi["caller_return"]["native_rva"]
            .as_str()
            .map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()),
        caller_return_bits: abi["caller_return"]["address_bits"].as_u64().unwrap(),
        state: parse(&e["snapshot"]),
    }
}
fn check(r: &Json) {
    let c = context(r);
    let before = c.clone();
    let out = replay(&c).unwrap();
    assert_eq!(c, before);
    assert_eq!(
        out.steps,
        r["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| expected_step(e, 0))
            .collect::<Vec<_>>()
    );
    assert_eq!(out.final_state, parse(&r["final"]));
    assert_eq!(out.completed, vec![out.final_state.clone()]);
    let mut counts = c.service_counts.clone();
    for e in r["events"].as_array().unwrap() {
        *counts
            .entry(service(e["kind"].as_str().unwrap()))
            .or_default() += 1;
    }
    assert_eq!(out.service_counts, counts);
}
fn supported(r: &Json) -> bool {
    r["returned"] == true && r["options"]["mutations"].is_null()
}
fn normal() -> Context {
    context(
        report()["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| supported(r))
            .unwrap(),
    )
}
#[test]
fn matches_every_supported_native_reward_context_and_baseline() {
    let mut n = 0;
    for r in report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report()["failure_baselines"].as_array().unwrap())
    {
        if supported(r) {
            check(r);
            n += 1;
        }
    }
    assert_eq!(n, 34);
}
#[test]
fn matches_complete_native_retained_sequences_and_cumulative_ordinals() {
    let mut n = 0;
    for seq in report()["retained_sequences"].as_array().unwrap() {
        let rows = seq["calls"].as_array().unwrap();
        if !rows.iter().all(supported) {
            continue;
        }
        let mut c = context(&rows[0]);
        c.calls = rows.iter().map(|r| context(r).calls[0].clone()).collect();
        let before = c.clone();
        let out = replay(&c).unwrap();
        assert_eq!(c, before);
        let steps = rows
            .iter()
            .enumerate()
            .flat_map(|(i, r)| {
                r["events"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(move |e| expected_step(e, i))
            })
            .collect::<Vec<_>>();
        assert_eq!(out.steps, steps);
        assert_eq!(
            out.completed,
            rows.iter().map(|r| parse(&r["final"])).collect::<Vec<_>>()
        );
        assert_eq!(out.final_state, parse(&rows.last().unwrap()["final"]));
        n += 1;
    }
    assert_eq!(n, 2);
}
#[test]
fn rejects_invalid_nominal_storage_and_history_atomically() {
    let c = normal();
    let mut cases = Vec::new();
    let mut x = c.clone();
    x.version.push('!');
    cases.push(x);
    let mut x = c.clone();
    x.storage_verified = false;
    cases.push(x);
    let mut x = c.clone();
    x.inert_services_verified = false;
    cases.push(x);
    let mut x = c.clone();
    x.normal_completion_verified = false;
    cases.push(x);
    let mut x = c.clone();
    x.owner = 0;
    cases.push(x);
    let mut x = c.clone();
    x.calls[0].input_data = 0;
    cases.push(x);
    let mut x = c.clone();
    x.calls[0].input_data = id("bluff");
    cases.push(x);
    let mut x = c.clone();
    x.calls[0].gameobject_result = id("data0");
    cases.push(x);
    let mut x = c.clone();
    x.state.records.push(x.state.records[0].clone());
    cases.push(x);
    let mut x = c.clone();
    x.state.records[1].identity = x.owner + 1;
    cases.push(x);
    let mut x = c.clone();
    x.state.records[1].identity = u64::MAX - 127;
    cases.push(x);
    let mut x = c.clone();
    x.state.records[0].bytes.pop();
    cases.push(x);
    let mut x = c.clone();
    x.state.games.pop();
    cases.push(x);
    let mut x = c.clone();
    x.state.activation_requests.push(id("state_action"));
    cases.push(x);
    let mut x = c.clone();
    x.state
        .callbacks
        .push([id("gameobject0"), id("state_method"), 0, 0]);
    cases.push(x);
    let mut x = c.clone();
    x.state.reveal_requests.push([x.owner, 1, 0, 0]);
    cases.push(x);
    let mut x = c.clone();
    x.service_counts.insert(Service::Barrier, u64::MAX);
    cases.push(x);
    let mut x = c.clone();
    x.native_base = u64::MAX;
    cases.push(x);
    let mut x = c.clone();
    x.callback_entry_bits = 1;
    cases.push(x);
    let mut x = c.clone();
    x.state.native_phases.push(NativePhase::Entry {
        input_data: id("acteds0"),
    });
    cases.push(x);
    for x in cases {
        let before = x.clone();
        assert_eq!(replay(&x), Err(LedgerError::InvalidContext));
        assert_eq!(x, before);
    }
}
#[test]
fn reserves_future_full_snapshots_and_retained_histories_before_replay() {
    let mut c = normal();
    let one = c.calls[0].clone();
    c.calls = vec![one.clone(); 6];
    assert!(replay(&c).is_ok());
    c.calls = vec![one; 7];
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = normal();
    c.state.native_phases = vec![NativePhase::CurrentStateLoad { bits: 1 }; 129];
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
}
#[test]
fn retains_prior_bytes_and_uses_explicit_outputs_and_independent_volatile_bits() {
    let mut c = normal();
    c.calls[0].gameobject_result = id("gameobject1");
    c.calls[0].entry_r8_bits = 0x8000000000000008;
    c.calls[0].entry_r9_bits = 0x9000000000000009;
    c.volatile_return_bits = [11, 12, 13, 14];
    let before = c.clone();
    let out = replay(&c).unwrap();
    assert_eq!(c, before);
    assert_eq!(
        out.steps[0].raw_args,
        [
            id("acteds0"),
            0,
            c.calls[0].entry_r8_bits,
            c.calls[0].entry_r9_bits
        ]
    );
    assert_eq!(out.steps[1].raw_args, [id("gameobject1"), 0, 0, 14]);
    assert!(out.steps[2..].iter().all(|s| s.raw_args[2..] == [13, 14]));
    assert!(out.final_state.games[0].active);
    assert!(!out.final_state.games[1].active);
    for r in &c.state.records {
        if r.identity != c.owner {
            assert_eq!(record(&out.final_state, r.identity), r);
        } else {
            for (i, b) in r.bytes.iter().enumerate() {
                if ![0x50..0x68, 0xDC..0xE8, 0xF8..0xFC]
                    .iter()
                    .any(|range| range.contains(&i))
                {
                    assert_eq!(*b, record(&out.final_state, r.identity).bytes[i]);
                }
            }
        }
    }
}
