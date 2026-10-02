use super::*;
use serde_json::{json, Value as Json};
use std::sync::OnceLock;
const NAMES: [&str; 14] = [
    "owner",
    "other_owner",
    "owner_class",
    "loc0",
    "loc1",
    "locale_class",
    "code",
    "other_code",
    "name",
    "other_name",
    "i_was",
    "other_i_was",
    "replacement",
    "entries",
];
const VOL: [&str; 7] = ["RAX", "RCX", "RDX", "R8", "R9", "R10", "R11"];
fn report() -> &'static Json {
    static R: OnceLock<Json> = OnceLock::new();
    R.get_or_init(||serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_loc_text.json")).unwrap())
}
fn id(n: &str) -> Identity {
    0x3_0000_0000 + 0xA00000 + NAMES.iter().position(|&x| x == n).unwrap() as u64 * 0x1000
}
fn reference(v: &Json) -> Identity {
    v.as_str().map_or(0, id)
}
fn snap(v: &Json) -> &Json {
    v["snapshot_sha256"]
        .as_str()
        .map_or(v, |h| &report()["snapshot_blobs"][h])
}
fn regs(v: &Json) -> [u64; 7] {
    VOL.map(|n| v[n].as_u64().unwrap())
}
fn xmm(v: &Json) -> [String; 6] {
    std::array::from_fn(|i| v[i].as_str().unwrap().to_owned())
}
fn method(v: &Json) -> Method {
    serde_json::from_value(v.clone()).unwrap()
}
fn entry(v: &Json) -> Entry {
    Entry {
        method: method(&v["method"]),
        volatile_registers: regs(&v["volatile_registers"]),
        volatile_xmm_hex: xmm(&v["volatile_xmm_hex"]),
    }
}
fn completed(kind: &str, args: &Json) -> Completed {
    match kind {
        "find_locale" => {
            assert_eq!(args[2], 0);
            Completed::FindLocale {
                owner: reference(&args[0]),
                code: reference(&args[1]),
                result: reference(&args[3]),
            }
        }
        "is_null_or_empty" => {
            assert_eq!(args[1], 0);
            Completed::IsNullOrEmpty {
                input: reference(&args[0]),
                result_bits: args[2].as_u64().unwrap(),
            }
        }
        _ => panic!("unexpected kind"),
    }
}
fn parse(v: &Json) -> State {
    let v = snap(v);
    State {
        records: NAMES
            .into_iter()
            .map(|n| {
                let memory = &v["memory"][n];
                let h = memory.as_str().unwrap_or_else(|| {
                    report()["memory_blobs"][memory["memory_sha256"].as_str().unwrap()]
                        .as_str()
                        .unwrap()
                });
                Record {
                    identity: id(n),
                    kind: match n {
                        "owner" | "other_owner" => Kind::Owner,
                        "loc0" | "loc1" => Kind::LocaleLoc,
                        "owner_class" | "locale_class" => Kind::Class,
                        "entries" => Kind::Entries,
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
            .map(entry)
            .collect(),
        service_history: v["service_history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| completed(h["kind"].as_str().unwrap(), &h["args"]))
            .collect(),
    }
}
fn counts(v: &Json) -> BTreeMap<Service, u64> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                match k.as_str() {
                    "find_locale" => Service::FindLocale,
                    "is_null_or_empty" => Service::IsNullOrEmpty,
                    _ => panic!("count"),
                },
                v.as_u64().unwrap(),
            )
        })
        .collect()
}
fn context(rows: &[&Json]) -> Context {
    Context {
        version: CHARACTER_LOC_TEXT_NATIVE_V1.into(),
        state: parse(&rows[0]["initial"]),
        calls: rows
            .iter()
            .map(|r| {
                let mut call = Call {
                    entry: Entry {
                        method: method(&r["method"]),
                        volatile_registers: regs(&r["entry_volatile_registers"]),
                        volatile_xmm_hex: xmm(&r["entry_volatile_xmm_hex"]),
                    },
                    search_result: 0,
                    empty_result_bits: 0,
                };
                for e in r["events"].as_array().unwrap() {
                    match completed(e["kind"].as_str().unwrap(), &e["args"]) {
                        Completed::FindLocale { result, .. } => call.search_result = result,
                        Completed::IsNullOrEmpty { result_bits, .. } => {
                            call.empty_result_bits = result_bits
                        }
                    }
                }
                call
            })
            .collect(),
        service_counts: counts(&rows[0]["service_counts_before"]),
        native_base: 0x180000000,
        volatile_return_registers: [0xFACE123456789090; 7],
        volatile_return_xmm_hex: std::array::from_fn(|i| {
            format!("{:032x}", (1u128 << 127) | i as u128)
        }),
        storage_verified: true,
        inert_services_verified: true,
        normal_completion_verified: true,
    }
}
fn eligible(r: &Json) -> bool {
    r["returned"] == true
        && r["options"]["mutations"].is_null()
        && r["options"]["failure"].is_null()
}
fn matches(rows: &[&Json]) {
    let c = context(rows);
    let unchanged = c.clone();
    let got = replay(&c).unwrap();
    assert_eq!(c, unchanged);
    let mut index = 0;
    for (i, r) in rows.iter().enumerate() {
        assert!(eligible(r));
        for e in r["events"].as_array().unwrap() {
            let step = &got.steps[index];
            index += 1;
            assert_eq!(step.call, i);
            assert_eq!(step.ordinal, e["ordinal"].as_u64().unwrap());
            assert_eq!(
                step.event,
                completed(e["kind"].as_str().unwrap(), &e["args"])
            );
            assert_eq!(step.volatile_registers, regs(&e["volatile_registers"]));
            assert_eq!(step.volatile_xmm_hex, xmm(&e["volatile_xmm_hex"]));
            assert_eq!(
                step.native_site_rva,
                u32::from_str_radix(
                    e["native_site"].as_str().unwrap().trim_start_matches("0x"),
                    16
                )
                .unwrap()
            );
            assert_eq!(
                step.caller_return_bits,
                e["caller_return_bits"].as_u64().unwrap()
            );
            assert!(
                step.state == parse(&e["snapshot"]),
                "complete service-entry state"
            );
            let raw: [u64; 4] = std::array::from_fn(|j| e["raw_args"][j].as_u64().unwrap());
            assert_eq!(
                raw,
                [
                    step.volatile_registers[1],
                    step.volatile_registers[2],
                    step.volatile_registers[3],
                    step.volatile_registers[4]
                ]
            );
        }
        assert_eq!(got.return_bits[i], r["result_bits"].as_u64().unwrap());
        assert_eq!(got.final_registers[i], regs(&r["final_volatile_registers"]));
        assert_eq!(got.final_xmm_hex[i], xmm(&r["final_volatile_xmm_hex"]));
        assert!(
            got.completed[i] == parse(&r["final"]),
            "complete final state"
        );
    }
    assert_eq!(index, got.steps.len());
    assert_eq!(
        got.service_counts,
        counts(&rows.last().unwrap()["service_counts_after"])
    );
    assert!(got.final_state == parse(&rows.last().unwrap()["final"]));
}
#[test]
fn full_native_normal_cases_match_all_registers_bytes_and_history() {
    let rows = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| eligible(r))
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 102);
    for r in rows {
        matches(&[r]);
    }
}
#[test]
fn retained_native_sequences_and_resumed_suffixes_match() {
    let mut full = 0;
    let mut suffix = 0;
    for seq in report()["retained_sequences"].as_array().unwrap() {
        let rows = seq.as_array().unwrap().iter().collect::<Vec<_>>();
        if rows.iter().all(|r| eligible(r)) {
            matches(&rows);
            full += 1;
        } else {
            assert!(rows[1..].iter().all(|r| eligible(r)));
            matches(&rows[1..]);
            suffix += 1;
        }
    }
    assert_eq!((full, suffix), (3, 3));
}
#[test]
fn unsupported_storage_and_provenance_reject_atomically() {
    let c = context(&[&report()["cases"][0]]);
    for variant in 0..10 {
        let mut bad = c.clone();
        match variant {
            0 => bad.inert_services_verified = false,
            1 => bad.calls[0].search_result = id("name"),
            2 => bad.calls[0].entry.volatile_registers[1] = id("loc0"),
            3 => bad.calls[0].entry.volatile_registers[2] = id("entries"),
            4 => bad
                .state
                .records
                .iter_mut()
                .find(|r| r.identity == id("loc0"))
                .unwrap()
                .bytes[0x20..0x28]
                .copy_from_slice(&id("entries").to_le_bytes()),
            5 => bad.state.records[0].identity = u64::MAX - 1,
            6 => bad.state.records[1].identity = bad.state.records[0].identity + 8,
            7 => bad.native_base = u64::MAX,
            8 => bad
                .service_counts
                .insert(Service::FindLocale, u64::MAX)
                .map(|_| ())
                .unwrap_or(()),
            _ => bad.calls[0].entry.volatile_xmm_hex[0] = "g".repeat(32),
        }
        let before = bad.clone();
        assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
        assert_eq!(bad, before);
    }
}
#[test]
fn complete_future_history_and_snapshot_capacity_is_bounded() {
    let c = context(&[&report()["cases"][0]]);
    let mut huge = c.clone();
    huge.calls = vec![c.calls[0].clone(); 17];
    assert_eq!(replay(&huge), Err(LedgerError::Capacity));
    let mut huge = c.clone();
    huge.state.records[0].bytes = vec![0; 8192];
    assert_eq!(replay(&huge), Err(LedgerError::Capacity));
    let mut boundary = 0;
    for n in 1..=16 {
        let mut probe = c.clone();
        probe.calls = vec![c.calls[0].clone(); n];
        match replay(&probe) {
            Ok(_) => boundary = n,
            Err(e) => assert_eq!(e, LedgerError::Capacity),
        }
    }
    assert!(boundary > 0 && boundary < 16);
}
#[test]
fn strict_schema_and_explicit_nullable_finder_contract() {
    let row = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| {
            eligible(r)
                && r["options"]["owner"].is_null()
                && r["entry_volatile_registers"]["RCX"] == 0
                && r["options"]["search_result"].is_null()
        })
        .unwrap();
    matches(&[row]);
    let c = context(&[row]);
    let mut v = serde_json::to_value(&c).unwrap();
    assert_eq!(serde_json::from_value::<Context>(v.clone()).unwrap(), c);
    v["real_locale_lookup"] = json!(true);
    assert!(serde_json::from_value::<Context>(v).is_err());
}
