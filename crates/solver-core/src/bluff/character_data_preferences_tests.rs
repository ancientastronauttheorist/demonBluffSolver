use super::*;
use serde_json::{json, Value as Json};
use std::sync::OnceLock;
const NAMES: [&str; 30] = [
    "owner",
    "other_owner",
    "data_class",
    "saved0",
    "saved1",
    "list0",
    "list1",
    "array0",
    "array1",
    "pref0",
    "pref1",
    "pref2",
    "target_id",
    "other_id",
    "skin_id0",
    "skin_id1",
    "skin0",
    "skin1",
    "old_skin",
    "string_class",
    "exception",
    "clone_enum",
    "mi0",
    "mi1",
    "mi2",
    "mi3",
    "other_mi0",
    "other_mi1",
    "other_mi2",
    "other_mi3",
];
const VOL: [&str; 7] = ["RAX", "RCX", "RDX", "R8", "R9", "R10", "R11"];
const BASE: u64 = 0x180000000;
fn report() -> &'static Json {
    static R: OnceLock<Json> = OnceLock::new();
    R.get_or_init(||serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_preferences.json")).unwrap())
}
fn id(n: &str) -> Identity {
    match n {
        "scratch" => 0x400000000 + 0x18008 - 0x68 + 0x20,
        "enum_output" => id("scratch") + 8,
        "enum_state" => id("scratch") + 32,
        _ => 0x300000000 + 0xA50000 + NAMES.iter().position(|x| *x == n).unwrap() as u64 * 0x1000,
    }
}
fn reference(v: &Json) -> Identity {
    v.as_str().map_or(0, id)
}
fn slot(n: &str) -> Slot {
    match n {
        "Method$System.Collections.Generic.List.Enumerator<CharacterPreference>.Dispose()" => {
            Slot::Dispose
        }
        "Method$System.Collections.Generic.List.Enumerator<CharacterPreference>.MoveNext()" => {
            Slot::MoveNext
        }
        "Method$System.Collections.Generic.List.Enumerator<CharacterPreference>.get_Current()" => {
            Slot::Current
        }
        "Method$System.Collections.Generic.List<CharacterPreference>.GetEnumerator()" => {
            Slot::GetEnumerator
        }
        _ => panic!("slot {n}"),
    }
}
fn snap(v: &Json) -> &Json {
    v["snapshot_sha256"]
        .as_str()
        .map_or(v, |h| &report()["snapshot_blobs"][h])
}
fn hex(v: &str) -> Vec<u8> {
    v.as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
fn regs(v: &Json) -> [u64; 7] {
    VOL.map(|n| v[n].as_u64().unwrap())
}
fn xmm(v: &Json) -> [String; 6] {
    std::array::from_fn(|i| v[i].as_str().unwrap().to_owned())
}
fn entry(v: &Json) -> Entry {
    Entry {
        owner: reference(&v["owner"]),
        synthetic_cleanup: v["synthetic_cleanup"].as_bool().unwrap(),
        volatile_registers: regs(&v["volatile_registers"]),
        volatile_xmm_hex: xmm(&v["volatile_xmm_hex"]),
    }
}
fn event(kind: &str, a: &Json) -> Completed {
    match kind {
        "metadata" => Completed::Metadata {
            slot: slot(a[0].as_str().unwrap()),
            value: reference(&a[1]),
        },
        "reference_barrier" => {
            assert_eq!(a[1], 0xC0);
            assert!(a[2].is_null());
            Completed::ReferenceBarrier {
                owner: reference(&a[0]),
            }
        }
        "character_preferences" => {
            assert_eq!(a[0], 0);
            Completed::CharacterPreferences {
                result: reference(&a[1]),
            }
        }
        "get_enumerator" => Completed::GetEnumerator {
            output: reference(&a[0]),
            list: reference(&a[1]),
            method: reference(&a[2]),
            bytes: hex(a[3].as_str().unwrap()).try_into().unwrap(),
            entries: a[4].as_array().unwrap().iter().map(reference).collect(),
        },
        "move_next" => Completed::MoveNext {
            owner: reference(&a[0]),
            method: reference(&a[1]),
            current: reference(&a[2]),
            result_bits: a[3].as_u64().unwrap(),
        },
        "string_equality" => {
            assert_eq!(a[2], 0);
            Completed::StringEquality {
                a: reference(&a[0]),
                b: reference(&a[1]),
                result_bits: a[3].as_u64().unwrap(),
            }
        }
        "load_skin" => {
            assert_eq!(a[2], 0);
            Completed::LoadSkin {
                owner: reference(&a[0]),
                skin_id: reference(&a[1]),
            }
        }
        "dispose" => Completed::Dispose {
            owner: reference(&a[0]),
            method: reference(&a[1]),
        },
        _ => panic!("unsupported event {kind}"),
    }
}
fn parse(v: &Json) -> State {
    let v = snap(v);
    let supplied = &v["supplied_state"];
    State {
        records: NAMES
            .into_iter()
            .chain(["scratch"])
            .map(|n| {
                let raw = &v["memory"][n];
                let h = raw.as_str().unwrap_or_else(|| {
                    report()["memory_blobs"][raw["memory_sha256"].as_str().unwrap()]
                        .as_str()
                        .unwrap()
                });
                let kind = match n {
                    "owner" | "other_owner" => Kind::Owner,
                    "data_class" => Kind::DataClass,
                    "saved0" | "saved1" => Kind::SavedCharacters,
                    "list0" | "list1" => Kind::List,
                    "array0" | "array1" => Kind::Array,
                    "pref0" | "pref1" | "pref2" => Kind::Preference,
                    "string_class" => Kind::StringClass,
                    "skin0" | "skin1" | "old_skin" => Kind::Skin,
                    "exception" => Kind::Exception,
                    "clone_enum" => Kind::Enumerator,
                    "scratch" => Kind::Scratch,
                    "mi0" | "other_mi0" => Kind::MethodInfo(Slot::Dispose),
                    "mi1" | "other_mi1" => Kind::MethodInfo(Slot::MoveNext),
                    "mi2" | "other_mi2" => Kind::MethodInfo(Slot::Current),
                    "mi3" | "other_mi3" => Kind::MethodInfo(Slot::GetEnumerator),
                    _ => Kind::String,
                };
                Record {
                    identity: id(n),
                    kind,
                    bytes: hex(h),
                }
            })
            .collect(),
        metadata_slots: v["metadata_slots"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, v)| (slot(n), reference(v)))
            .collect(),
        metadata_flag: v["metadata_flag"].as_u64().unwrap() as u8,
        native_entries: supplied["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(entry)
            .collect(),
        service_history: supplied["history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| event(v["kind"].as_str().unwrap(), &v["args"]))
            .collect(),
        iterator: (!supplied["iterator"].is_null()).then(|| {
            let it = &supplied["iterator"];
            IteratorState {
                list: reference(&it["list"]),
                entries: it["entries"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(reference)
                    .collect(),
                cursor: it["cursor"].as_u64().unwrap() as u32,
            }
        }),
    }
}
fn context(rows: &[&Json]) -> Context {
    Context {
        version: CHARACTER_DATA_PREFERENCES_NATIVE_V1.into(),
        state: parse(&rows[0]["initial"]),
        calls: rows
            .iter()
            .map(|r| {
                let o = &r["options"];
                let rr = regs(&r["entry_volatile_registers"]);
                let mut c = Call {
                    entry: Entry {
                        owner: rr[1],
                        synthetic_cleanup: false,
                        volatile_registers: rr,
                        volatile_xmm_hex: xmm(&r["entry_volatile_xmm_hex"]),
                    },
                    saved_result: 0,
                    enumerator_bytes: [0; 24],
                    enumerator_entries: Vec::new(),
                    moves: Vec::new(),
                    equality_results: Vec::new(),
                    barrier_return_bits: o["barrier_return_bits"]
                        .as_u64()
                        .unwrap_or(0xBADD123456789056),
                    get_return_bits: o["get_return_bits"].as_u64().unwrap_or(id("enum_output")),
                    load_skin_return_bits: o["load_skin_return_bits"]
                        .as_u64()
                        .unwrap_or(0x10AD123456789098),
                    dispose_return_bits: o["dispose_return_bits"]
                        .as_u64()
                        .unwrap_or(0xD15E1234567890AB),
                };
                for e in r["events"].as_array().unwrap() {
                    match event(e["kind"].as_str().unwrap(), &e["args"]) {
                        Completed::CharacterPreferences { result } => c.saved_result = result,
                        Completed::GetEnumerator { bytes, entries, .. } => {
                            c.enumerator_bytes = bytes;
                            c.enumerator_entries = entries
                        }
                        Completed::MoveNext {
                            current,
                            result_bits,
                            ..
                        } => c.moves.push(MoveOutput {
                            current,
                            result_bits,
                        }),
                        Completed::StringEquality { result_bits, .. } => {
                            c.equality_results.push(result_bits)
                        }
                        _ => (),
                    }
                }
                c
            })
            .collect(),
        native_base: BASE,
        scratch: id("scratch"),
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
        && r["synthetic_cleanup"] == false
        && r["options"]["mutations"].is_null()
        && r["options"]["failure"].is_null()
}
fn matches(rows: &[&Json]) {
    assert!(rows.iter().all(|r| eligible(r)));
    let c = context(rows);
    let before = c.clone();
    let out = replay(&c).unwrap();
    assert_eq!(c, before);
    let mut index = 0;
    for (i, r) in rows.iter().enumerate() {
        for e in r["events"].as_array().unwrap() {
            let step = &out.steps[index];
            index += 1;
            assert_eq!(step.call, i);
            assert_eq!(step.ordinal, e["ordinal"].as_u64().unwrap() as usize);
            assert_eq!(step.event, event(e["kind"].as_str().unwrap(), &e["args"]));
            assert_eq!(step.state, parse(&e["snapshot"]));
            assert_eq!(step.volatile_registers, regs(&e["volatile_registers"]));
            assert_eq!(step.volatile_xmm_hex, xmm(&e["volatile_xmm_hex"]));
            assert_eq!(
                step.caller_return_bits,
                e["caller_return_bits"].as_u64().unwrap()
            );
            assert_eq!(
                BASE + u64::from(step.native_site_rva) + 5,
                step.caller_return_bits
            );
            assert_eq!(
                json!([
                    step.volatile_registers[1],
                    step.volatile_registers[2],
                    step.volatile_registers[3],
                    step.volatile_registers[4]
                ]),
                e["raw_args"]
            );
            assert_eq!(e["synthetic_cleanup"], false);
        }
        assert_eq!(out.completed[i], parse(&r["final"]));
        assert_eq!(out.final_registers[i], regs(&r["final_volatile_registers"]));
        assert_eq!(out.final_xmm_hex[i], xmm(&r["final_volatile_xmm_hex"]));
        assert_eq!(
            out.final_registers[i][0],
            r["return_bits"].as_u64().unwrap()
        );
    }
    assert_eq!(index, out.steps.len());
    assert_eq!(out.final_state, *out.completed.last().unwrap());
}
fn normal_row() -> &'static Json {
    report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| {
            eligible(r)
                && r["options"]["entries"]
                    .as_array()
                    .is_some_and(|v| v.len() == 3)
                && r["options"]["warm_byte"] == 0
        })
        .unwrap()
}
#[test]
fn all_84_normal_inert_native_cases_match_full_snapshots_history_and_abi() {
    let rows: Vec<_> = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| eligible(r))
        .collect();
    assert_eq!(rows.len(), 84);
    for r in rows {
        matches(&[r]);
    }
}
#[test]
fn all_eight_supported_sequence_rows_and_cold_recovery_suffix_match() {
    let mut count = 0;
    let mut cold_pair = 0;
    for sequence in report()["retained_sequences"].as_array().unwrap() {
        let rows = sequence.as_array().unwrap();
        for r in rows.iter().filter(|r| eligible(r)) {
            matches(&[r]);
            count += 1;
        }
        for pair in rows.windows(2) {
            if eligible(&pair[0]) && eligible(&pair[1]) {
                assert_eq!(parse(&pair[0]["final"]), parse(&pair[1]["initial"]));
                matches(&[&pair[0], &pair[1]]);
                cold_pair += 1;
                assert!(!parse(&pair[0]["initial"]).native_entries.is_empty());
                assert!(!parse(&pair[0]["initial"]).service_history.is_empty());
            }
        }
    }
    assert_eq!((count, cold_pair), (8, 1));
}
#[test]
fn supplied_low_bytes_duplicate_matches_and_void_residues_are_independent() {
    let mut c = context(&[normal_row()]);
    c.calls[0]
        .moves
        .iter_mut()
        .take(3)
        .for_each(|m| m.result_bits = 0xFFFFFFFFFFFFFF80);
    c.calls[0].equality_results = vec![0x123456789ABCDEFF, 0xFFFFFFFFFFFFFF00, 0x8000000000000080];
    c.calls[0].get_return_bits = 0;
    c.calls[0].dispose_return_bits = 0x0123456789ABCDEF;
    c.volatile_return_registers = [0x1020304050607080; 7];
    c.volatile_return_xmm_hex =
        std::array::from_fn(|i| format!("{:032x}", (1u128 << 100) | i as u128));
    let out = replay(&c).unwrap();
    let loads: Vec<_> = out
        .steps
        .iter()
        .filter_map(|s| {
            if let Completed::LoadSkin { skin_id, .. } = s.event {
                Some(skin_id)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(loads, vec![id("skin_id0"), id("skin_id1")]);
    assert_eq!(out.final_registers[0][0], 0x0123456789ABCDEF);
    assert_eq!(out.final_xmm_hex[0], c.volatile_return_xmm_hex);
    assert_eq!(q(&out.final_state, id("owner"), 0xC0), 0);
    assert_eq!(out.final_state.iterator.as_ref().unwrap().cursor, 4);
    let move_step = out
        .steps
        .iter()
        .find(|s| s.event.service() == Service::MoveNext)
        .unwrap();
    let bytes = &c.calls[0].enumerator_bytes;
    assert_eq!(
        move_step.volatile_xmm_hex[0],
        format!(
            "{:032x}",
            u128::from_le_bytes(bytes[..16].try_into().unwrap())
        )
    );
    assert_eq!(
        move_step.volatile_xmm_hex[1],
        format!(
            "{:032x}",
            u64::from_le_bytes(bytes[16..].try_into().unwrap())
        )
    );
    let mut early = c.clone();
    early.calls[0].moves.truncate(1);
    early.calls[0].moves[0].result_bits = 0xFFFFFFFFFFFFFF00;
    early.calls[0].equality_results.clear();
    let early = replay(&early).unwrap();
    assert!(!early
        .steps
        .iter()
        .any(|s| s.event.service() == Service::StringEquality));
    assert_eq!(early.final_state.iterator.as_ref().unwrap().cursor, 1);
}
#[test]
fn strict_nominal_guard_failures_are_atomic() {
    let good = context(&[normal_row()]);
    for i in 0..15 {
        let mut bad = good.clone();
        match i {
            0 => bad.inert_services_verified = false,
            1 => bad.calls[0].entry.owner = id("pref0"),
            2 => bad.calls[0].saved_result = id("list0"),
            3 => {
                bad.state
                    .metadata_slots
                    .insert(Slot::GetEnumerator, id("mi0"));
            }
            4 => bad.calls[0].moves[0].current = 0,
            5 => bad.calls[0].moves.last_mut().unwrap().result_bits = 1,
            6 => bad.calls[0].enumerator_bytes[..8].copy_from_slice(&id("skin0").to_le_bytes()),
            7 => bad.scratch = id("clone_enum"),
            8 => bad.native_base = u64::MAX,
            9 => bad.calls[0].entry.synthetic_cleanup = true,
            10 => bad.calls[0].entry.volatile_xmm_hex[0] = "g".repeat(32),
            11 => bad.state.records[1].identity = bad.state.records[0].identity + 8,
            12 => bad.state.records[0].identity = u64::MAX - 1,
            13 => bad
                .state
                .records
                .iter_mut()
                .find(|r| r.identity == id("pref0"))
                .unwrap()
                .bytes[0x10..0x18]
                .copy_from_slice(&id("skin0").to_le_bytes()),
            _ => bad.state.service_history.push(Completed::Dispose {
                owner: id("clone_enum"),
                method: id("mi0"),
            }),
        }
        let before = bad.clone();
        assert_eq!(
            replay(&bad),
            Err(LedgerError::InvalidContext),
            "variant {i}"
        );
        assert_eq!(bad, before);
    }
}
#[test]
fn future_complete_entries_histories_and_trace_snapshots_are_budgeted() {
    let good = context(&[normal_row()]);
    let mut boundary = good.clone();
    boundary.state.native_entries = vec![good.calls[0].entry.clone(); 60];
    boundary.calls = vec![good.calls[0].clone(); 8];
    let (_, work) = budget(&boundary).unwrap();
    let events = 8 + boundary.calls[0].moves.len() + boundary.calls[0].equality_results.len() * 2;
    let snapshots = 2 + 8 * (events + 1);
    let added_entries = 8 * entry_units(&good.calls[0].entry).unwrap();
    assert!(work > 8_388_608);
    assert!(
        work - added_entries * snapshots <= 8_388_608,
        "complete future entries make this cross the limit"
    );
    let before = boundary.clone();
    assert_eq!(replay(&boundary), Err(LedgerError::Capacity));
    assert_eq!(boundary, before);
    let mut smaller = boundary.clone();
    smaller.calls.truncate(1);
    assert!(replay(&smaller).is_ok());
    let mut huge = good.clone();
    huge.calls[0].entry.volatile_xmm_hex[0] = "0".repeat(65537);
    let before = huge.clone();
    assert_eq!(replay(&huge), Err(LedgerError::Capacity));
    assert_eq!(huge, before);
    let mut huge = good;
    huge.calls = vec![huge.calls[0].clone(); 9];
    assert_eq!(replay(&huge), Err(LedgerError::Capacity));
}
#[test]
fn strict_schema_rejects_hidden_services_or_unknown_fields() {
    let c = context(&[normal_row()]);
    let value = serde_json::to_value(&c).unwrap();
    assert_eq!(serde_json::from_value::<Context>(value.clone()).unwrap(), c);
    let mut bad = value.clone();
    bad["execute_real_load_skin"] = json!(true);
    assert!(serde_json::from_value::<Context>(bad).is_err());
    let mut bad = value;
    bad["calls"][0]["callback"] = json!("write");
    assert!(serde_json::from_value::<Context>(bad).is_err());
}
