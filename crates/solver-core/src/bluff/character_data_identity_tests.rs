use super::*;
use serde_json::{json, Value as Json};
use std::sync::OnceLock;
fn report() -> &'static Json {
    static R: OnceLock<Json> = OnceLock::new();
    R.get_or_init(||serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_identity.json")).unwrap())
}
fn names() -> Vec<String> {
    [
        "owner",
        "other_owner",
        "data_class",
        "array_class",
        "other_array_class",
        "element_class",
        "other_element_class",
        "int_class",
        "other_int_class",
        "object_class",
        "array",
        "other_array",
        "old_id",
        "other_id",
        "name",
        "other_name",
        "format_literal",
        "other_literal",
        "formatted",
        "other_formatted",
        "exception",
    ]
    .into_iter()
    .map(str::to_owned)
    .chain((0..8).map(|i| format!("box{i}")))
    .chain(["other_box".to_owned(), "stack_window".to_owned()])
    .collect()
}
fn id(n: &str) -> Identity {
    if n == "stack_window" {
        0x400017FA0
    } else {
        0x300000000 + 0x380000 + names().iter().position(|s| s == n).unwrap() as u64 * 0x1000
    }
}
fn reference(v: &Json) -> Option<Identity> {
    v.as_str().map(id)
}
fn required(v: &Json) -> Identity {
    reference(v).unwrap()
}
fn kind(n: &str) -> Kind {
    match n {
        "owner" | "other_owner" => Kind::Data,
        "data_class" => Kind::DataClass,
        "array" | "other_array" => Kind::Array,
        "array_class" | "other_array_class" => Kind::ArrayClass,
        "element_class" | "other_element_class" => Kind::ElementClass,
        "int_class" | "other_int_class" => Kind::IntClass,
        "object_class" => Kind::ObjectClass,
        "exception" => Kind::Exception,
        "stack_window" => Kind::Stack,
        _ if n.starts_with("box") || n == "other_box" => Kind::Box,
        _ => Kind::Text,
    }
}
fn slot(n: &str) -> Slot {
    match n {
        "int_TypeInfo" => Slot::IntType,
        "object[]_TypeInfo" => Slot::ArrayType,
        "format_literal" => Slot::FormatLiteral,
        _ => panic!("slot"),
    }
}
fn service(n: &str) -> Service {
    match n {
        "metadata" => Service::Metadata,
        "is_empty" => Service::IsEmpty,
        "array_allocate" => Service::ArrayAllocate,
        "object_name" => Service::ObjectName,
        "random" => Service::Random,
        "box" => Service::Box,
        "type_check" => Service::TypeCheck,
        "reference_barrier" => Service::ReferenceBarrier,
        "format" => Service::Format,
        "make_exception" => Service::MakeException,
        "raise_exception" => Service::RaiseException,
        "null_exception" => Service::NullException,
        "bounds_exception" => Service::BoundsException,
        _ => panic!("service"),
    }
}
fn completed(s: Service, a: &Json) -> Completed {
    match s {
        Service::Metadata => Completed::Metadata {
            slot: slot(a[0].as_str().unwrap()),
            value: required(&a[1]),
        },
        Service::IsEmpty => Completed::IsEmpty {
            value: reference(&a[0]),
            method: a[1].as_u64().unwrap(),
        },
        Service::ArrayAllocate => Completed::ArrayAllocate {
            class: required(&a[0]),
            length: a[1].as_u64().unwrap(),
            result: reference(&a[2]),
        },
        Service::ObjectName => Completed::ObjectName {
            owner: required(&a[0]),
            method: a[1].as_u64().unwrap(),
            result: reference(&a[2]),
        },
        Service::Random => Completed::Random {
            index: a[0].as_u64().unwrap() as u8,
            min: a[1].as_i64().unwrap() as i32,
            max: a[2].as_i64().unwrap() as i32,
            method: a[3].as_u64().unwrap(),
            return_bits: a[4].as_u64().unwrap(),
        },
        Service::Box => Completed::Box {
            index: a[0].as_u64().unwrap() as u8,
            class: required(&a[1]),
            scratch: a[2].as_u64().unwrap(),
            value_bits: a[3].as_u64().unwrap() as u32,
            result: reference(&a[4]),
        },
        Service::TypeCheck => Completed::TypeCheck {
            index: a[0].as_u64().unwrap() as u8,
            input: required(&a[1]),
            target: required(&a[2]),
            result: reference(&a[3]),
        },
        Service::ReferenceBarrier => Completed::ReferenceBarrier {
            index: a[0].as_u64().unwrap() as u8,
            owner: required(&a[1]),
            offset: a[2].as_u64().unwrap() as usize,
            value: reference(&a[3]),
        },
        Service::Format => Completed::Format {
            literal: required(&a[0]),
            array: required(&a[1]),
            method: a[2].as_u64().unwrap(),
            result: reference(&a[3]),
        },
        Service::MakeException => Completed::MakeException {
            index: a[0].as_u64().unwrap() as u8,
            result: reference(&a[1]),
        },
        Service::RaiseException => Completed::RaiseException {
            index: a[0].as_u64().unwrap() as u8,
            exception: reference(&a[1]),
            method: a[2].as_u64().unwrap(),
        },
        Service::NullException => {
            assert_eq!(a, &json!([]));
            Completed::NullException {}
        }
        Service::BoundsException => {
            assert_eq!(a, &json!([]));
            Completed::BoundsException {}
        }
    }
}
const VOL: [&str; 7] = ["RAX", "RCX", "RDX", "R8", "R9", "R10", "R11"];
fn regs(v: &Json) -> [u64; 7] {
    VOL.map(|n| v[n].as_u64().unwrap())
}
fn xmm(v: &Json) -> [String; 6] {
    std::array::from_fn(|i| v[i].as_str().unwrap().to_owned())
}
fn entry(v: &Json) -> Entry {
    let r = regs(&v["volatile_registers"]);
    assert_eq!(ptr(reference(&v["owner"])), r[1]);
    Entry {
        volatile_registers: r,
        volatile_xmm_hex: xmm(&v["volatile_xmm_hex"]),
    }
}
fn snap(v: &Json) -> &Json {
    v["snapshot_sha256"]
        .as_str()
        .map_or(v, |h| &report()["snapshot_blobs"][h])
}
fn parse(v: &Json) -> State {
    let v = snap(v);
    let raw = &v["supplied_state"];
    State {
        records: names()
            .into_iter()
            .map(|n| {
                let m = &v["memory"][&n];
                let h = m.as_str().unwrap_or_else(|| {
                    report()["memory_blobs"][m["memory_sha256"].as_str().unwrap()]
                        .as_str()
                        .unwrap()
                });
                assert_eq!(h.len() % 2, 0);
                Record {
                    identity: id(&n),
                    kind: kind(&n),
                    bytes: h
                        .as_bytes()
                        .chunks_exact(2)
                        .map(|x| u8::from_str_radix(std::str::from_utf8(x).unwrap(), 16).unwrap())
                        .collect(),
                }
            })
            .collect(),
        metadata_slots: v["metadata_slots"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, v)| (slot(n), required(v)))
            .collect(),
        metadata_flag: v["metadata_flag"].as_u64().unwrap() as u8,
        native_entries: raw["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(entry)
            .collect(),
        service_history: raw
            .as_object()
            .unwrap()
            .iter()
            .filter(|(n, _)| n.as_str() != "entries")
            .map(|(n, a)| {
                let s = service(n);
                (
                    s,
                    a.as_array()
                        .unwrap()
                        .iter()
                        .map(|a| completed(s, a))
                        .collect(),
                )
            })
            .collect(),
    }
}
fn output(o: &Json, k: &str, default: &str) -> Option<Identity> {
    o.get(k).map_or(Some(id(default)), reference)
}
fn context(rows: &[&Json]) -> Context {
    Context {
        version: CHARACTER_DATA_IDENTITY_NATIVE_V1.into(),
        state: parse(&rows[0]["initial"]),
        calls: rows
            .iter()
            .map(|r| {
                let o = &r["options"];
                let name = output(o, "name_result", "name");
                let boxes = std::array::from_fn(|j| {
                    o["box_results"]
                        .get(j)
                        .map_or(Some(id(&format!("box{j}"))), reference)
                });
                Call {
                    entry: Entry {
                        volatile_registers: regs(&r["entry_volatile_registers"]),
                        volatile_xmm_hex: xmm(&r["entry_volatile_xmm_hex"]),
                    },
                    empty_return_bits: o["empty_return_bits"]
                        .as_u64()
                        .unwrap_or(0xAABBCCDDEE000001),
                    array_result: output(o, "array_result", "array"),
                    name_result: name,
                    random_return_bits: std::array::from_fn(|j| {
                        o["random_return_bits"][j]
                            .as_u64()
                            .unwrap_or(0xFFFFFFFF00000000 + j as u64)
                    }),
                    box_results: boxes,
                    type_results: std::array::from_fn(|j| {
                        o["type_results"]
                            .get(j.to_string())
                            .map_or(if j == 0 { name } else { boxes[j - 1] }, reference)
                    }),
                    format_result: output(o, "format_result", "formatted"),
                    barrier_return_bits: o["barrier_return_bits"]
                        .as_u64()
                        .unwrap_or(0xBADD123456789034),
                }
            })
            .collect(),
        native_base: 0x180000000,
        stack_window: id("stack_window"),
        return_sentinel: 0x500000000,
        nonvolatile_registers: std::array::from_fn(|i| 0xFAB0000000000000 + i as u64),
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
    let before = c.clone();
    let got = replay(&c).unwrap();
    assert_eq!(c, before);
    let mut index = 0;
    for (call, r) in rows.iter().enumerate() {
        let mut counts = BTreeMap::new();
        for e in r["events"].as_array().unwrap() {
            let s = service(e["kind"].as_str().unwrap());
            let n = counts.entry(s).or_insert(0);
            *n += 1;
            let step = &got.steps[index];
            index += 1;
            assert_eq!((step.call, step.ordinal), (call, *n));
            assert_eq!(step.event, completed(s, &e["args"]));
            assert_eq!(step.state, parse(&e["snapshot"]));
            assert_eq!(step.volatile_registers, regs(&e["volatile_registers"]));
            assert_eq!(step.volatile_xmm_hex, xmm(&e["volatile_xmm_hex"]));
            assert_eq!(json!(step.raw_args), e["raw_args"]);
            let caller =
                u64::from_str_radix(e["caller"].as_str().unwrap().trim_start_matches("0x"), 16)
                    .unwrap()
                    + c.native_base;
            assert_eq!(step.caller_return_bits, caller);
            assert_eq!(u64::from(step.native_site_rva), caller - c.native_base - 5);
            assert_eq!(step.rsp_bits, e["rsp_bits"].as_u64().unwrap());
            assert_eq!(e["caller_kind"], "native_return");
            assert_eq!(e["native_phase"], "CharacterData.GenerateCharacterId");
        }
        assert_eq!(got.completed[call], parse(&r["final"]));
        assert_eq!(
            got.final_registers[call],
            regs(&r["final_volatile_registers"])
        );
        assert_eq!(got.final_xmm_hex[call], xmm(&r["final_volatile_xmm_hex"]));
        assert_eq!(
            got.final_rsp_bits[call],
            r["final_rsp_bits"].as_u64().unwrap()
        );
        assert_eq!(
            got.final_registers[call][0],
            r["return_bits"].as_u64().unwrap()
        );
    }
    assert_eq!(index, got.steps.len());
    assert_eq!(got.final_state, *got.completed.last().unwrap());
}
#[test]
fn all_46_inert_normal_identity_fixtures_match_complete_storage_abi_and_histories() {
    let rows: Vec<_> = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| eligible(r))
        .collect();
    assert_eq!(rows.len(), 46);
    for r in rows {
        assert_eq!(parse(&r["initial"]).records.len(), 31);
        matches(&[r]);
    }
}
#[test]
fn all_15_inert_retained_rows_and_three_complete_sequences_preserve_prior_state() {
    let mut count = 0;
    let mut whole = 0;
    for s in report()["retained_sequences"].as_array().unwrap() {
        let rows = s.as_array().unwrap();
        for (i, r) in rows.iter().enumerate() {
            if eligible(r) {
                if i > 0 {
                    assert_eq!(snap(&rows[i - 1]["final"]), snap(&r["initial"]));
                }
                matches(&[r]);
                count += 1;
            }
        }
        if rows.iter().all(eligible) {
            matches(&rows.iter().collect::<Vec<_>>());
            whole += 1;
        }
    }
    assert_eq!(count, 15);
    assert_eq!(whole, 3);
}
fn original() -> Context {
    context(&[report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| eligible(r) && r["options"]["empty_return_bits"].as_u64().unwrap() as u8 != 0)
        .unwrap()])
}
fn setq(c: &mut Context, n: &str, off: usize, v: u64) {
    c.state
        .records
        .iter_mut()
        .find(|r| r.identity == id(n))
        .unwrap()
        .bytes[off..off + 8]
        .copy_from_slice(&v.to_le_bytes());
}
#[test]
fn guards_nominal_references_layouts_and_all_calls_reject_atomically() {
    let original = original();
    let mut bad = Vec::new();
    let mut c = original.clone();
    c.version = "wrong".into();
    bad.push(c);
    let mut c = original.clone();
    c.storage_verified = false;
    bad.push(c);
    let mut c = original.clone();
    c.inert_services_verified = false;
    bad.push(c);
    let mut c = original.clone();
    c.normal_completion_verified = false;
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].entry.volatile_registers[1] = 0;
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].array_result = None;
    bad.push(c);
    let mut c = original.clone();
    setq(&mut c, "array", 0x18, 8);
    bad.push(c);
    let mut c = original.clone();
    setq(&mut c, "array", 0, 0);
    bad.push(c);
    let mut c = original.clone();
    setq(&mut c, "array_class", 0x40, id("int_class"));
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].name_result = Some(id("box0"));
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].type_results[8] = None;
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].format_result = Some(id("array"));
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].box_results[0] = Some(0);
    bad.push(c);
    let mut c = original.clone();
    c.state
        .metadata_slots
        .insert(Slot::IntType, id("array_class"));
    bad.push(c);
    let mut c = original.clone();
    c.state.records[0].bytes.pop();
    bad.push(c);
    let mut c = original.clone();
    c.state.records[1].identity = c.state.records[0].identity + 8;
    bad.push(c);
    let mut c = original.clone();
    c.state.records[0].identity = u64::MAX - 8;
    bad.push(c);
    let mut c = original.clone();
    c.native_base = u64::MAX - 8;
    bad.push(c);
    // The old inclusive flag check accepted MAX itself; its exclusive endpoint
    // must also fit. Root and flag windows may not alias any nominal record.
    let mut c = original.clone();
    c.native_base = u64::MAX - 0x288C4DC;
    bad.push(c);
    for slot in SLOTS {
        for offset in [-7i64, 0, 255] {
            let mut c = original.clone();
            c.native_base = (id("int_class") as i64 + offset - i64::from(slot.rva())) as u64;
            bad.push(c);
        }
    }
    for record in ["int_class", "stack_window"] {
        let mut c = original.clone();
        c.native_base = id(record) - 0x288C4DC;
        bad.push(c);
    }
    let mut c = original.clone();
    c.return_sentinel = 0;
    bad.push(c);
    let mut c = original.clone();
    setq(&mut c, "stack_window", 0x68, 0);
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].entry.volatile_xmm_hex[0] = "z".repeat(32);
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].entry.volatile_xmm_hex[0] = "A".repeat(32);
    bad.push(c);
    let mut c = original.clone();
    c.volatile_return_xmm_hex[0] = "F".repeat(32);
    bad.push(c);
    let mut c = original.clone();
    c.state.service_history.remove(&Service::BoundsException);
    bad.push(c);
    let mut c = original.clone();
    c.state
        .service_history
        .get_mut(&Service::Box)
        .unwrap()
        .push(Completed::Box {
            index: 8,
            class: id("int_class"),
            scratch: id("stack_window"),
            value_bits: 0,
            result: None,
        });
    bad.push(c);
    let mut c = original.clone();
    c.calls.push(c.calls[0].clone());
    c.calls[1].array_result = None;
    bad.push(c);
    for c in bad {
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
}
#[test]
fn capacity_bounds_full_future_entries_histories_abi_trace_and_snapshots() {
    let original = original();
    let mut c = original.clone();
    c.calls = vec![c.calls[0].clone(); 9];
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = original.clone();
    c.calls[0].entry.volatile_xmm_hex[0] = "0".repeat(65536);
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut c = original.clone();
    c.state
        .service_history
        .get_mut(&Service::IsEmpty)
        .unwrap()
        .resize(
            513,
            Completed::IsEmpty {
                value: None,
                method: 0,
            },
        );
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut maximum = 0;
    for n in 1..=8 {
        let mut c = original.clone();
        c.calls = vec![c.calls[0].clone(); n];
        if replay(&c).is_ok() {
            maximum = n;
        } else {
            assert_eq!(replay(&c), Err(LedgerError::Capacity));
        }
    }
    assert!(maximum >= 2);
    let mut c = original.clone();
    c.calls = vec![c.calls[0].clone(); maximum];
    c.state.native_entries = vec![c.calls[0].entry.clone(); 64];
    assert!(units(&c).unwrap() <= 65536);
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
}
#[test]
fn strict_nested_serde_and_explicit_volatile_residues_do_not_invent_public_ids() {
    let c = original();
    let mut v = serde_json::to_value(&c).unwrap();
    v["state"]["records"][0]["unknown"] = json!(1);
    assert!(serde_json::from_value::<Context>(v).is_err());
    let mut v = serde_json::to_value(&c).unwrap();
    v["calls"][0]["entry"]["unknown"] = json!(1);
    assert!(serde_json::from_value::<Context>(v).is_err());
    let mut v = serde_json::to_value(&c).unwrap();
    v["calls"][0].as_object_mut().unwrap().remove("name_result");
    assert!(serde_json::from_value::<Context>(v).is_err());
    let mut v = serde_json::to_value(&c).unwrap();
    v["state"]["service_history"]["IsEmpty"] = json!([{"IsEmpty":{"method":0}}]);
    assert!(serde_json::from_value::<Context>(v).is_err());
    let mut v = serde_json::to_value(&c).unwrap();
    v["state"]["service_history"]["IsEmpty"] =
        json!([{"IsEmpty":{"value":null,"method":0,"unknown":true}}]);
    assert!(serde_json::from_value::<Context>(v).is_err());
    let v = serde_json::to_value(&c).unwrap();
    assert_eq!(serde_json::from_value::<Context>(v).unwrap(), c);
    let mut c = c;
    c.volatile_return_registers = [0xABCDEF9876543201; 7];
    c.volatile_return_xmm_hex =
        std::array::from_fn(|i| format!("{:032x}", 0xFEDCBA9876543210u128 + i as u128));
    let got = replay(&c).unwrap();
    assert_eq!(got.final_registers[0][0], c.calls[0].barrier_return_bits);
    assert_eq!(
        &got.final_registers[0][1..],
        &c.volatile_return_registers[1..]
    );
    assert_eq!(got.final_xmm_hex[0], c.volatile_return_xmm_hex);
    let mut skip = c.clone();
    skip.calls[0].empty_return_bits = 0xFFFFFFFFFFFFFF00;
    skip.calls[0].array_result = None;
    let got = replay(&skip).unwrap();
    assert_eq!(got.final_registers[0][0], 0xFFFFFFFFFFFFFF00);
    let b = skip
        .state
        .records
        .iter()
        .find(|r| r.identity == id("owner"))
        .unwrap();
    let a = got
        .final_state
        .records
        .iter()
        .find(|r| r.identity == id("owner"))
        .unwrap();
    assert_eq!(a.bytes, b.bytes);
    assert!(!got
        .steps
        .iter()
        .any(|s| s.event.service() == Service::ArrayAllocate));
}
