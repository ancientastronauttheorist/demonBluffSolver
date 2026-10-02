use super::*;
use serde_json::{json, Value as Json};
use std::sync::OnceLock;
fn report() -> &'static Json {
    static R: OnceLock<Json> = OnceLock::new();
    R.get_or_init(||serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_data_constructor.json")).unwrap())
}
fn names() -> Vec<String> {
    let mut names: Vec<String> = ["owner", "other_owner", "data_class", "old_skin", "new_skin"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    names.extend((0..6).map(|i| format!("list{i}")));
    names.extend((0..6).map(|i| format!("prior{i}")));
    for kind in [
        "AchievementData",
        "CharacterData",
        "ECharacterStatus",
        "ECharacterTag",
        "SkinData",
    ] {
        names.extend(
            ["class", "method", "other_class", "other_method"]
                .map(|category| format!("{kind}:{category}")),
        );
    }
    names
}
fn id(n: &str) -> Identity {
    0x3_0000_0000 + 0x280000 + names().iter().position(|s| s == n).unwrap() as u64 * 0x1000
}
fn reference(v: &Json) -> Identity {
    v.as_str().map_or(0, id)
}
fn generic(n: &str) -> Generic {
    match n {
        "CharacterData" => Generic::CharacterData,
        "SkinData" => Generic::SkinData,
        "AchievementData" => Generic::AchievementData,
        "ECharacterStatus" => Generic::ECharacterStatus,
        "ECharacterTag" => Generic::ECharacterTag,
        _ => panic!("generic"),
    }
}
fn kind(n: &str) -> Kind {
    match n {
        "owner" | "other_owner" => Kind::Data,
        "data_class" => Kind::DataClass,
        "old_skin" | "new_skin" => Kind::Skin,
        _ => {
            if let Some((g, c)) = n.split_once(':') {
                if c.ends_with("class") {
                    Kind::TypeInfo(generic(g))
                } else {
                    Kind::MethodInfo(generic(g))
                }
            } else {
                let i = n.chars().last().unwrap().to_digit(10).unwrap() as usize;
                Kind::List(GENERICS[i])
            }
        }
    }
}
fn slot(n: &str) -> Slot {
    let ctor = n.starts_with("Method$");
    let name = n.split('<').nth(1).unwrap().split('>').next().unwrap();
    match (generic(name), ctor) {
        (Generic::CharacterData, false) => Slot::CharacterDataType,
        (Generic::CharacterData, true) => Slot::CharacterDataCtor,
        (Generic::SkinData, false) => Slot::SkinDataType,
        (Generic::SkinData, true) => Slot::SkinDataCtor,
        (Generic::AchievementData, false) => Slot::AchievementDataType,
        (Generic::AchievementData, true) => Slot::AchievementDataCtor,
        (Generic::ECharacterStatus, false) => Slot::StatusType,
        (Generic::ECharacterStatus, true) => Slot::StatusCtor,
        (Generic::ECharacterTag, false) => Slot::TagType,
        (Generic::ECharacterTag, true) => Slot::TagCtor,
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
    assert_eq!(
        v["owner"].as_str().map_or(0, id),
        v["volatile_registers"]["RCX"].as_u64().unwrap()
    );
    Entry {
        volatile_registers: regs(&v["volatile_registers"]),
        volatile_xmm_hex: xmm(&v["volatile_xmm_hex"]),
    }
}
fn completed(service: Service, a: &Json) -> Completed {
    match service {
        Service::Metadata => Completed::Metadata {
            slot: slot(a[0].as_str().unwrap()),
            value: reference(&a[1]),
        },
        Service::Allocation => Completed::Allocation {
            index: a[0].as_u64().unwrap() as u8,
            class: reference(&a[1]),
            result: reference(&a[2]),
        },
        Service::ListConstructor => Completed::ListConstructor {
            index: a[0].as_u64().unwrap() as u8,
            owner: reference(&a[1]),
            method: reference(&a[2]),
        },
        Service::ReferenceBarrier => {
            let index = a[0].as_u64().unwrap() as usize;
            assert_eq!(a[3], OFFSETS[index]);
            assert_eq!(
                a[2],
                [
                    "bundledCharacters",
                    "skins",
                    "achievements",
                    "additionalStatuses",
                    "tags",
                    "canAppearIf"
                ][index]
            );
            Completed::ReferenceBarrier {
                index: index as u8,
                owner: reference(&a[1]),
                value: reference(&a[4]),
            }
        }
        Service::BaseConstructor => {
            assert_eq!(a[1], 0);
            Completed::BaseConstructor {
                owner: reference(&a[0]),
            }
        }
    }
}
fn service(n: &str) -> Service {
    match n {
        "metadata" => Service::Metadata,
        "allocation" => Service::Allocation,
        "list_constructor" => Service::ListConstructor,
        "reference_barrier" => Service::ReferenceBarrier,
        "base_constructor" => Service::BaseConstructor,
        _ => panic!("service"),
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
                Record {
                    identity: id(&n),
                    kind: kind(&n),
                    bytes: h
                        .as_bytes()
                        .chunks_exact(2)
                        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
                        .collect(),
                }
            })
            .collect(),
        metadata_slots: v["metadata_slots"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, p)| (slot(n), reference(p)))
            .collect(),
        metadata_flag: v["metadata_flag"].as_u64().unwrap() as u8,
        native_entries: raw["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(entry)
            .collect(),
        service_history: [
            (Service::Metadata, "metadata"),
            (Service::Allocation, "allocations"),
            (Service::ListConstructor, "list_constructors"),
            (Service::ReferenceBarrier, "barriers"),
            (Service::BaseConstructor, "base_constructors"),
        ]
        .map(|(s, n)| {
            (
                s,
                raw[n]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|a| completed(s, a))
                    .collect(),
            )
        })
        .into_iter()
        .collect(),
    }
}
fn context(rows: &[&Json]) -> Context {
    Context {
        version: CHARACTER_DATA_CONSTRUCTOR_NATIVE_V1.into(),
        state: parse(&rows[0]["initial"]),
        calls: rows
            .iter()
            .map(|r| {
                let o = &r["options"];
                let allocations = std::array::from_fn(|i| {
                    o["allocation_records"][i]
                        .as_str()
                        .map_or_else(|| id(&format!("list{i}")), id)
                });
                Call {
                    entry: Entry {
                        volatile_registers: regs(&r["entry_volatile_registers"]),
                        volatile_xmm_hex: xmm(&r["entry_volatile_xmm_hex"]),
                    },
                    allocation_results: allocations,
                    list_ctor_return_bits: o["list_ctor_return_bits"]
                        .as_u64()
                        .unwrap_or(0xF00D123456789012),
                    barrier_return_bits: o["barrier_return_bits"]
                        .as_u64()
                        .unwrap_or(0xBADD123456789034),
                    base_return_bits: o["base_return_bits"].as_u64().unwrap_or(0xABCD123456789056),
                }
            })
            .collect(),
        native_base: 0x180000000,
        return_sentinel: 0x500000000,
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
        let mut ordinals = BTreeMap::new();
        for e in r["events"].as_array().unwrap() {
            let s = service(e["kind"].as_str().unwrap());
            let ordinal = ordinals.entry(s).or_insert(0);
            *ordinal += 1;
            let step = &got.steps[index];
            index += 1;
            assert_eq!(step.call, call);
            assert_eq!(step.ordinal, *ordinal);
            assert_eq!(step.event, completed(s, &e["args"]));
            assert_eq!(step.state, parse(&e["snapshot"]));
            assert_eq!(step.volatile_registers, regs(&e["volatile_registers"]));
            assert_eq!(step.volatile_xmm_hex, xmm(&e["volatile_xmm_hex"]));
            let raw: [u64; 4] = [
                step.volatile_registers[1],
                step.volatile_registers[2],
                step.volatile_registers[3],
                step.volatile_registers[4],
            ];
            assert_eq!(json!(raw), e["raw_args"]);
            let caller =
                u64::from_str_radix(e["caller"].as_str().unwrap().trim_start_matches("0x"), 16)
                    .unwrap()
                    + c.native_base;
            assert_eq!(step.caller_return_bits, caller);
            assert_eq!(
                step.native_site_rva,
                s.ne(&Service::BaseConstructor)
                    .then_some((caller - c.native_base - 5) as u32)
            );
            assert_eq!(
                e["caller_kind"],
                if s == Service::BaseConstructor {
                    "fixture_return_sentinel"
                } else {
                    "native_return"
                }
            );
            assert_eq!(e["native_phase"], "CharacterData.ctor");
        }
        assert_eq!(got.completed[call], parse(&r["final"]));
        assert_eq!(
            got.final_registers[call],
            regs(&r["final_volatile_registers"])
        );
        assert_eq!(got.final_xmm_hex[call], xmm(&r["final_volatile_xmm_hex"]));
        assert_eq!(
            got.final_registers[call][0],
            r["return_bits"].as_u64().unwrap()
        );
    }
    assert_eq!(index, got.steps.len());
    assert_eq!(got.final_state, *got.completed.last().unwrap());
}
#[test]
fn all_66_normal_inert_constructor_fixtures_match_complete_storage_and_abi() {
    let rows: Vec<_> = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| eligible(r))
        .collect();
    assert_eq!(rows.len(), 66);
    for row in rows {
        matches(&[row]);
    }
}
#[test]
fn normal_recovery_preserves_complete_failed_prefix_history_and_aliases() {
    let mut resumed = 0;
    for seq in report()["retained_sequences"].as_array().unwrap() {
        let seq = seq.as_array().unwrap();
        if eligible(&seq[2]) {
            assert_eq!(snap(&seq[1]["final"]), snap(&seq[2]["initial"]));
            assert!(!parse(&seq[2]["initial"]).native_entries.is_empty());
            matches(&[&seq[2]]);
            resumed += 1;
        }
    }
    assert_eq!(resumed, 3);
    let r = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| eligible(r))
        .unwrap();
    let mut c = context(&[r]);
    let call = c.calls[0].clone();
    c.calls.push(call);
    let got = replay(&c).unwrap();
    assert_eq!(got.completed[0].metadata_flag, 1);
    assert_eq!(
        got.steps
            .iter()
            .filter(|s| s.call == 1 && s.event.service() == Service::Metadata)
            .count(),
        0
    );
    assert_eq!(got.final_state.native_entries.len(), 2);
    assert_eq!(
        got.final_state.service_history[&Service::Allocation].len(),
        12
    );
}
#[test]
fn invalid_contexts_reject_atomically_before_cloning_or_partial_stores() {
    let r = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| eligible(r))
        .unwrap();
    let original = context(&[r]);
    let mut bad = Vec::new();
    let mut c = original.clone();
    c.version = "wrong".into();
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
    c.calls[0].allocation_results[1] = id("list0");
    bad.push(c);
    let mut c = original.clone();
    c.state
        .metadata_slots
        .insert(Slot::CharacterDataCtor, id("SkinData:method"));
    bad.push(c);
    let mut c = original.clone();
    c.state.records[0].bytes.pop();
    bad.push(c);
    let mut c = original.clone();
    c.state.records[1].identity = c.state.records[0].identity + 16;
    bad.push(c);
    let mut c = original.clone();
    c.state.records[0].identity = u64::MAX - 16;
    bad.push(c);
    let mut c = original.clone();
    c.native_base = u64::MAX - 10;
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].entry.volatile_xmm_hex[0] = "z".repeat(32);
    bad.push(c);
    let mut c = original.clone();
    c.state
        .service_history
        .get_mut(&Service::Allocation)
        .unwrap()
        .push(Completed::Allocation {
            index: 6,
            class: id("CharacterData:class"),
            result: id("list0"),
        });
    bad.push(c);
    for c in bad {
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
}
#[test]
fn capacity_accounts_for_all_retained_registers_history_and_future_snapshots() {
    let r = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| eligible(r))
        .unwrap();
    let original = context(&[r]);
    let mut c = original.clone();
    c.calls = vec![c.calls[0].clone(); 9];
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = original.clone();
    c.calls[0].entry.volatile_xmm_hex[5] = "0".repeat(65536);
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut c = original.clone();
    c.state
        .service_history
        .get_mut(&Service::BaseConstructor)
        .unwrap()
        .resize(513, Completed::BaseConstructor { owner: id("owner") });
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut c = original.clone();
    c.calls = vec![c.calls[0].clone(); 8];
    assert!(replay(&c).is_ok());
    let mut expanded = c.clone();
    let entry = expanded.calls[0].entry.clone();
    expanded.state.native_entries = vec![entry; 64];
    let units = units(&expanded).unwrap();
    assert!(units <= 65536);
    assert!((units + 8 * 374) * (8 * 30 + 2) > 4_194_304);
    assert_eq!(replay(&expanded), Err(LedgerError::Capacity));
}
#[test]
fn strict_serialization_and_boolean_stores_preserve_opaque_defaults() {
    let r = report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| eligible(r) && r["options"]["seed_byte"] == 165)
        .unwrap();
    let c = context(&[r]);
    let got = replay(&c).unwrap();
    let owner = id("owner");
    let before = c
        .state
        .records
        .iter()
        .find(|r| r.identity == owner)
        .unwrap();
    let after = got
        .final_state
        .records
        .iter()
        .find(|r| r.identity == owner)
        .unwrap();
    assert_eq!(after.bytes[0x13C], 1);
    assert_eq!(after.bytes[0x13E], 1);
    assert_eq!(after.bytes[0x13D], before.bytes[0x13D]);
    assert_eq!(&after.bytes[0xC0..0xC8], &before.bytes[0xC0..0xC8]);
    let mut value = serde_json::to_value(&c).unwrap();
    value["state"]["unexpected"] = json!(true);
    assert!(serde_json::from_value::<Context>(value).is_err());
    let mut value = serde_json::to_value(&c).unwrap();
    value["calls"][0]["entry"]["unexpected"] = json!(true);
    assert!(serde_json::from_value::<Context>(value).is_err());
    let value = serde_json::to_value(&c).unwrap();
    assert_eq!(serde_json::from_value::<Context>(value).unwrap(), c);
}
