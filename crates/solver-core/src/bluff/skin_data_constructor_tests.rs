use super::*;
use std::sync::OnceLock;
fn report() -> &'static Value {
    static R: OnceLock<Value> = OnceLock::new();
    R.get_or_init(||{
        let raw:Value=serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_skin_data_constructor.json")).unwrap();
        fn expand(v:&Value,r:&Value)->Value {
            if let Some(o)=v.as_object() {
                if o.len()==1 {
                    for (key,pool) in [("snapshot_sha256","snapshot_blobs"),("state_map_sha256","state_map_blobs"),("memory_map_sha256","memory_map_blobs"),("history_sha256","history_blobs"),("memory_sha256","memory_blobs")] {
                        if let Some(hash)=o.get(key).and_then(Value::as_str) {return expand(&r[pool][hash],r)}
                    }
                }
                return Value::Object(o.iter().map(|(k,v)|(k.clone(),expand(v,r))).collect())
            }
            if let Some(a)=v.as_array(){return Value::Array(a.iter().map(|v|expand(v,r)).collect())}v.clone()
        }
        expand(&raw,&raw)
    })
}
fn bytes(v: &Value) -> Vec<u8> {
    v.as_str()
        .unwrap()
        .as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
fn state(v: &Value) -> State {
    State {
        records: WINDOWS
            .iter()
            .map(|n| {
                (
                    (*n).into(),
                    Record {
                        identity: report()["physical_layout"][*n]["pointer"].as_u64().unwrap(),
                        bytes: bytes(&v["memory"][*n]),
                    },
                )
            })
            .collect(),
        native_entries: v["native_entries"].as_array().unwrap().clone(),
        requests: v["requests"].as_array().unwrap().clone(),
    }
}
fn context(rows: &[&Value]) -> Context {
    Context {
        version: SKIN_DATA_CONSTRUCTOR_NATIVE_V1.into(),
        state: state(&rows[0]["initial"]),
        calls: rows
            .iter()
            .map(|r| {
                let o = &r["options"];
                Call {
                    registers: serde_json::from_value(r["entry_registers"].clone()).unwrap(),
                    return_bits: o["return_bits"].as_u64().unwrap_or(0xD00D1234567890AB),
                    volatile_poison: o["volatile_poison"].as_u64().unwrap_or(0xFACE123456789090),
                    stop_base: o["stop_base"].as_bool().unwrap_or(false),
                    base_writes: o["base_writes"].as_array().map_or_else(Vec::new, |ws| {
                        ws.iter()
                            .map(|w| Write {
                                window: w[0].as_str().unwrap().into(),
                                offset: w[1].as_u64().unwrap() as usize,
                                bytes: bytes(&w[2]),
                            })
                            .collect()
                    }),
                }
            })
            .collect(),
        native_base: 0x180000000,
        return_sentinel: 0x500000000,
        entry_sp: 0x400018008,
        storage_verified: true,
        supplied_base_verified: true,
    }
}
fn matches(rows: &[&Value]) {
    let c = context(rows);
    let before = c.clone();
    let got = replay(&c).unwrap();
    assert_eq!(c, before);
    for (i, r) in rows.iter().enumerate() {
        let event = &r["events"][0];
        let step = &got.steps[i];
        assert_eq!(step.call, i);
        assert_eq!(step.state, state(&event["snapshot"]));
        let mut expected = event.clone();
        expected.as_object_mut().unwrap().remove("snapshot");
        assert_eq!(step.event, expected);
        assert_eq!(got.completed[i], state(&r["final"]));
        assert_eq!(
            serde_json::to_value(&got.final_registers[i]).unwrap(),
            r["final_registers"]
        );
        assert_eq!(step.event["volatile_registers"]["RDX"], 0);
        assert_eq!(step.event["caller_return_bits"], 0x500000000u64);
        assert_eq!(step.event["entry_sp"], c.entry_sp);
    }
    assert_eq!(got.final_state, *got.completed.last().unwrap());
}
#[test]
fn all_32_normal_native_profiles_match_complete_physical_state_and_abi() {
    let mut n = 0;
    for r in report()["cases"].as_array().unwrap() {
        if r["returned"] == true {
            matches(&[r]);
            n += 1;
        }
    }
    assert_eq!(n, 32);
}
#[test]
fn all_eight_retained_rows_match_or_reject_and_recovery_preserves_history() {
    let mut normal = 0;
    let mut stopped = 0;
    for seq in report()["sequences"].as_array().unwrap() {
        let a = seq.as_array().unwrap();
        assert_eq!(a[0]["final"], a[1]["initial"]);
        for r in a {
            if r["returned"] == true {
                matches(&[r]);
                normal += 1;
            } else {
                let c = context(&[r]);
                assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
                stopped += 1;
            }
        }
        if a.iter().all(|r| r["returned"] == true) {
            matches(&[&a[0], &a[1]]);
        } else {
            let c = context(&[&a[0], &a[1]]);
            assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        }
    }
    assert_eq!((normal, stopped), (7, 1));
}
#[test]
fn every_supplied_stop_rejects_atomically() {
    for r in report()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["returned"] == false)
        .chain(
            report()["stops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| &s["result"]),
        )
    {
        let c = context(&[r]);
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
}
#[test]
fn nominal_extents_metadata_abi_and_write_permissions_reject_before_replay() {
    let original = context(&[&report()["cases"][0]]);
    let mut bad = Vec::new();
    let mut c = original.clone();
    c.version = "other".into();
    bad.push(c);
    let mut c = original.clone();
    c.supplied_base_verified = false;
    bad.push(c);
    let mut c = original.clone();
    c.calls[0]
        .registers
        .insert("RCX".into(), json!(c.state.records["skin_class"].identity));
    bad.push(c);
    let mut c = original.clone();
    c.calls[0]
        .registers
        .insert("XMM0".into(), json!("Z".repeat(32)));
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].registers.insert("extra".into(), json!(0));
    bad.push(c);
    let mut c = original.clone();
    c.state.records.get_mut("skin0").unwrap().bytes.pop();
    bad.push(c);
    let mut c = original.clone();
    c.state.records.get_mut("art").unwrap().identity = c.state.records["animated"].identity + 1;
    bad.push(c);
    let mut c = original.clone();
    c.state.records.get_mut("skin0").unwrap().bytes[0..8].copy_from_slice(&0u64.to_le_bytes());
    bad.push(c);
    let mut c = original.clone();
    c.calls[0].base_writes.push(Write {
        window: "skin0".into(),
        offset: 0x18,
        bytes: vec![0; 8],
    });
    bad.push(c);
    let mut c = original.clone();
    c.entry_sp = u64::MAX - 7;
    bad.push(c);
    let mut c = original.clone();
    c.native_base = u64::MAX;
    bad.push(c);
    let mut c = context(&[&report()["sequences"][0][1]]);
    c.state.native_entries[0]["unexpected"] = json!(true);
    bad.push(c);
    let mut c = context(&[&report()["sequences"][0][1]]);
    c.state.requests[0]["raw_args"][1] = json!(1);
    bad.push(c);
    for c in bad {
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
}
#[test]
fn future_capacity_and_strict_serialization_are_bounded() {
    let original = context(&[&report()["cases"][0]]);
    let mut c = original.clone();
    c.calls = vec![c.calls[0].clone(); 9];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut c = original.clone();
    c.calls[0]
        .registers
        .insert("XMM0".into(), json!("0".repeat(131072)));
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut c = original.clone();
    c.calls = vec![c.calls[0].clone(); 8];
    assert!(units(&c).is_some());
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    c.calls.pop();
    assert!(replay(&c).is_ok());
    let mut c = context(&[&report()["sequences"][0][1]]);
    c.calls = vec![c.calls[0].clone(); 8];
    c.state.native_entries = vec![c.state.native_entries[0].clone(); 64];
    c.state.requests = vec![c.state.requests[0].clone(); 64];
    assert!(units(&c).is_none());
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = original.clone();
    c.state
        .native_entries
        .push(json!({"oversized": "x".repeat(131072)}));
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = original.clone();
    let mut deep = json!(0);
    for _ in 0..40 {
        deep = Value::Array(vec![deep]);
    }
    c.state.requests.push(deep);
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = original.clone();
    c.state.native_entries = vec![json!(null); 64];
    assert!(units(&c).is_some());
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut v = serde_json::to_value(&original).unwrap();
    v["unexpected"] = json!(true);
    assert!(serde_json::from_value::<Context>(v).is_err());
    let mut v = serde_json::to_value(&original).unwrap();
    v["calls"][0]["unexpected"] = json!(true);
    assert!(serde_json::from_value::<Context>(v).is_err());
    assert_eq!(
        serde_json::from_value::<Context>(serde_json::to_value(&original).unwrap()).unwrap(),
        original
    );
}
