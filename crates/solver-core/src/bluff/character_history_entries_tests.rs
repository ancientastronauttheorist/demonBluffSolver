use super::super::character_initialization::Statuses;
use super::*;
use serde_json::{json, Value};
use std::sync::OnceLock;

const ARENA: u64 = 0x2_0000_0000;

fn native_report() -> &'static Value {
    static REPORT: OnceLock<Value> = OnceLock::new();
    REPORT.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_history_entries.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    })
}

fn id(v: &Value) -> Option<u64> {
    if v.is_null() {
        return None;
    }
    Some(
        ARENA
            + match v.as_str().unwrap() {
                "history" => 0x60000,
                "history_array" => 0x61000,
                "hover" => 0xd0000,
                "hover_array" => 0xd1000,
                "data" => 0x2000,
                "register_as" => 0xd2000,
                "prior_info" => 0x70000,
                "info" => 0x71000,
                "old" => 0x72000,
                "UnityEngine.Object_TypeInfo" => 0x73000,
                other => panic!("unknown identity {other}"),
            },
    )
}

fn symbol(p: Option<u64>) -> Value {
    match p {
        None => Value::Null,
        Some(p) => {
            for name in [
                "history",
                "history_array",
                "hover",
                "hover_array",
                "data",
                "register_as",
                "prior_info",
                "info",
                "old",
                "UnityEngine.Object_TypeInfo",
            ] {
                if id(&json!(name)) == Some(p) {
                    return json!(name);
                }
            }
            panic!("unknown pointer {p:x}")
        }
    }
}

fn actor() -> Actor {
    Actor {
        identity: ARENA + 0x1000,
        data: Some(ARENA + 0x2000),
        bluff: Some(103),
        register_as: Some(ARENA + 0xd2000),
        trailer: Some(105),
        runtime: Some(106),
        dead_prefab: None,
        revealed: true,
        uses: -71,
        previous: -101,
        state: 30,
        killed_hidden: true,
        killed_demon: true,
        alignment: -73,
        id: 97,
        started: true,
        role: Some(107),
        bluff_role: Some(108),
        saved_act: Some(ARENA + 0x72000),
        infos: vec![],
        info_version: 0,
        statuses: Statuses {
            active: vec![10, 30],
            version: u32::MAX,
            resistances: vec![40, 60],
            target: Some(109),
        },
        state_callback: Some(110),
    }
}

fn call(case: &Value) -> Call {
    match case["method"].as_str().unwrap() {
        "AddOnHoverInfo" => Call::AddOnHoverInfo {
            info: if case["options"]["null_info"] == true {
                None
            } else {
                Some(ARENA + 0x71000)
            },
        },
        "ClearRecentMemory" => Call::ClearRecentMemory,
        "GetCurrentActedInfo" => Call::GetCurrentActedInfo,
        "GetCharacterType" => Call::GetCharacterType {
            unity_null_return_bits: 0xABC000
                | u64::from(
                    case["options"]["register"] == "absent"
                        || case["options"]["register"] == "destroyed",
                ),
        },
        other => panic!("{other}"),
    }
}

fn from_native(case: &Value) -> Option<Context> {
    let n = &case["initial"];
    let mut a = actor();
    a.data = id(&n["data_ref"]);
    a.register_as = id(&n["register_as_ref"]);
    a.saved_act = id(&n["saved_speech"]);
    let history = id(&n["history_ref"])?;
    let hover = id(&n["hover_ref"])?;
    let mut lists = Vec::new();
    for key in ["history", "hover"] {
        let l = &n[key];
        let backing_array = id(&l["backing"])?;
        let count_bits = l["count_bits"].as_u64().unwrap() as u32;
        let capacity = l["capacity"].as_u64().unwrap() as u32;
        if count_bits > i32::MAX as u32 || count_bits > capacity {
            return None;
        }
        lists.push(ReferenceList {
            identity: id(&l["identity"]).unwrap(),
            backing_array,
            capacity,
            count_bits,
            version: l["version"].as_u64().unwrap() as u32,
            slots: l["slots"].as_array().unwrap().iter().map(id).collect(),
        });
    }
    let h = lists.iter().find(|l| l.identity == history).unwrap();
    a.infos = h.slots[..h.count_bits as usize].to_vec();
    a.info_version = h.version;
    let mut metadata_flags = [0; 14];
    for (i, rva) in METADATA_FLAG_RVAS.iter().enumerate() {
        metadata_flags[i] = n["metadata_flags"][format!("{rva:#x}")].as_u64().unwrap() as u8;
    }
    let result = Context {
        version: CHARACTER_HISTORY_ENTRIES_NATIVE_V1.into(),
        state: State {
            actor: a,
            history,
            hover,
            lists,
            assets: vec![
                DataAsset {
                    identity: ARENA + 0x2000,
                    type_bits: n["data_type_bits"].as_u64().unwrap() as u32,
                    live: true,
                },
                DataAsset {
                    identity: ARENA + 0xd2000,
                    type_bits: n["register_type_bits"].as_u64().unwrap() as u32,
                    live: case["options"]["register"] != "destroyed",
                },
            ],
            metadata_flags,
            object_class_initialized: n["object_class_initialized"].as_u64().unwrap() as u32,
        },
        object_class: ARENA + 0x73000,
        info_identities: vec![ARENA + 0x70000, ARENA + 0x71000],
        calls: vec![call(case)],
        services: Services {
            runtime_verified_inert: true,
            gc_verified_inert: true,
            unity_liveness_verified_inert: true,
            storage_verified: true,
            normal_completion_verified: true,
        },
    };
    if matches!(&result.calls[0], Call::AddOnHoverInfo { .. }) {
        let l = result
            .state
            .lists
            .iter()
            .find(|l| l.identity == hover)
            .unwrap();
        if l.count_bits >= l.capacity {
            return None;
        }
    }
    Some(result)
}

fn snapshot(s: &State) -> Value {
    let list = |name: &str| {
        let identity = id(&json!(name)).unwrap();
        let l = s.lists.iter().find(|l| l.identity == identity).unwrap();
        json!({"identity": name, "backing": symbol(Some(l.backing_array)), "count_bits": l.count_bits,
               "version": l.version, "capacity": l.capacity,
               "slots": l.slots.iter().map(|p| symbol(*p)).collect::<Vec<_>>()})
    };
    let flags: serde_json::Map<_, _> = METADATA_FLAG_RVAS
        .iter()
        .enumerate()
        .map(|(i, rva)| (format!("{rva:#x}"), json!(s.metadata_flags[i])))
        .collect();
    json!({"history_ref": symbol(Some(s.history)), "hover_ref": symbol(Some(s.hover)),
           "history": list("history"), "hover": list("hover"),
           "data_ref": symbol(s.actor.data), "register_as_ref": symbol(s.actor.register_as),
           "data_type_bits": s.assets[0].type_bits, "register_type_bits": s.assets[1].type_bits,
           "object_class_initialized": s.object_class_initialized,
           "saved_speech": symbol(s.actor.saved_act), "metadata_flags": flags})
}

fn event(e: &Event) -> Value {
    match e {
        Event::MetadataService { slot_rva } => {
            json!({"kind": "metadata_service", "args": [slot_rva]})
        }
        Event::ClassInitializationService { class } => {
            json!({"kind": "class_initialization_service", "args": [symbol(Some(*class))]})
        }
        Event::RegisterUnityNullService { object } => {
            json!({"kind": "register_unity_null_service", "args": [symbol(*object)]})
        }
        Event::BarrierService { address, value } => {
            json!({"kind": "barrier_service", "args": [address - ARENA, symbol(*value)]})
        }
    }
}

fn compare(c: &Context, r: &Replay, case: &Value, previous_native_events: usize) {
    assert_eq!(
        snapshot(&r.state),
        case["final"],
        "{} {:?}",
        case["method"],
        case["options"]
    );
    let mut untouched = r.state.actor.clone();
    untouched.infos = c.state.actor.infos.clone();
    untouched.info_version = c.state.actor.info_version;
    assert_eq!(untouched, c.state.actor, "unrelated Actor fields changed");
    let h = r
        .state
        .lists
        .iter()
        .find(|l| l.identity == r.state.history)
        .unwrap();
    assert_eq!(r.state.actor.infos, h.slots[..h.count_bits as usize]);
    assert_eq!(r.state.actor.info_version, h.version);
    let events = &case["events"].as_array().unwrap()[previous_native_events..];
    assert_eq!(r.steps.len(), events.len());
    for (step, native) in r.steps.iter().zip(events) {
        let expected = json!({"kind": native["kind"], "args": native["args"]});
        assert_eq!(event(&step.event), expected);
        assert_eq!(snapshot(&step.state), native["snapshot"]);
    }
    match &r.calls[0].result {
        Return::Void => assert!(case["result"].is_null()),
        Return::Info { identity } => assert_eq!(symbol(*identity), case["result"]),
        Return::Type { bits } => assert_eq!(json!(bits), case["result"]),
    }
}

#[test]
fn matches_supported_normal_native_cases_and_retained_alias_sequences() {
    let report = native_report();
    assert_eq!(report["case_count"], 116);
    let mut compared = 0;
    for case in report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report["failure_baselines"].as_array().unwrap())
    {
        if case["returned"] != true {
            continue;
        }
        let Some(c) = from_native(case) else {
            continue;
        };
        compare(&c, &replay(&c).unwrap(), case, 0);
        compared += 1;
    }
    assert_eq!(compared, 93);
    for sequence in report["retained_sequences"].as_array().unwrap() {
        let native = sequence["calls"].as_array().unwrap();
        let mut c = from_native(&native[0]).unwrap();
        let mut previous = 0;
        for case in native {
            c.calls = vec![call(case)];
            let r = replay(&c).unwrap();
            compare(&c, &r, case, previous);
            previous = case["events"].as_array().unwrap().len();
            c.state = r.state;
        }
        // A retained batch must agree with the same supplied native call order.
        let mut batch = from_native(&native[0]).unwrap();
        batch.calls = native.iter().map(call).collect();
        let r = replay(&batch).unwrap();
        for (result, case) in r.calls.iter().zip(native) {
            assert_eq!(snapshot(&result.state), case["final"]);
        }
    }
}

fn context() -> Context {
    from_native(&native_report()["failure_baselines"][0]).unwrap()
}

#[test]
fn alias_null_pointer_wrapping_and_barrier_order_are_physical() {
    let mut c = context();
    c.state.hover = c.state.history;
    c.state.lists[0].version = u32::MAX;
    c.state.actor.info_version = u32::MAX;
    c.calls = vec![
        Call::AddOnHoverInfo { info: None },
        Call::GetCurrentActedInfo,
        Call::ClearRecentMemory,
    ];
    let r = replay(&c).unwrap();
    assert_eq!(r.calls[1].result, Return::Info { identity: None });
    assert_eq!(r.calls[0].state.actor.info_version, 0);
    assert_eq!(r.state.actor.info_version, 1);
    assert_eq!(r.state.lists[1], c.state.lists[1]);
    let barriers: Vec<_> = r
        .steps
        .iter()
        .filter(|s| matches!(s.event, Event::BarrierService { .. }))
        .collect();
    assert_eq!(barriers[0].state.actor.info_version, 0);
    assert_eq!(barriers[1].state.actor.info_version, 0);
    assert_eq!(barriers[1].state.actor.infos.len(), 1);
    assert_eq!(r.state.actor.saved_act, c.state.actor.saved_act);
}

#[test]
fn type_result_uses_low_byte_and_preserves_native_dwords_and_warm_bytes() {
    let mut c = context();
    c.state.metadata_flags = [0x80; 14];
    c.state.object_class_initialized = 0xdead_beef;
    c.state.assets[1].type_bits = 0x8000_0000;
    c.calls = vec![Call::GetCharacterType {
        unity_null_return_bits: 0xffff_ffff_ffff_ff00,
    }];
    let r = replay(&c).unwrap();
    assert_eq!(r.calls[0].result, Return::Type { bits: 0x8000_0000 });
    assert_eq!(r.state.metadata_flags, [0x80; 14]);
    assert_eq!(r.state.object_class_initialized, 0xdead_beef);
    assert_eq!(r.steps.len(), 1);
    c.state.assets[1].live = false;
    c.state.assets[0].type_bits = u32::MAX;
    c.calls = vec![Call::GetCharacterType {
        unity_null_return_bits: 0x1234_ffff_ffff_ff80,
    }];
    assert_eq!(
        replay(&c).unwrap().calls[0].result,
        Return::Type { bits: u32::MAX }
    );
}

#[test]
fn rejects_invalid_storage_typed_aliases_growth_and_unverified_services() {
    let base = context();
    let mutations: Vec<Box<dyn Fn(&mut Context)>> = vec![
        Box::new(|c| c.services.runtime_verified_inert = false),
        Box::new(|c| c.services.gc_verified_inert = false),
        Box::new(|c| c.services.unity_liveness_verified_inert = false),
        Box::new(|c| c.services.storage_verified = false),
        Box::new(|c| c.services.normal_completion_verified = false),
        Box::new(|c| c.version = "unverified".into()),
        Box::new(|c| c.state.lists[1].backing_array = c.state.lists[0].backing_array),
        Box::new(|c| c.state.lists[1].identity = c.state.lists[0].identity),
        Box::new(|c| c.state.lists[0].count_bits = u32::MAX),
        Box::new(|c| c.state.lists[1].capacity = 1),
        Box::new(|c| c.state.lists[0].capacity = 99),
        Box::new(|c| c.state.lists[0].backing_array = 0),
        Box::new(|c| c.state.lists[0].backing_array = u64::MAX - 8),
        Box::new(|c| c.state.lists[0].slots[7] = Some(0)),
        Box::new(|c| c.state.lists[0].slots[7] = Some(321)),
        Box::new(|c| c.info_identities[0] = c.state.actor.identity),
        Box::new(|c| c.info_identities[0] = c.state.assets[0].identity),
        Box::new(|c| c.state.actor.saved_act = Some(c.info_identities[0])),
        Box::new(|c| c.state.actor.runtime = Some(c.state.assets[0].identity)),
        Box::new(|c| c.state.actor.bluff = Some(c.info_identities[0])),
        Box::new(|c| c.state.actor.trailer = Some(c.info_identities[0])),
        Box::new(|c| c.state.actor.trailer = c.state.actor.data),
        Box::new(|c| c.state.actor.role = Some(c.state.actor.identity)),
        Box::new(|c| c.state.actor.infos.clear()),
        Box::new(|c| c.state.actor.info_version = 71),
        Box::new(|c| c.object_class = c.state.lists[0].identity),
        Box::new(|c| c.state.history = c.state.history as u32 as u64),
        Box::new(|c| {
            c.state.actor.register_as = c.state.actor.register_as.map(|p| p as u32 as u64)
        }),
        Box::new(|c| c.calls = vec![Call::AddOnHoverInfo { info: Some(321) }]),
        Box::new(|c| {
            c.calls = vec![Call::GetCharacterType {
                unity_null_return_bits: 1,
            }]
        }),
    ];
    for mutate in mutations {
        let mut c = base.clone();
        mutate(&mut c);
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    }
    let mut c = base;
    c.calls = vec![Call::ClearRecentMemory, Call::GetCurrentActedInfo];
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
}

#[test]
fn bounds_aggregate_retained_snapshot_work_and_rejects_unknown_claims() {
    let mut c = context();
    c.calls = vec![Call::ClearRecentMemory; 32];
    c.state.actor.statuses.active = vec![10; 3000];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut v = serde_json::to_value(context()).unwrap();
    v["real_runtime"] = json!(true);
    assert!(serde_json::from_value::<Context>(v).is_err());
}

#[test]
fn same_asset_and_self_status_target_aliases_remain_legal() {
    let mut c = context();
    c.state.actor.register_as = c.state.actor.data;
    c.state.actor.bluff = c.state.actor.data;
    c.state.actor.statuses.target = Some(c.state.actor.identity);
    c.calls = vec![Call::GetCharacterType {
        unity_null_return_bits: 0xABC000,
    }];
    assert_eq!(
        replay(&c).unwrap().calls[0].result,
        Return::Type { bits: 10 }
    );
}
