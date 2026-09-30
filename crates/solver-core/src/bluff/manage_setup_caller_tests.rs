use super::*;
fn id(prefix: &str, value: &Value) -> Option<String> {
    value.as_u64().map(|v| format!("{prefix}{v}"))
}
fn fixture(input: &Value) -> Context {
    let cold = input["cold"].as_bool().unwrap_or(false);
    let values = |name: &str, default: Value| {
        input
            .get(name)
            .cloned()
            .unwrap_or(default)
            .as_array()
            .unwrap()
            .clone()
    };
    let board = values("board", json!([0, 1, 2]))
        .iter()
        .map(|v| id("card", v))
        .collect::<Vec<_>>();
    let roster = values("roster", json!([0, 0, 1]))
        .iter()
        .map(|v| id("data", v))
        .collect();
    let order = values("order", json!([0, 1]))
        .iter()
        .map(|v| id("data", v))
        .collect::<Vec<_>>();
    let mut classes = BTreeMap::new();
    for name in ["Ordinary", "Alchemist", "Poisoner", "Puzzlemaster", "Other"] {
        classes.insert(name.into(), vec![name.into()]);
    }
    let mut roles = ["Ordinary", "Alchemist", "Poisoner", "Puzzlemaster", "Other"]
        .iter()
        .enumerate()
        .map(|(i, name)| (format!("role{i}"), name.to_string()))
        .collect::<BTreeMap<_, _>>();
    if let Some(subclass) = input["subclass"].as_u64() {
        classes.insert(
            "Subclass".into(),
            vec![ALL_MATCH[subclass as usize - 1].into(), "Subclass".into()],
        );
        roles.insert("role0".into(), "Subclass".into());
    }
    let mut data_roles = (0..5)
        .map(|i| (format!("data{i}"), Some(format!("role{i}"))))
        .collect::<BTreeMap<_, _>>();
    data_roles.insert(
        "data0".into(),
        if input["role"] == "null" {
            None
        } else {
            Some(format!("role{}", input["role"].as_u64().unwrap_or(0)))
        },
    );
    let mut c = Context {
        version: MANAGE_SETUP_CALLER_NATIVE_V1.into(),
        pinned_class_hierarchies: true,
        stable_occurrence_services: true,
        reference_equality: true,
        supplied_init_data_write_only: true,
        supplied_gateway_effects_only: true,
        caller_metadata_initialized: !cold,
        shuffle_metadata_initialized: !cold,
        math_initialized: !cold,
        gameplay_initialized: !cold,
        object_initialized: !cold,
        state: State {
            board: if input["null_board"] == true {
                None
            } else {
                Some("board".into())
            },
            order: if input["null_order"] == true {
                None
            } else {
                Some("order".into())
            },
            callback: if input["no_callback"] == true {
                None
            } else {
                Some("callback".into())
            },
            lists: BTreeMap::from([
                (
                    "board".into(),
                    List {
                        count: board.len() as i32,
                        items: board,
                    },
                ),
                (
                    "alternate".into(),
                    List {
                        count: 2,
                        items: vec![Some("card3".into()), Some("card4".into())],
                    },
                ),
            ]),
            arrays: BTreeMap::from([
                (
                    "order".into(),
                    Array {
                        length: order.len() as i32,
                        items: order,
                    },
                ),
                (
                    "other_order".into(),
                    Array {
                        length: 0,
                        items: vec![],
                    },
                ),
            ]),
            identities: (0..5)
                .map(|i| (format!("card{i}"), Some(format!("data{i}"))))
                .collect(),
            data_roles,
            iterator_state: 0xeeeeeeee,
        },
        roster: if input["null_roster"] == true {
            None
        } else {
            Some(roster)
        },
        roles,
        classes,
        callbacks: BTreeSet::from(["callback".into()]),
        failure: None,
        after: vec![],
    };
    if let Some(f) = input["fail"].as_array() {
        c.failure = Some(Failure {
            gateway: serde_json::from_value(f[0].clone()).unwrap(),
            occurrence: f[1].as_u64().unwrap() as u16,
        });
    }
    for a in input["mutations"].as_array().into_iter().flatten() {
        let mutation = match a[2].as_str().unwrap() {
            "board" => Mutation::Board {
                value: if a[3] == "null" {
                    None
                } else {
                    Some("alternate".into())
                },
            },
            "order" => Mutation::Order {
                value: Some("other_order".into()),
            },
            "order_length" => Mutation::ArrayLength {
                array: "order".into(),
                value: a[3].as_i64().unwrap() as i32,
            },
            "role" => Mutation::Role {
                data: "data0".into(),
                role: id("role", &a[3]),
            },
            "identity" => Mutation::Identity {
                card: "card1".into(),
                data: id("data", &a[3]),
            },
            "callback" => Mutation::Callback {
                value: if a[3] == 0 {
                    None
                } else {
                    Some("callback".into())
                },
            },
            _ => panic!("unknown native fixture mutation"),
        };
        c.after.push(AfterGateway {
            gateway: serde_json::from_value(a[0].clone()).unwrap(),
            occurrence: a[1].as_u64().unwrap() as u16,
            mutation,
        });
    }
    c
}
fn project(s: &Snapshot) -> Value {
    json!({"board":s.state.board,"order":s.state.order,"identities":s.state.identities.values().collect::<Vec<_>>(),"effects":s.effects,"iterator_state":s.state.iterator_state})
}
#[test]
fn all_140_native_full_caller_fixtures_match() {
    let report:Value=serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_manage_setup_caller.json")).unwrap();
    assert_eq!(report["cases_passed"], 140);
    for (index, case) in report["cases"].as_array().unwrap().iter().enumerate() {
        let context = fixture(&case["input"]);
        let r = replay(&context).unwrap();
        let events = r
            .events
            .iter()
            .map(|e| {
                let mut value = serde_json::to_value(e).unwrap();
                value["snapshot"] = project(&e.snapshot);
                value
            })
            .collect::<Vec<_>>();
        assert_eq!(
            json!(events),
            case["events"],
            "fixture {index}: {}",
            case["input"]
        );
        assert_eq!(project(&r.final_state), case["final"], "fixture {index}");
        assert_eq!(json!(r.error), case["error"], "fixture {index}");
        assert_eq!(json!(r.returned), case["returned"], "fixture {index}");
    }
}
#[test]
fn signed_count_is_captured_before_math_gateway_and_wraps_exactly() {
    for count in [i32::MIN, -7, -1, 0, 1, i32::MAX] {
        let mut c = fixture(&json!({"cold":true}));
        c.state.lists.get_mut("board").unwrap().count = count;
        c.after.push(AfterGateway {
            gateway: Gateway::ClassInit,
            occurrence: 1,
            mutation: Mutation::ListCount {
                list: "board".into(),
                value: 99,
            },
        });
        let r = replay(&c).unwrap();
        let inits = r
            .events
            .iter()
            .filter(|e| e.kind == Gateway::Init)
            .collect::<Vec<_>>();
        assert_eq!(
            inits[0].arguments["display_id"],
            json!(count.wrapping_abs() as u32)
        );
        assert_eq!(inits[1].arguments["display_id"], 98);
        assert_eq!(inits[2].arguments["display_id"], 97);
    }
}
#[test]
fn publication_argument_captured_before_class_init_but_next_pass_rereads() {
    let mut c = fixture(&json!({"cold":true}));
    c.after.push(AfterGateway {
        gateway: Gateway::ClassInit,
        occurrence: 2,
        mutation: Mutation::Board {
            value: Some("alternate".into()),
        },
    });
    let r = replay(&c).unwrap();
    assert_eq!(
        r.events
            .iter()
            .find(|e| e.kind == Gateway::Publish)
            .unwrap()
            .arguments["source"],
        "board"
    );
    assert_eq!(
        r.events
            .iter()
            .filter(|e| e.kind == Gateway::ActInit)
            .map(|e| e.arguments["card"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["card3", "card4"]
    );
}
#[test]
fn null_publication_is_forwarded_before_next_enumerator_guard() {
    let mut c = fixture(&json!({}));
    c.after.push(AfterGateway {
        gateway: Gateway::Dispose,
        occurrence: 1,
        mutation: Mutation::Board { value: None },
    });
    let r = replay(&c).unwrap();
    assert_eq!(r.error.as_deref(), Some("null"));
    assert!(r
        .events
        .iter()
        .find(|e| e.kind == Gateway::Publish)
        .unwrap()
        .arguments["source"]
        .is_null());
}
#[test]
fn failed_init_has_no_synthetic_data_write() {
    let mut c = fixture(&json!({}));
    c.roster.as_mut().unwrap()[0] = Some("data4".into());
    c.failure = Some(Failure {
        gateway: Gateway::Init,
        occurrence: 1,
    });
    let r = replay(&c).unwrap();
    assert_eq!(
        r.final_state.state.identities["card0"],
        Some("data0".into())
    );
    assert_eq!(
        r.final_state.effects,
        vec![Gateway::Positions, Gateway::Unique, Gateway::Duplicates]
    );
}
#[test]
fn registry_identity_not_role_name_controls_equality() {
    let mut c = fixture(&json!({}));
    c.state
        .data_roles
        .insert("same_class_distinct_asset".into(), Some("role0".into()));
    c.state.arrays.get_mut("order").unwrap().items = vec![Some("same_class_distinct_asset".into())];
    c.state.arrays.get_mut("order").unwrap().length = 1;
    let r = replay(&c).unwrap();
    assert!(!r.events.iter().any(|e| e.kind == Gateway::ActStart));
}
#[test]
fn malformed_registries_provenance_and_bounds_reject() {
    let base = fixture(&json!({}));
    let mut contexts = vec![];
    let mut c = base.clone();
    c.reference_equality = false;
    contexts.push(c);
    let mut c = base.clone();
    c.version = "other".into();
    contexts.push(c);
    let mut c = base.clone();
    c.state.board = Some("missing".into());
    contexts.push(c);
    let mut c = base.clone();
    c.state.identities.insert("board".into(), None);
    contexts.push(c);
    let mut c = base.clone();
    c.roles.insert("role0".into(), "missing".into());
    contexts.push(c);
    let mut c = base.clone();
    c.classes.insert(
        "Child".into(),
        vec!["Ordinary".into(), "Alchemist".into(), "Child".into()],
    );
    contexts.push(c);
    let mut c = base.clone();
    c.state.arrays.get_mut("order").unwrap().length = 3;
    contexts.push(c);
    let mut c = base.clone();
    c.after.push(AfterGateway {
        gateway: Gateway::Init,
        occurrence: 0,
        mutation: Mutation::Board { value: None },
    });
    contexts.push(c);
    let mut c = base.clone();
    c.failure = Some(Failure {
        gateway: Gateway::Init,
        occurrence: 0,
    });
    contexts.push(c);
    for c in contexts {
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    }
    let mut c = base;
    c.state
        .lists
        .get_mut("board")
        .unwrap()
        .items
        .resize(33, None);
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}
#[test]
fn retained_snapshot_budget_rejects_whole_replay() {
    let mut c = fixture(&json!({}));
    c.state.lists.get_mut("board").unwrap().items = vec![Some("card0".into()); 32];
    c.state.lists.get_mut("board").unwrap().count = 32;
    c.roster = Some(vec![Some("data0".into()); 32]);
    c.state
        .data_roles
        .insert("data0".into(), Some("role1".into()));
    c.state.arrays.get_mut("order").unwrap().items = vec![Some("data0".into()); 32];
    c.state.arrays.get_mut("order").unwrap().length = 32;
    for i in 0..30 {
        c.state.lists.insert(
            format!("list{i}"),
            List {
                items: vec![Some("card0".into()); 32],
                count: 32,
            },
        );
    }
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}
