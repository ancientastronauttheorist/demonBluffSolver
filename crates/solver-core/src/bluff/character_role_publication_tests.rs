use super::super::character_initialization::Statuses;
use super::*;
use serde_json::{json, Value};

fn context() -> Context {
    Context {
        version: CHARACTER_ROLE_PUBLICATION_NATIVE_V1.into(),
        actor: Actor {
            identity: 1,
            data: Some(20),
            bluff: None,
            register_as: None,
            trailer: None,
            runtime: None,
            dead_prefab: None,
            revealed: false,
            uses: 1,
            previous: 5,
            state: 10,
            killed_hidden: false,
            killed_demon: false,
            alignment: 10,
            id: 2,
            started: true,
            role: None,
            bluff_role: None,
            saved_act: Some(103),
            infos: vec![Some(201)],
            info_version: 9,
            statuses: Statuses {
                active: vec![],
                version: 17,
                resistances: vec![],
                target: None,
            },
            state_callback: None,
        },
        act: true,
        raw_bluff: None,
        data_assets: vec![
            DataAsset {
                identity: 20,
                picking: false,
            },
            DataAsset {
                identity: 30,
                picking: true,
            },
        ],
        history: HistoryStorage {
            identity: 2,
            backing_array: 3,
            capacity: 8,
        },
        strings: [
            (100, "authored clue"),
            (101, "authored replacement"),
            (102, ""),
            (103, "old speech"),
            (104, "prior clue"),
            (105, "Fixture"),
            (106, "Character: Fixture"),
            (107, "authored trailer"),
        ]
        .into_iter()
        .map(|(identity, text)| ManagedString {
            identity,
            units: text.encode_utf16().collect(),
        })
        .collect(),
        infos: vec![
            ActedInfo {
                identity: 200,
                description: Some(100),
                references: Some(300),
            },
            ActedInfo {
                identity: 201,
                description: Some(104),
                references: Some(300),
            },
        ],
        reference_lists: vec![ReferenceList {
            identity: 300,
            backing_array: None,
            version: 42,
            entries: vec![],
        }],
        result_iterators: vec![ResultIterator {
            identity: 500,
            actor: 1,
            info: Some(200),
            trigger_bits: 30,
            delay_bits: RESULT_WAIT_BITS,
            state: 1,
            current: 501,
        }],
        result_waits: vec![WaitObject {
            identity: 501,
            seconds_bits: RESULT_WAIT_BITS,
        }],
        allocations: vec![Allocation {
            result_iterator: 500,
            speech_iterator: 600,
            speech_wait: Some(601),
        }],
        preappend_callback: None,
        info_revealed_callback: Some(700),
        trailer_mode: false,
        trailer_text: None,
        ui: Ui {
            acted_component: 400,
            acted_version: 401,
            blank_text: 402,
            blank_text_value: Some(103),
            layout_array: 403,
            layouts: vec![404, 404],
            first_game_object: 410,
            log_game_object: Some(410),
            show_game_object: 410,
            name_text: 105,
            log_text: 106,
            pickable: 411,
            objects: vec![
                UiObject {
                    identity: 410,
                    active: false,
                },
                UiObject {
                    identity: 411,
                    active: true,
                },
            ],
        },
        result_resume_order: vec![500],
        speech_resume_order: vec![600, 600],
        services: Services {
            runtime_and_metadata_verified: true,
            callback_captures_and_first_yield_verified: true,
            list_storage_and_capacity_verified: true,
            preappend_callbacks_inert: true,
            global_callbacks_inert: true,
            ui_and_other_services_inert: true,
            unity_liveness_verified: true,
            trailer_lookup_stable_verified: true,
            supplied_resume_order_verified: true,
            normal_completion_verified: true,
        },
    }
}

fn id(label: &Value) -> Option<u64> {
    match label.as_str() {
        None => None,
        Some("original") => Some(100),
        Some("replacement") => Some(101),
        Some("empty") => Some(102),
        Some("old") => Some(103),
        Some("prior") => Some(104),
        Some("trailer") => Some(107),
        Some("info") => Some(200),
        Some("prior_info") => Some(201),
        Some("references") => Some(300),
        Some(other) => panic!("unsupported native fixture label {other}"),
    }
}

fn plan(c: &mut Context) {
    let waiting = speech_waits(c);
    c.allocations.clear();
    c.speech_resume_order.clear();
    for (index, result) in c.result_resume_order.iter().enumerate() {
        let r = c
            .result_iterators
            .iter()
            .find(|r| r.identity == *result)
            .unwrap();
        if !publishable(c, r) {
            continue;
        }
        let speech = 600 + index as u64 * 10;
        c.allocations.push(Allocation {
            result_iterator: *result,
            speech_iterator: speech,
            speech_wait: waiting.then_some(speech + 1),
        });
        c.speech_resume_order.push(speech);
        if waiting {
            c.speech_resume_order.push(speech);
        }
    }
}

fn native_input(native: &Value) -> Context {
    let mut c = context();
    let options = &native["options"];
    c.actor.uses = options["uses"].as_i64().unwrap_or(1) as i32;
    c.actor.state = options["state"].as_i64().unwrap_or(10) as i32;
    c.actor.revealed = options["revealed"] == true;
    c.data_assets[0].picking = options["picking"] == true;
    c.data_assets[1].picking = options["bluff_picking"] != false;
    if options["bluff"] == true {
        c.actor.bluff = Some(30);
        c.raw_bluff = Some(ObjectReference {
            identity: 30,
            live: options["destroyed_bluff"] != true,
        });
    }
    c.act = options["act"] != false;
    c.result_iterators[0].trigger_bits = options["trigger"].as_u64().unwrap_or(30) as u32;
    if options["null_info"] == true {
        c.result_iterators[0].info = None;
    }
    if options["null_description"] == true {
        c.infos[0].description = None;
    }
    if options["empty_description"] == true {
        c.infos[0].description = Some(102);
    }
    if options["no_event"] == true {
        c.info_revealed_callback = None;
    }
    if options["null_game_call"] == 2 {
        c.ui.log_game_object = None;
    }
    c.trailer_mode = options["trailer"] == true;
    if c.trailer_mode {
        c.trailer_text = if options["null_trailer_text"] == true {
            None
        } else if options["empty_trailer"] == true {
            Some(102)
        } else {
            Some(107)
        };
    }
    plan(&mut c);
    c
}

fn api(event: &Event) -> Option<&'static str> {
    match event {
        Event::Preappend { .. } => Some("preappend_delegate_service"),
        Event::Barrier { .. } => Some("barrier_service"),
        Event::InfoRevealed { .. } => Some("info_revealed_delegate_service"),
        Event::AllocateSpeech { .. } | Event::AllocateWait { .. } => Some("allocate_service"),
        Event::RegisterSpeech { .. } => Some("speech_registration_service"),
        Event::TrailerLookup { .. } => Some("trailer_lookup_service"),
        Event::GameObject { .. } => Some("game_object_service"),
        Event::Name { .. } => Some("name_service"),
        Event::ConcatLog { .. } => Some("concat_service"),
        Event::Log { .. } => Some("log_service"),
        Event::SetText { .. } => Some("text_setter_service"),
        Event::UnityNull { .. } => Some("unity_null_service"),
        Event::ConstructWait { .. } => Some("wait_constructor_service"),
        Event::SetActive { .. } => Some("set_active_service"),
        Event::Show { .. } => Some("show_version_service"),
        Event::Rebuild { .. } => Some("layout_service"),
        _ => None,
    }
}

fn assert_native(c: &Context, native: &Value) -> Replay {
    let out = replay(c).unwrap();
    let final_state = &native["final"];
    assert_eq!(
        out.context.actor.infos,
        final_state["history"]
            .as_array()
            .unwrap()
            .iter()
            .map(id)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        out.context.actor.info_version as u64,
        final_state["history_version"].as_u64().unwrap()
    );
    assert_eq!(
        out.context.actor.uses as u32 as u64,
        final_state["uses_bits"].as_u64().unwrap()
    );
    assert_eq!(
        out.context.actor.saved_act,
        id(&final_state["saved_speech"])
    );
    assert_eq!(out.context.ui.blank_text_value, id(&final_state["text"]));
    assert_eq!(out.context.infos, c.infos);
    assert_eq!(out.context.reference_lists, c.reference_lists);
    assert_eq!(out.context.actor.statuses, c.actor.statuses);
    assert_eq!(out.context.strings, c.strings);
    let mut expected_actor = c.actor.clone();
    expected_actor.infos = out.context.actor.infos.clone();
    expected_actor.info_version = out.context.actor.info_version;
    expected_actor.uses = out.context.actor.uses;
    expected_actor.saved_act = out.context.actor.saved_act;
    assert_eq!(out.context.actor, expected_actor);
    assert!(out
        .context
        .result_iterators
        .iter()
        .all(|r| r.state == -1 && r.delay_bits == 0));
    let native_speech = final_state["speech_iterators"].as_array().unwrap();
    assert_eq!(out.speech_iterators.len(), native_speech.len());
    for (speech, native) in out.speech_iterators.iter().zip(native_speech) {
        assert_eq!(
            speech.state as u32 as u64,
            native["state"].as_u64().unwrap()
        );
        assert_eq!(Some(speech.description), id(&native["description"]));
        assert_eq!(speech.current.is_some(), !native["current"].is_null());
    }
    let returned: Vec<_> = out
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Return { iterator, value }
                if out.speech_iterators.iter().any(|s| s.identity == *iterator) =>
            {
                Some(u64::from(*value))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        returned,
        final_state["speech_resumes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["result"].as_u64().unwrap())
            .collect::<Vec<_>>()
    );
    let shown: Vec<_> = out
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Show { text, .. } => Some(Some(*text)),
            _ => None,
        })
        .collect();
    assert_eq!(
        shown,
        final_state["shown"]
            .as_array()
            .unwrap()
            .iter()
            .map(id)
            .collect::<Vec<_>>()
    );
    for (name, identity) in [("game", 410), ("pickable", 411)] {
        if let Some(value) = final_state["active"][name].as_bool() {
            assert_eq!(
                out.context
                    .ui
                    .objects
                    .iter()
                    .find(|o| o.identity == identity)
                    .unwrap()
                    .active,
                value
            );
        }
    }
    if let Some(events) = native["events"].as_array() {
        let expected: Vec<_> = events
            .iter()
            .filter(|e| {
                e["snapshot"]["completed_first_steps"]
                    .as_array()
                    .unwrap()
                    .len()
                    == c.result_iterators.len()
            })
            .map(|e| e["kind"].as_str().unwrap())
            .filter(|kind| !["metadata_service", "class_initialization_service"].contains(kind))
            .collect();
        let actual: Vec<_> = out.events.iter().filter_map(api).collect();
        assert_eq!(actual, expected);
    }
    out
}

#[test]
fn matches_normal_inert_native_publication_corpus() {
    let report: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_role_publication.json"))).unwrap();
    assert_eq!(report["case_count"], 158);
    assert_eq!(report["failure_case_count"], 196);
    assert_eq!(report["speech_wait_bits"], SPEECH_WAIT_BITS);
    let mut checked = 0;
    for native in report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report["failure_baselines"].as_array().unwrap())
    {
        let options = &native["options"];
        if native["returned"] != true
            || options["about"] == true
            || options["event_mutation"] == true
            || options["capacity"] == 1
        {
            continue;
        }
        assert_native(&native_input(native), native);
        checked += 1;
    }
    assert!(checked >= 75, "only {checked} eligible fixtures checked");
    for sequence in report["explicit_order_sequences"].as_array().unwrap() {
        let native = &sequence["publication"];
        let mut c = context();
        c.actor.uses = 2;
        c.result_iterators[0].trigger_bits = 3;
        c.result_iterators.push(ResultIterator {
            identity: 510,
            actor: 1,
            info: Some(201),
            trigger_bits: 30,
            delay_bits: 0,
            state: 1,
            current: 511,
        });
        c.result_waits.push(WaitObject {
            identity: 511,
            seconds_bits: 0,
        });
        c.result_resume_order = if native["options"]["reverse_results"] == true {
            vec![510, 500]
        } else {
            vec![500, 510]
        };
        plan(&mut c);
        assert_native(&c, native);
    }
}

#[test]
fn aliases_versions_dword_wrap_and_ui_order_survive() {
    let mut c = context();
    c.actor.info_version = u32::MAX;
    c.actor.uses = i32::MIN;
    c.preappend_callback = Some(701);
    let out = replay(&c).unwrap();
    assert_eq!(out.context.actor.info_version, 0);
    assert_eq!(out.context.actor.uses, i32::MAX);
    assert_eq!(out.context.reference_lists[0].version, 42);
    let locate = |pred: fn(&Event) -> bool| out.events.iter().position(pred).unwrap();
    assert!(
        locate(|e| matches!(e, Event::Preappend { .. }))
            < locate(|e| matches!(e, Event::HistoryAppend { .. }))
    );
    assert!(
        locate(|e| matches!(e, Event::HistoryAppend { .. }))
            < locate(|e| matches!(e, Event::Uses { .. }))
    );
    assert!(
        locate(|e| matches!(e, Event::Uses { .. }))
            < locate(|e| matches!(e, Event::InfoRevealed { .. }))
    );
    assert!(
        locate(|e| matches!(e, Event::RegisterSpeech { .. }))
            < locate(|e| matches!(e, Event::SetText { .. }))
    );
    assert!(
        locate(|e| matches!(e, Event::SetText { .. }))
            < locate(|e| matches!(e, Event::SaveSpeech { .. }))
    );
    assert!(
        locate(|e| matches!(e, Event::SaveSpeech { .. }))
            < locate(|e| matches!(e, Event::ConstructWait { .. }))
    );
    assert_eq!(out.waits.last().unwrap().seconds_bits, SPEECH_WAIT_BITS);
    assert_eq!(out.context.result_iterators[0].current, 501);
    assert_eq!(out.speech_iterators[0].current, Some(601));
    c.result_iterators.push(ResultIterator {
        identity: 510,
        current: 511,
        ..c.result_iterators[0].clone()
    });
    c.result_waits.push(WaitObject {
        identity: 511,
        seconds_bits: 0,
    });
    c.result_resume_order.push(510);
    plan(&mut c);
    let mut out = replay(&c).unwrap();
    assert_eq!(
        out.context.actor.infos,
        vec![Some(201), Some(200), Some(200)]
    );
    out.context.infos[0].description = Some(101);
    assert!(out.context.actor.infos[1..].iter().all(|id| out
        .context
        .infos
        .iter()
        .find(|info| Some(info.identity) == *id)
        .unwrap()
        .description
        == Some(101)));
    assert_eq!(out.context.actor.saved_act, Some(100));
}

#[test]
fn preserves_utf16_and_early_gates_without_normalizing_text() {
    let mut c = context();
    c.strings[0].units = vec![0, 0xD800, 0xDC00, 0xDFFF];
    assert_eq!(replay(&c).unwrap().context.strings, c.strings);
    c.act = false;
    c.result_iterators[0].info = None;
    plan(&mut c);
    let out = replay(&c).unwrap();
    assert_eq!(out.context.actor.infos, c.actor.infos);
    assert_eq!(out.context.actor.uses, c.actor.uses);
    assert!(out.speech_iterators.is_empty());
    assert!(!out
        .events
        .iter()
        .any(|e| matches!(e, Event::StringEmpty { .. })));
}

#[test]
fn rejects_callback_layout_identity_collisions_but_preserves_layout_occurrences() {
    let base = context();
    assert_eq!(base.ui.layouts, vec![404, 404]);
    let out = replay(&base).unwrap();
    let rebuilt: Vec<_> = out
        .events
        .iter()
        .filter_map(|event| match event {
            Event::Rebuild { rect } => Some(*rect),
            _ => None,
        })
        .collect();
    assert_eq!(rebuilt, vec![404, 404]);
    for preappend in [false, true] {
        let mut c = base.clone();
        if preappend {
            c.preappend_callback = Some(c.ui.layouts[0]);
        } else {
            c.info_revealed_callback = Some(c.ui.layouts[0]);
        }
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
}

#[test]
fn rejects_unverified_and_unsupported_contexts_atomically() {
    let base = context();
    for key in [
        "runtime_and_metadata_verified",
        "callback_captures_and_first_yield_verified",
        "list_storage_and_capacity_verified",
        "preappend_callbacks_inert",
        "global_callbacks_inert",
        "ui_and_other_services_inert",
        "unity_liveness_verified",
        "trailer_lookup_stable_verified",
        "supplied_resume_order_verified",
        "normal_completion_verified",
    ] {
        let mut value = serde_json::to_value(&base).unwrap();
        value["services"][key] = json!(false);
        let c: Context = serde_json::from_value(value).unwrap();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext), "{key}");
    }
    let mut cases = Vec::new();
    let mut c = base.clone();
    c.version.push_str("_unknown");
    cases.push(c);
    let mut c = base.clone();
    c.allocations[0].speech_iterator = c.actor.identity;
    cases.push(c);
    let mut c = base.clone();
    c.allocations[0].speech_wait = Some(c.history.backing_array);
    cases.push(c);
    let mut c = base.clone();
    c.result_waits[0].seconds_bits = 0x80000000;
    cases.push(c);
    let mut c = base.clone();
    c.result_iterators[0].delay_bits = 1;
    cases.push(c);
    let mut c = base.clone();
    c.result_iterators[0].state = 0;
    cases.push(c);
    let mut c = base.clone();
    c.result_resume_order.push(500);
    cases.push(c);
    let mut c = base.clone();
    c.speech_resume_order.pop();
    cases.push(c);
    let mut c = base.clone();
    c.speech_resume_order.push(600);
    cases.push(c);
    let mut c = base.clone();
    c.history.capacity = 1;
    cases.push(c);
    let mut c = base.clone();
    c.infos[0].description = Some(0);
    cases.push(c);
    let mut c = base.clone();
    c.reference_lists[0].backing_array = Some(c.history.backing_array);
    cases.push(c);
    let mut c = base.clone();
    c.actor.bluff = Some(30);
    cases.push(c);
    let mut c = base.clone();
    c.strings[6].units.push(0);
    cases.push(c);
    for c in cases {
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
    let mut c = base.clone();
    c.history.capacity = MAX_RETAINED / 2;
    c.strings[0].units = vec![1; MAX_RETAINED / 2];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut unknown = serde_json::to_value(&base).unwrap();
    unknown["real_scheduler_readiness"] = json!(true);
    assert!(serde_json::from_value::<Context>(unknown).is_err());
}
