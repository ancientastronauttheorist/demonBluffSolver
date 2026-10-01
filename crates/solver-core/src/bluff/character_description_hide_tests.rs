use super::super::character_initialization::{ObjectReference, Statuses};
use super::*;
use serde_json::{json, Value};

fn id(name: &str) -> Identity {
    0x3_0000_0000
        + match name {
            "actor" => 0x10000,
            "data" => 0x16000,
            "bluff" => 0x17000,
            "ui_static" => 0x18000,
            "left_acted" => 0x30000,
            "left_version" => 0x31000,
            "left_game" => 0x32000,
            "acted" => 0x33000,
            "speech" => 0x34000,
            "characters_static" => 0x35000,
            "characters" => 0x36000,
            "hide_callback" => 0x37000,
            "hint_callback" => 0x38000,
            "hint_replacement" => 0x39000,
            "action_target" => 0x3A000,
            "action_method" => 0x3B000,
            _ => panic!("unknown label {name}"),
        }
}

fn label(pointer: Option<Identity>) -> Value {
    let Some(pointer) = pointer else {
        return Value::Null;
    };
    let name = [
        "actor",
        "data",
        "bluff",
        "ui_static",
        "left_acted",
        "left_version",
        "left_game",
        "acted",
        "speech",
        "characters_static",
        "characters",
        "hide_callback",
        "hint_callback",
        "hint_replacement",
        "action_target",
        "action_method",
    ]
    .into_iter()
    .find(|n| id(n) == pointer)
    .unwrap();
    json!(name)
}

fn pointer(value: &Value) -> Option<Identity> {
    value.as_str().map(id)
}

fn method(row: &Value) -> Method {
    match row["method"].as_str().unwrap() {
        "OnHover" => Method::OnHover,
        "HideDescription" => Method::HideDescription,
        _ => panic!("method"),
    }
}

fn action(value: &Value) -> Action {
    Action {
        identity: pointer(&value["delegate"]).unwrap(),
        target: pointer(&value["target"]),
        method_info: pointer(&value["method"]).unwrap(),
    }
}

fn fixture(rows: &[&Value]) -> Context {
    let initial = &rows[0]["initial"];
    let d = &initial["description"];
    let options = &rows[0]["options"];
    // Additional logical Actor semantics are authored valid retained inputs.
    // They are not decoded from the native fixture's A5 diagnostic padding.
    let actor = Actor {
        identity: id("actor"),
        data: Some(id("data")),
        bluff: Some(id("bluff")),
        register_as: None,
        trailer: None,
        runtime: None,
        dead_prefab: None,
        revealed: false,
        uses: 7,
        previous: 10,
        state: 5,
        killed_hidden: true,
        killed_demon: false,
        alignment: 10,
        id: 17,
        started: true,
        role: None,
        bluff_role: None,
        saved_act: pointer(&d["speech"]),
        infos: vec![],
        info_version: u32::MAX,
        statuses: Statuses {
            active: vec![10, 30],
            version: u32::MAX,
            resistances: vec![55],
            target: Some(id("actor")),
        },
        state_callback: None,
    };
    let target = if options["null_action_target"] == true {
        None
    } else if options["alias_action_left"] == true {
        Some(id("left_acted"))
    } else {
        Some(id("action_target"))
    };
    let header = |value: &Value| {
        pointer(value).map(|identity| Action {
            identity,
            target,
            method_info: id("action_method"),
        })
    };
    let state = State {
        actor,
        hover_bits: initial["actor"]["hover_bits"].as_u64().unwrap() as u8,
        left_bits: d["left_bits"].as_u64().unwrap() as u8,
        ui_objects: vec![UiObject {
            identity: id("left_game"),
            active: d["supplied_game_active"].as_bool().unwrap(),
        }],
        stopped_components: d["stopped_components"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| pointer(v).unwrap())
            .collect(),
        restored_speech: d["restored_speech"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| RestoredSpeech {
                component: id("acted"),
                text: pointer(v),
            })
            .collect(),
        highlight_clears: vec![id("characters"); d["highlight_clears"].as_u64().unwrap() as usize],
        action_calls: d["action_calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(action)
            .collect(),
    };
    Context {
        version: CHARACTER_DESCRIPTION_HIDE_NATIVE_V1.into(),
        state,
        bindings: Bindings {
            left_acted: pointer(&d["left_acted"]),
            left_version: pointer(&d["left_version"]),
            left_game_object: Some(id("left_game")),
            acted: pointer(&d["acted"]),
            characters_static: Some(id("characters_static")),
            characters: pointer(&d["characters"]),
            ui_events_static: Some(id("ui_static")),
            hide_custom_hint: header(&d["hide_callback"]),
            hide_hint: header(&d["hint_callback"]),
        },
        calls: rows.iter().map(|row| method(row)).collect(),
        services: Services {
            retained_actor_projection_verified: true,
            native_runtime_and_bindings_verified: true,
            metadata_and_class_state_verified: true,
            object_bindings_verified: true,
            callbacks_and_services_inert: true,
            normal_completion_verified: true,
        },
    }
}

fn report() -> Value {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_description_hide.json")).unwrap()
}

fn description(state: &State) -> Value {
    json!({"hover_bits":state.hover_bits,"left_bits":state.left_bits,
        "speech":label(state.actor.saved_act),
        "supplied_game_active":state.ui_objects.iter().find(|o|o.identity==id("left_game")).unwrap().active,
        "stopped_components":state.stopped_components.iter().map(|p|label(Some(*p))).collect::<Vec<_>>(),
        "restored_speech":state.restored_speech.iter().map(|p|label(p.text)).collect::<Vec<_>>(),
        "highlight_clears":state.highlight_clears.len(),
        "action_calls":state.action_calls.iter().map(|a|json!({"delegate":label(Some(a.identity)),"target":label(a.target),"method":label(Some(a.method_info))})).collect::<Vec<_>>()})
}

fn native_description(snapshot: &Value) -> Value {
    let d = &snapshot["description"];
    json!({"hover_bits":snapshot["actor"]["hover_bits"],"left_bits":d["left_bits"],
        "speech":d["speech"],"supplied_game_active":d["supplied_game_active"],
        "stopped_components":d["stopped_components"],"restored_speech":d["restored_speech"],
        "highlight_clears":d["highlight_clears"],"action_calls":d["action_calls"]})
}

fn event(event: &Event) -> Option<Value> {
    let (kind, args) = match event {
        Event::StoreHover { .. } => return None,
        Event::StopAllCoroutines { component } => (
            "supplied_stop_all_coroutines",
            json!([label(Some(*component))]),
        ),
        Event::GetGameObject { component, result } => (
            "supplied_game_object",
            json!([label(Some(*component)), label(Some(*result))]),
        ),
        Event::SetActive { object, active } => {
            ("supplied_set_active", json!([label(Some(*object)), active]))
        }
        Event::ActedAct { component, text } => (
            "supplied_acted_act",
            json!([label(Some(*component)), label(*text)]),
        ),
        Event::DisableHighlightAll { characters } => (
            "supplied_disable_highlight_all",
            json!([label(Some(*characters))]),
        ),
        Event::HideCustomHint { action } => (
            "supplied_hide_action",
            json!([{"delegate":label(Some(action.identity)),"target":label(action.target),"method":label(Some(action.method_info))}]),
        ),
        Event::HideHint { action } => (
            "supplied_hint_action",
            json!([{"delegate":label(Some(action.identity)),"target":label(action.target),"method":label(Some(action.method_info))}]),
        ),
    };
    Some(json!({"kind":kind,"args":args}))
}

fn verify(rows: &[&Value]) {
    let c = fixture(rows);
    let out = replay(&c).unwrap();
    assert_eq!(out.state.actor, c.state.actor);
    for step in &out.steps {
        assert_eq!(step.state.actor, c.state.actor);
    }
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(
            description(&out.calls[i].state),
            native_description(&row["final"])
        );
        let native = row["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["kind"].as_str().unwrap().starts_with("supplied_"))
            .collect::<Vec<_>>();
        let steps = out
            .steps
            .iter()
            .filter(|s| s.call == i && event(&s.event).is_some())
            .collect::<Vec<_>>();
        assert_eq!(steps.len(), native.len());
        for (step, expected) in steps.iter().zip(native) {
            assert_eq!(
                event(&step.event).unwrap(),
                json!({"kind":expected["kind"],"args":expected["args"]})
            );
            assert_eq!(
                description(&step.state),
                native_description(&expected["snapshot"])
            );
        }
        // A genuine diagnostic oracle: only the native hover byte can change.
        // It is not interpreted as a typed full Actor initializer.
        let before = row["initial"]["description"]["actor_bytes_hex"]
            .as_str()
            .unwrap()
            .as_bytes();
        let after = row["final"]["description"]["actor_bytes_hex"]
            .as_str()
            .unwrap()
            .as_bytes();
        assert_eq!(before.len(), 1024);
        assert_eq!(after.len(), 1024);
        for offset in 0..512 {
            if method(row) != Method::OnHover || offset != 0x190 {
                assert_eq!(
                    &before[offset * 2..offset * 2 + 2],
                    &after[offset * 2..offset * 2 + 2]
                );
            } else {
                assert_eq!(&after[offset * 2..offset * 2 + 2], b"01");
            }
        }
    }
}

#[test]
fn normal_native_corpus_and_retained_sequences() {
    let r = report();
    let mut count = 0;
    for row in r["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(r["failure_baselines"].as_array().unwrap())
    {
        if row["returned"] != true || row["options"].get("mutation_phase").is_some() {
            continue;
        }
        verify(&[row]);
        count += 1;
    }
    assert_eq!(count, 42);
    for sequence in r["retained_sequences"].as_array().unwrap() {
        let rows = sequence.as_array().unwrap().iter().collect::<Vec<_>>();
        verify(&rows);
    }
    assert_eq!(r["retained_sequences"].as_array().unwrap().len(), 2);
}

fn base() -> Context {
    let r = report();
    fixture(&[&r["failure_baselines"][0]])
}

#[test]
fn physical_aliases_and_retained_actor_types() {
    let mut c = base();
    c.state.actor.dead_prefab = Some(ObjectReference {
        identity: id("left_game"),
        live: false,
    });
    c.state.actor.register_as = c.state.actor.data;
    c.state.actor.bluff = c.state.actor.data;
    c.state.actor.statuses.target = Some(c.state.actor.identity);
    c.bindings.acted = c.bindings.left_acted;
    let custom = c.bindings.hide_custom_hint.unwrap();
    c.bindings.hide_hint = Some(custom);
    c.bindings.hide_custom_hint.as_mut().unwrap().target = c.bindings.left_acted;
    c.bindings.hide_hint = c.bindings.hide_custom_hint;
    let out = replay(&c).unwrap();
    assert_eq!(out.state.actor, c.state.actor);
    assert_eq!(out.state.action_calls[0], out.state.action_calls[1]);
    assert_eq!(out.state.restored_speech[0].component, id("left_acted"));
    for field in 0..10 {
        let mut invalid = base();
        let component = id("left_version");
        match field {
            0 => invalid.state.actor.saved_act = Some(component),
            1 => invalid.state.actor.infos.push(Some(id("acted"))),
            2 => invalid.state.actor.data = Some(id("left_game")),
            3 => invalid.state.actor.role = Some(id("characters")),
            4 => invalid.state.actor.trailer = Some(id("acted")),
            5 => invalid.state.actor.runtime = Some(id("hint_callback")),
            6 => invalid.state.actor.state_callback = Some(component),
            7 => invalid.state.actor.statuses.target = Some(component),
            8 => invalid.bindings.characters_static = Some(id("left_game")),
            _ => invalid.bindings.hide_hint.as_mut().unwrap().method_info = id("acted"),
        }
        assert_eq!(replay(&invalid), Err(LedgerError::InvalidContext));
    }
    let mut inconsistent = base();
    inconsistent.bindings.hide_hint = inconsistent.bindings.hide_custom_hint;
    inconsistent.bindings.hide_hint.as_mut().unwrap().target = None;
    assert_eq!(replay(&inconsistent), Err(LedgerError::InvalidContext));
}

#[test]
fn hover_uses_no_description_bindings_and_raw_byte_gates() {
    let mut c = base();
    c.calls = vec![Method::OnHover, Method::OnHover];
    c.state.hover_bits = 0xFF;
    c.bindings = Bindings {
        left_acted: None,
        left_version: None,
        left_game_object: None,
        acted: None,
        characters_static: None,
        characters: None,
        ui_events_static: None,
        hide_custom_hint: None,
        hide_hint: None,
    };
    c.state.ui_objects.clear();
    c.services.native_runtime_and_bindings_verified = false;
    c.services.metadata_and_class_state_verified = false;
    c.services.object_bindings_verified = false;
    c.services.callbacks_and_services_inert = false;
    let out = replay(&c).unwrap();
    assert_eq!(out.state.hover_bits, 1);
    assert_eq!(out.state.actor, c.state.actor);
    assert_eq!(out.steps.len(), 2);
    assert!(out.state.action_calls.is_empty());
    c.calls.push(Method::HideDescription);
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    let mut zero = base();
    zero.state.left_bits = 0;
    zero.bindings.left_acted = None;
    zero.bindings.left_version = None;
    zero.bindings.left_game_object = None;
    zero.bindings.acted = None;
    zero.state.ui_objects.clear();
    let out = replay(&zero).unwrap();
    assert_eq!(out.steps.len(), 3);
    for value in [1, 0x80, 0xFF] {
        let mut c = base();
        c.state.left_bits = value;
        assert_eq!(replay(&c).unwrap().steps.len(), 7);
    }
    let mut encoded = serde_json::to_value(base()).unwrap();
    encoded["state"]["left_bits"] = json!(256);
    assert!(serde_json::from_value::<Context>(encoded).is_err());
}

#[test]
fn reject_missing_storage_mutation_failure_and_runtime_provenance() {
    for flag in 0..6 {
        let mut c = base();
        match flag {
            0 => c.services.retained_actor_projection_verified = false,
            1 => c.services.native_runtime_and_bindings_verified = false,
            2 => c.services.metadata_and_class_state_verified = false,
            3 => c.services.object_bindings_verified = false,
            4 => c.services.callbacks_and_services_inert = false,
            _ => c.services.normal_completion_verified = false,
        }
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    }
    for missing in 0..7 {
        let mut c = base();
        match missing {
            0 => c.bindings.left_acted = None,
            1 => c.bindings.left_version = None,
            2 => c.bindings.left_game_object = None,
            3 => c.bindings.acted = None,
            4 => c.bindings.characters = None,
            5 => c.bindings.ui_events_static = None,
            _ => c.state.ui_objects.clear(),
        }
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    }
    let mut null_text = base();
    null_text.state.actor.saved_act = None;
    assert_eq!(
        replay(&null_text).unwrap().state.restored_speech[0].text,
        None
    );
    let mut duplicate = base();
    duplicate
        .state
        .ui_objects
        .push(duplicate.state.ui_objects[0].clone());
    assert_eq!(replay(&duplicate), Err(LedgerError::InvalidContext));
    let mut null_header = base();
    null_header
        .bindings
        .hide_custom_hint
        .as_mut()
        .unwrap()
        .target = None;
    assert!(replay(&null_header).is_ok());
    null_header
        .bindings
        .hide_custom_hint
        .as_mut()
        .unwrap()
        .target = Some(0);
    assert_eq!(replay(&null_header), Err(LedgerError::InvalidContext));
}

#[test]
fn whole_future_snapshot_budget_and_retained_effects() {
    let mut c = base();
    c.calls = vec![Method::OnHover; 32];
    c.state.actor.statuses.active.clear();
    c.state.actor.statuses.resistances.clear();
    c.state.actor.infos = vec![None; 533];
    assert!(budget(&c).is_ok());
    assert!(replay(&c).is_ok());
    c.state.actor.infos.push(None);
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    c.state.actor.infos.clear();
    c.calls.push(Method::OnHover);
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut c = base();
    c.calls = vec![Method::HideDescription; 32];
    let out = replay(&c).unwrap();
    assert_eq!(out.state.stopped_components.len(), 32);
    assert_eq!(out.state.restored_speech.len(), 32);
    assert_eq!(out.state.highlight_clears.len(), 32);
    assert_eq!(out.state.action_calls.len(), 64);
    assert_eq!(out.state.actor, c.state.actor);
    c.state = out.state;
    c.calls = vec![Method::HideDescription];
    let second = replay(&c).unwrap();
    assert_eq!(second.state.action_calls.len(), 66);
    assert_eq!(&second.state.action_calls[..64], &c.state.action_calls);
    c.state.action_calls = vec![c.bindings.hide_custom_hint.unwrap(); MAX_RETAINED];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}
