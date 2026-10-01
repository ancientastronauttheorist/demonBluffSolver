use super::*;
use serde_json::{json, Value};

const LABELS: [&str; 25] = [
    "view",
    "data",
    "bluff",
    "bg",
    "bgs",
    "border0",
    "border1",
    "art",
    "clipping",
    "text",
    "canvas",
    "borders",
    "anim_id",
    "background_sprite",
    "art_sprite",
    "name",
    "upper_name",
    "old supplied text",
    "go_art",
    "go_clipping",
    "image_class",
    "text_class",
    "color_method",
    "text_method",
    "DG.Tweening.DOTween_TypeInfo",
];
const SET_ID: &str =
    "Method$DG.Tweening.TweenSettingsExtensions.SetId<TweenerCore<float, float, FloatOptions>>()";

fn id(label: &str) -> Identity {
    if label == "view" {
        return 0x14000;
    }
    if label == SET_ID {
        return 0xc0000;
    }
    if let Some(index) = label.strip_prefix("tween") {
        return 0xd0000 + index.parse::<u64>().unwrap() * 0x100;
    }
    0x50000 + LABELS.iter().position(|&v| v == label).unwrap() as u64 * 0x1000
}

fn label(pointer: Option<Identity>) -> Value {
    let Some(pointer) = pointer else {
        return Value::Null;
    };
    if pointer == id(SET_ID) {
        return json!(SET_ID);
    }
    for name in LABELS {
        if pointer == id(name) {
            return json!(name);
        }
    }
    if pointer >= 0xd0000 && (pointer - 0xd0000) % 0x100 == 0 {
        return json!(format!("tween{}", (pointer - 0xd0000) / 0x100));
    }
    panic!("unknown pointer {pointer:x}");
}

fn reference(v: &Value) -> Option<Identity> {
    v.as_str().map(id)
}
fn u32_bits(v: &Value) -> [u32; 4] {
    std::array::from_fn(|i| v[i].as_u64().unwrap() as u32)
}
fn ranges() -> Vec<RetainedRange> {
    vec![RetainedRange {
        offset: 0x100,
        bytes: vec![0x51, 0xa2, 0xff],
    }]
}

fn context(fixture: &Value) -> Context {
    let initial = &fixture["initial"];
    let view = &initial["view"];
    let options = &fixture["options"];
    let bg_bits = [0x3f000001, 0x80000000, 0x7fc01234, 0x3f800000];
    let data = [
        ("data", bg_bits),
        ("bluff", [0x3e000001, 0x3f000002, 0x3f000003, 0x3f000004]),
    ]
    .into_iter()
    .map(|(name, bits)| DataAsset {
        identity: id(name),
        name: Some(id("name")),
        background_sprite: if options["null_background_sprite"] == true {
            None
        } else {
            Some(id("background_sprite"))
        },
        bg_color_bits: bits,
        border_color_bits: [bits[3], bits[2], bits[1], bits[0]],
        retained: vec![RetainedRange {
            offset: 0x118,
            bytes: vec![0x34, 0x12],
        }],
    })
    .collect();
    let images = ["bg", "bgs", "border0", "border1", "art", "clipping"]
        .into_iter()
        .map(|name| Image {
            identity: id(name),
            class: id("image_class"),
            color_bits: u32_bits(&initial["supplied_images"][name]["color_bits"]),
            sprite: reference(&initial["supplied_images"][name]["sprite"]),
            game_object: match name {
                "art" => Some(id("go_art")),
                "clipping" => Some(id("go_clipping")),
                _ => None,
            },
            retained: ranges(),
        })
        .collect();
    let art_type = options["art_type_bits"].as_u64().unwrap_or(0);
    let call = match fixture["method"].as_str().unwrap() {
        "AnimateIn" => Call::AnimateIn {
            kill_rdx_before_dl: 0xabcdef1234567890,
            tween: if options["null_tween"] == true {
                None
            } else {
                Some(id("tween0"))
            },
        },
        "AnimateOut" => Call::AnimateOut {
            kill_rdx_before_dl: 0xabcdef1234567890,
            tween: if options["null_tween"] == true {
                None
            } else {
                Some(id("tween0"))
            },
        },
        "Init" => Call::Init {
            data: id("data"),
            uppercase_result: if options["null_upper_result"] == true {
                None
            } else {
                Some(id("upper_name"))
            },
            art_result: if options["null_art_sprite"] == true {
                None
            } else {
                Some(id("art_sprite"))
            },
            art_type_return_bits: 0xface000000000000 | art_type,
        },
        "SetupArt" => Call::SetupArt {
            sprite: if options["null_argument"] == true {
                None
            } else {
                Some(id("art_sprite"))
            },
            type_register_bits: 0xface000000000000 | art_type,
        },
        _ => panic!("standalone method"),
    };
    Context {
        version: CHARACTER_VIEW_PRESENTATION_NATIVE_V1.into(),
        state: State {
            view: View {
                identity: id("view"),
                bg: reference(&view["bg"]),
                data: reference(&view["data"]),
                text: reference(&view["text"]),
                art: reference(&view["art"]),
                clipping: reference(&view["clipping"]),
                bgs: reference(&view["bgs"]),
                borders: reference(&view["borders"]),
                canvas: reference(&view["canvas"]),
                anim_id: reference(&view["anim_id"]),
                white_bg: Some(id("background_sprite")),
                locked_bg_bits: [0x80000000, 0x7fc0abcd, 0x12345678, 0xffffffff],
                locked_art_bits: [0xabcdef01; 4],
                retained: vec![RetainedRange {
                    offset: 0x90,
                    bytes: vec![0x32, 0x65],
                }],
            },
            data,
            images,
            texts: vec![Text {
                identity: id("text"),
                class: id("text_class"),
                value: reference(&initial["supplied_text"]),
                retained: ranges(),
            }],
            canvases: vec![OpaqueObject {
                identity: id("canvas"),
                retained: ranges(),
            }],
            sprites: ["background_sprite", "art_sprite"]
                .into_iter()
                .map(|name| OpaqueObject {
                    identity: id(name),
                    retained: ranges(),
                })
                .collect(),
            strings: ["name", "upper_name", "anim_id", "old supplied text"]
                .into_iter()
                .map(|name| StringObject {
                    identity: id(name),
                    units: name.encode_utf16().collect(),
                })
                .collect(),
            arrays: vec![ImageArray {
                identity: id("borders"),
                length_bits: initial["border_length_bits"].as_u64().unwrap(),
                slots: initial["border_slots"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(reference)
                    .collect(),
                retained: vec![RetainedRange {
                    offset: 0x38,
                    bytes: vec![1, 2, 3],
                }],
            }],
            game_objects: ["go_art", "go_clipping"]
                .into_iter()
                .map(|name| GameObject {
                    identity: id(name),
                    active: initial["supplied_game_objects"][name].as_bool().unwrap(),
                    retained: ranges(),
                })
                .collect(),
            metadata_flags: std::array::from_fn(|i| {
                initial["metadata_flags"][format!("0x{:x}", FLAG_RVAS[i])]
                    .as_u64()
                    .unwrap() as u8
            }),
            dotween_class_initialized: initial["dotween_class_initialized"].as_u64().unwrap()
                as u32,
            tween_requests: vec![],
            native_entries: vec![],
        },
        dotween_class: id("DG.Tweening.DOTween_TypeInfo"),
        image_class: id("image_class"),
        text_class: id("text_class"),
        color_method: id("color_method"),
        text_method: id("text_method"),
        set_id_method: id(SET_ID),
        tweens: vec![id("tween0"), id("tween1"), id("tween2")],
        calls: vec![call],
        services: Services {
            runtime_verified_inert: true,
            gc_verified_inert: true,
            ui_verified_inert: true,
            text_verified_inert: true,
            art_verified_inert: true,
            dotween_verified_inert: true,
            storage_verified: true,
            normal_completion_verified: true,
        },
    }
}

fn event(e: &Event) -> Value {
    let (kind, args) = match e {
        Event::MetadataService { slot_rva } => ("metadata_service", json!([slot_rva])),
        Event::ClassInitializationService { class } => {
            ("class_initialization_service", json!([label(Some(*class))]))
        }
        Event::DotweenKillService {
            id,
            rdx_bits,
            method_bits,
        } => (
            "dotween_kill_service",
            json!([label(*id), rdx_bits, method_bits]),
        ),
        Event::DofadeService {
            canvas,
            end_bits,
            duration_bits,
        } => (
            "dofade_service",
            json!([label(*canvas), end_bits, duration_bits]),
        ),
        Event::SetIdService { tween, id, method } => (
            "set_id_service",
            json!([label(*tween), label(*id), label(Some(*method))]),
        ),
        Event::ViewDataBarrierService { address, data } => (
            "view_data_barrier_service",
            json!([address, label(Some(*data))]),
        ),
        Event::ImageColorService {
            image,
            bits,
            method,
        } => (
            "image_color_service",
            json!([label(Some(*image)), bits, label(Some(*method))]),
        ),
        Event::ImageSpriteService { image, sprite } => (
            "image_sprite_service",
            json!([label(Some(*image)), label(*sprite)]),
        ),
        Event::UppercaseService { input, result } => (
            "uppercase_service",
            json!([label(Some(*input)), label(*result)]),
        ),
        Event::TextSetterService {
            text,
            value,
            method,
        } => (
            "text_setter_service",
            json!([label(Some(*text)), label(*value), label(Some(*method))]),
        ),
        Event::GetArtService { data } => ("get_art_service", json!([label(Some(*data))])),
        Event::GetArtTypeService { data } => ("get_art_type_service", json!([label(Some(*data))])),
        Event::ImageGameObjectService { image, result } => (
            "image_game_object_service",
            json!([label(Some(*image)), label(Some(*result))]),
        ),
        Event::ImageSetActiveService {
            game_object,
            rdx_bits,
            value,
        } => (
            "image_set_active_service",
            json!([label(Some(*game_object)), rdx_bits, u8::from(*value)]),
        ),
    };
    json!({"kind":kind,"args":args})
}

fn snapshot(s: &State, native_initial: &Value) -> Value {
    // Untouched Character/caller globals are outside this standalone contract.
    // Preserve that fixture envelope while projecting every modeled value.
    let mut out = native_initial.clone();
    let v = &s.view;
    out["view"] = json!({"bg":label(v.bg),"data":label(v.data),"text":label(v.text),"art":label(v.art),"clipping":label(v.clipping),"bgs":label(v.bgs),"borders":label(v.borders),"canvas":label(v.canvas),"anim_id":label(v.anim_id)});
    for (i, rva) in FLAG_RVAS.iter().enumerate() {
        out["metadata_flags"][format!("0x{rva:x}")] = json!(s.metadata_flags[i]);
    }
    out["dotween_class_initialized"] = json!(s.dotween_class_initialized);
    let a = &s.arrays[0];
    out["border_length_bits"] = json!(a.length_bits);
    out["border_slots"] = json!(a.slots.iter().map(|&p| label(p)).collect::<Vec<_>>());
    for i in &s.images {
        let name = label(Some(i.identity));
        out["supplied_images"][name.as_str().unwrap()] =
            json!({"color_bits":i.color_bits,"sprite":label(i.sprite)});
    }
    out["supplied_text"] = label(s.texts[0].value);
    for go in &s.game_objects {
        let name = label(Some(go.identity));
        out["supplied_game_objects"][name.as_str().unwrap()] = json!(go.active);
    }
    for (key, kind) in [
        ("supplied_kill_requests", "dotween_kill_service"),
        ("supplied_tween_requests", "dofade_service"),
        ("supplied_set_id_requests", "set_id_service"),
    ] {
        out[key] = json!(s
            .tween_requests
            .iter()
            .map(event)
            .filter(|v| v["kind"] == kind)
            .map(|v| v["args"].clone())
            .collect::<Vec<_>>());
    }
    out["native_view_entries"] = json!(s
        .native_entries
        .iter()
        .map(|e| {
            let mut entry = json!({"method":e.method,"view":label(Some(e.view))});
            if e.method == Method::Init {
                entry["data"] = label(e.data);
            }
            if e.method == Method::SetupArt {
                entry["sprite"] = label(e.sprite);
                entry["type_bits"] = json!(e.type_bits.unwrap());
            }
            entry
        })
        .collect::<Vec<_>>());
    out
}

fn corpus() -> Value {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_view_presentation.json")).unwrap()
}

fn supported(f: &Value) -> bool {
    f["returned"] == true
        && f["options"].get("view_mutation_phase").is_none()
        && f["initial"]["border_length_bits"].as_u64().unwrap() as u32 as i32 >= 0
}

#[test]
fn supported_native_corpus_matches_every_service_entry_and_final_state() {
    let report = corpus();
    let mut count = 0;
    let mut methods = BTreeSet::new();
    for fixture in report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| supported(f))
    {
        let c = context(fixture);
        let r = replay(&c)
            .unwrap_or_else(|_| panic!("rejected {} {}", fixture["method"], fixture["options"]));
        methods.insert(fixture["method"].as_str().unwrap());
        count += 1;
        assert_eq!(
            r.steps.len(),
            fixture["events"].as_array().unwrap().len(),
            "{}",
            fixture["options"]
        );
        for (step, native) in r.steps.iter().zip(fixture["events"].as_array().unwrap()) {
            assert_eq!(
                event(&step.event),
                json!({"kind":native["kind"],"args":native["args"]}),
                "{} {}",
                fixture["method"],
                fixture["options"]
            );
            assert_eq!(
                snapshot(&step.state, &fixture["initial"]),
                native["snapshot"],
                "{} {} at {}",
                fixture["method"],
                fixture["options"],
                native["kind"]
            );
        }
        assert_eq!(snapshot(&r.state, &fixture["initial"]), fixture["final"]);
        assert_eq!(r.completed, vec![r.state.clone()]);
        // All data, canvas/string/sprite storage and unconsumed view fields remain.
        assert_eq!(r.state.data, c.state.data);
        assert_eq!(r.state.canvases, c.state.canvases);
        assert_eq!(r.state.sprites, c.state.sprites);
        assert_eq!(r.state.strings, c.state.strings);
        assert_eq!(r.state.arrays, c.state.arrays);
        assert_eq!(r.state.view.retained, c.state.view.retained);
        assert_eq!(r.state.view.locked_bg_bits, c.state.view.locked_bg_bits);
        assert_eq!(r.state.view.locked_art_bits, c.state.view.locked_art_bits);
        assert_eq!(r.state.view.white_bg, c.state.view.white_bg);
    }
    assert_eq!(methods.len(), 4);
    assert_eq!(count, 166);
}

fn basic(method: &str) -> Context {
    let report = corpus();
    let f = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["method"] == method && supported(f))
        .unwrap();
    context(f)
}

#[test]
fn explicit_retained_calls_preserve_warm_flags_requests_and_aliases() {
    let mut c = basic("AnimateIn");
    c.state.metadata_flags = [0; 5];
    c.state.dotween_class_initialized = 0;
    let init = basic("Init").calls.remove(0);
    c.calls = vec![
        c.calls.remove(0),
        Call::AnimateOut {
            kill_rdx_before_dl: 0xface0000000000ff,
            tween: Some(id("tween1")),
        },
        init,
        Call::AnimateIn {
            kill_rdx_before_dl: 0,
            tween: None,
        },
    ];
    c.state.view.bg = c.state.view.art;
    c.state.view.bgs = c.state.view.art;
    c.state.view.clipping = c.state.view.art;
    c.state.arrays[0].slots[1] = c.state.arrays[0].slots[0];
    let r = replay(&c).unwrap();
    assert_eq!(r.completed.len(), 4);
    assert_eq!(r.state.tween_requests.len(), 9);
    assert_eq!(r.state.metadata_flags, [0, 0, 0, 1, 1]);
    assert_eq!(r.state.dotween_class_initialized, 1);
    assert_eq!(
        r.steps
            .iter()
            .filter(|s| matches!(s.event, Event::MetadataService { .. }))
            .count(),
        4
    );
    assert_eq!(
        r.steps
            .iter()
            .filter(|s| matches!(s.event, Event::ClassInitializationService { .. }))
            .count(),
        1
    );
    let art = r
        .state
        .images
        .iter()
        .find(|i| i.identity == id("art"))
        .unwrap();
    assert_eq!(art.color_bits, WHITE);
    assert_eq!(art.sprite, Some(id("art_sprite")));
    assert!(
        !r.state
            .game_objects
            .iter()
            .find(|g| g.identity == id("go_art"))
            .unwrap()
            .active
    );
    assert_eq!(r.state.native_entries.len(), 5);
    let retained = Context {
        state: r.state.clone(),
        calls: vec![],
        ..c
    };
    assert_eq!(replay(&retained).unwrap().state, r.state);
}

#[test]
fn width_bits_and_barrier_observe_actual_native_order() {
    let mut c = basic("Init");
    c.state.view.data = Some(id("bluff"));
    if let Call::Init {
        art_type_return_bits,
        ..
    } = &mut c.calls[0]
    {
        *art_type_return_bits = 0xdeadbeef0000000a;
    }
    let r = replay(&c).unwrap();
    assert_eq!(r.steps[0].state.view.data, Some(id("data")));
    assert!(matches!(r.steps[0].event,Event::ViewDataBarrierService{data,..} if data==id("data")));
    assert_eq!(r.state.native_entries[1].type_bits, Some(10));
    assert_eq!(
        r.state
            .images
            .iter()
            .find(|i| i.identity == id("clipping"))
            .unwrap()
            .sprite,
        Some(id("art_sprite"))
    );
    assert!(
        r.state
            .game_objects
            .iter()
            .find(|g| g.identity == id("go_clipping"))
            .unwrap()
            .active
    );
    let mut c = basic("AnimateOut");
    c.state.metadata_flags = [0x80; 5];
    c.state.dotween_class_initialized = 0xdeadbeef;
    let r = replay(&c).unwrap();
    assert_eq!(r.state.metadata_flags, [0x80; 5]);
    assert_eq!(r.state.dotween_class_initialized, 0xdeadbeef);
    assert!(matches!(
        r.steps[0].event,
        Event::DotweenKillService {
            rdx_bits: 0xabcdef1234567801,
            ..
        }
    ));
    assert!(matches!(
        r.steps[1].event,
        Event::DofadeService {
            end_bits: 0,
            duration_bits: FADE_DURATION_BITS,
            ..
        }
    ));
}

#[test]
fn incompatible_physical_types_nulls_and_unverified_services_rejected() {
    let c = basic("Init");
    let rejected = |v: Context| assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
    let mut v = c.clone();
    v.state.view.data = v.state.view.art;
    rejected(v);
    let mut v = c.clone();
    v.state.view.white_bg = v.state.view.anim_id;
    rejected(v);
    let mut v = c.clone();
    v.state.view.text = v.state.view.canvas;
    rejected(v);
    let mut v = c.clone();
    v.state.images[0].class = v.text_class;
    rejected(v);
    let mut v = c.clone();
    v.state.images[0].identity = v.dotween_class;
    rejected(v);
    let mut v = c.clone();
    v.state.images[0].sprite = Some(id("data"));
    rejected(v);
    let mut v = c.clone();
    v.state.view.art = None;
    rejected(v);
    let mut v = c.clone();
    v.state.arrays[0].slots[0] = None;
    // basic selects a valid zero-length native profile. A null diagnostic
    // backing slot is legal until the caller's consumed length includes it.
    assert!(replay(&v).is_ok());
    v.state.arrays[0].length_bits = (v.state.arrays[0].length_bits & !0xffff_ffff) | 1;
    rejected(v);
    let mut v = c.clone();
    v.state.arrays[0].length_bits = 0x80000000;
    rejected(v);
    let mut v = c.clone();
    v.state.arrays[0].length_bits = 4;
    rejected(v);
    let mut v = c.clone();
    v.state.images[4].game_object = None;
    rejected(v);
    let mut v = c.clone();
    v.state.data[0].name = None;
    rejected(v);
    let mut v = c.clone();
    v.services.art_verified_inert = false;
    rejected(v);
    let mut v = c.clone();
    v.services.dotween_verified_inert = false;
    rejected(v);
    let mut v = c.clone();
    v.services.runtime_verified_inert = false;
    rejected(v);
    let mut v = c.clone();
    v.services.gc_verified_inert = false;
    rejected(v);
    let mut v = c.clone();
    v.services.ui_verified_inert = false;
    rejected(v);
    let mut v = c.clone();
    v.services.text_verified_inert = false;
    rejected(v);
    let mut v = c.clone();
    v.services.storage_verified = false;
    rejected(v);
    let mut v = c.clone();
    v.services.normal_completion_verified = false;
    rejected(v);
    let mut v = c.clone();
    v.version.push('x');
    rejected(v);
    let mut v = c.clone();
    v.state.native_entries.push(NativeEntry {
        method: Method::Init,
        view: id("view"),
        data: Some(id("art")),
        sprite: None,
        type_bits: None,
    });
    rejected(v);
    let mut v = c;
    v.state.tween_requests.push(Event::SetIdService {
        tween: Some(id("art")),
        id: Some(id("anim_id")),
        method: id(SET_ID),
    });
    rejected(v);
}

#[test]
fn retained_ranges_cannot_contradict_consumed_fields_and_boundaries_are_exact() {
    let c = basic("Init");
    for offset in [0x20, 0x27, 0x28, 0x60, 0x68, 0x8f] {
        let mut v = c.clone();
        v.state.view.retained = vec![RetainedRange {
            offset,
            bytes: vec![0],
        }];
        assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
    }
    for offset in [0x28, 0x2f, 0xb8, 0xbf, 0xf8, 0x117] {
        let mut v = c.clone();
        v.state.data[0].retained = vec![RetainedRange {
            offset,
            bytes: vec![0],
        }];
        assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
    }
    for offset in [0x18, 0x1f, 0x20, 0x37] {
        let mut v = c.clone();
        v.state.arrays[0].retained = vec![RetainedRange {
            offset,
            bytes: vec![0],
        }];
        assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
    }
    let mut v = c.clone();
    v.state.view.retained = vec![
        RetainedRange {
            offset: 0x1f,
            bytes: vec![0],
        },
        RetainedRange {
            offset: 0x90,
            bytes: vec![0],
        },
    ];
    assert!(replay(&v).is_ok());
    let mut v = c.clone();
    v.state.data[0].retained = vec![
        RetainedRange {
            offset: 0x27,
            bytes: vec![0],
        },
        RetainedRange {
            offset: 0x30,
            bytes: vec![0],
        },
        RetainedRange {
            offset: 0x118,
            bytes: vec![0],
        },
    ];
    assert!(replay(&v).is_ok());
    let mut v = c;
    v.state.view.retained = vec![
        RetainedRange {
            offset: 0x100,
            bytes: vec![0, 1],
        },
        RetainedRange {
            offset: 0x101,
            bytes: vec![0],
        },
    ];
    assert_eq!(replay(&v).unwrap_err(), LedgerError::InvalidContext);
}

#[test]
fn aggregate_future_snapshots_are_bounded_before_replay() {
    let mut c = basic("Init");
    c.state.view.retained = vec![RetainedRange {
        offset: 0x1000,
        bytes: vec![0; 7000],
    }];
    assert!(replay(&c).is_ok());
    c.calls = vec![c.calls[0].clone(); 3];
    assert_eq!(replay(&c).unwrap_err(), LedgerError::Capacity);
    let mut c = basic("SetupArt");
    c.state.strings[0].units = vec![0; MAX_UNITS];
    assert_eq!(replay(&c).unwrap_err(), LedgerError::Capacity);
    let mut c = basic("AnimateIn");
    c.calls = vec![c.calls[0].clone(); 17];
    assert_eq!(replay(&c).unwrap_err(), LedgerError::Capacity);
}
