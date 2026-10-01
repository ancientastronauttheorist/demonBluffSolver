//! Guarded offline replay of CharacterView AnimateIn/AnimateOut/Init/SetupArt.
//! Only standalone, normally completing calls with independently verified inert
//! runtime/GC and named supplied UI, text, art and DOTween services are accepted.
//! Supplied effects update fixture bookkeeping, not Unity/CLR/DOTween internals.
//! No disguise join, mutation, stop, unwinding, scheduler or rendering is modeled.
//! References preserve physical identity; repeated Image/array slots are legal.
//! Retained memory ranges exclude modeled native fields. Array slots beyond the
//! supplied low-DWORD length are untouched diagnostic backing memory, not valid
//! managed elements. Negative lengths are outside this conservative profile.
//! Unconsumed data/runtime bytes are retained without classifying embedded
//! references. Class/method tokens and supplied virtual dispatch are verified
//! inputs; this module does not implement their metadata or object internals.

use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CHARACTER_VIEW_PRESENTATION_NATIVE_V1: &str = "character_view_presentation_native_v1";
pub const FLAG_RVAS: [u32; 5] = [0x288c0e7, 0x288c186, 0x288c187, 0x288c1dc, 0x288c1dd];
pub const SET_ID_METHOD_RVA: u32 = 0x271c038;
pub const DOTWEEN_CLASS_SLOT_RVA: u32 = 0x26e3790;
pub const WHITE: [u32; 4] = [0x3f800000; 4];
pub const FADE_DURATION_BITS: u32 = 0x3e4ccccd;
const MAX_UNITS: usize = 16_384;
const MAX_WORK: usize = 262_144;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetainedRange {
    pub offset: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    pub identity: Identity,
    pub bg: Option<Identity>,
    pub data: Option<Identity>,
    pub text: Option<Identity>,
    pub art: Option<Identity>,
    pub clipping: Option<Identity>,
    pub bgs: Option<Identity>,
    pub borders: Option<Identity>,
    pub canvas: Option<Identity>,
    pub anim_id: Option<Identity>,
    pub white_bg: Option<Identity>,
    pub locked_bg_bits: [u32; 4],
    pub locked_art_bits: [u32; 4],
    pub retained: Vec<RetainedRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataAsset {
    pub identity: Identity,
    pub name: Option<Identity>,
    pub background_sprite: Option<Identity>,
    pub bg_color_bits: [u32; 4],
    pub border_color_bits: [u32; 4],
    pub retained: Vec<RetainedRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub identity: Identity,
    pub class: Identity,
    pub color_bits: [u32; 4],
    pub sprite: Option<Identity>,
    /// Supplied Component.get_gameObject result; never reconstructed here.
    pub game_object: Option<Identity>,
    pub retained: Vec<RetainedRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Text {
    pub identity: Identity,
    pub class: Identity,
    pub value: Option<Identity>,
    pub retained: Vec<RetainedRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StringObject {
    pub identity: Identity,
    pub units: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpaqueObject {
    pub identity: Identity,
    pub retained: Vec<RetainedRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameObject {
    pub identity: Identity,
    pub active: bool,
    pub retained: Vec<RetainedRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageArray {
    pub identity: Identity,
    pub length_bits: u64,
    pub slots: Vec<Option<Identity>>,
    pub retained: Vec<RetainedRange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    AnimateIn,
    AnimateOut,
    Init,
    SetupArt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "method", deny_unknown_fields)]
pub enum Call {
    AnimateIn {
        kill_rdx_before_dl: u64,
        tween: Option<Identity>,
    },
    AnimateOut {
        kill_rdx_before_dl: u64,
        tween: Option<Identity>,
    },
    Init {
        data: Identity,
        uppercase_result: Option<Identity>,
        art_result: Option<Identity>,
        /// Actual supplied return register. Init forwards EAX through R8D.
        art_type_return_bits: u64,
    },
    SetupArt {
        sprite: Option<Identity>,
        type_register_bits: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Services {
    pub runtime_verified_inert: bool,
    pub gc_verified_inert: bool,
    pub ui_verified_inert: bool,
    pub text_verified_inert: bool,
    pub art_verified_inert: bool,
    pub dotween_verified_inert: bool,
    pub storage_verified: bool,
    pub normal_completion_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    MetadataService {
        slot_rva: u32,
    },
    ClassInitializationService {
        class: Identity,
    },
    DotweenKillService {
        id: Option<Identity>,
        rdx_bits: u64,
        method_bits: u64,
    },
    DofadeService {
        canvas: Option<Identity>,
        end_bits: u32,
        duration_bits: u32,
    },
    SetIdService {
        tween: Option<Identity>,
        id: Option<Identity>,
        method: Identity,
    },
    ViewDataBarrierService {
        address: Identity,
        data: Identity,
    },
    ImageColorService {
        image: Identity,
        bits: [u32; 4],
        method: Identity,
    },
    ImageSpriteService {
        image: Identity,
        sprite: Option<Identity>,
    },
    UppercaseService {
        input: Identity,
        result: Option<Identity>,
    },
    TextSetterService {
        text: Identity,
        value: Option<Identity>,
        method: Identity,
    },
    GetArtService {
        data: Identity,
    },
    GetArtTypeService {
        data: Identity,
    },
    ImageGameObjectService {
        image: Identity,
        result: Identity,
    },
    ImageSetActiveService {
        game_object: Identity,
        rdx_bits: u64,
        value: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeEntry {
    pub method: Method,
    pub view: Identity,
    pub data: Option<Identity>,
    pub sprite: Option<Identity>,
    pub type_bits: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub view: View,
    pub data: Vec<DataAsset>,
    pub images: Vec<Image>,
    pub texts: Vec<Text>,
    pub canvases: Vec<OpaqueObject>,
    pub sprites: Vec<OpaqueObject>,
    pub strings: Vec<StringObject>,
    pub arrays: Vec<ImageArray>,
    pub game_objects: Vec<GameObject>,
    pub metadata_flags: [u8; 5],
    pub dotween_class_initialized: u32,
    /// Prior supplied tween requests retained by explicit caller batches.
    pub tween_requests: Vec<Event>,
    pub native_entries: Vec<NativeEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub dotween_class: Identity,
    pub image_class: Identity,
    pub text_class: Identity,
    pub color_method: Identity,
    pub text_method: Identity,
    pub set_id_method: Identity,
    /// Supplied tween object identities. Their implementation/storage is opaque.
    pub tweens: Vec<Identity>,
    pub calls: Vec<Call>,
    pub services: Services,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Step {
    pub event: Event,
    /// Exact state before applying the supplied service effect.
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub state: State,
    pub steps: Vec<Step>,
    pub completed: Vec<State>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    View,
    Data,
    Image,
    Text,
    Canvas,
    Sprite,
    String,
    Array,
    GameObject,
    Class,
    Method,
    Tween,
}

fn bind(map: &mut BTreeMap<Identity, Kind>, id: Identity, kind: Kind) -> Result<(), LedgerError> {
    if id == 0 || map.get(&id).is_some_and(|previous| *previous != kind) {
        return Err(LedgerError::InvalidContext);
    }
    map.insert(id, kind);
    Ok(())
}

fn ranges_valid(ranges: &[RetainedRange], excluded: &[(u64, u64)]) -> bool {
    ranges.iter().enumerate().all(|(i, r)| {
        let start = u64::from(r.offset);
        let Some(end) = start.checked_add(r.bytes.len() as u64) else {
            return false;
        };
        end <= u64::from(u32::MAX) + 1
            && (start == end
                || !excluded.iter().any(|&(a, b)| start < b && a < end)
                    && !ranges[..i].iter().any(|p| {
                        start < u64::from(p.offset) + p.bytes.len() as u64
                            && u64::from(p.offset) < end
                    }))
    })
}

fn range_units(r: &[RetainedRange]) -> Option<usize> {
    r.iter()
        .try_fold(0usize, |n, v| n.checked_add(v.bytes.len())?.checked_add(2))
}

fn validate(c: &Context) -> Result<(), LedgerError> {
    // Include future request/entry growth, every event snapshot and final output,
    // before creating any maps, allocating identities or cloning retained state.
    let s = &c.state;
    if c.calls.len() > 16 {
        return Err(LedgerError::Capacity);
    }
    let units = (|| {
        let mut n = 64usize
            .checked_add(c.version.len())?
            .checked_add(range_units(&s.view.retained)?)?;
        for d in &s.data {
            n = n.checked_add(24)?.checked_add(range_units(&d.retained)?)?;
        }
        for i in &s.images {
            n = n.checked_add(16)?.checked_add(range_units(&i.retained)?)?;
        }
        for t in &s.texts {
            n = n.checked_add(8)?.checked_add(range_units(&t.retained)?)?;
        }
        for v in s.canvases.iter().chain(&s.sprites) {
            n = n.checked_add(4)?.checked_add(range_units(&v.retained)?)?;
        }
        for v in &s.strings {
            n = n.checked_add(2)?.checked_add(v.units.len())?;
        }
        for v in &s.arrays {
            n = n
                .checked_add(4)?
                .checked_add(v.slots.len())?
                .checked_add(range_units(&v.retained)?)?;
        }
        for v in &s.game_objects {
            n = n.checked_add(4)?.checked_add(range_units(&v.retained)?)?;
        }
        n = n.checked_add(s.tween_requests.len().checked_mul(8)?)?;
        n = n.checked_add(s.native_entries.len().checked_mul(6)?)?;
        n = n
            .checked_add(c.tweens.len())?
            .checked_add(c.calls.len().checked_mul(8)?)?;
        n.checked_add(c.calls.len().checked_mul(36)?)
    })();
    let snapshots = c.calls.iter().try_fold(2usize, |n, call| {
        let events = if matches!(call, Call::Init { .. }) {
            s.arrays
                .iter()
                .map(|a| a.slots.len())
                .max()
                .unwrap_or(0)
                .checked_add(16)?
        } else {
            6
        };
        n.checked_add(events)?.checked_add(1)
    });
    if c.calls.len() > 16
        || units.is_none_or(|n| n > MAX_UNITS)
        || units
            .zip(snapshots)
            .and_then(|(n, m)| n.checked_mul(m))
            .is_none_or(|n| n > MAX_WORK)
    {
        return Err(LedgerError::Capacity);
    }
    let svc = &c.services;
    if c.version != CHARACTER_VIEW_PRESENTATION_NATIVE_V1
        || !svc.runtime_verified_inert
        || !svc.gc_verified_inert
        || !svc.ui_verified_inert
        || !svc.text_verified_inert
        || !svc.art_verified_inert
        || !svc.dotween_verified_inert
        || !svc.storage_verified
        || !svc.normal_completion_verified
        || !ranges_valid(&s.view.retained, &[(0x20, 0x90)])
        || s.view.identity.checked_add(0x90).is_none()
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut kinds = BTreeMap::new();
    bind(&mut kinds, s.view.identity, Kind::View)?;
    let mut unique = BTreeSet::new();
    macro_rules! records {
        ($records:expr, $kind:expr) => {
            for v in $records {
                if !unique.insert(v.identity) {
                    return Err(LedgerError::InvalidContext);
                }
                bind(&mut kinds, v.identity, $kind)?;
            }
        };
    }
    records!(&s.data, Kind::Data);
    records!(&s.images, Kind::Image);
    records!(&s.texts, Kind::Text);
    records!(&s.canvases, Kind::Canvas);
    records!(&s.sprites, Kind::Sprite);
    records!(&s.strings, Kind::String);
    records!(&s.arrays, Kind::Array);
    records!(&s.game_objects, Kind::GameObject);
    for id in [c.dotween_class, c.image_class, c.text_class] {
        bind(&mut kinds, id, Kind::Class)?;
    }
    if c.image_class == c.text_class
        || c.image_class == c.dotween_class
        || c.text_class == c.dotween_class
    {
        return Err(LedgerError::InvalidContext);
    }
    for id in [c.color_method, c.text_method, c.set_id_method] {
        bind(&mut kinds, id, Kind::Method)?;
    }
    if [c.color_method, c.text_method, c.set_id_method]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .len()
        != 3
    {
        return Err(LedgerError::InvalidContext);
    }
    for &id in &c.tweens {
        if !unique.insert(id) {
            return Err(LedgerError::InvalidContext);
        }
        bind(&mut kinds, id, Kind::Tween)?;
    }
    let reference = |id: Option<Identity>, kind| id.is_none_or(|id| kinds.get(&id) == Some(&kind));
    let v = &s.view;
    if ![v.bg, v.art, v.clipping, v.bgs]
        .into_iter()
        .all(|id| reference(id, Kind::Image))
        || !reference(v.data, Kind::Data)
        || !reference(v.text, Kind::Text)
        || !reference(v.borders, Kind::Array)
        || !reference(v.canvas, Kind::Canvas)
        || !reference(v.anim_id, Kind::String)
        || !reference(v.white_bg, Kind::Sprite)
    {
        return Err(LedgerError::InvalidContext);
    }
    for d in &s.data {
        if !reference(d.name, Kind::String)
            || !reference(d.background_sprite, Kind::Sprite)
            || !ranges_valid(&d.retained, &[(0x28, 0x30), (0xb8, 0xc0), (0xf8, 0x118)])
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    for i in &s.images {
        if i.class != c.image_class
            || !reference(i.sprite, Kind::Sprite)
            || !reference(i.game_object, Kind::GameObject)
            || !ranges_valid(&i.retained, &[(0, 8)])
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    for t in &s.texts {
        if t.class != c.text_class
            || !reference(t.value, Kind::String)
            || !ranges_valid(&t.retained, &[(0, 8)])
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    for a in &s.arrays {
        let len = a.length_bits as u32 as i32;
        if len < 0
            || len as usize > a.slots.len()
            || a.slots.len() > 64
            || !a.slots.iter().all(|&id| reference(id, Kind::Image))
            || !ranges_valid(&a.retained, &[(0x18, 0x20 + 8 * a.slots.len() as u64)])
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if s.canvases
        .iter()
        .chain(&s.sprites)
        .any(|v| !ranges_valid(&v.retained, &[]))
        || s.game_objects
            .iter()
            .any(|v| !ranges_valid(&v.retained, &[]))
    {
        return Err(LedgerError::InvalidContext);
    }
    let setup_valid = || {
        [v.art, v.clipping].into_iter().all(|id| {
            id.and_then(|id| s.images.iter().find(|i| i.identity == id))
                .is_some_and(|i| i.game_object.is_some())
        })
    };
    for call in &c.calls {
        let valid = match call {
            Call::AnimateIn { tween, .. } | Call::AnimateOut { tween, .. } => {
                reference(*tween, Kind::Tween)
            }
            Call::SetupArt { sprite, .. } => reference(*sprite, Kind::Sprite) && setup_valid(),
            Call::Init {
                data,
                uppercase_result,
                art_result,
                ..
            } => {
                reference(Some(*data), Kind::Data)
                    && reference(*uppercase_result, Kind::String)
                    && reference(*art_result, Kind::Sprite)
                    && setup_valid()
                    && [v.bg, v.bgs, v.art, v.clipping, v.text, v.borders]
                        .into_iter()
                        .all(|id| id.is_some())
                    && s.data
                        .iter()
                        .find(|d| d.identity == *data)
                        .is_some_and(|d| d.name.is_some())
                    && v.borders
                        .and_then(|id| s.arrays.iter().find(|a| a.identity == id))
                        .is_some_and(|a| {
                            a.slots[..a.length_bits as u32 as usize]
                                .iter()
                                .all(Option::is_some)
                        })
            }
        };
        if !valid {
            return Err(LedgerError::InvalidContext);
        }
    }
    // Retained chronology must already satisfy this exact standalone profile.
    for e in &s.native_entries {
        let valid = e.view == v.identity
            && match e.method {
                Method::AnimateIn | Method::AnimateOut => {
                    e.data.is_none() && e.sprite.is_none() && e.type_bits.is_none()
                }
                Method::Init => {
                    e.data.is_some()
                        && reference(e.data, Kind::Data)
                        && e.sprite.is_none()
                        && e.type_bits.is_none()
                }
                Method::SetupArt => {
                    e.data.is_none() && reference(e.sprite, Kind::Sprite) && e.type_bits.is_some()
                }
            };
        if !valid {
            return Err(LedgerError::InvalidContext);
        }
    }
    for e in &s.tween_requests {
        let valid = match e {
            Event::DotweenKillService {
                id,
                rdx_bits,
                method_bits,
            } => reference(*id, Kind::String) && rdx_bits & 255 == 1 && *method_bits == 0,
            Event::DofadeService {
                canvas,
                end_bits,
                duration_bits,
            } => {
                reference(*canvas, Kind::Canvas)
                    && [0, WHITE[0]].contains(end_bits)
                    && *duration_bits == FADE_DURATION_BITS
            }
            Event::SetIdService { tween, id, method } => {
                reference(*tween, Kind::Tween)
                    && reference(*id, Kind::String)
                    && *method == c.set_id_method
            }
            _ => false,
        };
        if !valid {
            return Err(LedgerError::InvalidContext);
        }
    }
    Ok(())
}

fn observe(s: &State, steps: &mut Vec<Step>, event: Event) {
    steps.push(Step {
        event,
        state: s.clone(),
    });
}

fn color(c: &Context, s: &mut State, steps: &mut Vec<Step>, id: Identity, bits: [u32; 4]) {
    observe(
        s,
        steps,
        Event::ImageColorService {
            image: id,
            bits,
            method: c.color_method,
        },
    );
    s.images
        .iter_mut()
        .find(|i| i.identity == id)
        .expect("validated image")
        .color_bits = bits;
}

fn sprite(s: &mut State, steps: &mut Vec<Step>, id: Identity, sprite: Option<Identity>) {
    observe(s, steps, Event::ImageSpriteService { image: id, sprite });
    s.images
        .iter_mut()
        .find(|i| i.identity == id)
        .expect("validated image")
        .sprite = sprite;
}

fn setup(s: &mut State, steps: &mut Vec<Step>, art_sprite: Option<Identity>, type_bits: u32) {
    s.native_entries.push(NativeEntry {
        method: Method::SetupArt,
        view: s.view.identity,
        data: None,
        sprite: art_sprite,
        type_bits: Some(type_bits),
    });
    let (selected, other) = if type_bits == 10 {
        (s.view.clipping, s.view.art)
    } else {
        (s.view.art, s.view.clipping)
    };
    let selected = selected.expect("normal profile");
    let other = other.expect("normal profile");
    let go = s
        .images
        .iter()
        .find(|i| i.identity == selected)
        .unwrap()
        .game_object
        .unwrap();
    observe(
        s,
        steps,
        Event::ImageGameObjectService {
            image: selected,
            result: go,
        },
    );
    observe(
        s,
        steps,
        Event::ImageSetActiveService {
            game_object: go,
            rdx_bits: 1,
            value: true,
        },
    );
    s.game_objects
        .iter_mut()
        .find(|g| g.identity == go)
        .unwrap()
        .active = true;
    sprite(s, steps, selected, art_sprite);
    let go = s
        .images
        .iter()
        .find(|i| i.identity == other)
        .unwrap()
        .game_object
        .unwrap();
    observe(
        s,
        steps,
        Event::ImageGameObjectService {
            image: other,
            result: go,
        },
    );
    observe(
        s,
        steps,
        Event::ImageSetActiveService {
            game_object: go,
            rdx_bits: 0,
            value: false,
        },
    );
    s.game_objects
        .iter_mut()
        .find(|g| g.identity == go)
        .unwrap()
        .active = false;
}

pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut s = c.state.clone();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    for call in &c.calls {
        match call {
            Call::AnimateIn {
                kill_rdx_before_dl,
                tween,
            }
            | Call::AnimateOut {
                kill_rdx_before_dl,
                tween,
            } => {
                let fade_in = matches!(call, Call::AnimateIn { .. });
                s.native_entries.push(NativeEntry {
                    method: if fade_in {
                        Method::AnimateIn
                    } else {
                        Method::AnimateOut
                    },
                    view: s.view.identity,
                    data: None,
                    sprite: None,
                    type_bits: None,
                });
                let flag = if fade_in { 3 } else { 4 };
                if s.metadata_flags[flag] == 0 {
                    observe(
                        &s,
                        &mut steps,
                        Event::MetadataService {
                            slot_rva: DOTWEEN_CLASS_SLOT_RVA,
                        },
                    );
                    observe(
                        &s,
                        &mut steps,
                        Event::MetadataService {
                            slot_rva: SET_ID_METHOD_RVA,
                        },
                    );
                    s.metadata_flags[flag] = 1;
                }
                let id = s.view.anim_id;
                if s.dotween_class_initialized == 0 {
                    observe(
                        &s,
                        &mut steps,
                        Event::ClassInitializationService {
                            class: c.dotween_class,
                        },
                    );
                    s.dotween_class_initialized = 1;
                }
                let e = Event::DotweenKillService {
                    id,
                    rdx_bits: (kill_rdx_before_dl & !255) | 1,
                    method_bits: 0,
                };
                observe(&s, &mut steps, e.clone());
                s.tween_requests.push(e);
                let e = Event::DofadeService {
                    canvas: s.view.canvas,
                    end_bits: if fade_in { WHITE[0] } else { 0 },
                    duration_bits: FADE_DURATION_BITS,
                };
                observe(&s, &mut steps, e.clone());
                s.tween_requests.push(e);
                let e = Event::SetIdService {
                    tween: *tween,
                    id: s.view.anim_id,
                    method: c.set_id_method,
                };
                observe(&s, &mut steps, e.clone());
                s.tween_requests.push(e);
            }
            Call::SetupArt {
                sprite,
                type_register_bits,
            } => setup(&mut s, &mut steps, *sprite, *type_register_bits as u32),
            Call::Init {
                data,
                uppercase_result,
                art_result,
                art_type_return_bits,
            } => {
                s.native_entries.push(NativeEntry {
                    method: Method::Init,
                    view: s.view.identity,
                    data: Some(*data),
                    sprite: None,
                    type_bits: None,
                });
                s.view.data = Some(*data);
                observe(
                    &s,
                    &mut steps,
                    Event::ViewDataBarrierService {
                        address: s.view.identity + 0x28,
                        data: *data,
                    },
                );
                let d = s.data.iter().find(|d| d.identity == *data).unwrap();
                let (bg_bits, border_bits, background, name) = (
                    d.bg_color_bits,
                    d.border_color_bits,
                    d.background_sprite,
                    d.name.unwrap(),
                );
                let id = s.view.bgs.unwrap();
                color(c, &mut s, &mut steps, id, bg_bits);
                let array = s
                    .arrays
                    .iter()
                    .position(|a| Some(a.identity) == s.view.borders)
                    .unwrap();
                let len = s.arrays[array].length_bits as u32 as usize;
                for i in 0..len {
                    let id = s.arrays[array].slots[i].unwrap();
                    color(c, &mut s, &mut steps, id, border_bits);
                }
                let id = s.view.bg.unwrap();
                sprite(&mut s, &mut steps, id, background);
                let text = s.view.text.unwrap();
                observe(
                    &s,
                    &mut steps,
                    Event::UppercaseService {
                        input: name,
                        result: *uppercase_result,
                    },
                );
                observe(
                    &s,
                    &mut steps,
                    Event::TextSetterService {
                        text,
                        value: *uppercase_result,
                        method: c.text_method,
                    },
                );
                s.texts
                    .iter_mut()
                    .find(|t| t.identity == text)
                    .unwrap()
                    .value = *uppercase_result;
                let id = s.view.art.unwrap();
                color(c, &mut s, &mut steps, id, WHITE);
                let id = s.view.clipping.unwrap();
                color(c, &mut s, &mut steps, id, WHITE);
                observe(&s, &mut steps, Event::GetArtService { data: *data });
                observe(&s, &mut steps, Event::GetArtTypeService { data: *data });
                setup(
                    &mut s,
                    &mut steps,
                    *art_result,
                    *art_type_return_bits as u32,
                );
                let id = s.view.bg.unwrap();
                color(c, &mut s, &mut steps, id, WHITE);
            }
        }
        completed.push(s.clone());
    }
    Ok(Replay {
        state: s,
        steps,
        completed,
    })
}

#[cfg(test)]
#[path = "character_view_presentation_tests.rs"]
mod tests;
