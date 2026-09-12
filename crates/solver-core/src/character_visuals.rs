//! Offline caller replay for Characters rotation and highlight loops.
//! Transform, Unity equality, enumeration and highlight effects are explicit
//! stable services. No engine rendering, tweening or scheduler is reconstructed.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHARACTER_VISUALS_NATIVE_V1: &str = "character_visuals_native_v1";
pub const MAX_ENTRIES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Rotate,
    Highlight,
    Disable,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gateway {
    Enumerator,
    MoveNext,
    Transform,
    LocalEuler,
    WorldEuler,
    ClassInit,
    UnityNonnull,
    Show,
    Disable,
    Dispose,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailurePoint {
    pub gateway: Gateway,
    pub occurrence: u16,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Character {
    pub identity: u16,
    pub transform: Option<u16>,
    pub icon: Option<u16>,
    pub icon_transform: Option<u16>,
    pub highlight: Option<u16>,
    /// Explicit Object.op_Inequality result, including destroyed-object cases.
    pub unity_nonnull: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub rule_version: String,
    pub operation: Operation,
    pub stable_services: bool,
    pub native_mxcsr: u32,
    pub board: Option<Vec<Option<u16>>>,
    pub selected: Option<Vec<Option<u16>>>,
    pub characters: Vec<Character>,
    /// Native signed size read for the rotation divisor. A mismatch is only
    /// supported as an explicitly admitted adversarial enumerator fixture.
    pub board_count: i32,
    pub allow_adversarial_count: bool,
    pub zero_vector_bits: [u32; 3],
    pub object_initialized: bool,
    pub null_unity_nonnull: bool,
    pub failure: Option<FailurePoint>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Board,
    Selected,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Enumerator { source: Source },
    MoveNext,
    Transform { target: u16 },
    LocalEuler { target: u16, bits: [u32; 3] },
    WorldEuler { target: u16, bits: [u32; 3] },
    ClassInit,
    UnityNonnull { target: Option<u16>, result: bool },
    Show { target: u16 },
    Disable { target: u16 },
    Dispose,
}
impl Event {
    fn gateway(&self) -> Gateway {
        match self {
            Self::Enumerator { .. } => Gateway::Enumerator,
            Self::MoveNext => Gateway::MoveNext,
            Self::Transform { .. } => Gateway::Transform,
            Self::LocalEuler { .. } => Gateway::LocalEuler,
            Self::WorldEuler { .. } => Gateway::WorldEuler,
            Self::ClassInit => Gateway::ClassInit,
            Self::UnityNonnull { .. } => Gateway::UnityNonnull,
            Self::Show { .. } => Gateway::Show,
            Self::Disable { .. } => Gateway::Disable,
            Self::Dispose => Gateway::Dispose,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Failure {
    Null,
    Gateway(FailurePoint),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replay {
    pub events: Vec<Event>,
    /// Successful setter effects keyed by transform identity. Local and world
    /// channels stay distinct; parent-space composition is an engine boundary.
    pub local_rotations: BTreeMap<u16, [u32; 3]>,
    pub world_rotations: BTreeMap<u16, [u32; 3]>,
    pub highlights: Vec<(Gateway, u16)>,
    pub object_initialized: bool,
    pub error: Option<Failure>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsupported {
    Context,
    Capacity,
    FloatingEnvironment,
}

fn mxcsr() -> Option<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut value = 0u32;
        // Read only: replay never changes caller floating-point controls.
        unsafe {
            core::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut value, options(nostack, preserves_flags));
        }
        Some(value)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}
fn native_angle(count: i32, index: i32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        let result: f32;
        // Preserve the audited CVTSI2SS / DIVSS / MULSS ordering, including
        // the exact x86 indefinite NaN for index zero times infinity.
        unsafe {
            core::arch::asm!(
                "cvtsi2ss {divisor}, {count:e}",
                "divss {step}, {divisor}",
                "cvtsi2ss {result}, {index:e}",
                "mulss {result}, {step}",
                divisor = out(xmm_reg) _, step = inout(xmm_reg) 360.0f32 => _,
                result = out(xmm_reg) result, count = in(reg) count, index = in(reg) index,
                options(nostack, nomem, preserves_flags)
            );
        }
        result.to_bits()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (count, index);
        unreachable!("rejected before arithmetic")
    }
}

impl Replay {
    fn emit(&mut self, c: &Context, e: Event) -> Result<(), Failure> {
        let gateway = e.gateway();
        self.events.push(e);
        let occurrence = self
            .events
            .iter()
            .filter(|e| e.gateway() == gateway)
            .count() as u16;
        let point = FailurePoint {
            gateway,
            occurrence,
        };
        if c.failure == Some(point) {
            Err(Failure::Gateway(point))
        } else {
            Ok(())
        }
    }
    fn execute(&mut self, c: &Context) -> Result<(), Failure> {
        let (source, list) = if c.operation == Operation::Highlight {
            (Source::Selected, &c.selected)
        } else {
            (Source::Board, &c.board)
        };
        let list = list.as_ref().ok_or(Failure::Null)?;
        self.emit(c, Event::Enumerator { source })?;
        for (index, identity) in list.iter().enumerate() {
            self.emit(c, Event::MoveNext)?;
            let character =
                identity.and_then(|id| c.characters.iter().find(|ch| ch.identity == id));
            if c.operation == Operation::Rotate {
                let ch = character.ok_or(Failure::Null)?;
                self.emit(
                    c,
                    Event::Transform {
                        target: ch.identity,
                    },
                )?;
                let transform = ch.transform.ok_or(Failure::Null)?;
                let bits = [0, 0, native_angle(c.board_count, index as i32)];
                self.emit(
                    c,
                    Event::LocalEuler {
                        target: transform,
                        bits,
                    },
                )?;
                self.local_rotations.insert(transform, bits);
                let icon = ch.icon.ok_or(Failure::Null)?;
                self.emit(c, Event::Transform { target: icon })?;
                let target = ch.icon_transform.ok_or(Failure::Null)?;
                let bits = c.zero_vector_bits;
                self.emit(c, Event::WorldEuler { target, bits })?;
                self.world_rotations.insert(target, bits);
            } else {
                if !self.object_initialized {
                    self.emit(c, Event::ClassInit)?;
                    self.object_initialized = true;
                }
                let result = character.map_or(c.null_unity_nonnull, |ch| ch.unity_nonnull);
                self.emit(
                    c,
                    Event::UnityNonnull {
                        target: *identity,
                        result,
                    },
                )?;
                if !result {
                    continue;
                }
                let target = character
                    .ok_or(Failure::Null)?
                    .highlight
                    .ok_or(Failure::Null)?;
                let event = if c.operation == Operation::Highlight {
                    Event::Show { target }
                } else {
                    Event::Disable { target }
                };
                let gateway = event.gateway();
                self.emit(c, event)?;
                self.highlights.push((gateway, target));
            }
        }
        self.emit(c, Event::MoveNext)?;
        self.emit(c, Event::Dispose)
    }
}

pub fn replay(c: &Context) -> Result<Replay, Unsupported> {
    if c.rule_version != CHARACTER_VISUALS_NATIVE_V1
        || !c.stable_services
        || c.failure.is_some_and(|f| f.occurrence == 0)
        || (!c.allow_adversarial_count
            && c.board
                .as_ref()
                .is_some_and(|l| c.board_count != l.len() as i32))
    {
        return Err(Unsupported::Context);
    }
    if c.characters.len() > MAX_ENTRIES
        || c.board.as_ref().is_some_and(|l| l.len() > MAX_ENTRIES)
        || c.selected.as_ref().is_some_and(|l| l.len() > MAX_ENTRIES)
    {
        return Err(Unsupported::Capacity);
    }
    let mut identities = std::collections::BTreeSet::new();
    for ch in &c.characters {
        if !identities.insert(ch.identity) {
            return Err(Unsupported::Context);
        }
    }
    if c.board
        .iter()
        .chain(c.selected.iter())
        .flatten()
        .flatten()
        .any(|id| !identities.contains(id))
    {
        return Err(Unsupported::Context);
    }
    // Every stable get_transform provider must return the same reference for
    // the same component, including shared icons and Transform self-identity.
    let mut transforms = BTreeMap::new();
    for ch in &c.characters {
        for (id, result) in std::iter::once((ch.identity, ch.transform))
            .chain(ch.icon.map(|id| (id, ch.icon_transform)))
        {
            if transforms
                .insert(id, result)
                .is_some_and(|previous| previous != result)
            {
                return Err(Unsupported::Context);
            }
        }
    }
    if c.operation == Operation::Rotate
        && (c.native_mxcsr & !0x3f != 0x1f80 || !mxcsr().is_some_and(|v| v & !0x3f == 0x1f80))
    {
        return Err(Unsupported::FloatingEnvironment);
    }
    let mut out = Replay {
        events: vec![],
        local_rotations: BTreeMap::new(),
        world_rotations: BTreeMap::new(),
        highlights: vec![],
        object_initialized: c.object_initialized,
        error: None,
    };
    out.error = out.execute(c).err();
    Ok(out)
}

#[cfg(test)]
#[path = "character_visual_tests.rs"]
mod tests;
