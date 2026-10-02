//! Guarded inert normal CharacterData.GenerateCharacterId caller replay.
//! Whole RNG/boxing/cast/format/runtime services are explicit supplied inputs.
use super::character_initialization::Identity;
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHARACTER_DATA_IDENTITY_NATIVE_V1: &str = "character_data_identity_native_v1";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Data,
    DataClass,
    Array,
    ArrayClass,
    ElementClass,
    IntClass,
    ObjectClass,
    Text,
    Box,
    Exception,
    Stack,
}
impl Kind {
    fn size(self) -> usize {
        match self {
            Self::Data => 512,
            Self::DataClass
            | Self::ArrayClass
            | Self::ElementClass
            | Self::IntClass
            | Self::ObjectClass => 256,
            Self::Stack => 152,
            _ => 128,
        }
    }
    fn object(self) -> bool {
        matches!(self, Self::Text | Self::Box)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub identity: Identity,
    pub kind: Kind,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Slot {
    IntType,
    ArrayType,
    FormatLiteral,
}
impl Slot {
    pub fn rva(self) -> u32 {
        match self {
            Self::IntType => 0x2707130,
            Self::ArrayType => 0x2720110,
            Self::FormatLiteral => 0x26EEE60,
        }
    }
    fn kind(self) -> Kind {
        match self {
            Self::IntType => Kind::IntClass,
            Self::ArrayType => Kind::ArrayClass,
            Self::FormatLiteral => Kind::Text,
        }
    }
}
const SLOTS: [Slot; 3] = [Slot::IntType, Slot::ArrayType, Slot::FormatLiteral];
const META: [u32; 3] = [0x3B453A, 0x3B4546, 0x3B4552];
const RNG: [u32; 8] = [
    0x3B45E3, 0x3B463F, 0x3B469B, 0x3B46F7, 0x3B4753, 0x3B47AF, 0x3B480B, 0x3B4867,
];
const BOX: [u32; 8] = [
    0x3B45F8, 0x3B4654, 0x3B46B0, 0x3B470C, 0x3B4768, 0x3B47C4, 0x3B4820, 0x3B487C,
];
const SCRATCH: [usize; 8] = [0x60, 0x70, 0x78, 0x20, 0x24, 0x28, 0x2C, 0x30];
const CAST: [u32; 9] = [
    0x3B45B3, 0x3B460F, 0x3B466B, 0x3B46C7, 0x3B4723, 0x3B477F, 0x3B47DB, 0x3B4837, 0x3B4893,
];
const BARRIER: [u32; 10] = [
    0x3B45D5, 0x3B4631, 0x3B468D, 0x3B46E9, 0x3B4745, 0x3B47A1, 0x3B47FD, 0x3B4859, 0x3B48B5,
    0x3B48D5,
];
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// RAX, RCX, RDX, R8, R9, R10, R11.
    pub volatile_registers: [u64; 7],
    pub volatile_xmm_hex: [String; 6],
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Service {
    Metadata,
    IsEmpty,
    ArrayAllocate,
    ObjectName,
    Random,
    Box,
    TypeCheck,
    ReferenceBarrier,
    Format,
    MakeException,
    RaiseException,
    NullException,
    BoundsException,
}
const SERVICES: [Service; 13] = [
    Service::Metadata,
    Service::IsEmpty,
    Service::ArrayAllocate,
    Service::ObjectName,
    Service::Random,
    Service::Box,
    Service::TypeCheck,
    Service::ReferenceBarrier,
    Service::Format,
    Service::MakeException,
    Service::RaiseException,
    Service::NullException,
    Service::BoundsException,
];
// Nullable supplied values must be explicitly present as null or an identity;
// a missing field is not evidence for a successful service's null output.
fn required_nullable<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Identity>, D::Error> {
    Option::<Identity>::deserialize(deserializer)
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Completed {
    Metadata {
        slot: Slot,
        value: Identity,
    },
    IsEmpty {
        #[serde(deserialize_with = "required_nullable")]
        value: Option<Identity>,
        method: u64,
    },
    ArrayAllocate {
        class: Identity,
        length: u64,
        #[serde(deserialize_with = "required_nullable")]
        result: Option<Identity>,
    },
    ObjectName {
        owner: Identity,
        method: u64,
        #[serde(deserialize_with = "required_nullable")]
        result: Option<Identity>,
    },
    Random {
        index: u8,
        min: i32,
        max: i32,
        method: u64,
        return_bits: u64,
    },
    Box {
        index: u8,
        class: Identity,
        scratch: Identity,
        value_bits: u32,
        #[serde(deserialize_with = "required_nullable")]
        result: Option<Identity>,
    },
    TypeCheck {
        index: u8,
        input: Identity,
        target: Identity,
        #[serde(deserialize_with = "required_nullable")]
        result: Option<Identity>,
    },
    ReferenceBarrier {
        index: u8,
        owner: Identity,
        offset: usize,
        #[serde(deserialize_with = "required_nullable")]
        value: Option<Identity>,
    },
    Format {
        literal: Identity,
        array: Identity,
        method: u64,
        #[serde(deserialize_with = "required_nullable")]
        result: Option<Identity>,
    },
    MakeException {
        index: u8,
        #[serde(deserialize_with = "required_nullable")]
        result: Option<Identity>,
    },
    RaiseException {
        index: u8,
        #[serde(deserialize_with = "required_nullable")]
        exception: Option<Identity>,
        method: u64,
    },
    NullException {},
    BoundsException {},
}
impl Completed {
    fn service(&self) -> Service {
        match self {
            Self::Metadata { .. } => Service::Metadata,
            Self::IsEmpty { .. } => Service::IsEmpty,
            Self::ArrayAllocate { .. } => Service::ArrayAllocate,
            Self::ObjectName { .. } => Service::ObjectName,
            Self::Random { .. } => Service::Random,
            Self::Box { .. } => Service::Box,
            Self::TypeCheck { .. } => Service::TypeCheck,
            Self::ReferenceBarrier { .. } => Service::ReferenceBarrier,
            Self::Format { .. } => Service::Format,
            Self::MakeException { .. } => Service::MakeException,
            Self::RaiseException { .. } => Service::RaiseException,
            Self::NullException { .. } => Service::NullException,
            Self::BoundsException { .. } => Service::BoundsException,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: Vec<Record>,
    pub metadata_slots: BTreeMap<Slot, Identity>,
    pub metadata_flag: u8,
    pub native_entries: Vec<Entry>,
    pub service_history: BTreeMap<Service, Vec<Completed>>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub entry: Entry,
    pub empty_return_bits: u64,
    #[serde(deserialize_with = "required_nullable")]
    pub array_result: Option<Identity>,
    #[serde(deserialize_with = "required_nullable")]
    pub name_result: Option<Identity>,
    pub random_return_bits: [u64; 8],
    pub box_results: [Option<Identity>; 8],
    pub type_results: [Option<Identity>; 9],
    #[serde(deserialize_with = "required_nullable")]
    pub format_result: Option<Identity>,
    pub barrier_return_bits: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub calls: Vec<Call>,
    pub native_base: u64,
    pub stack_window: Identity,
    pub return_sentinel: u64,
    /// RBX, RBP, RSI, RDI, R12, R13, R14, R15; native saves only RBX/RSI/RDI here.
    pub nonvolatile_registers: [u64; 8],
    pub volatile_return_registers: [u64; 7],
    pub volatile_return_xmm_hex: [String; 6],
    pub storage_verified: bool,
    pub inert_services_verified: bool,
    pub normal_completion_verified: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub call: usize,
    pub ordinal: usize,
    pub event: Completed,
    pub volatile_registers: [u64; 7],
    pub volatile_xmm_hex: [String; 6],
    pub raw_args: [u64; 4],
    pub native_site_rva: u32,
    pub caller_return_bits: u64,
    pub rsp_bits: u64,
    pub state: State,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replay {
    pub steps: Vec<Step>,
    pub completed: Vec<State>,
    pub final_registers: Vec<[u64; 7]>,
    pub final_xmm_hex: Vec<[String; 6]>,
    pub final_rsp_bits: Vec<u64>,
    pub final_state: State,
}
fn xmm_valid(v: &[String; 6]) -> bool {
    v.iter().all(|s| {
        s.len() == 32
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}
fn units(c: &Context) -> Option<usize> {
    let mut n = 128usize.checked_add(c.version.len())?;
    for r in &c.state.records {
        n = n.checked_add(4)?.checked_add(r.bytes.len())?;
    }
    for e in c
        .state
        .native_entries
        .iter()
        .chain(c.calls.iter().map(|v| &v.entry))
    {
        n = n.checked_add(8)?;
        for x in &e.volatile_xmm_hex {
            n = n.checked_add(x.len())?;
        }
    }
    for x in &c.volatile_return_xmm_hex {
        n = n.checked_add(x.len())?;
    }
    n = n.checked_add(c.state.metadata_slots.len().checked_mul(3)?)?;
    for h in c.state.service_history.values() {
        n = n.checked_add(2)?.checked_add(h.len().checked_mul(6)?)?;
    }
    n.checked_add(c.calls.len().checked_mul(48)?)
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    let n = units(c);
    // Each future entry is 200 units; at most 42 complete history records use
    // six units each. 42 entry snapshots + completion per call, initial/final,
    // plus 224 raw-register/event/caller units per step and 200 final ABI units
    // per call, separately from the complete snapshot cloning allowance.
    let work = n.and_then(|v| {
        v.checked_add(c.calls.len().checked_mul(452)?)?
            .checked_mul(c.calls.len().checked_mul(43)?.checked_add(2)?)?
            .checked_add(
                c.calls
                    .len()
                    .checked_mul(42usize.checked_mul(224)?.checked_add(200)?)?,
            )
    });
    if c.state.records.len() > 64
        || c.calls.len() > 8
        || c.state.native_entries.len() > 64
        || c.state.service_history.len() > 13
        || c.state.service_history.values().any(|h| h.len() > 512)
        || n.is_none_or(|v| v > 65536)
        || work.is_none_or(|v| v > 4_194_304)
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != CHARACTER_DATA_IDENTITY_NATIVE_V1
        || !c.storage_verified
        || !c.inert_services_verified
        || !c.normal_completion_verified
        || c.native_base == 0
        || c.native_base.checked_add(0x288C4DD).is_none()
        || c.return_sentinel == 0
        || c.stack_window % 16 != 0
        || !xmm_valid(&c.volatile_return_xmm_hex)
        || c.state.service_history.len() != 13
        || SERVICES
            .iter()
            .any(|s| !c.state.service_history.contains_key(s))
    {
        return Err(LedgerError::InvalidContext);
    }
    // Logical roots and the flag are separate physical storage. The exclusive
    // highest endpoint above validates every following addition before use.
    let global_windows = [
        (c.native_base + u64::from(Slot::IntType.rva()), 8),
        (c.native_base + u64::from(Slot::ArrayType.rva()), 8),
        (c.native_base + u64::from(Slot::FormatLiteral.rva()), 8),
        (c.native_base + 0x288C4DC, 1),
    ];
    let mut records = BTreeMap::new();
    for r in &c.state.records {
        if r.identity == 0
            || r.bytes.len() != r.kind.size()
            || records.insert(r.identity, r).is_some()
        {
            return Err(LedgerError::InvalidContext);
        }
        let Some(end) = r.identity.checked_add(r.bytes.len() as u64) else {
            return Err(LedgerError::InvalidContext);
        };
        if global_windows
            .iter()
            .any(|(start, size)| r.identity < start + size && *start < end)
        {
            return Err(LedgerError::InvalidContext);
        }
        if c.state.records.iter().any(|o| {
            o.identity != r.identity
                && o.identity < end
                && o.identity
                    .checked_add(o.bytes.len() as u64)
                    .is_none_or(|v| v > r.identity)
        }) {
            return Err(LedgerError::InvalidContext);
        }
    }
    let reference = |id: Identity, k: Kind| records.get(&id).is_some_and(|r| r.kind == k);
    let optional = |id: Option<Identity>, k: Kind| id.is_none_or(|v| reference(v, k));
    let object = |id: Identity| records.get(&id).is_some_and(|r| r.kind.object());
    let optional_object = |id: Option<Identity>| id.is_none_or(object);
    if !reference(c.stack_window, Kind::Stack)
        || c.state
            .records
            .iter()
            .filter(|r| r.kind == Kind::Stack)
            .count()
            != 1
        || c.state.metadata_slots.len() != 3
        || SLOTS.iter().any(|s| {
            !c.state
                .metadata_slots
                .get(s)
                .is_some_and(|v| reference(*v, s.kind()))
        })
    {
        return Err(LedgerError::InvalidContext);
    }
    let q = |id: Identity, off: usize| {
        u64::from_le_bytes(records[&id].bytes[off..off + 8].try_into().unwrap())
    };
    if q(c.stack_window, 0x68) != c.return_sentinel {
        return Err(LedgerError::InvalidContext);
    }
    let entry_valid = |e: &Entry, nullable: bool| {
        xmm_valid(&e.volatile_xmm_hex)
            && (nullable && e.volatile_registers[1] == 0
                || reference(e.volatile_registers[1], Kind::Data))
    };
    if c.state.native_entries.iter().any(|e| !entry_valid(e, true)) {
        return Err(LedgerError::InvalidContext);
    }
    for r in &c.state.records {
        if r.kind == Kind::Data
            && !optional(
                (q(r.identity, 0x18) != 0).then_some(q(r.identity, 0x18)),
                Kind::Text,
            )
        {
            return Err(LedgerError::InvalidContext);
        }
        if r.kind == Kind::Array {
            if !optional(
                (q(r.identity, 0) != 0).then_some(q(r.identity, 0)),
                Kind::ArrayClass,
            ) {
                return Err(LedgerError::InvalidContext);
            }
            for j in 0..9 {
                let v = q(r.identity, 0x20 + j * 8);
                if v != 0 && !object(v) {
                    return Err(LedgerError::InvalidContext);
                }
            }
        }
        if r.kind == Kind::ArrayClass
            && !optional(
                (q(r.identity, 0x40) != 0).then_some(q(r.identity, 0x40)),
                Kind::ElementClass,
            )
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    for call in &c.calls {
        if !entry_valid(&call.entry, false)
            || !optional(call.array_result, Kind::Array)
            || !optional(call.name_result, Kind::Text)
            || !optional(call.format_result, Kind::Text)
            || call.box_results.iter().any(|v| !optional_object(*v))
            || call.type_results.iter().any(|v| !optional_object(*v))
        {
            return Err(LedgerError::InvalidContext);
        }
        if call.empty_return_bits as u8 != 0 {
            let Some(array) = call.array_result else {
                return Err(LedgerError::InvalidContext);
            };
            let class = q(array, 0);
            if !reference(class, Kind::ArrayClass)
                || !reference(q(class, 0x40), Kind::ElementClass)
                || (q(array, 0x18) as u32) < 9
            {
                return Err(LedgerError::InvalidContext);
            }
            let inputs = std::iter::once(call.name_result).chain(call.box_results);
            if inputs
                .zip(call.type_results)
                .any(|(input, output)| input.is_some() && output.is_none())
            {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    for (service, history) in &c.state.service_history {
        for e in history {
            let valid = e.service() == *service
                && match e {
                    Completed::Metadata { slot, value } => reference(*value, slot.kind()),
                    Completed::IsEmpty { value, method } => {
                        *method == 0 && optional(*value, Kind::Text)
                    }
                    Completed::ArrayAllocate {
                        class,
                        length,
                        result,
                    } => {
                        reference(*class, Kind::ArrayClass)
                            && *length == 9
                            && optional(*result, Kind::Array)
                    }
                    Completed::ObjectName {
                        owner,
                        method,
                        result,
                    } => {
                        reference(*owner, Kind::Data)
                            && *method == 0
                            && optional(*result, Kind::Text)
                    }
                    Completed::Random {
                        index,
                        min,
                        max,
                        method,
                        ..
                    } => *index < 8 && *min == 0 && *max == 10 && *method == 0,
                    Completed::Box {
                        index,
                        class,
                        scratch,
                        result,
                        ..
                    } => {
                        usize::from(*index) < 8
                            && reference(*class, Kind::IntClass)
                            && *scratch
                                == c.stack_window + 0x10 + SCRATCH[usize::from(*index)] as u64
                            && optional_object(*result)
                    }
                    Completed::TypeCheck {
                        index,
                        input,
                        target,
                        result,
                    } => {
                        *index < 9
                            && object(*input)
                            && reference(*target, Kind::ElementClass)
                            && optional_object(*result)
                    }
                    Completed::ReferenceBarrier {
                        index,
                        owner,
                        offset,
                        value,
                    } => {
                        if *index < 9 {
                            reference(*owner, Kind::Array)
                                && *offset == 0x20 + usize::from(*index) * 8
                                && optional_object(*value)
                        } else {
                            *index == 9
                                && reference(*owner, Kind::Data)
                                && *offset == 0x18
                                && optional(*value, Kind::Text)
                        }
                    }
                    Completed::Format {
                        literal,
                        array,
                        method,
                        result,
                    } => {
                        reference(*literal, Kind::Text)
                            && reference(*array, Kind::Array)
                            && *method == 0
                            && optional(*result, Kind::Text)
                    }
                    Completed::MakeException { index, result } => {
                        *index < 9 && optional(*result, Kind::Exception)
                    }
                    Completed::RaiseException {
                        index,
                        exception,
                        method,
                    } => *index < 9 && *method == 0 && optional(*exception, Kind::Exception),
                    Completed::NullException {} | Completed::BoundsException {} => true,
                };
            if !valid {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    Ok(())
}
fn write(s: &mut State, id: Identity, off: usize, bytes: &[u8]) {
    s.records
        .iter_mut()
        .find(|r| r.identity == id)
        .expect("validated storage")
        .bytes[off..off + bytes.len()]
        .copy_from_slice(bytes);
}
fn q(s: &State, id: Identity, off: usize) -> u64 {
    u64::from_le_bytes(
        s.records
            .iter()
            .find(|r| r.identity == id)
            .expect("validated reference")
            .bytes[off..off + 8]
            .try_into()
            .unwrap(),
    )
}
fn ptr(v: Option<Identity>) -> u64 {
    v.unwrap_or(0)
}
fn emit(
    steps: &mut Vec<Step>,
    s: &mut State,
    c: &Context,
    call: usize,
    counts: &mut BTreeMap<Service, usize>,
    e: Completed,
    regs: [u64; 7],
    xmm: [String; 6],
    site: u32,
) {
    let service = e.service();
    let count = counts.entry(service).or_default();
    *count += 1;
    let caller = c.native_base + u64::from(site) + 5;
    write(s, c.stack_window, 8, &caller.to_le_bytes());
    steps.push(Step {
        call,
        ordinal: *count,
        event: e.clone(),
        volatile_registers: regs,
        volatile_xmm_hex: xmm,
        raw_args: [regs[1], regs[2], regs[3], regs[4]],
        native_site_rva: site,
        caller_return_bits: caller,
        rsp_bits: c.stack_window + 8,
        state: s.clone(),
    });
    s.service_history
        .get_mut(&service)
        .expect("complete history shape")
        .push(e);
}
/// Validate every current call/history/reference and total future cloning work
/// before cloning the initial state or reaching any modeled store.
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut s = c.state.clone();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    let mut final_registers = Vec::new();
    let mut final_xmm_hex = Vec::new();
    let mut final_rsp_bits = Vec::new();
    for (i, call) in c.calls.iter().enumerate() {
        s.native_entries.push(call.entry.clone());
        let owner = call.entry.volatile_registers[1];
        let mut regs = call.entry.volatile_registers;
        let mut xmm = call.entry.volatile_xmm_hex.clone();
        let mut counts = BTreeMap::new();
        write(
            &mut s,
            c.stack_window,
            0x60,
            &c.nonvolatile_registers[2].to_le_bytes(),
        );
        write(
            &mut s,
            c.stack_window,
            0x58,
            &c.nonvolatile_registers[3].to_le_bytes(),
        );
        if s.metadata_flag == 0 {
            for (slot, site) in SLOTS.into_iter().zip(META) {
                regs[1] = c.native_base + u64::from(slot.rva());
                let v = s.metadata_slots[&slot];
                emit(
                    &mut steps,
                    &mut s,
                    c,
                    i,
                    &mut counts,
                    Completed::Metadata { slot, value: v },
                    regs,
                    xmm,
                    site,
                );
                regs = c.volatile_return_registers;
                regs[0] = v;
                xmm = c.volatile_return_xmm_hex.clone();
            }
            s.metadata_flag = 1;
        }
        regs[1] = q(&s, owner, 0x18);
        regs[2] = 0;
        emit(
            &mut steps,
            &mut s,
            c,
            i,
            &mut counts,
            Completed::IsEmpty {
                value: (regs[1] != 0).then_some(regs[1]),
                method: 0,
            },
            regs,
            xmm,
            0x3B4568,
        );
        regs = c.volatile_return_registers;
        regs[0] = call.empty_return_bits;
        xmm = c.volatile_return_xmm_hex.clone();
        if call.empty_return_bits as u8 != 0 {
            regs[1] = s.metadata_slots[&Slot::ArrayType];
            regs[2] = 9;
            write(
                &mut s,
                c.stack_window,
                0x50,
                &c.nonvolatile_registers[0].to_le_bytes(),
            );
            let array = call.array_result.expect("normal generation validated");
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::ArrayAllocate {
                    class: regs[1],
                    length: 9,
                    result: Some(array),
                },
                regs,
                xmm,
                0x3B4586,
            );
            regs = c.volatile_return_registers;
            regs[0] = array;
            xmm = c.volatile_return_xmm_hex.clone();
            regs[1] = owner;
            regs[2] = 0;
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::ObjectName {
                    owner,
                    method: 0,
                    result: call.name_result,
                },
                regs,
                xmm,
                0x3B4593,
            );
            regs = c.volatile_return_registers;
            regs[0] = ptr(call.name_result);
            xmm = c.volatile_return_xmm_hex.clone();
            for j in 0..9 {
                let captured = if j == 0 {
                    call.name_result
                } else {
                    let n = j - 1;
                    regs[1] = 0;
                    regs[2] = 10;
                    regs[3] = 0;
                    let v = call.random_return_bits[n];
                    emit(
                        &mut steps,
                        &mut s,
                        c,
                        i,
                        &mut counts,
                        Completed::Random {
                            index: n as u8,
                            min: 0,
                            max: 10,
                            method: 0,
                            return_bits: v,
                        },
                        regs,
                        xmm,
                        RNG[n],
                    );
                    regs = c.volatile_return_registers;
                    regs[0] = v;
                    xmm = c.volatile_return_xmm_hex.clone();
                    regs[1] = s.metadata_slots[&Slot::IntType];
                    regs[2] = c.stack_window + 0x10 + SCRATCH[n] as u64;
                    write(
                        &mut s,
                        c.stack_window,
                        0x10 + SCRATCH[n],
                        &(v as u32).to_le_bytes(),
                    );
                    emit(
                        &mut steps,
                        &mut s,
                        c,
                        i,
                        &mut counts,
                        Completed::Box {
                            index: n as u8,
                            class: regs[1],
                            scratch: regs[2],
                            value_bits: v as u32,
                            result: call.box_results[n],
                        },
                        regs,
                        xmm,
                        BOX[n],
                    );
                    regs = c.volatile_return_registers;
                    regs[0] = ptr(call.box_results[n]);
                    xmm = c.volatile_return_xmm_hex.clone();
                    call.box_results[n]
                };
                if let Some(input) = captured {
                    regs[1] = input;
                    regs[2] = q(&s, q(&s, array, 0), 0x40);
                    emit(
                        &mut steps,
                        &mut s,
                        c,
                        i,
                        &mut counts,
                        Completed::TypeCheck {
                            index: j as u8,
                            input,
                            target: regs[2],
                            result: call.type_results[j],
                        },
                        regs,
                        xmm,
                        CAST[j],
                    );
                    regs = c.volatile_return_registers;
                    regs[0] = call.type_results[j].expect("normal nonnull cast validated");
                    xmm = c.volatile_return_xmm_hex.clone();
                }
                // Each gate consumes the current low DWORD; inert service shape
                // was validated for all nine stores before any state was cloned.
                debug_assert!(q(&s, array, 0x18) as u32 > j as u32);
                let off = 0x20 + j * 8;
                regs[1] = array + off as u64;
                regs[2] = ptr(captured);
                write(&mut s, array, off, &regs[2].to_le_bytes());
                emit(
                    &mut steps,
                    &mut s,
                    c,
                    i,
                    &mut counts,
                    Completed::ReferenceBarrier {
                        index: j as u8,
                        owner: array,
                        offset: off,
                        value: captured,
                    },
                    regs,
                    xmm,
                    BARRIER[j],
                );
                regs = c.volatile_return_registers;
                regs[0] = call.barrier_return_bits;
                xmm = c.volatile_return_xmm_hex.clone();
            }
            regs[1] = s.metadata_slots[&Slot::FormatLiteral];
            regs[2] = array;
            regs[3] = 0;
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::Format {
                    literal: regs[1],
                    array,
                    method: 0,
                    result: call.format_result,
                },
                regs,
                xmm,
                0x3B48C7,
            );
            regs = c.volatile_return_registers;
            regs[0] = ptr(call.format_result);
            xmm = c.volatile_return_xmm_hex.clone();
            regs[2] = ptr(call.format_result);
            write(&mut s, owner, 0x18, &regs[2].to_le_bytes());
            regs[1] = owner + 0x18;
            emit(
                &mut steps,
                &mut s,
                c,
                i,
                &mut counts,
                Completed::ReferenceBarrier {
                    index: 9,
                    owner,
                    offset: 0x18,
                    value: call.format_result,
                },
                regs,
                xmm,
                BARRIER[9],
            );
            regs = c.volatile_return_registers;
            regs[0] = call.barrier_return_bits;
            xmm = c.volatile_return_xmm_hex.clone();
        }
        final_registers.push(regs);
        final_xmm_hex.push(xmm);
        final_rsp_bits.push(c.stack_window + 0x70);
        completed.push(s.clone());
    }
    Ok(Replay {
        steps,
        completed,
        final_registers,
        final_xmm_hex,
        final_rsp_bits,
        final_state: s,
    })
}
#[cfg(test)]
#[path = "character_data_identity_tests.rs"]
mod tests;
