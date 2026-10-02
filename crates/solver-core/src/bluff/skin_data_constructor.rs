//! Exact nominal SkinData EDX-clear/tail wrapper. Unity base implementation,
//! allocation and runtime construction remain whole supplied contracts.
use super::ledger::LedgerError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
pub const SKIN_DATA_CONSTRUCTOR_NATIVE_V1: &str = "skin_data_constructor_native_v1";
const GPRS: [&str; 16] = [
    "RAX", "RCX", "RDX", "R8", "R9", "R10", "R11", "RBX", "RBP", "RSI", "RDI", "R12", "R13", "R14",
    "R15", "RSP",
];
const WINDOWS: [&str; 14] = [
    "skin0",
    "skin1",
    "skin_class",
    "skin_id",
    "artist",
    "link",
    "flavor",
    "notes",
    "art",
    "animated",
    "locked",
    "unlock",
    "character_data",
    "native_stack",
];
pub type Registers = BTreeMap<String, Value>;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub identity: u64,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub records: BTreeMap<String, Record>,
    /// Exact native ABI/history objects, validated recursively before cloning.
    pub native_entries: Vec<Value>,
    pub requests: Vec<Value>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Write {
    pub window: String,
    pub offset: usize,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub registers: Registers,
    pub return_bits: u64,
    pub volatile_poison: u64,
    pub base_writes: Vec<Write>,
    pub stop_base: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub version: String,
    pub state: State,
    pub calls: Vec<Call>,
    pub native_base: u64,
    pub return_sentinel: u64,
    pub entry_sp: u64,
    pub storage_verified: bool,
    pub supplied_base_verified: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub call: usize,
    pub event: Value,
    pub state: State,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replay {
    pub steps: Vec<Step>,
    pub completed: Vec<State>,
    pub final_registers: Vec<Registers>,
    pub final_state: State,
}
fn size(n: &str) -> usize {
    match n {
        "skin0" | "skin1" | "native_stack" => 256,
        "character_data" => 512,
        _ => 128,
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn exact(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|m| m.len() == keys.len() && keys.iter().all(|k| m.contains_key(*k)))
}
fn canonical(v: &Value, length: usize) -> bool {
    v.as_str().is_some_and(|s| {
        s.len() == length
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}
fn regs_valid(r: &Registers) -> bool {
    r.len() == 32
        && GPRS
            .iter()
            .all(|n| r.get(*n).is_some_and(|v| v.as_u64().is_some()))
        && (0..16).all(|i| r.get(&format!("XMM{i}")).is_some_and(|v| canonical(v, 32)))
}
fn abi(r: &Registers) -> Value {
    let ints: BTreeMap<_, _> = GPRS[..7]
        .iter()
        .map(|n| (*n, r[*n].as_u64().unwrap()))
        .collect();
    let raw: BTreeMap<_, _> = GPRS[..7]
        .iter()
        .map(|n| (*n, format!("{:016x}", r[*n].as_u64().unwrap())))
        .collect();
    let xmm: Vec<_> = (0..6).map(|i| r[&format!("XMM{i}")].clone()).collect();
    json!({"volatile_registers":ints,"volatile_gpr_hex":raw,"volatile_xmm_hex":xmm,"all_registers":r})
}
fn valid_abi(v: &Value) -> bool {
    let Some(r) = v.get("all_registers").and_then(Value::as_object) else {
        return false;
    };
    let r: Registers = r.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    regs_valid(&r)
        && [
            "volatile_registers",
            "volatile_gpr_hex",
            "volatile_xmm_hex",
            "all_registers",
        ]
        .iter()
        .all(|k| v[*k] == abi(&r)[*k])
}
fn entry(c: &Context, r: &Registers) -> Value {
    let mut v = abi(r);
    v["owner"] = r["RCX"].clone();
    v["native_entry"] = json!(c.native_base + 0x373A40);
    v
}
fn nominal(s: &State, owner: u64) -> bool {
    ["skin0", "skin1"]
        .iter()
        .any(|n| s.records[*n].identity == owner)
}
fn writes_valid(w: &[Write], s: &State) -> bool {
    w.len()<=8 && w.iter().all(|w|s.records.get(&w.window).is_some_and(|r|
        !w.bytes.is_empty() && w.offset.checked_add(w.bytes.len()).is_some_and(|end|end<=r.bytes.len())
        // This bounded corpus explicitly supplies only these two owner effects.
        && w.window=="skin0" && ((w.offset==0x54 && w.bytes.len()==16)||(w.offset==0x48 && w.bytes.len()==8))))
}
fn units(c: &Context) -> Option<usize> {
    fn add(n: &mut usize, cost: usize) -> Option<()> {
        *n = n.checked_add(cost)?;
        (*n <= 131072).then_some(())
    }
    fn value(n: &mut usize, v: &Value, depth: usize) -> Option<()> {
        if depth > 32 {
            return None;
        }
        add(n, 32)?;
        match v {
            Value::String(s) => add(n, s.len().checked_mul(6)?)?,
            Value::Array(a) => {
                add(n, a.len().checked_mul(32)?)?;
                for item in a {
                    value(n, item, depth + 1)?;
                }
            }
            Value::Object(o) => {
                add(n, o.len().checked_mul(64)?)?;
                for (key, item) in o {
                    add(n, key.len().checked_mul(6)?)?;
                    value(n, item, depth + 1)?;
                }
            }
            _ => add(n, 24)?,
        }
        Some(())
    }
    let mut n = 256usize;
    add(&mut n, c.version.len())?;
    for (name, r) in &c.state.records {
        add(&mut n, 32)?;
        add(&mut n, name.len())?;
        add(&mut n, r.bytes.len())?;
    }
    for v in c.state.native_entries.iter().chain(&c.state.requests) {
        value(&mut n, v, 0)?;
    }
    for call in &c.calls {
        add(&mut n, 192)?;
        add(&mut n, call.registers.len().checked_mul(64)?)?;
        for (key, v) in &call.registers {
            add(&mut n, key.len().checked_mul(6)?)?;
            value(&mut n, v, 0)?;
        }
        for w in &call.base_writes {
            add(&mut n, 64)?;
            add(&mut n, w.window.len())?;
            add(&mut n, w.bytes.len())?;
        }
    }
    Some(n)
}
fn validate(c: &Context) -> Result<(), LedgerError> {
    // Account for growing entry/completion histories plus every event and final
    // snapshot before the first clone; borrowed depth-limited traversal allocates
    // nothing and rejects excess bytes or container sizes before visiting them.
    let input_units = units(c);
    let budget = input_units.and_then(|n| {
        n.checked_add(c.calls.len().checked_mul(24576)?)?
            .checked_mul(c.calls.len().checked_mul(2)?.checked_add(2)?)
    });
    if c.calls.len() > 8
        || c.state.records.len() > 14
        || c.state
            .native_entries
            .len()
            .checked_add(c.calls.len())
            .is_none_or(|n| n > 64)
        || c.state
            .requests
            .len()
            .checked_add(c.calls.len())
            .is_none_or(|n| n > 64)
        || input_units.is_none()
        || budget.is_none_or(|n| n > 4_194_304)
    {
        return Err(LedgerError::Capacity);
    }
    if c.version != SKIN_DATA_CONSTRUCTOR_NATIVE_V1
        || !c.storage_verified
        || !c.supplied_base_verified
        || c.native_base == 0
        || c.native_base.checked_add(0x1C8A5C0).is_none()
        || c.return_sentinel == 0
        || c.entry_sp.checked_add(8).is_none()
        || c.entry_sp < 128
        || c.entry_sp % 16 != 8
        || c.state.records.len() != 14
        || WINDOWS.iter().any(|n| !c.state.records.contains_key(*n))
    {
        return Err(LedgerError::InvalidContext);
    }
    for (name, r) in &c.state.records {
        let Some(end) = r.identity.checked_add(r.bytes.len() as u64) else {
            return Err(LedgerError::InvalidContext);
        };
        if r.identity == 0
            || r.bytes.len() != size(name)
            || c.state.records.iter().any(|(n, o)| {
                n != name
                    && o.identity < end
                    && o.identity
                        .checked_add(o.bytes.len() as u64)
                        .is_none_or(|e| e > r.identity)
            })
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let stack = &c.state.records["native_stack"];
    if stack.identity != c.entry_sp - 128
        || stack.bytes[128..136] != c.return_sentinel.to_le_bytes()
    {
        return Err(LedgerError::InvalidContext);
    }
    // Nominal field references are accepted only at exact diagnostic record starts.
    for skin in ["skin0", "skin1"] {
        let r = &c.state.records[skin];
        for (offset, target) in [
            (0, "skin_class"),
            (0x18, "skin_id"),
            (0x20, "artist"),
            (0x28, "link"),
            (0x38, "art"),
            (0x40, "animated"),
            (0x48, "locked"),
            (0x68, "unlock"),
            (0x70, "flavor"),
            (0x78, "notes"),
            (0x80, "character_data"),
        ] {
            let p = u64::from_le_bytes(r.bytes[offset..offset + 8].try_into().unwrap());
            if p != c.state.records[target].identity && !(offset != 0 && p == 0) {
                return Err(LedgerError::InvalidContext);
            }
        }
    }
    if c.calls.iter().any(|call| {
        call.stop_base
            || !regs_valid(&call.registers)
            || !nominal(&c.state, call.registers["RCX"].as_u64().unwrap_or(0))
            || call.registers["RSP"] != json!(c.entry_sp)
            || !writes_valid(&call.base_writes, &c.state)
    }) {
        return Err(LedgerError::InvalidContext);
    }
    for e in &c.state.native_entries {
        if !exact(
            e,
            &[
                "owner",
                "native_entry",
                "volatile_registers",
                "volatile_gpr_hex",
                "volatile_xmm_hex",
                "all_registers",
            ],
        ) || !valid_abi(e)
            || e["native_entry"] != json!(c.native_base + 0x373A40)
            || e["owner"] != e["all_registers"]["RCX"]
            || !e["owner"].as_u64().is_some_and(|p| nominal(&c.state, p))
            || e["all_registers"]["RSP"] != json!(c.entry_sp)
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    let mut prior_entry = 0;
    for h in &c.state.requests {
        if !exact(
            h,
            &[
                "kind",
                "ordinal",
                "site",
                "raw_args",
                "return_bits",
                "writes",
                "normal_callee_abi",
            ],
        ) || h["kind"] != "scriptable_object_constructor"
            || h["ordinal"] != 1
            || h["site"] != "0x373a42"
            || h["return_bits"].as_u64().is_none()
        {
            return Err(LedgerError::InvalidContext);
        }
        let Some(args) = h["raw_args"].as_array() else {
            return Err(LedgerError::InvalidContext);
        };
        let a = &h["normal_callee_abi"];
        let Some(p) = a["preserved"].as_object() else {
            return Err(LedgerError::InvalidContext);
        };
        if args.len() != 4
            || args.iter().any(|v| v.as_u64().is_none())
            || args[1] != 0
            || !nominal(&c.state, args[0].as_u64().unwrap())
            || !exact(a, &["entry_sp", "return_sp", "caller", "preserved"])
            || a["entry_sp"] != json!(c.entry_sp)
            || a["return_sp"] != json!(c.entry_sp + 8)
            || a["caller"] != json!(c.return_sentinel)
            || p.len() != 18
            || GPRS[7..15]
                .iter()
                .any(|n| p.get(*n).is_none_or(|v| v.as_u64().is_none()))
            || (6..16).any(|i| p.get(&format!("XMM{i}")).is_none_or(|v| !canonical(v, 32)))
        {
            return Err(LedgerError::InvalidContext);
        }
        let Some(ws) = h["writes"].as_array() else {
            return Err(LedgerError::InvalidContext);
        };
        let mut writes = Vec::new();
        for w in ws {
            let Some(w) = w.as_array().filter(|v| v.len() == 3) else {
                return Err(LedgerError::InvalidContext);
            };
            let (Some(n), Some(off), Some(raw)) = (w[0].as_str(), w[1].as_u64(), w[2].as_str())
            else {
                return Err(LedgerError::InvalidContext);
            };
            if raw.len() % 2 != 0
                || !raw
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(LedgerError::InvalidContext);
            }
            let bytes = raw
                .as_bytes()
                .chunks_exact(2)
                .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
                .collect();
            let Ok(offset) = usize::try_from(off) else {
                return Err(LedgerError::InvalidContext);
            };
            writes.push(Write {
                window: n.into(),
                offset,
                bytes,
            });
        }
        if !writes_valid(&writes, &c.state) {
            return Err(LedgerError::InvalidContext);
        }
        // Completed history is chronological; unmatched entries are earlier
        // supplied stops. Match both consumed and preserved register evidence.
        let mut found = false;
        while prior_entry < c.state.native_entries.len() {
            let e = &c.state.native_entries[prior_entry]["all_registers"];
            prior_entry += 1;
            if args[0] == e["RCX"]
                && args[2] == e["R8"]
                && args[3] == e["R9"]
                && GPRS[7..15].iter().all(|n| p[*n] == e[*n])
                && (6..16).all(|i| p[&format!("XMM{i}")] == e[&format!("XMM{i}")])
            {
                found = true;
                break;
            }
        }
        if !found {
            return Err(LedgerError::InvalidContext);
        }
    }
    Ok(())
}
/// Validate all future calls and retained storage before producing any result.
/// The supplied base contract may author the two explicitly supported writes.
pub fn replay(c: &Context) -> Result<Replay, LedgerError> {
    validate(c)?;
    let mut state = c.state.clone();
    let mut steps = Vec::new();
    let mut completed = Vec::new();
    let mut final_registers = Vec::new();
    for (i, call) in c.calls.iter().enumerate() {
        state.native_entries.push(entry(c, &call.registers));
        let mut r = call.registers.clone();
        r.insert("RDX".into(), json!(0));
        let args = [
            r["RCX"].clone(),
            r["RDX"].clone(),
            r["R8"].clone(),
            r["R9"].clone(),
        ];
        let mut e = abi(&r);
        e["kind"] = json!("scriptable_object_constructor");
        e["ordinal"] = json!(1);
        e["native_site"] = json!("0x373a42");
        e["gateway"] = json!(c.native_base + 0x1C8A5C0);
        e["caller_return_bits"] = json!(c.return_sentinel);
        e["entry_sp"] = json!(c.entry_sp);
        e["raw_args"] = json!(args);
        steps.push(Step {
            call: i,
            event: e,
            state: state.clone(),
        });
        let mut writes = Vec::new();
        for w in &call.base_writes {
            state.records.get_mut(&w.window).unwrap().bytes[w.offset..w.offset + w.bytes.len()]
                .copy_from_slice(&w.bytes);
            writes.push(json!([w.window, w.offset, hex(&w.bytes)]));
        }
        let preserved: BTreeMap<String, Value> = GPRS[7..15]
            .iter()
            .map(|n| ((*n).into(), r[*n].clone()))
            .chain((6..16).map(|i| (format!("XMM{i}"), r[&format!("XMM{i}")].clone())))
            .collect();
        state.requests.push(json!({"kind":"scriptable_object_constructor","ordinal":1,"site":"0x373a42","raw_args":args,"return_bits":call.return_bits,"writes":writes,"normal_callee_abi":{"entry_sp":c.entry_sp,"return_sp":c.entry_sp+8,"caller":c.return_sentinel,"preserved":preserved}}));
        for n in &GPRS[1..7] {
            r.insert((*n).into(), json!(call.volatile_poison));
        }
        for j in 0..6 {
            r.insert(
                format!("XMM{j}"),
                json!(format!("{:032x}", (1u128 << 127) | j as u128)),
            );
        }
        r.insert("RAX".into(), json!(call.return_bits));
        r.insert("RSP".into(), json!(c.entry_sp + 8));
        completed.push(state.clone());
        final_registers.push(r);
    }
    Ok(Replay {
        steps,
        completed,
        final_registers,
        final_state: state,
    })
}
#[cfg(test)]
#[path = "skin_data_constructor_tests.rs"]
mod tests;
