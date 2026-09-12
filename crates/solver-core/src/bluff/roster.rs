//! Offline logical-list replay of seven native Gameplay roster operations.
//!
//! Reference identity, stable callbacks, and sufficient native Add capacity
//! are required provenance. List size/version and logical contents are modeled;
//! stale backing slots after a failing Array.Clear and managed exception
//! unwinding are outside this contract. Starting-pool requests are explicit
//! services: this does not compose their internal lazy selection or Unity RNG.
use super::ledger::{LedgerError, Probability};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const ROSTER_NATIVE_V1: &str = "gameplay_roster_native_v1";
const MAX_ENTRIES: usize = 4096;
const MAX_PATHS: usize = 65_536;
const MAX_RETAINED: usize = 1_048_576;
const TYPES: [i32; 4] = [10, 20, 30, 100];
pub type ListId = u16;
pub type AssetId = u16;
pub type Occurrences = Vec<Option<AssetId>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RosterList {
    pub items: Occurrences,
    pub version: u32,
    pub capacity: u16,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RosterAsset {
    pub character_type: i32,
    pub starting_alignment: i32,
    pub can_appear_if: Option<ListId>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RosterCharacter {
    pub data: Option<AssetId>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RosterMethod {
    GetAllCurrentCharacters,
    CleanupCharactersList,
    FilterIfCanAppearCharacters { character_type: i32 },
    GetNotInPlayCharacters,
    GetNotInDeckCharacters,
    GetScriptCharactersOfAlignment { alignment: i32 },
    UpdateCurrentCharacters,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RosterServiceFailures {
    pub append_range: Option<u16>,
    pub remove: Option<u16>,
    pub add: Option<u16>,
    pub clear: Option<u16>,
    pub typed_pool: Option<u16>,
    pub class_initializer: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RosterContext {
    pub rule_version: String,
    pub reference_equality: bool,
    pub services_preserve_inputs: bool,
    pub list_add_capacity_sufficient: bool,
    pub allocation_capacity: u16,
    pub method: RosterMethod,
    pub assets: BTreeMap<AssetId, RosterAsset>,
    pub lists: BTreeMap<ListId, RosterList>,
    pub rosters: [Option<ListId>; 4],
    pub input: Option<ListId>,
    pub output: Option<ListId>,
    pub current_characters: Option<Vec<Option<RosterCharacter>>>,
    pub starting_pools: [Option<Occurrences>; 4],
    /// town, demon, outs, minion, then their four disguised counterparts.
    pub current_script: Option<[i32; 8]>,
    pub project_available: bool,
    pub gameplay_class_initialized: bool,
    pub service_failures: RosterServiceFailures,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RosterFailure {
    Null,
    Collection,
    RemoveService,
    AddService,
    ClearService,
    TypedPoolService,
    ClassInitializer,
    EnumeratorVersion,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RosterDraw {
    pub character_type: i32,
    pub width: u16,
    pub occurrence_index: u16,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RosterPath {
    pub probability: Probability,
    pub lists: BTreeMap<ListId, RosterList>,
    pub allocations: Vec<ListId>,
    pub returned_list: Option<ListId>,
    pub failure: Option<RosterFailure>,
    pub typed_pool_requests: Vec<i32>,
    pub draws: Vec<RosterDraw>,
    pub adds: Vec<(ListId, Option<AssetId>)>,
    pub removes: Vec<(ListId, Option<AssetId>)>,
    pub append_ranges: Vec<(ListId, Option<ListId>)>,
    pub clears: Vec<ListId>,
    pub class_initialized: bool,
}
fn fail(path: &mut RosterPath, error: RosterFailure) {
    path.failure = Some(error);
}
fn valid(context: &RosterContext) -> Result<(), LedgerError> {
    let f = &context.service_failures;
    if context.rule_version != ROSTER_NATIVE_V1
        || !context.reference_equality
        || !context.services_preserve_inputs
        || !context.list_add_capacity_sufficient
        || context.allocation_capacity as usize > MAX_ENTRIES
        || context.assets.len() > MAX_ENTRIES
        || context.lists.len() > MAX_ENTRIES
        || [f.append_range, f.remove, f.add, f.clear, f.typed_pool]
            .into_iter()
            .flatten()
            .any(|n| n == 0 || n as usize > MAX_ENTRIES)
        || context
            .rosters
            .iter()
            .chain([&context.input, &context.output])
            .flatten()
            .any(|id| !context.lists.contains_key(id))
        || context
            .assets
            .values()
            .filter_map(|a| a.can_appear_if)
            .any(|id| !context.lists.contains_key(&id))
    {
        return Err(LedgerError::InvalidContext);
    }
    let mut total = 0;
    for list in context.lists.values() {
        if list.items.len() > list.capacity as usize || list.capacity as usize > MAX_ENTRIES {
            return Err(LedgerError::InvalidContext);
        }
    }
    for items in context
        .lists
        .values()
        .map(|l| &l.items)
        .chain(context.starting_pools.iter().flatten())
    {
        if items.len() > MAX_ENTRIES {
            return Err(LedgerError::Capacity);
        }
        if items
            .iter()
            .flatten()
            .any(|id| !context.assets.contains_key(id))
        {
            return Err(LedgerError::InvalidContext);
        }
        total += items.len();
    }
    if let Some(current) = &context.current_characters {
        if current.len() > MAX_ENTRIES {
            return Err(LedgerError::Capacity);
        }
        if current
            .iter()
            .flatten()
            .filter_map(|c| c.data)
            .any(|id| !context.assets.contains_key(&id))
        {
            return Err(LedgerError::InvalidContext);
        }
    }
    if total > MAX_RETAINED {
        return Err(LedgerError::Capacity);
    }
    Ok(())
}
fn allocate(
    context: &RosterContext,
    path: &mut RosterPath,
    items: Occurrences,
) -> Result<ListId, LedgerError> {
    if items.len() > context.allocation_capacity as usize {
        return Err(LedgerError::Capacity);
    }
    let id = (0..=u16::MAX)
        .find(|id| !path.lists.contains_key(id))
        .ok_or(LedgerError::Capacity)?;
    path.lists.insert(
        id,
        RosterList {
            items,
            version: 0,
            capacity: context.allocation_capacity,
        },
    );
    path.allocations.push(id);
    Ok(id)
}
fn append(
    context: &RosterContext,
    path: &mut RosterPath,
    dest: ListId,
    source: Option<ListId>,
    items: Option<Occurrences>,
) -> Result<(), LedgerError> {
    path.append_ranges.push((dest, source));
    if context.service_failures.append_range == Some(path.append_ranges.len() as u16)
        || items.is_none()
    {
        fail(path, RosterFailure::Collection);
        return Ok(());
    }
    let values = items.unwrap();
    let list = path.lists.get_mut(&dest).unwrap();
    if list.items.len() + values.len() > list.capacity as usize {
        return Err(LedgerError::Capacity);
    }
    list.items.extend(values);
    list.version = list.version.wrapping_add(1);
    Ok(())
}
fn add(
    context: &RosterContext,
    path: &mut RosterPath,
    list_id: ListId,
    asset: Option<AssetId>,
) -> Result<(), LedgerError> {
    path.adds.push((list_id, asset));
    if context.service_failures.add == Some(path.adds.len() as u16) {
        fail(path, RosterFailure::AddService);
        return Ok(());
    }
    let list = path.lists.get_mut(&list_id).unwrap();
    if list.items.len() == list.capacity as usize {
        return Err(LedgerError::Capacity);
    }
    list.version = list.version.wrapping_add(1);
    list.items.push(asset);
    Ok(())
}
fn remove(context: &RosterContext, path: &mut RosterPath, list_id: ListId, asset: Option<AssetId>) {
    path.removes.push((list_id, asset));
    if context.service_failures.remove == Some(path.removes.len() as u16) {
        fail(path, RosterFailure::RemoveService);
        return;
    }
    let list = path.lists.get_mut(&list_id).unwrap();
    if let Some(index) = list.items.iter().position(|a| *a == asset) {
        list.items.remove(index);
        list.version = list.version.wrapping_add(1);
    }
}
fn aggregate(context: &RosterContext, path: &mut RosterPath) -> Result<ListId, LedgerError> {
    let dest = allocate(context, path, Vec::new())?;
    for source in context.rosters {
        let items = source.map(|id| path.lists[&id].items.clone());
        append(context, path, dest, source, items)?;
        if path.failure.is_some() {
            break;
        }
    }
    Ok(dest)
}
fn ensure_class(context: &RosterContext, path: &mut RosterPath) {
    if !path.class_initialized {
        if context.service_failures.class_initializer {
            fail(path, RosterFailure::ClassInitializer);
        } else {
            path.class_initialized = true;
        }
    }
}
fn typed_pool(context: &RosterContext, path: &mut RosterPath, kind: i32) -> Option<Occurrences> {
    if !context.project_available {
        fail(path, RosterFailure::Null);
        return None;
    }
    path.typed_pool_requests.push(kind);
    if context.service_failures.typed_pool == Some(path.typed_pool_requests.len() as u16) {
        fail(path, RosterFailure::TypedPoolService);
        return None;
    }
    context.starting_pools[TYPES.iter().position(|t| *t == kind).unwrap()].clone()
}
fn exclude_current(context: &RosterContext, path: &mut RosterPath, result: ListId) {
    ensure_class(context, path);
    if path.failure.is_some() {
        return;
    }
    let Some(current) = &context.current_characters else {
        fail(path, RosterFailure::Null);
        return;
    };
    for character in current {
        let Some(character) = character else {
            fail(path, RosterFailure::Null);
            return;
        };
        remove(context, path, result, character.data);
        if path.failure.is_some() {
            return;
        }
    }
    path.returned_list = Some(result);
}
fn filter(
    context: &RosterContext,
    path: &mut RosterPath,
    kind: i32,
    input: Option<ListId>,
    output: Option<ListId>,
) {
    let Some(input) = input else {
        fail(path, RosterFailure::Null);
        return;
    };
    let version = path.lists[&input].version;
    let mut index = 0;
    loop {
        let list = &path.lists[&input];
        if list.version != version {
            fail(path, RosterFailure::EnumeratorVersion);
            return;
        }
        let Some(asset) = list.items.get(index).copied() else {
            return;
        };
        index += 1;
        let Some(asset) = asset else {
            fail(path, RosterFailure::Null);
            return;
        };
        let data = &context.assets[&asset];
        if data.character_type != kind {
            continue;
        }
        let Some(deps) = data.can_appear_if else {
            fail(path, RosterFailure::Null);
            return;
        };
        let deps = &path.lists[&deps].items;
        if deps.is_empty() || deps.iter().any(|d| path.lists[&input].items.contains(d)) {
            continue;
        }
        if let Some(faction) = TYPES.iter().position(|t| *t == kind) {
            let Some(roster) = context.rosters[faction] else {
                fail(path, RosterFailure::Null);
                return;
            };
            remove(context, path, roster, Some(asset));
            if path.failure.is_some() {
                return;
            }
        }
        let Some(out) = output else {
            fail(path, RosterFailure::Null);
            return;
        };
        remove(context, path, out, Some(asset));
        if path.failure.is_some() {
            return;
        }
    }
}
fn update(context: &RosterContext, path: &mut RosterPath) -> Result<(), LedgerError> {
    for id in context.rosters.iter().rev() {
        let Some(id) = id else {
            fail(path, RosterFailure::Null);
            return Ok(());
        };
        let list = path.lists.get_mut(id).unwrap();
        let count = list.items.len();
        list.version = list.version.wrapping_add(1);
        list.items.clear();
        if count > 0 {
            path.clears.push(*id);
            if context.service_failures.clear == Some(path.clears.len() as u16) {
                fail(path, RosterFailure::ClearService);
                return Ok(());
            }
        }
    }
    let Some(input) = context.input else {
        fail(path, RosterFailure::Null);
        return Ok(());
    };
    let version = path.lists[&input].version;
    let mut index = 0;
    loop {
        let list = &path.lists[&input];
        if list.version != version {
            fail(path, RosterFailure::EnumeratorVersion);
            return Ok(());
        }
        let Some(asset) = list.items.get(index).copied() else {
            return Ok(());
        };
        index += 1;
        let Some(asset) = asset else {
            fail(path, RosterFailure::Null);
            return Ok(());
        };
        if let Some(faction) = TYPES
            .iter()
            .position(|t| *t == context.assets[&asset].character_type)
        {
            let Some(roster) = context.rosters[faction] else {
                fail(path, RosterFailure::Null);
                return Ok(());
            };
            add(context, path, roster, Some(asset))?;
            if path.failure.is_some() {
                return Ok(());
            }
        }
    }
}
fn alignment(
    context: &RosterContext,
    path: &mut RosterPath,
    wanted: i32,
) -> Result<(), LedgerError> {
    // Native allocates both lists before aggregating into the first.
    let all = allocate(context, path, Vec::new())?;
    let output = allocate(context, path, Vec::new())?;
    for source in context.rosters {
        let items = source.map(|id| path.lists[&id].items.clone());
        append(context, path, all, source, items)?;
        if path.failure.is_some() {
            return Ok(());
        }
    }
    for asset in path.lists[&all].items.clone() {
        let Some(asset) = asset else {
            fail(path, RosterFailure::Null);
            return Ok(());
        };
        if context.assets[&asset].starting_alignment == wanted {
            add(context, path, output, Some(asset))?;
            if path.failure.is_some() {
                return Ok(());
            }
        }
    }
    path.returned_list = Some(output);
    Ok(())
}
fn retain(
    out: &mut Vec<RosterPath>,
    path: RosterPath,
    cost: &mut usize,
) -> Result<(), LedgerError> {
    if out.len() == MAX_PATHS {
        return Err(LedgerError::Capacity);
    }
    let entries = path
        .lists
        .values()
        .map(|l| l.items.len() + 1)
        .sum::<usize>()
        + path.draws.len()
        + path.adds.len()
        + path.removes.len()
        + path.append_ranges.len()
        + path.clears.len();
    *cost = cost.checked_add(entries).ok_or(LedgerError::Capacity)?;
    if *cost > MAX_RETAINED {
        return Err(LedgerError::Capacity);
    }
    out.push(path);
    Ok(())
}
fn cleanup_phase(
    context: &RosterContext,
    mut path: RosterPath,
    snapshot: ListId,
    kind: i32,
) -> Result<Vec<RosterPath>, LedgerError> {
    if path.failure.is_some() {
        return Ok(vec![path]);
    }
    let pool = typed_pool(context, &mut path, kind);
    if path.failure.is_some() {
        return Ok(vec![path]);
    }
    let candidate = allocate(context, &mut path, Vec::new())?;
    let Some(pool) = pool else {
        fail(&mut path, RosterFailure::Collection);
        return Ok(vec![path]);
    };
    if pool.len() > context.allocation_capacity as usize {
        return Err(LedgerError::Capacity);
    }
    path.lists.get_mut(&candidate).unwrap().items = pool;
    filter(context, &mut path, kind, Some(snapshot), Some(candidate));
    if path.failure.is_some() {
        return Ok(vec![path]);
    }
    ensure_class(context, &mut path);
    if path.failure.is_some() {
        return Ok(vec![path]);
    }
    let Some(counts) = context.current_script else {
        fail(&mut path, RosterFailure::Null);
        return Ok(vec![path]);
    };
    let faction = TYPES.iter().position(|t| *t == kind).unwrap();
    let Some(roster) = context.rosters[faction] else {
        fail(&mut path, RosterFailure::Null);
        return Ok(vec![path]);
    };
    let count_index = match kind {
        10 => 0,
        100 => 1,
        20 => 2,
        _ => 3,
    };
    let deficit = counts[count_index]
        .max(counts[count_index + 4])
        .wrapping_sub(path.lists[&roster].items.len() as i32);
    let steps = if deficit > 0 {
        (deficit as usize).min(path.lists[&candidate].items.len())
    } else {
        0
    };
    let mut paths = vec![path];
    for _ in 0..steps {
        let mut next = Vec::new();
        let mut cost = 0;
        for path in paths {
            if path.failure.is_some() || path.lists[&candidate].items.is_empty() {
                retain(&mut next, path, &mut cost)?;
                continue;
            }
            let choices = &path.lists[&candidate].items;
            if next.len() + choices.len() > MAX_PATHS {
                return Err(LedgerError::Capacity);
            }
            for (index, asset) in choices.iter().enumerate() {
                let mut branch = path.clone();
                branch.probability = branch.probability.multiply(1, choices.len() as u64)?;
                branch.draws.push(RosterDraw {
                    character_type: kind,
                    width: choices.len() as u16,
                    occurrence_index: index as u16,
                });
                add(context, &mut branch, roster, *asset)?;
                if branch.failure.is_none() {
                    remove(context, &mut branch, candidate, *asset);
                }
                retain(&mut next, branch, &mut cost)?;
            }
        }
        paths = next;
    }
    Ok(paths)
}

/// Replay known immutable input identities without consulting live game state.
/// Failed native paths retain their unconditional mass and logical mutations.
/// Unsupported capacity/support/provenance rejects the whole request.
pub fn replay_roster(context: &RosterContext) -> Result<Vec<RosterPath>, LedgerError> {
    valid(context)?;
    let mut path = RosterPath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        lists: context.lists.clone(),
        allocations: Vec::new(),
        returned_list: None,
        failure: None,
        typed_pool_requests: Vec::new(),
        draws: Vec::new(),
        adds: Vec::new(),
        removes: Vec::new(),
        append_ranges: Vec::new(),
        clears: Vec::new(),
        class_initialized: context.gameplay_class_initialized,
    };
    match context.method {
        RosterMethod::GetAllCurrentCharacters => {
            let result = aggregate(context, &mut path)?;
            if path.failure.is_none() {
                path.returned_list = Some(result);
            }
        }
        RosterMethod::GetNotInPlayCharacters => {
            let result = aggregate(context, &mut path)?;
            if path.failure.is_none() {
                exclude_current(context, &mut path, result);
            }
        }
        RosterMethod::GetNotInDeckCharacters => {
            let result = allocate(context, &mut path, Vec::new())?;
            for kind in [100, 20, 30, 10] {
                let pool = typed_pool(context, &mut path, kind);
                if path.failure.is_some() {
                    break;
                }
                append(context, &mut path, result, None, pool)?;
                if path.failure.is_some() {
                    break;
                }
            }
            if path.failure.is_none() {
                exclude_current(context, &mut path, result);
            }
        }
        RosterMethod::FilterIfCanAppearCharacters { character_type } => {
            filter(
                context,
                &mut path,
                character_type,
                context.input,
                context.output,
            );
            if path.failure.is_none() {
                path.returned_list = context.output;
            }
        }
        RosterMethod::GetScriptCharactersOfAlignment { alignment: wanted } => {
            alignment(context, &mut path, wanted)?
        }
        RosterMethod::UpdateCurrentCharacters => update(context, &mut path)?,
        RosterMethod::CleanupCharactersList => {
            allocate(context, &mut path, Vec::new())?;
            let all = aggregate(context, &mut path)?;
            if path.failure.is_some() {
                return Ok(vec![path]);
            }
            let items = path.lists[&all].items.clone();
            let snapshot = allocate(context, &mut path, items)?;
            let mut paths = vec![path];
            for kind in [20, 30, 100, 10] {
                let mut next = Vec::new();
                let mut cost = 0;
                for p in paths {
                    for p in cleanup_phase(context, p, snapshot, kind)? {
                        retain(&mut next, p, &mut cost)?;
                    }
                }
                paths = next;
            }
            return Ok(paths);
        }
    }
    let mut result = Vec::new();
    let mut cost = 0;
    retain(&mut result, path, &mut cost)?;
    Ok(result)
}

#[cfg(test)]
#[path = "roster_tests.rs"]
mod tests;
