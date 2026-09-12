//! Offline filter caller replay with distinct supplied managed/Unity equality results.
//! Collection services preserve occurrences; RemoveAll commits after every predicate
//! succeeds. That fixture contract does not claim native RemoveAll rollback.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CHARACTER_FILTER_NATIVE_V1: &str = "character_filter_native_v1";
const LIMIT: usize = 4096;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CharacterFilterMethod {
    Status,
    Revealed,
    Unique,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterCharacter {
    pub state: i32,
    /// None means a native null status wrapper or inner list, never unknown data.
    pub statuses: Option<Vec<i32>>,
    pub data: Option<u16>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterGateway {
    Allocate,
    Copy,
    Enumerator,
    MoveNext,
    Add,
    StatusContains,
    ManagedContains,
    RemoveAll,
    UnityEqual,
    ClassInit,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterServiceFailure {
    pub gateway: FilterGateway,
    pub occurrence: u16,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CharacterFilterContext {
    pub rule_version: String,
    pub metadata_initialized: bool,
    pub services_preserve_input_contents: bool,
    pub remove_all_commits_after_predicates: bool,
    pub method: CharacterFilterMethod,
    /// Character labels for Status/Revealed; data labels for Unique.
    pub input: Option<Vec<Option<u16>>>,
    pub characters: BTreeMap<u16, FilterCharacter>,
    pub board: Option<Vec<Option<u16>>>,
    pub requested_status: i32,
    pub gameplay_initialized: bool,
    pub unity_object_initialized: bool,
    pub input_version: u32,
    /// Explicit fixture callback mutation; content stays stable.
    pub advance_input_version_after_add: bool,
    /// Actual responses in call order, independently supplied. No reference,
    /// destroyed-object, string, or null-equivalence policy is inferred.
    pub managed_contains: Vec<bool>,
    pub unity_equals: Vec<bool>,
    pub fail_at: Option<FilterServiceFailure>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "gateway", rename_all = "snake_case")]
pub enum CharacterFilterFailure {
    Null,
    CollectionNull,
    Version,
    Gateway(FilterGateway),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CharacterFilterResult {
    pub output_prefix: Vec<Option<u16>>,
    pub returned: bool,
    pub failure: Option<CharacterFilterFailure>,
    pub events: Vec<FilterGateway>,
    /// Attempted query operands, including a query whose service fails.
    pub managed_contains_queries: Vec<Option<u16>>,
    pub unity_equal_queries: Vec<(Option<u16>, Option<u16>)>,
    pub managed_contains_consumed: usize,
    pub unity_equals_consumed: usize,
    pub input_version: u32,
    pub gameplay_initialized: bool,
    pub unity_object_initialized: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterContextError {
    InvalidContext,
    Capacity,
}
struct Replay<'a> {
    c: &'a CharacterFilterContext,
    out: CharacterFilterResult,
    counts: [usize; 10],
}
impl Replay<'_> {
    fn event(&mut self, gateway: FilterGateway) -> Result<(), CharacterFilterFailure> {
        self.out.events.push(gateway);
        self.counts[gateway as usize] += 1;
        let occurrence = self.counts[gateway as usize];
        if self
            .c
            .fail_at
            .is_some_and(|f| f.gateway == gateway && usize::from(f.occurrence) == occurrence)
        {
            return Err(CharacterFilterFailure::Gateway(gateway));
        }
        Ok(())
    }
    fn apply(&mut self) -> Result<(), CharacterFilterFailure> {
        use FilterGateway as G;
        self.event(G::Allocate)?;
        if self.c.method == CharacterFilterMethod::Unique {
            self.event(G::Copy)?;
            self.out.output_prefix = self
                .c
                .input
                .clone()
                .ok_or(CharacterFilterFailure::CollectionNull)?;
            if !self.out.gameplay_initialized {
                self.event(G::ClassInit)?;
                self.out.gameplay_initialized = true;
            }
            let board = self.c.board.as_ref().ok_or(CharacterFilterFailure::Null)?;
            self.event(G::Enumerator)?;
            for character in board {
                self.event(G::MoveNext)?;
                let id = character.ok_or(CharacterFilterFailure::Null)?;
                let captured = self.c.characters[&id].data;
                self.out.managed_contains_queries.push(captured);
                self.event(G::ManagedContains)?;
                let contains = self.c.managed_contains[self.out.managed_contains_consumed];
                self.out.managed_contains_consumed += 1;
                if !contains {
                    continue;
                }
                self.event(G::Allocate)?; // closure
                self.event(G::Allocate)?; // predicate
                self.event(G::RemoveAll)?;
                let original = self.out.output_prefix.clone();
                let mut kept = Vec::new();
                for item in original {
                    if !self.out.unity_object_initialized {
                        self.event(G::ClassInit)?;
                        self.out.unity_object_initialized = true;
                    }
                    self.out.unity_equal_queries.push((item, captured));
                    self.event(G::UnityEqual)?;
                    let equal = self.c.unity_equals[self.out.unity_equals_consumed];
                    self.out.unity_equals_consumed += 1;
                    if !equal {
                        kept.push(item);
                    }
                }
                self.out.output_prefix = kept;
            }
            self.event(G::MoveNext)?;
        } else {
            let input = self.c.input.as_ref().ok_or(CharacterFilterFailure::Null)?;
            self.event(G::Enumerator)?;
            let version = self.out.input_version;
            for character in input {
                self.event(G::MoveNext)?;
                if version != self.out.input_version {
                    return Err(CharacterFilterFailure::Version);
                }
                let id = character.ok_or(CharacterFilterFailure::Null)?;
                let ch = &self.c.characters[&id];
                let keep = if self.c.method == CharacterFilterMethod::Revealed {
                    ch.state != 5
                } else {
                    let statuses = ch.statuses.as_ref().ok_or(CharacterFilterFailure::Null)?;
                    self.event(G::StatusContains)?;
                    statuses.contains(&self.c.requested_status)
                };
                if keep {
                    self.event(G::Add)?;
                    self.out.output_prefix.push(Some(id));
                    if self.c.advance_input_version_after_add {
                        self.out.input_version = self.out.input_version.wrapping_add(1);
                    }
                }
            }
            self.event(G::MoveNext)?;
            if version != self.out.input_version {
                return Err(CharacterFilterFailure::Version);
            }
        }
        self.out.returned = true;
        Ok(())
    }
}
/// Missing response provenance rejects the whole invocation. Response arrays may
/// include unused suffixes, but must cover every potentially reached occurrence.
pub fn replay_character_filter(
    c: &CharacterFilterContext,
) -> Result<CharacterFilterResult, FilterContextError> {
    if c.rule_version != CHARACTER_FILTER_NATIVE_V1
        || !c.metadata_initialized
        || !c.services_preserve_input_contents
        || !c.remove_all_commits_after_predicates
        || c.fail_at.is_some_and(|f| f.occurrence == 0)
    {
        return Err(FilterContextError::InvalidContext);
    }
    let input_len = c.input.as_ref().map_or(0, Vec::len);
    let board_len = c.board.as_ref().map_or(0, Vec::len);
    if input_len > LIMIT
        || board_len > LIMIT
        || c.characters.len() > LIMIT
        || c.managed_contains.len() > LIMIT
        || c.unity_equals.len() > 65_536
        || c.characters
            .values()
            .any(|v| v.statuses.as_ref().is_some_and(|s| s.len() > LIMIT))
    {
        return Err(FilterContextError::Capacity);
    }
    if c.board
        .iter()
        .flatten()
        .flatten()
        .any(|id| !c.characters.contains_key(id))
        || (c.method != CharacterFilterMethod::Unique
            && c.input
                .iter()
                .flatten()
                .flatten()
                .any(|id| !c.characters.contains_key(id)))
    {
        return Err(FilterContextError::InvalidContext);
    }
    // Preflight bounds response indexing without allocating a speculative event
    // history. Supplied false gates require no equality responses.
    if c.method == CharacterFilterMethod::Unique && c.input.is_some() && c.board.is_some() {
        if c.managed_contains.len() < board_len {
            return Err(FilterContextError::InvalidContext);
        }
        let mut remaining = input_len;
        let mut equality = 0;
        for (&contains, character) in c.managed_contains.iter().zip(c.board.as_ref().unwrap()) {
            if character.is_none() {
                break;
            }
            if contains {
                let end = equality + remaining;
                if end > c.unity_equals.len() {
                    return Err(FilterContextError::InvalidContext);
                }
                remaining = c.unity_equals[equality..end]
                    .iter()
                    .filter(|b| !**b)
                    .count();
                equality = end;
            }
        }
    }
    let mut replay = Replay {
        c,
        counts: [0; 10],
        out: CharacterFilterResult {
            output_prefix: vec![],
            returned: false,
            failure: None,
            events: vec![],
            managed_contains_queries: vec![],
            unity_equal_queries: vec![],
            managed_contains_consumed: 0,
            unity_equals_consumed: 0,
            input_version: c.input_version,
            gameplay_initialized: c.gameplay_initialized,
            unity_object_initialized: c.unity_object_initialized,
        },
    };
    replay.out.failure = replay.apply().err();
    Ok(replay.out)
}
#[cfg(test)]
#[path = "character_filter_tests.rs"]
mod tests;
