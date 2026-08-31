//! Native Baker Day-action history validation.
//!
//! Baker rewrites happen synchronously in reveal order.  They therefore need
//! one shared history witness: validating each Baker clue independently loses
//! the initial Villager multiset, while counting final Baker appearances as
//! ordinary roles rejects legitimate chains.

use std::collections::{HashMap, HashSet};

use crate::knowledge_base::{
    baker_preserved_runtime_class, get_card, normalize_role, BakerPreservedRuntimeClass, Faction,
};
use crate::types::{BoardCountProvenance, CardInfo, GameState, Scenario};

use super::{info_pos, info_str, known_evil_role, truth_status, TruthStatus};

pub(super) const BAKER_CURRENT_RULE: &str = "baker_day_reveal_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RuntimeState {
    NotBaker,
    Null,
    Baker(u8),
    Incompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalProjection {
    Unknown,
    Baker,
    Other(u8),
}

#[derive(Debug, Clone, Copy)]
struct SeatSpec {
    definite_villager: bool,
    optional_villager: bool,
    final_projection: FinalProjection,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SeatState {
    physical_villager: bool,
    initial_role: Option<u8>,
    current_role: Option<u8>,
    runtime: RuntimeState,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SearchState {
    seats: Vec<SeatState>,
    initial_counts: Vec<u8>,
    revealed: Vec<bool>,
    erased_roles: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Observation {
    Missing,
    Original,
    Named(u8),
    Interrupted,
    Invalid,
}

struct Problem<'a> {
    state: &'a GameState,
    scenario: &'a Scenario,
    exact: bool,
    role_names: Vec<String>,
    role_indices: HashMap<String, u8>,
    role_caps: Vec<u8>,
    baker_role: Option<u8>,
    specs: Vec<SeatSpec>,
    required_optional_villagers: Option<usize>,
    anonymous_erased_villagers: usize,
    required_erased_role: Option<u8>,
}

impl<'a> Problem<'a> {
    fn role_index(&self, role: &str) -> Option<u8> {
        self.role_indices.get(&normalize_role(role)).copied()
    }

    fn card(&self, position: u8) -> Option<&CardInfo> {
        self.state.card_at(position)
    }

    fn observation(&self, card: &CardInfo) -> Observation {
        let interrupted = card
            .info_parsed
            .get("shut_up_target")
            .and_then(|value| value.as_u64())
            .is_some_and(|position| position > 0 && position <= u64::from(self.state.n_cards));
        if interrupted {
            // A native Rambler replacement suppresses Baker's own text. Fresh
            // capture must therefore contain exactly one of these surfaces.
            if self.exact && card.info_parsed.contains_key("original_role") {
                return Observation::Invalid;
            }
            return Observation::Interrupted;
        }

        let Some(raw) = card.info_parsed.get("original_role") else {
            return if self.exact {
                Observation::Invalid
            } else {
                Observation::Missing
            };
        };
        let Some(claimed) = raw.as_str() else {
            return if self.exact {
                Observation::Invalid
            } else {
                Observation::Missing
            };
        };
        let normalized = normalize_role(claimed);
        if normalized == "original" {
            return Observation::Original;
        }
        if normalized.is_empty() || matches!(normalized.as_str(), "none" | "unknown") {
            return if self.exact {
                Observation::Invalid
            } else {
                Observation::Missing
            };
        }
        let Some(role) = self.role_index(claimed) else {
            return if self.exact {
                Observation::Invalid
            } else {
                Observation::Missing
            };
        };
        // Prefer the serialized pool faction so archived roles whose public
        // faction later changed keep their historical Baker semantics.
        if !role_is_state_villager(claimed, self.state) {
            return if self.exact {
                Observation::Invalid
            } else {
                Observation::Missing
            };
        }
        Observation::Named(role)
    }

    fn output_possible(
        &self,
        runtime: RuntimeState,
        truth: TruthStatus,
        puppet: bool,
        observation: Observation,
    ) -> bool {
        if observation == Observation::Invalid {
            return false;
        }
        if observation == Observation::Missing {
            return !self.exact;
        }

        if truth == TruthStatus::Truthful {
            let expected = match runtime {
                RuntimeState::Null if puppet => self.baker_role.map(Observation::Named),
                RuntimeState::Null => Some(Observation::Original),
                RuntimeState::Baker(role) => Some(Observation::Named(role)),
                RuntimeState::Incompatible | RuntimeState::NotBaker => None,
            };
            return match observation {
                Observation::Interrupted => expected.is_some(),
                // Archived parsing collapsed "I was a Baker" into the same
                // sentinel used for "I am the original Baker".
                Observation::Original if !self.exact => {
                    expected == Some(Observation::Original)
                        || expected == self.baker_role.map(Observation::Named)
                }
                _ => expected == Some(observation),
            };
        }

        let remaining_pool = |claimed: Option<u8>| -> bool {
            match runtime {
                RuntimeState::Incompatible | RuntimeState::NotBaker => false,
                RuntimeState::Null => claimed.map_or_else(
                    || self.role_caps.iter().any(|count| *count > 0),
                    |role| self.role_caps[usize::from(role)] > 0,
                ),
                RuntimeState::Baker(original) => claimed.map_or_else(
                    || {
                        self.role_caps
                            .iter()
                            .enumerate()
                            .any(|(index, count)| *count > u8::from(index == usize::from(original)))
                    },
                    |role| self.role_caps[usize::from(role)] > u8::from(role == original),
                ),
            }
        };

        match observation {
            Observation::Interrupted => remaining_pool(None),
            Observation::Named(role) => remaining_pool(Some(role)),
            Observation::Original if !self.exact => self
                .baker_role
                .is_some_and(|role| remaining_pool(Some(role))),
            Observation::Original | Observation::Missing | Observation::Invalid => false,
        }
    }
}

fn role_is_state_villager(role: &str, state: &GameState) -> bool {
    let normalized = normalize_role(role);
    if state
        .deck
        .villagers
        .iter()
        .any(|candidate| normalize_role(candidate) == normalized)
    {
        return true;
    }
    if state
        .deck
        .outcasts
        .iter()
        .chain(state.deck.minions.iter())
        .chain(state.deck.demons.iter())
        .any(|candidate| normalize_role(candidate) == normalized)
    {
        return false;
    }
    get_card(role).is_some_and(|card| card.faction == Faction::Villager)
}

fn observed_final_role<'a>(position: u8, state: &'a GameState) -> Option<&'a str> {
    state
        .executed_good_roles
        .get(&position)
        .map(String::as_str)
        .or_else(|| {
            state
                .card_at(position)
                .map(|card| card.apparent_role.as_str())
        })
}

fn excluded_from_good_villagers(position: u8, scenario: &Scenario, state: &GameState) -> bool {
    known_evil_role(position, scenario, state).is_some()
        || scenario.puppet_position == Some(position)
        || scenario.drunk_position == Some(position)
        || scenario.doppelganger_position == Some(position)
        || scenario.chancellor_added_outcast_position() == Some(position)
}

fn add_initial_role(problem: &Problem<'_>, search: &mut SearchState, role: u8) -> bool {
    let index = usize::from(role);
    if index >= search.initial_counts.len()
        || search.initial_counts[index] >= problem.role_caps[index]
    {
        return false;
    }
    search.initial_counts[index] += 1;
    true
}

fn initialize_for_shaman_previous(
    problem: &Problem<'_>,
    target_previous: Option<u8>,
) -> Option<SearchState> {
    let mut search = SearchState {
        seats: problem
            .specs
            .iter()
            .map(|spec| SeatState {
                physical_villager: spec.definite_villager,
                initial_role: None,
                current_role: None,
                runtime: RuntimeState::NotBaker,
            })
            .collect(),
        initial_counts: vec![0; problem.role_caps.len()],
        revealed: vec![false; usize::from(problem.state.n_cards) + 1],
        erased_roles: 0,
    };

    let shaman = problem.scenario.shaman_trace.as_ref();
    let source = shaman.map(|trace| trace.source_position);
    let target = shaman.map(|trace| trace.target_position);
    let copied_role = shaman.and_then(|trace| problem.role_index(&trace.copied_role));

    // Puppeteer overwrites a real Villager and keeps that victim's former
    // identity only as Puppet's visible bluff. It is not a later Baker target,
    // but the erased identity still consumes the initial Villager multiset.
    if let Some(puppet_position) = problem.scenario.puppet_position {
        if let Some(role) = observed_final_role(puppet_position, problem.state) {
            if !role_is_state_villager(role, problem.state) {
                if problem.exact {
                    return None;
                }
            } else {
                let initial = problem.role_index(role)?;
                if !add_initial_role(problem, &mut search, initial) {
                    return None;
                }
            }
        }
    }

    for position in 1..=problem.state.n_cards {
        let index = usize::from(position);
        if !problem.specs[index].definite_villager {
            continue;
        }

        let (initial, current, runtime) = if source == Some(position) {
            let copied = copied_role?;
            (
                copied,
                copied,
                if Some(copied) == problem.baker_role {
                    RuntimeState::Null
                } else {
                    RuntimeState::NotBaker
                },
            )
        } else if target == Some(position) {
            let previous = target_previous?;
            let copied = copied_role?;
            let runtime = if Some(copied) == problem.baker_role {
                match baker_preserved_runtime_class(&problem.role_names[usize::from(previous)]) {
                    BakerPreservedRuntimeClass::Null => RuntimeState::Null,
                    BakerPreservedRuntimeClass::Alchemist
                    | BakerPreservedRuntimeClass::Enlightened => RuntimeState::Incompatible,
                }
            } else {
                RuntimeState::NotBaker
            };
            (previous, copied, runtime)
        } else {
            match problem.specs[index].final_projection {
                FinalProjection::Other(role) => (role, role, RuntimeState::NotBaker),
                FinalProjection::Baker | FinalProjection::Unknown => continue,
            }
        };

        if !add_initial_role(problem, &mut search, initial) {
            return None;
        }
        search.seats[index].initial_role = Some(initial);
        search.seats[index].current_role = Some(current);
        search.seats[index].runtime = runtime;
    }
    Some(search)
}

fn initial_villager_assignments(problem: &Problem<'_>, initial: SearchState) -> Vec<SearchState> {
    let Some(required) = problem.required_optional_villagers else {
        return vec![initial];
    };
    let candidates: Vec<usize> = problem
        .specs
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(index, spec)| spec.optional_villager.then_some(index))
        .collect();
    if required > candidates.len() {
        return Vec::new();
    }

    fn choose(
        candidates: &[usize],
        offset: usize,
        remaining: usize,
        current: &mut SearchState,
        results: &mut Vec<SearchState>,
    ) {
        if remaining == 0 {
            results.push(current.clone());
            return;
        }
        if candidates.len().saturating_sub(offset) < remaining {
            return;
        }
        for candidate_index in offset..=candidates.len() - remaining {
            let seat_index = candidates[candidate_index];
            current.seats[seat_index].physical_villager = true;
            choose(
                candidates,
                candidate_index + 1,
                remaining - 1,
                current,
                results,
            );
            current.seats[seat_index].physical_villager = false;
        }
    }

    let mut current = initial;
    let mut results = Vec::new();
    choose(&candidates, 0, required, &mut current, &mut results);
    results
}

fn activate_with_initial_role(
    problem: &Problem<'_>,
    search: &SearchState,
    position: u8,
    role: u8,
) -> Option<SearchState> {
    let index = usize::from(position);
    let mut next = search.clone();
    if next.seats[index].initial_role.is_some() {
        return (next.seats[index].initial_role == Some(role)).then_some(next);
    }
    if !add_initial_role(problem, &mut next, role) {
        return None;
    }
    next.seats[index].physical_villager = true;
    next.seats[index].initial_role = Some(role);
    next.seats[index].current_role = Some(role);
    next.seats[index].runtime = if Some(role) == problem.baker_role {
        RuntimeState::Null
    } else {
        RuntimeState::NotBaker
    };
    Some(next)
}

fn ensure_baker_actor(
    problem: &Problem<'_>,
    search: &SearchState,
    position: u8,
) -> Option<SearchState> {
    let index = usize::from(position);
    if !search.seats[index].physical_villager {
        return Some(search.clone());
    }
    let baker = problem.baker_role?;
    match search.seats[index].current_role {
        Some(role) if role == baker => Some(search.clone()),
        Some(_) => None,
        None => activate_with_initial_role(problem, search, position, baker),
    }
}

fn actor_runtime(search: &SearchState, position: u8) -> RuntimeState {
    let seat = &search.seats[usize::from(position)];
    if seat.physical_villager {
        seat.runtime
    } else {
        // Puppet, Drunk, Doppelganger, and ordinary Evil Baker appearances
        // reach Baker with null runtimeData in the represented scenarios.
        RuntimeState::Null
    }
}

fn conversion_states(
    problem: &Problem<'_>,
    search: &SearchState,
    source_position: u8,
) -> Vec<SearchState> {
    let mut definite_pool_exists = false;
    let mut allowed_targets = Vec::new();

    for position in 1..=problem.state.n_cards {
        if position == source_position || search.revealed[usize::from(position)] {
            continue;
        }
        let index = usize::from(position);
        let seat = &search.seats[index];
        let spec = problem.specs[index];
        if seat.physical_villager {
            definite_pool_exists = true;
            if !matches!(spec.final_projection, FinalProjection::Other(_)) {
                allowed_targets.push(position);
            }
        } else if problem.required_optional_villagers.is_none() && spec.optional_villager {
            allowed_targets.push(position);
        }
    }

    let mut results = Vec::new();
    if !definite_pool_exists {
        // Every still-unknown seat may be an Outcast/not selected Villager, so
        // a native empty candidate pool remains an existential possibility.
        results.push(search.clone());
    }

    for target in allowed_targets {
        let index = usize::from(target);
        let bases: Vec<SearchState> = if search.seats[index].physical_villager
            && search.seats[index].current_role.is_some()
        {
            vec![search.clone()]
        } else {
            (0..problem.role_names.len())
                .filter_map(|role| activate_with_initial_role(problem, search, target, role as u8))
                .collect()
        };

        for mut next in bases {
            let previous = next.seats[index]
                .current_role
                .expect("activated Baker target has a current Villager role");
            if previous < 32 {
                next.erased_roles |= 1u32 << previous;
            }
            let Some(baker) = problem.baker_role else {
                continue;
            };
            next.seats[index].current_role = Some(baker);
            next.seats[index].runtime = RuntimeState::Baker(previous);
            results.push(next);
        }
    }

    results
}

fn process_baker(problem: &Problem<'_>, search: &SearchState, card: &CardInfo) -> Vec<SearchState> {
    let Some(ready) = ensure_baker_actor(problem, search, card.position) else {
        return Vec::new();
    };
    let runtime = actor_runtime(&ready, card.position);
    let truth = truth_status(card.position, problem.scenario, problem.state);
    let puppet = problem.scenario.puppet_position == Some(card.position);
    let observation = problem.observation(card);
    if !problem.output_possible(runtime, truth, puppet, observation) {
        return Vec::new();
    }

    let working_conversion = truth == TruthStatus::Truthful
        && !puppet
        && !matches!(runtime, RuntimeState::Incompatible | RuntimeState::NotBaker);
    if !working_conversion {
        return vec![ready];
    }
    conversion_states(problem, &ready, card.position)
}

fn is_current_poet_medium(card: &CardInfo) -> bool {
    normalize_role(&card.apparent_role) == "poet"
        && card.info_parsed.get("poet_variant").and_then(serde_json::Value::as_str)
            == Some("public_current")
        && card.info_parsed.get("copied_role").and_then(serde_json::Value::as_str)
            == Some("Medium")
}

fn process_medium(
    problem: &Problem<'_>,
    search: &SearchState,
    card: &CardInfo,
) -> Vec<SearchState> {
    let Some(target) = info_pos(&card.info_parsed, "good_position") else {
        return vec![search.clone()];
    };
    let Some(claimed) = info_str(&card.info_parsed, "good_role") else {
        return vec![search.clone()];
    };
    if target == 0 || target > problem.state.n_cards {
        return vec![search.clone()];
    }
    let target_index = usize::from(target);
    if problem.specs[target_index].final_projection != FinalProjection::Baker
        || !search.seats[target_index].physical_villager
        || truth_status(card.position, problem.scenario, problem.state) == TruthStatus::Lying
    {
        return vec![search.clone()];
    }
    let Some(claimed_role) = problem.role_index(claimed) else {
        return Vec::new();
    };

    if search.seats[target_index].current_role.is_none() {
        return activate_with_initial_role(problem, search, target, claimed_role)
            .into_iter()
            .collect();
    }

    let current = search.seats[target_index].current_role.unwrap();
    let predecessor = match search.seats[target_index].runtime {
        RuntimeState::Baker(role) => Some(role),
        _ => None,
    };
    let current_contract = match normalize_role(&card.apparent_role).as_str() {
        "medium" => {
            card.info_parsed
                .get("medium_variant")
                .and_then(serde_json::Value::as_str)
                == Some("public_current")
        }
        "poet" => is_current_poet_medium(card),
        _ => false,
    };
    if current == claimed_role || (!current_contract && predecessor == Some(claimed_role)) {
        vec![search.clone()]
    } else {
        Vec::new()
    }
}

fn process_reveal(problem: &Problem<'_>, search: &SearchState, position: u8) -> Vec<SearchState> {
    let mut revealed = search.clone();
    revealed.revealed[usize::from(position)] = true;
    let Some(card) = problem.card(position) else {
        return vec![revealed];
    };
    match normalize_role(&card.apparent_role).as_str() {
        "baker" => process_baker(problem, &revealed, card),
        "medium" => process_medium(problem, &revealed, card),
        "poet" if is_current_poet_medium(card) => {
            process_medium(problem, &revealed, card)
        }
        _ => vec![revealed],
    }
}

fn final_state_valid(problem: &Problem<'_>, search: &SearchState) -> bool {
    let mut completed_counts = search.initial_counts.clone();
    let mut unassigned_villagers = problem.anonymous_erased_villagers;
    let mut consume_unassigned = |role: u8| -> bool {
        let index = usize::from(role);
        if index >= completed_counts.len() || completed_counts[index] >= problem.role_caps[index] {
            return false;
        }
        completed_counts[index] += 1;
        true
    };

    for position in 1..=problem.state.n_cards {
        let index = usize::from(position);
        if !search.seats[index].physical_villager {
            continue;
        }
        match problem.specs[index].final_projection {
            FinalProjection::Baker => match search.seats[index].current_role {
                Some(role) if Some(role) == problem.baker_role => {}
                None => {
                    let Some(baker) = problem.baker_role else {
                        return false;
                    };
                    if !consume_unassigned(baker) {
                        return false;
                    }
                }
                Some(_) => return false,
            },
            FinalProjection::Other(expected) => match search.seats[index].current_role {
                Some(role) if role == expected => {}
                None if consume_unassigned(expected) => {}
                None | Some(_) => return false,
            },
            FinalProjection::Unknown => {
                if search.seats[index].initial_role.is_none() {
                    unassigned_villagers += 1;
                }
            }
        }
    }
    let remaining_capacity: usize = problem
        .role_caps
        .iter()
        .zip(completed_counts.iter())
        .map(|(capacity, used)| usize::from(capacity.saturating_sub(*used)))
        .sum();
    remaining_capacity >= unassigned_villagers
        && problem
            .required_erased_role
            .is_none_or(|role| role < 32 && search.erased_roles & (1u32 << role) != 0)
}

fn exact_search(
    problem: &Problem<'_>,
    order: &[u8],
    index: usize,
    search: SearchState,
    seen: &mut HashSet<(usize, SearchState)>,
) -> bool {
    if index == order.len() {
        return final_state_valid(problem, &search);
    }
    if !seen.insert((index, search.clone())) {
        return false;
    }
    process_reveal(problem, &search, order[index])
        .into_iter()
        .any(|next| exact_search(problem, order, index + 1, next, seen))
}

fn legacy_search(
    problem: &Problem<'_>,
    events: &[u8],
    remaining: u16,
    search: SearchState,
    seen: &mut HashSet<(u16, SearchState)>,
) -> bool {
    if remaining == 0 {
        return final_state_valid(problem, &search);
    }
    if !seen.insert((remaining, search.clone())) {
        return false;
    }
    events.iter().enumerate().any(|(event_index, position)| {
        let bit = 1u16 << event_index;
        remaining & bit != 0
            && process_reveal(problem, &search, *position)
                .into_iter()
                .any(|next| legacy_search(problem, events, remaining ^ bit, next, seen))
    })
}

fn build_problem<'a>(
    scenario: &'a Scenario,
    state: &'a GameState,
    required_erased_role: Option<&str>,
) -> Option<Problem<'a>> {
    let mut role_names = Vec::new();
    let mut role_indices = HashMap::new();
    let mut role_caps = Vec::new();
    for role in &state.deck.villagers {
        let normalized = normalize_role(role);
        if let Some(index) = role_indices.get(&normalized).copied() {
            role_caps[usize::from(index)] += 1;
        } else {
            let index = u8::try_from(role_names.len()).ok()?;
            role_indices.insert(normalized.clone(), index);
            role_names.push(normalized);
            role_caps.push(1);
        }
    }
    let baker_role = role_indices.get("baker").copied();
    let required_erased_role =
        required_erased_role.and_then(|role| role_indices.get(&normalize_role(role)).copied());
    if required_erased_role.is_some_and(|role| role >= 32) {
        return None;
    }
    let exact = state.baker_rule_version.as_deref() == Some(BAKER_CURRENT_RULE);
    let generated = scenario.chancellor_added_outcast_position();

    let mut specs = vec![
        SeatSpec {
            definite_villager: false,
            optional_villager: false,
            final_projection: FinalProjection::Unknown,
        };
        usize::from(state.n_cards) + 1
    ];
    let shaman_endpoints = scenario
        .shaman_trace
        .as_ref()
        .map(|trace| HashSet::from([trace.source_position, trace.target_position]));

    for position in 1..=state.n_cards {
        let index = usize::from(position);
        if excluded_from_good_villagers(position, scenario, state) {
            continue;
        }
        let observed = observed_final_role(position, state);
        let endpoint = shaman_endpoints
            .as_ref()
            .is_some_and(|positions| positions.contains(&position));
        let start_proves_villager = scenario.corrupted.contains(&position)
            || scenario.pd_corrupted == Some(position)
            || scenario.alchemist_cures.contains_key(&position);
        let definite = endpoint
            || start_proves_villager
            || observed.is_some_and(|role| role_is_state_villager(role, state));
        let optional = !definite && observed.is_none();
        let final_projection = observed
            .filter(|role| role_is_state_villager(role, state))
            .and_then(|role| {
                if normalize_role(role) == "baker" {
                    Some(FinalProjection::Baker)
                } else {
                    role_indices
                        .get(&normalize_role(role))
                        .copied()
                        .map(FinalProjection::Other)
                }
            })
            .unwrap_or(FinalProjection::Unknown);
        specs[index] = SeatSpec {
            definite_villager: definite,
            optional_villager: optional,
            final_projection,
        };
    }

    // The trusted pre-Start V header includes natural Drunk/Doppelganger
    // identities. Those disguisers are not real-Villager Baker candidates,
    // while later Chancellor and Puppeteer replacements each remove one real
    // Villager from the post-Start board. Resolve the remaining hidden seats
    // existentially inside this history instead of multiplying Scenarios.
    let required_optional_villagers = if state.board_count_provenance
        == BoardCountProvenance::TrustedPreStart
    {
        let mut natural_disguisers = HashSet::new();
        for position in [scenario.doppelganger_position, scenario.drunk_position]
            .into_iter()
            .flatten()
        {
            if Some(position) != generated {
                natural_disguisers.insert(position);
            }
        }
        if scenario.puppet_position.is_some_and(|position| {
            Some(position) == generated || natural_disguisers.contains(&position)
        }) {
            return None;
        }

        let reductions = natural_disguisers.len()
            + usize::from(scenario.puppet_position.is_some())
            + usize::from(generated.is_some());
        let post_start_total = usize::from(state.board_villager_count?).checked_sub(reductions)?;
        let definite = specs
            .iter()
            .skip(1)
            .filter(|spec| spec.definite_villager)
            .count();
        let required = post_start_total.checked_sub(definite)?;
        if required
            > specs
                .iter()
                .skip(1)
                .filter(|spec| spec.optional_villager)
                .count()
        {
            return None;
        }
        Some(required)
    } else {
        None
    };
    let puppet_role_is_known = scenario
        .puppet_position
        .and_then(|position| observed_final_role(position, state))
        .is_some_and(|role| role_is_state_villager(role, state));
    let anonymous_erased_villagers = usize::from(generated.is_some())
        + usize::from(scenario.puppet_position.is_some() && !puppet_role_is_known);

    Some(Problem {
        state,
        scenario,
        exact,
        role_names,
        role_indices,
        role_caps,
        baker_role,
        specs,
        required_optional_villagers,
        anonymous_erased_villagers,
        required_erased_role,
    })
}

fn coherent_exact_order(problem: &Problem<'_>) -> bool {
    let order_set: HashSet<u8> = problem.state.reveal_order.iter().copied().collect();
    let card_positions: Vec<u8> = problem
        .state
        .cards
        .iter()
        .map(|card| card.position)
        .collect();
    let unique_cards: HashSet<u8> = card_positions.iter().copied().collect();
    order_set.len() == problem.state.reveal_order.len()
        && unique_cards.len() == card_positions.len()
        && order_set
            .iter()
            .all(|position| *position > 0 && *position <= problem.state.n_cards)
        && card_positions.iter().all(|position| {
            *position > 0 && *position <= problem.state.n_cards && order_set.contains(position)
        })
}

fn require_pre_day_role(
    problem: &Problem<'_>,
    search: SearchState,
    position: u8,
    role: u8,
) -> Option<SearchState> {
    if position == 0 || position > problem.state.n_cards {
        return None;
    }
    let seat = &search.seats[usize::from(position)];
    if !seat.physical_villager {
        return None;
    }
    match seat.current_role {
        Some(current) => (current == role).then_some(search),
        None => activate_with_initial_role(problem, &search, position, role),
    }
}

fn history_exists_with_pre_day_role(
    scenario: &Scenario,
    state: &GameState,
    required_erased_role: Option<&str>,
    required_pre_day_role: Option<(u8, &str)>,
) -> bool {
    let baker_cards: Vec<&CardInfo> = state
        .cards
        .iter()
        .filter(|card| normalize_role(&card.apparent_role) == "baker")
        .collect();
    if baker_cards.is_empty() && required_erased_role.is_some() {
        // Shaman/Puppet/Chancellor erasures have their own native allowances.
        // Baker itself cannot erase a role until an apparent Baker Day action
        // has actually been revealed and captured.
        return false;
    }
    if baker_cards.is_empty()
        && required_erased_role.is_none()
        && required_pre_day_role.is_none()
        && scenario.shaman_trace.is_none()
    {
        return true;
    }
    let Some(problem) = build_problem(scenario, state, required_erased_role) else {
        return false;
    };
    if required_erased_role.is_some() && problem.required_erased_role.is_none() {
        return false;
    }
    let required_pre_day_role = match required_pre_day_role {
        Some((position, role)) => {
            let Some(role) = problem.role_index(role) else {
                return false;
            };
            Some((position, role))
        }
        None => None,
    };
    if problem.baker_role.is_none() && !baker_cards.is_empty() {
        // An authored Baker appearance needs the Baker script asset even when
        // the physical actor is an Evil/Outcast disguise.
        return false;
    }
    if problem.exact && !coherent_exact_order(&problem) {
        return false;
    }

    let previous_roles: Vec<Option<u8>> = if let Some(trace) = scenario.shaman_trace.as_ref() {
        trace
            .target_previous_roles
            .iter()
            .filter_map(|role| problem.role_index(role).map(Some))
            .collect()
    } else {
        vec![None]
    };
    if previous_roles.is_empty() {
        return false;
    }

    previous_roles.into_iter().any(|previous| {
        let Some(initial) = initialize_for_shaman_previous(&problem, previous) else {
            return false;
        };
        initial_villager_assignments(&problem, initial)
            .into_iter()
            .any(|initial| {
                let constrained = match required_pre_day_role {
                    Some((position, role)) => {
                        require_pre_day_role(&problem, initial, position, role)
                    }
                    None => Some(initial),
                };
                let Some(mut initial) = constrained else {
                    return false;
                };
                for &position in &state.night_kills {
                    if position > 0 && position <= state.n_cards {
                        initial.revealed[usize::from(position)] = true;
                    }
                }

                if problem.exact {
                    let order_set: HashSet<u8> = state.reveal_order.iter().copied().collect();
                    for &position in &state.executed {
                        if position > 0
                            && position <= state.n_cards
                            && !order_set.contains(&position)
                        {
                            initial.revealed[usize::from(position)] = true;
                        }
                    }
                    exact_search(
                        &problem,
                        &state.reveal_order,
                        0,
                        initial,
                        &mut HashSet::new(),
                    )
                } else {
                    let mut events: Vec<u8> = state
                        .cards
                        .iter()
                        .filter(|card| {
                            let role = normalize_role(&card.apparent_role);
                            role == "baker" || role == "medium" || is_current_poet_medium(card)
                        })
                        .map(|card| card.position)
                        .collect();
                    events.sort_unstable();
                    events.dedup();
                    if events.len() > 16 {
                        return false;
                    }
                    let event_set: HashSet<u8> = events.iter().copied().collect();
                    for card in &state.cards {
                        if !event_set.contains(&card.position) {
                            initial.revealed[usize::from(card.position)] = true;
                        }
                    }
                    for &position in &state.executed {
                        if position > 0
                            && position <= state.n_cards
                            && !event_set.contains(&position)
                        {
                            initial.revealed[usize::from(position)] = true;
                        }
                    }
                    let remaining = if events.len() == 16 {
                        u16::MAX
                    } else {
                        (1u16 << events.len()) - 1
                    };
                    legacy_search(&problem, &events, remaining, initial, &mut HashSet::new())
                }
            })
    })
}

fn history_exists(
    scenario: &Scenario,
    state: &GameState,
    required_erased_role: Option<&str>,
) -> bool {
    history_exists_with_pre_day_role(scenario, state, required_erased_role, None)
}

pub(super) fn validate_baker_history(scenario: &Scenario, state: &GameState) -> bool {
    history_exists(scenario, state, None)
}

pub(super) fn baker_history_can_erase_role(
    scenario: &Scenario,
    state: &GameState,
    role: &str,
) -> bool {
    history_exists(scenario, state, Some(role))
}

/// Whether one concrete seat can have carried `role` after all native Start
/// mutations but before the first player-triggered Day reveal. This shared
/// chronology surface lets clean Doppelganger validation resolve both final
/// Baker identities and trusted hidden-seat occupancy against the same
/// physical-role multiset and reveal history as ordinary Baker validation.
pub(super) fn baker_history_supports_pre_day_role(
    scenario: &Scenario,
    state: &GameState,
    position: u8,
    role: &str,
) -> bool {
    history_exists_with_pre_day_role(scenario, state, None, Some((position, role)))
}

/// Whether a truthful Medium observation aimed at a final Good Baker is owned
/// by the shared Baker history rather than the scalar card validator.
pub(super) fn medium_uses_baker_history(
    target: u8,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    !excluded_from_good_villagers(target, scenario, state)
        && observed_final_role(target, state).is_some_and(|role| normalize_role(role) == "baker")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DeckComposition, ShamanTrace};
    use serde_json::json;

    fn card(position: u8, role: &str, info: serde_json::Value) -> CardInfo {
        CardInfo {
            position,
            apparent_role: role.to_string(),
            info_text: String::new(),
            info_parsed: info.as_object().unwrap().clone(),
        }
    }

    fn state(villagers: &[&str], cards: Vec<CardInfo>, order: &[u8]) -> GameState {
        GameState {
            n_cards: cards.len() as u8,
            deck: DeckComposition {
                villagers: villagers.iter().map(|role| (*role).to_string()).collect(),
                ..DeckComposition::default()
            },
            cards,
            reveal_order: order.to_vec(),
            baker_rule_version: Some(BAKER_CURRENT_RULE.to_string()),
            ..GameState::default()
        }
    }

    #[test]
    fn exact_day_chain_requires_the_verified_reveal_chronology() {
        let mut state = state(
            &["Baker", "Empress", "Judge"],
            vec![
                card(1, "Baker", json!({"original_role": "Judge"})),
                card(2, "Baker", json!({"original_role": "original"})),
                card(3, "Baker", json!({"original_role": "Empress"})),
            ],
            &[2, 3, 1],
        );
        assert!(validate_baker_history(&Scenario::default(), &state));

        state.reveal_order = vec![1, 2, 3];
        assert!(!validate_baker_history(&Scenario::default(), &state));
    }

    #[test]
    fn exact_marker_rejects_incoherent_order_and_missing_clues() {
        let mut state = state(
            &["Baker"],
            vec![card(1, "Baker", json!({"original_role": "original"}))],
            &[1],
        );
        assert!(validate_baker_history(&Scenario::default(), &state));

        state.reveal_order = vec![1, 1];
        assert!(!validate_baker_history(&Scenario::default(), &state));
        state.reveal_order = vec![2];
        assert!(!validate_baker_history(&Scenario::default(), &state));
        state.reveal_order = vec![1];
        state.cards[0].info_parsed.clear();
        assert!(!validate_baker_history(&Scenario::default(), &state));
    }

    #[test]
    fn exact_marker_covers_every_entered_card_and_rejects_dual_clue_surfaces() {
        let mut captured = state(
            &["Baker", "Medium"],
            vec![
                card(1, "Baker", json!({"original_role": "original"})),
                card(2, "Medium", json!({})),
            ],
            &[2, 1],
        );
        assert!(validate_baker_history(&Scenario::default(), &captured));

        captured.reveal_order = vec![1];
        assert!(!validate_baker_history(&Scenario::default(), &captured));

        captured = state(
            &["Baker"],
            vec![card(
                1,
                "Baker",
                json!({"original_role": "original", "shut_up_target": 1}),
            )],
            &[1],
        );
        assert!(!validate_baker_history(&Scenario::default(), &captured));
        captured.cards[0].info_parsed.remove("original_role");
        assert!(validate_baker_history(&Scenario::default(), &captured));
    }

    #[test]
    fn literal_baker_is_not_the_original_sentinel_in_current_capture() {
        let state = state(
            &["Baker", "Judge"],
            vec![
                card(1, "Baker", json!({"original_role": "original"})),
                card(2, "Baker", json!({"original_role": "Baker"})),
            ],
            &[1, 2],
        );
        assert!(!validate_baker_history(&Scenario::default(), &state));

        let mut lying_descendant = Scenario::default();
        lying_descendant.corrupted.insert(2);
        assert!(validate_baker_history(&lying_descendant, &state));
    }

    #[test]
    fn shaman_null_runtime_can_speak_but_incompatible_runtime_must_be_overwritten() {
        let cards = vec![
            card(1, "Baker", json!({"original_role": "Baker"})),
            card(2, "Baker", json!({"original_role": "original"})),
            card(3, "Scout", json!({})),
        ];
        let mut null_state = state(&["Baker", "Judge", "Scout"], cards.clone(), &[2, 1, 3]);
        null_state.deck.minions = vec!["Shaman".to_string()];
        let mut null = Scenario::default();
        null.evil_positions.insert(3, "Shaman".to_string());
        null.shaman_trace = Some(ShamanTrace {
            source_position: 1,
            target_position: 2,
            copied_role: "Baker".to_string(),
            target_previous_roles: vec!["Judge".to_string()],
        });
        assert!(validate_baker_history(&null, &null_state));

        let mut incompatible_state = state(
            &["Baker", "Alchemist", "Scout"],
            vec![
                card(1, "Baker", json!({"original_role": "original"})),
                card(2, "Baker", json!({"original_role": "Baker"})),
                card(3, "Scout", json!({})),
            ],
            &[2, 1, 3],
        );
        incompatible_state.deck.minions = vec!["Shaman".to_string()];
        let mut incompatible = Scenario::default();
        incompatible.evil_positions.insert(3, "Shaman".to_string());
        incompatible.shaman_trace = Some(ShamanTrace {
            source_position: 1,
            target_position: 2,
            copied_role: "Baker".to_string(),
            target_previous_roles: vec!["Alchemist".to_string()],
        });
        assert!(!validate_baker_history(&incompatible, &incompatible_state));

        incompatible_state.reveal_order = vec![1, 2, 3];
        assert!(validate_baker_history(&incompatible, &incompatible_state));

        incompatible_state.deck.villagers[1] = "Enlightened".to_string();
        incompatible
            .shaman_trace
            .as_mut()
            .unwrap()
            .target_previous_roles = vec!["Enlightened".to_string()];
        assert!(validate_baker_history(&incompatible, &incompatible_state));
    }

    #[test]
    fn shaman_initial_multiset_is_checked_without_an_apparent_baker() {
        let mut state = state(
            &["Scout", "Witness"],
            vec![
                card(1, "Scout", json!({})),
                card(2, "Scout", json!({})),
                card(3, "Shaman", json!({})),
            ],
            &[1, 2, 3],
        );
        state.deck.minions = vec!["Shaman".to_string()];
        let mut scenario = Scenario::default();
        scenario.evil_positions.insert(3, "Shaman".to_string());
        scenario.shaman_trace = Some(ShamanTrace {
            source_position: 1,
            target_position: 2,
            copied_role: "Scout".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });
        assert!(!validate_baker_history(&scenario, &state));

        scenario
            .shaman_trace
            .as_mut()
            .unwrap()
            .target_previous_roles = vec!["Witness".to_string()];
        assert!(validate_baker_history(&scenario, &state));

        state.deck.villagers.push("Knight".to_string());
        assert!(!baker_history_can_erase_role(&scenario, &state, "Knight"));
    }

    #[test]
    fn puppet_baker_consumes_the_erased_initial_baker_identity() {
        let mut state = state(
            &["Baker"],
            vec![
                card(1, "Baker", json!({"original_role": "Baker"})),
                card(2, "Baker", json!({"original_role": "original"})),
                card(3, "Puppeteer", json!({})),
            ],
            &[1, 2, 3],
        );
        state.deck.minions = vec!["Puppeteer".to_string()];
        let mut scenario = Scenario::default();
        scenario.puppet_position = Some(1);
        scenario.evil_positions.insert(3, "Puppeteer".to_string());
        assert!(!validate_baker_history(&scenario, &state));

        state.deck.villagers.push("Baker".to_string());
        assert!(validate_baker_history(&scenario, &state));
    }

    #[test]
    fn hidden_puppet_and_chancellor_reserve_an_erased_villager_identity() {
        let mut puppet_state = GameState {
            n_cards: 3,
            deck: DeckComposition {
                villagers: vec!["Baker".to_string()],
                minions: vec!["Puppeteer".to_string()],
                ..DeckComposition::default()
            },
            cards: vec![
                card(1, "Baker", json!({"original_role": "original"})),
                card(3, "Puppeteer", json!({})),
            ],
            reveal_order: vec![1, 3],
            baker_rule_version: Some(BAKER_CURRENT_RULE.to_string()),
            ..GameState::default()
        };
        let mut puppet = Scenario::default();
        puppet.puppet_position = Some(2);
        puppet.evil_positions.insert(3, "Puppeteer".to_string());
        assert!(!validate_baker_history(&puppet, &puppet_state));
        puppet_state.deck.villagers.push("Judge".to_string());
        assert!(validate_baker_history(&puppet, &puppet_state));

        let mut chancellor_state = GameState {
            n_cards: 3,
            deck: DeckComposition {
                villagers: vec!["Baker".to_string()],
                demons: vec!["Chancellor".to_string()],
                ..DeckComposition::default()
            },
            cards: vec![
                card(1, "Baker", json!({"original_role": "original"})),
                card(3, "Chancellor", json!({})),
            ],
            reveal_order: vec![1, 3],
            baker_rule_version: Some(BAKER_CURRENT_RULE.to_string()),
            ..GameState::default()
        };
        let mut chancellor = Scenario::default();
        chancellor
            .evil_positions
            .insert(3, "Chancellor".to_string());
        chancellor.chancellor_conversion = Some(2);
        assert!(!validate_baker_history(&chancellor, &chancellor_state));
        chancellor_state.deck.villagers.push("Judge".to_string());
        assert!(validate_baker_history(&chancellor, &chancellor_state));
    }

    #[test]
    fn trusted_villager_count_makes_hidden_conversion_mandatory_at_that_time() {
        let mut state = GameState {
            n_cards: 3,
            deck: DeckComposition {
                villagers: vec!["Baker".to_string(), "Knight".to_string()],
                minions: vec!["Minion".to_string()],
                ..DeckComposition::default()
            },
            cards: vec![card(1, "Baker", json!({"original_role": "original"}))],
            reveal_order: vec![1],
            baker_rule_version: Some(BAKER_CURRENT_RULE.to_string()),
            board_villager_count: Some(2),
            board_outcast_count: Some(0),
            board_count_provenance: BoardCountProvenance::TrustedPreStart,
            ..GameState::default()
        };
        let mut scenario = Scenario::default();
        scenario.evil_positions.insert(3, "Minion".to_string());

        assert!(validate_baker_history(&scenario, &state));
        assert!(baker_history_can_erase_role(&scenario, &state, "Knight"));

        state.reveal_order = vec![2, 1];
        assert!(validate_baker_history(&scenario, &state));
        assert!(!baker_history_can_erase_role(&scenario, &state, "Knight"));
    }

    #[test]
    fn medium_predecessor_timing_and_knight_erasure_share_the_history() {
        let mut state = state(
            &["Baker", "Knight", "Medium"],
            vec![
                card(1, "Baker", json!({"original_role": "Knight"})),
                card(2, "Baker", json!({"original_role": "original"})),
                card(
                    3,
                    "Medium",
                    json!({"good_position": 1, "good_role": "Baker"}),
                ),
            ],
            &[2, 3, 1],
        );
        assert!(validate_baker_history(&Scenario::default(), &state));
        assert!(baker_history_can_erase_role(
            &Scenario::default(),
            &state,
            "Knight",
        ));

        state.reveal_order = vec![3, 2, 1];
        assert!(!validate_baker_history(&Scenario::default(), &state));
        assert!(!baker_history_can_erase_role(
            &Scenario::default(),
            &state,
            "Knight",
        ));
    }

    #[test]
    fn legacy_missing_claim_is_unknown_and_original_keeps_old_baker_ambiguity() {
        let mut archived = state(
            &["Baker", "Judge"],
            vec![
                card(1, "Baker", json!({"original_role": "original"})),
                card(2, "Baker", json!({"original_role": "none"})),
            ],
            &[1, 2],
        );
        archived.baker_rule_version = None;
        assert!(validate_baker_history(&Scenario::default(), &archived));

        archived.cards[1].info_parsed = json!({"original_role": "Judge"})
            .as_object()
            .unwrap()
            .clone();
        assert!(validate_baker_history(&Scenario::default(), &archived));

        let mut redesigned_role = state(
            &["Baker", "Rambler"],
            vec![
                card(1, "Baker", json!({"original_role": "original"})),
                card(2, "Baker", json!({"original_role": "Rambler"})),
            ],
            &[1, 2],
        );
        redesigned_role.baker_rule_version = None;
        assert!(validate_baker_history(
            &Scenario::default(),
            &redesigned_role
        ));
    }
}
