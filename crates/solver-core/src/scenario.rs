/// Scenario generation: enumerate all possible evil placements and build
/// full scenarios with corruption variants.

use std::collections::{HashMap, HashSet};
use crate::corruption::{
    enumerate_post_twin_corruption, enumerate_pre_twin_corruption,
    enumerate_start_corruption, StartCorruptionContext, StartCorruptionOutcome,
};
use crate::geometry::adjacent_positions;
use crate::knowledge_base::{
    get_card, normalize_role, shaman_erased_role_class,
    BakerPreservedRuntimeClass, Faction,
};
use crate::twin::{enumerate_twin_traces, role_after_twin};
use crate::types::{
    BoardCountProvenance, ChancellorTrace, GameState, Scenario, ShamanTrace, TwinTrace,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct RawChancellorTrace {
    original_position: u8,
    added_outcast_position: u8,
    added_outcast_role: String,
    anchor_position: u8,
}

impl RawChancellorTrace {
    fn original_villager_position(&self, final_chancellor_position: u8) -> u8 {
        if self.original_position == self.added_outcast_position {
            final_chancellor_position
        } else {
            self.added_outcast_position
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ScenarioSemanticKey {
    corrupted: Vec<u8>,
    messed_up_by_evil: Vec<u8>,
    pd_target: Option<u8>,
    alchemist_counts: Vec<(u8, u8)>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    chancellor_added: Option<(u8, String)>,
    shaman_trace: Option<ShamanTrace>,
    twin_trace: Option<TwinTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StartContextSemanticKey {
    real_villagers_before_puppet: Vec<u8>,
    registered_villagers_at_pd_call: Vec<u8>,
    corruption_resistant_at_init: Vec<u8>,
    messed_up_resistant_at_init: Vec<u8>,
    true_alchemists: Vec<u8>,
    initial_messed_up_by_evil: Vec<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    plague_doctor_acts: bool,
    shaman_trace: Option<ShamanTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PendingStartKey {
    chancellor_added: Option<(u8, String)>,
    context: StartContextSemanticKey,
}

struct PendingStartContext {
    context: StartCorruptionContext,
    added_outcast_position: Option<u8>,
    added_outcast_role: Option<String>,
    original_positions: Vec<u8>,
    affected_anchor_positions: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EvilPlacement {
    // Ordinary generated Puppets are present in both surfaces. When Twin moved
    // Villager data onto its stable Evil body before Puppeteer acted, `roles`
    // retains that Twin origin and `puppet_position` overlays the same seat.
    roles: HashMap<u8, String>,
    puppet_position: Option<u8>,
}

impl std::ops::Deref for EvilPlacement {
    type Target = HashMap<u8, String>;

    fn deref(&self) -> &Self::Target {
        &self.roles
    }
}

fn merge_position_candidates(target: &mut Vec<u8>, candidates: &[u8]) {
    for &position in candidates {
        if !target.contains(&position) {
            target.push(position);
        }
    }
    target.sort_unstable();
}

/// Prefer the serialized pool's faction over the current knowledge base.
/// Frozen cases span role redesigns (notably Rambler's Villager -> Outcast
/// move), while live states also serialize the current pool explicitly.
fn role_faction_in_state(role: &str, state: &GameState) -> Option<Faction> {
    let wanted = normalize_role(role);
    if state
        .deck
        .villagers
        .iter()
        .any(|candidate| normalize_role(candidate) == wanted)
    {
        return Some(Faction::Villager);
    }
    if state
        .deck
        .outcasts
        .iter()
        .any(|candidate| normalize_role(candidate) == wanted)
    {
        return Some(Faction::Outcast);
    }
    if state
        .deck
        .minions
        .iter()
        .any(|candidate| normalize_role(candidate) == wanted)
    {
        return Some(Faction::Minion);
    }
    if state
        .deck
        .demons
        .iter()
        .any(|candidate| normalize_role(candidate) == wanted)
    {
        return Some(Faction::Demon);
    }
    get_card(role).map(|card| card.faction)
}

fn is_state_villager_role(role: &str, state: &GameState) -> bool {
    role_faction_in_state(role, state) == Some(Faction::Villager)
}

fn is_state_outcast_role(role: &str, state: &GameState) -> bool {
    role_faction_in_state(role, state) == Some(Faction::Outcast)
}

fn is_hud_villager_outcast(role: &str) -> bool {
    matches!(normalize_role(role).as_str(), "doppelganger" | "drunk")
}

#[allow(clippy::too_many_arguments)]
fn is_known_natural_ordinary_outcast(
    position: u8,
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    generated_position: Option<u8>,
) -> bool {
    if full_evil.contains_key(&position)
        || doppelganger_position == Some(position)
        || drunk_position == Some(position)
        || generated_position == Some(position)
    {
        return false;
    }
    let role = state
        .executed_good_roles
        .get(&position)
        .map(String::as_str)
        .or_else(|| state.card_at(position).map(|card| card.apparent_role.as_str()));
    role.is_some_and(|role| {
        is_state_outcast_role(role, state) && !is_hud_villager_outcast(role)
    })
}

/// Generate all candidate scenarios for the current game state.
pub fn build_scenarios(state: &GameState) -> Vec<Scenario> {
    let executed_role_branches = branch_untyped_executed_evil_roles(state);
    if executed_role_branches.is_empty() {
        return Vec::new();
    }

    if supports_ordered_twin_start_slice(state) {
        let mut ordered_scenarios = Vec::new();
        let mut ordered_complete = true;
        for executed_evil_roles in &executed_role_branches {
            let mut branch = state.clone();
            branch.executed_evil_roles = executed_evil_roles.clone();
            match build_scenarios_with_start_mode(&branch, true) {
                Some(mut scenarios) => ordered_scenarios.append(&mut scenarios),
                None => {
                    ordered_complete = false;
                    break;
                }
            }
        }
        if ordered_complete {
            return ordered_scenarios;
        }
    }

    let mut scenarios = Vec::new();
    for executed_evil_roles in executed_role_branches {
        let mut branch = state.clone();
        branch.executed_evil_roles = executed_evil_roles;
        scenarios.extend(
            build_scenarios_with_start_mode(&branch, false)
                .expect("legacy one-shot Start generation is infallible"),
        );
    }
    scenarios
}

/// Expand private stable-origin hypotheses for dead confirmed Evil seats whose
/// public execution record did not reveal an original role. The cloned maps
/// are construction inputs only: validation still receives the original
/// `GameState`, while each resulting `Scenario` retains its exact branch.
fn branch_untyped_executed_evil_roles(
    state: &GameState,
) -> Vec<HashMap<u8, String>> {
    let mut untyped_positions: Vec<u8> = state
        .confirmed_evil
        .iter()
        .copied()
        .filter(|position| state.executed.contains(position))
        .filter(|position| {
            state
                .executed_evil_roles
                .get(position)
                .is_none_or(|role| normalize_role(role) == "unknown")
        })
        .collect();
    for (&position, role) in &state.executed_evil_roles {
        if state.executed.contains(&position)
            && normalize_role(role) == "unknown"
            && !untyped_positions.contains(&position)
        {
            untyped_positions.push(position);
        }
    }
    untyped_positions.sort_unstable();
    untyped_positions.dedup();

    let mut remaining_authored = state.deck.evil_roles();
    let puppeteer_authored = remaining_authored
        .iter()
        .any(|role| normalize_role(role) == "puppeteer");
    // Some historical pools serialized Puppet beside Puppeteer even though
    // native creates it at Start. Keep at most one generated occurrence.
    if puppeteer_authored {
        if let Some(index) = remaining_authored
            .iter()
            .position(|role| normalize_role(role) == "puppet")
        {
            remaining_authored.remove(index);
        }
    }
    let mut generated_puppet_available = puppeteer_authored;

    for role in state.executed_evil_roles.values() {
        let normalized = normalize_role(role);
        if normalized == "unknown" {
            continue;
        }
        if normalized == "puppet" && generated_puppet_available {
            generated_puppet_available = false;
            continue;
        }
        let Some(index) = remaining_authored
            .iter()
            .position(|candidate| normalize_role(candidate) == normalized)
        else {
            // Exact stable-origin evidence outside the authored multiset is an
            // inconsistent state, not permission to invent an extra role.
            return Vec::new();
        };
        remaining_authored.remove(index);
    }
    if untyped_positions.is_empty() {
        return vec![state.executed_evil_roles.clone()];
    }

    let mut candidates = remaining_authored;
    if generated_puppet_available {
        candidates.push("Puppet".to_string());
    }
    if untyped_positions.len() > candidates.len() {
        return Vec::new();
    }

    let mut branches = Vec::new();
    let mut seen = HashSet::new();
    for chosen_indices in combinations_indices(candidates.len(), untyped_positions.len()) {
        let chosen_roles: Vec<String> = chosen_indices
            .iter()
            .map(|index| candidates[*index].clone())
            .collect();
        for roles in permutations_of(&chosen_roles) {
            let mut branch = state.executed_evil_roles.clone();
            for (&position, role) in untyped_positions.iter().zip(roles) {
                branch.insert(position, role);
            }
            let mut key: Vec<(u8, String)> = branch
                .iter()
                .map(|(&position, role)| (position, normalize_role(role)))
                .collect();
            key.sort_unstable();
            if seen.insert(key) {
                branches.push(branch);
            }
        }
    }
    branches
}

/// Build one state atomically in either legacy one-shot mode or the narrowly
/// supported ordered Twin slice. An incomplete ordered world aborts the entire
/// slice so the public wrapper can rebuild every world through the legacy path.
fn build_scenarios_with_start_mode(
    state: &GameState,
    ordered_twin_start: bool,
) -> Option<Vec<Scenario>> {
    let placements = generate_evil_placements(state);
    let mut scenarios = Vec::new();
    let n = state.n_cards;
    let mut common_twin_trace_width: Option<usize> = None;

    for generated_placement in &placements {
        let placement = &generated_placement.roles;
        if !apply_placement_constraints(placement, state) {
            continue;
        }

        // Find Puppet position
        let puppet_pos = generated_placement.puppet_position;

        // Build full evil set including executed evils
        let mut full_evil: HashMap<u8, String> = placement.clone();
        for (&pos, role) in &state.executed_evil_roles {
            full_evil.entry(pos).or_insert_with(|| role.clone());
        }
        for &pos in &state.confirmed_evil {
            if state.executed.contains(&pos) && !full_evil.contains_key(&pos) {
                full_evil.insert(pos, "Unknown".to_string());
            }
        }

        // `full_evil` is the structural evil-role map after Chancellor.
        // Ordered Twin current-data replay remains separate from it;
        // Chancellor's original physical seat is a hidden history variable.
        let final_chancellor_positions: Vec<u8> = full_evil.iter()
            .filter(|(_, role)| normalize_role(role) == "chancellor")
            .map(|(&position, _)| position)
            .collect();
        if final_chancellor_positions.len() > 1 {
            // The audited Standard pool contains one Chancellor. Multiple
            // interacting Baron swaps need a separate lifecycle model.
            continue;
        }
        let final_chancellor_position = final_chancellor_positions.first().copied();

        // `state.deck.outcasts` is the authoritative full role pool. Baa only
        // obscures one of those entries in the visual deck view; normalization
        // restores it before the state reaches the solver.
        let chancellor_present = final_chancellor_position.is_some();
        // Doppelganger candidates
        let has_doppelganger = state
            .deck
            .outcasts
            .iter()
            .any(|o| normalize_role(o) == "doppelganger");
        let (can_have_dopp, can_skip_dopp) = hidden_outcast_presence_flags(
            "Doppelganger",
            state,
            chancellor_present,
        );
        let mut dopp_candidates: Vec<Option<u8>> = if can_skip_dopp { vec![None] } else { vec![] };
        if has_doppelganger && can_have_dopp {
            for p in 1..=n {
                if full_evil.contains_key(&p) || puppet_pos == Some(p) { continue; }
                if let Some(card) = state.card_at(p) {
                    if is_state_villager_role(&card.apparent_role, state) {
                        dopp_candidates.push(Some(p));
                    }
                } else {
                    dopp_candidates.push(Some(p)); // Unrevealed
                }
            }
        }

        // Drunk candidates
        let has_drunk = state
            .deck
            .outcasts
            .iter()
            .any(|o| normalize_role(o) == "drunk");
        let drunk_already_known = state.cards.iter().any(|c| c.apparent_role == "Drunk");
        let (can_have_drunk, can_skip_drunk) = hidden_outcast_presence_flags(
            "Drunk",
            state,
            chancellor_present,
        );
        let mut drunk_candidates: Vec<Option<u8>> = if drunk_already_known || can_skip_drunk { vec![None] } else { vec![] };
        if has_drunk && !drunk_already_known && can_have_drunk {
            for p in 1..=n {
                if full_evil.contains_key(&p) || puppet_pos == Some(p) { continue; }
                if let Some(card) = state.card_at(p) {
                    if is_state_villager_role(&card.apparent_role, state) {
                        drunk_candidates.push(Some(p));
                    }
                } else {
                    drunk_candidates.push(Some(p));
                }
            }
        }

        // Identity hypotheses feed one ordered Start simulator. Random
        // Poisoner and Plague Doctor choices branch inside that simulator so
        // each actor sees the live statuses left by earlier actors.
        let mut seen: HashMap<ScenarioSemanticKey, usize> = HashMap::new();
        let mut placement_scenarios: Vec<Scenario> = Vec::new();

        for &dopp_pos_opt in &dopp_candidates {
            for &drunk_pos_opt in &drunk_candidates {
                if drunk_pos_opt.is_some() && drunk_pos_opt == dopp_pos_opt {
                    continue;
                }
                if final_chancellor_position.is_none()
                    && !natural_outcast_hypothesis_allows(
                        state,
                        &full_evil,
                        puppet_pos,
                        dopp_pos_opt,
                        drunk_pos_opt,
                        None,
                        None,
                    )
                {
                    continue;
                }
                let trace_variants: Vec<Option<RawChancellorTrace>> =
                    if let Some(final_chancellor_position) = final_chancellor_position {
                        enumerate_raw_chancellor_traces(
                            state,
                            &full_evil,
                            puppet_pos,
                            dopp_pos_opt,
                            drunk_pos_opt,
                            final_chancellor_position,
                        )
                        .into_iter()
                        .map(Some)
                        .collect()
                    } else {
                        vec![None]
                    };

                let mut pending_seen: HashMap<PendingStartKey, usize> = HashMap::new();
                let mut pending_contexts: Vec<PendingStartContext> = Vec::new();
                let mut pd_variants_cache: HashMap<(u8, String, u8), Vec<bool>> = HashMap::new();
                let mut context_variants_cache: HashMap<
                    (u8, u8, u8, bool),
                    Vec<StartCorruptionContext>,
                > = HashMap::new();
                for raw_trace in &trace_variants {
                    let pd_act_variants = if let Some(trace) = raw_trace.as_ref() {
                        let cache_key = (
                            trace.added_outcast_position,
                            normalize_role(&trace.added_outcast_role),
                            trace.anchor_position,
                        );
                        if let Some(cached) = pd_variants_cache.get(&cache_key) {
                            cached.clone()
                        } else {
                            let variants = plague_doctor_act_variants(
                                state,
                                &full_evil,
                                dopp_pos_opt,
                                drunk_pos_opt,
                                puppet_pos,
                                Some(trace),
                            );
                            pd_variants_cache.insert(cache_key, variants.clone());
                            variants
                        }
                    } else {
                        plague_doctor_act_variants(
                            state,
                            &full_evil,
                            dopp_pos_opt,
                            drunk_pos_opt,
                            puppet_pos,
                            None,
                        )
                    };

                    for &plague_doctor_acts in &pd_act_variants {
                        let context_variants = if let Some(trace) = raw_trace.as_ref() {
                            let cache_key = (
                                trace.original_position,
                                trace.added_outcast_position,
                                trace.anchor_position,
                                plague_doctor_acts,
                            );
                            if let Some(cached) = context_variants_cache.get(&cache_key) {
                                cached.clone()
                            } else {
                                let variants = build_chancellor_start_context_variants(
                                    state,
                                    &full_evil,
                                    dopp_pos_opt,
                                    drunk_pos_opt,
                                    puppet_pos,
                                    final_chancellor_position.expect("trace requires Chancellor"),
                                    trace,
                                    plague_doctor_acts,
                                );
                                context_variants_cache.insert(cache_key, variants.clone());
                                variants
                            }
                        } else {
                            let nk_alch_variants = night_killed_alchemist_variants(
                                state, &full_evil, dopp_pos_opt, drunk_pos_opt, puppet_pos, None,
                            );
                            nk_alch_variants
                                .iter()
                                .map(|nk_alchemists| {
                                    build_start_corruption_context(
                                        state,
                                        &full_evil,
                                        dopp_pos_opt,
                                        drunk_pos_opt,
                                        puppet_pos,
                                        None,
                                        nk_alchemists,
                                        plague_doctor_acts,
                                    )
                                })
                                .collect()
                        };

                        for base_context in context_variants {
                            for context in shaman_start_context_variants(
                                state,
                                &full_evil,
                                dopp_pos_opt,
                                drunk_pos_opt,
                                puppet_pos,
                                raw_trace.as_ref(),
                                plague_doctor_acts,
                                base_context,
                            )
                            .into_iter()
                            .filter(|context| {
                                start_context_matches_native_puppeteer_conversion(
                                    &full_evil,
                                    puppet_pos,
                                    context,
                                    state,
                                )
                            }) {
                                let chancellor_added = raw_trace.as_ref().map(|trace| {
                                    (
                                        trace.added_outcast_position,
                                        normalize_role(&trace.added_outcast_role),
                                    )
                                });
                                let key = PendingStartKey {
                                    chancellor_added,
                                    context: start_context_key(&context),
                                };
                                if let Some(&index) = pending_seen.get(&key) {
                                    if let Some(trace) = raw_trace.as_ref() {
                                        merge_position_candidates(
                                            &mut pending_contexts[index].original_positions,
                                            &[trace.original_position],
                                        );
                                        merge_position_candidates(
                                            &mut pending_contexts[index]
                                                .affected_anchor_positions,
                                            &[trace.anchor_position],
                                        );
                                    }
                                    continue;
                                }

                                let index = pending_contexts.len();
                                pending_contexts.push(PendingStartContext {
                                    context,
                                    added_outcast_position: raw_trace
                                        .as_ref()
                                        .map(|trace| trace.added_outcast_position),
                                    added_outcast_role: raw_trace
                                        .as_ref()
                                        .map(|trace| trace.added_outcast_role.clone()),
                                    original_positions: raw_trace
                                        .as_ref()
                                        .map(|trace| vec![trace.original_position])
                                        .unwrap_or_default(),
                                    affected_anchor_positions: raw_trace
                                        .as_ref()
                                        .map(|trace| vec![trace.anchor_position])
                                        .unwrap_or_default(),
                                });
                                pending_seen.insert(key, index);
                            }
                        }
                    }
                }

                for pending in pending_contexts {
                    let outcome_variants: Vec<(StartCorruptionOutcome, Option<TwinTrace>)> =
                        if ordered_twin_start {
                            if dopp_pos_opt.is_some()
                                || drunk_pos_opt.is_some()
                                || puppet_pos.is_some()
                                || pending.added_outcast_position.is_some()
                                || pending.added_outcast_role.is_some()
                                || !pending.original_positions.is_empty()
                                || !pending.affected_anchor_positions.is_empty()
                            {
                                return None;
                            }
                            let (trace_width, outcomes) =
                                enumerate_ordered_twin_start_outcomes(
                                    state,
                                    &full_evil,
                                    &pending.context,
                                )?;
                            if common_twin_trace_width
                                .is_some_and(|expected| expected != trace_width)
                            {
                                return None;
                            }
                            common_twin_trace_width = Some(trace_width);
                            outcomes
                                .into_iter()
                                .map(|(outcome, trace)| (outcome, Some(trace)))
                                .collect()
                        } else {
                            enumerate_start_corruption(
                                n,
                                &full_evil,
                                &pending.context,
                                state.pd_corruption_target,
                            )
                            .into_iter()
                            .map(|outcome| (outcome, None))
                            .collect()
                        };
                    for (outcome, twin_trace) in outcome_variants {
                        let mut corr_key: Vec<u8> = outcome.corrupted.iter().copied().collect();
                        corr_key.sort_unstable();
                        let mut affected_key: Vec<u8> = outcome
                            .messed_up_by_evil
                            .iter()
                            .copied()
                            .collect();
                        affected_key.sort_unstable();
                        let mut alch_key: Vec<(u8, u8)> = outcome
                            .alchemist_counts
                            .iter()
                            .map(|(&position, &count)| (position, count))
                            .collect();
                        alch_key.sort_unstable();
                        let chancellor_added = pending.added_outcast_position.zip(
                            pending
                                .added_outcast_role
                                .as_deref()
                                .map(normalize_role),
                        );
                        // A hidden PD choice is an internal Start-history
                        // variable. Final statuses and Alchemist counts retain
                        // every observable consequence, while only an
                        // explicitly supplied target is safe to serialize as a
                        // known identity. This also lets cured, otherwise
                        // equivalent target histories collapse to one world.
                        let represented_pd_target = state.pd_corruption_target;
                        let key = ScenarioSemanticKey {
                            corrupted: corr_key,
                            messed_up_by_evil: affected_key,
                            pd_target: represented_pd_target,
                            alchemist_counts: alch_key,
                            doppelganger_position: dopp_pos_opt,
                            drunk_position: drunk_pos_opt,
                            chancellor_added,
                            shaman_trace: outcome.shaman_trace.clone(),
                            twin_trace: twin_trace.clone(),
                        };

                        if let Some(&index) = seen.get(&key) {
                            if let Some(trace) =
                                placement_scenarios[index].chancellor_trace.as_mut()
                            {
                                merge_position_candidates(
                                    &mut trace.original_positions,
                                    &pending.original_positions,
                                );
                                merge_position_candidates(
                                    &mut trace.affected_anchor_positions,
                                    &pending.affected_anchor_positions,
                                );
                            }
                            continue;
                        }

                        let chancellor_trace = pending
                            .added_outcast_position
                            .zip(pending.added_outcast_role.clone())
                            .map(|(added_outcast_position, added_outcast_role)| {
                                ChancellorTrace {
                                    original_positions: pending.original_positions.clone(),
                                    added_outcast_position,
                                    added_outcast_role,
                                    affected_anchor_positions: pending
                                        .affected_anchor_positions
                                        .clone(),
                                }
                            });
                        let chancellor_conversion = pending.added_outcast_position;
                        let scenario = Scenario {
                            evil_positions: full_evil.clone(),
                            puppet_position: puppet_pos,
                            corrupted: outcome.corrupted,
                            pd_corrupted: represented_pd_target,
                            doppelganger_position: dopp_pos_opt,
                            drunk_position: drunk_pos_opt,
                            alchemist_cures: outcome.alchemist_counts,
                            messed_up_by_evil: outcome.messed_up_by_evil,
                            shaman_trace: outcome.shaman_trace,
                            chancellor_trace,
                            chancellor_conversion,
                            twin_trace,
                        };
                        let index = placement_scenarios.len();
                        placement_scenarios.push(scenario);
                        seen.insert(key, index);
                    }
                }
            }
        }
        scenarios.extend(placement_scenarios);
    }

    Some(scenarios)
}

fn is_ordered_twin_safe_role(role: &str) -> bool {
    let Some(card) = get_card(role) else {
        return false;
    };
    let role = normalize_role(card.name);
    match card.faction {
        Faction::Villager => role != "alchemist",
        Faction::Outcast => matches!(role.as_str(), "rambler" | "wretch" | "bombardier"),
        Faction::Minion => matches!(
            role.as_str(),
            "twinminion" | "witch" | "minion" | "poisoner"
        ),
        Faction::Demon => matches!(role.as_str(), "baa" | "pooka" | "lilis"),
    }
}

/// This checkpoint integrates only the identity-stable slice around Twin.
/// Later current-data consumers remain a hard gate until their contexts are
/// derived from the replayed post-Twin map.
fn supports_ordered_twin_start_slice(state: &GameState) -> bool {
    if state.n_cards == 0 {
        return false;
    }
    let roles = state.deck.all_roles();
    roles
        .iter()
        .any(|role| normalize_role(role) == "twinminion")
        && roles.iter().all(|role| is_ordered_twin_safe_role(role))
}

/// Exact structural current-role facts at Twin's ordered Start slot.
///
/// Final apparent card roles are intentionally excluded: on Twin boards they
/// can be post-swap bluffs or presentations and cannot identify the former
/// neighbor data. This first slice therefore admits only exact, non-Unknown
/// `evil_positions`; if a selected Demon's required neighbor is absent from
/// that structural map, the ordered state falls back atomically.
fn exact_pre_twin_structural_roles(
    state: &GameState,
    structural_roles: &HashMap<u8, String>,
) -> Option<HashMap<u8, String>> {
    let mut roles: HashMap<u8, String> = HashMap::new();
    let mut insert = |position: u8, role: &str| -> bool {
        if position == 0
            || position > state.n_cards
            || role.trim().is_empty()
            || !is_ordered_twin_safe_role(role)
        {
            return false;
        }
        match roles.get(&position) {
            Some(existing) => normalize_role(existing) == normalize_role(role),
            None => {
                roles.insert(position, role.to_string());
                true
            }
        }
    };

    for (&position, role) in structural_roles {
        if !insert(position, role) {
            return None;
        }
    }

    Some(roles)
}

/// Enumerate the exact ordered Start slice at one pending context boundary.
/// Status producers before Twin run once, each full Twin event replays current
/// role data, and the post-Twin status phase then consumes that branch.
fn enumerate_ordered_twin_start_outcomes(
    state: &GameState,
    pre_twin_current_roles: &HashMap<u8, String>,
    context: &StartCorruptionContext,
) -> Option<(usize, Vec<(StartCorruptionOutcome, TwinTrace)>)> {
    if state.pd_corruption_target.is_some()
        || context.drunk_position.is_some()
        || context.puppet_position.is_some()
        || context.plague_doctor_acts
        || context.shaman_trace.is_some()
        || !context.true_alchemist_positions.is_empty()
    {
        return None;
    }

    let current_roles = exact_pre_twin_structural_roles(state, pre_twin_current_roles)?;
    if !current_roles
        .values()
        .any(|role| normalize_role(role) == "twinminion")
    {
        return None;
    }

    // Initial ManageCharacters construction and the ordinary Start scanner use
    // descending displayed-ID order. Every physical card is alive at Start.
    let current_order: Vec<u8> = (1..=state.n_cards).rev().collect();
    let demon_positions: Vec<u8> = current_order
        .iter()
        .copied()
        .filter(|position| {
            current_roles
                .get(position)
                .and_then(|role| role_faction_in_state(role, state))
                == Some(Faction::Demon)
        })
        .collect();

    // The pure Twin enumerator skips malformed paths. Require every selected
    // Demon's two occurrence-sensitive neighbors before calling it so a
    // partial structural map can never masquerade as an exact trace set.
    for demon_position in &demon_positions {
        let anchor_index = current_order
            .iter()
            .position(|position| position == demon_position)
            .expect("Demon position came from current_order");
        let previous = current_order
            [(anchor_index + current_order.len() - 1) % current_order.len()];
        let next = current_order[(anchor_index + 1) % current_order.len()];
        if !current_roles.contains_key(&previous) || !current_roles.contains_key(&next) {
            return None;
        }
    }

    let traces = enumerate_twin_traces(&current_roles, &current_order, &current_order);
    let expected_width = if demon_positions.is_empty() {
        1
    } else {
        demon_positions.len() * 2
    };
    if traces.is_empty() || traces.len() != expected_width {
        return None;
    }

    let pre_twin_context = context.pre_twin_context();
    let pre_twin_outcomes =
        enumerate_pre_twin_corruption(state.n_cards, pre_twin_current_roles, &pre_twin_context);
    let mut outcomes = Vec::new();

    for pre_twin_outcome in pre_twin_outcomes {
        for trace in &traces {
            let post_twin_current_roles: Option<HashMap<u8, String>> = current_roles
                .keys()
                .copied()
                .map(|position| {
                    role_after_twin(position, &current_roles, trace)
                        .map(|role| (position, role))
                })
                .collect();
            let post_twin_current_roles = post_twin_current_roles?;
            if post_twin_current_roles
                .values()
                .any(|role| !is_ordered_twin_safe_role(role))
            {
                return None;
            }
            let post_twin_context = context.post_twin_context();

            for outcome in enumerate_post_twin_corruption(
                state.n_cards,
                &pre_twin_outcome,
                &post_twin_context,
                None,
            ) {
                outcomes.push((outcome, trace.clone()));
            }
        }
    }

    Some((traces.len(), outcomes))
}

/// Enumerate native Chancellor histories. `c` and the selected Outcast anchor
/// remain internal history variables; final scenarios later aggregate every
/// `c` that yields the same represented runtime outcome.
#[allow(clippy::too_many_arguments)]
fn enumerate_raw_chancellor_traces(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    puppet_position: Option<u8>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    final_chancellor_position: u8,
) -> Vec<RawChancellorTrace> {
    let mut role_candidates = Vec::new();
    let mut seen_roles = HashSet::new();
    for role in &state.deck.outcasts {
        let normalized = normalize_role(role);
        if seen_roles.insert(normalized) {
            role_candidates.push(role.clone());
        }
    }
    let mut traces = Vec::new();
    for added_outcast_role in role_candidates {
        for added_outcast_position in 1..=state.n_cards {
            if added_outcast_position == final_chancellor_position
                || full_evil.contains_key(&added_outcast_position)
                || puppet_position == Some(added_outcast_position)
                || !added_outcast_matches_final_position(
                    state,
                    added_outcast_position,
                    &added_outcast_role,
                    doppelganger_position,
                    drunk_position,
                )
            {
                continue;
            }

            // These compatibility checks depend on (a, r), not on the hidden
            // original Chancellor seat. Keeping them outside the `c` loop is
            // both probability-safe and important on large role pools.
            if final_board_has_another_true_role(
                state,
                full_evil,
                puppet_position,
                doppelganger_position,
                drunk_position,
                added_outcast_position,
                &added_outcast_role,
            ) {
                continue;
            }

            for anchor_position in
                adjacent_positions(final_chancellor_position, state.n_cards)
            {
                if anchor_position == final_chancellor_position
                    || !can_be_final_outcast_anchor(
                        state,
                        full_evil,
                        puppet_position,
                        doppelganger_position,
                        drunk_position,
                        added_outcast_position,
                        anchor_position,
                    )
                    || !natural_outcast_hypothesis_allows(
                        state,
                        full_evil,
                        puppet_position,
                        doppelganger_position,
                        drunk_position,
                        Some(&RawChancellorTrace {
                            original_position: 0,
                            added_outcast_position,
                            added_outcast_role: added_outcast_role.clone(),
                            anchor_position,
                        }),
                        None,
                    )
                {
                    continue;
                }

                for original_position in 1..=state.n_cards {
                    // Before the swap, c is Chancellor and f is the chosen
                    // neighbour. Neither can be the selected Outcast anchor.
                    if anchor_position == original_position {
                        continue;
                    }
                    traces.push(RawChancellorTrace {
                        original_position,
                        added_outcast_position,
                        added_outcast_role: added_outcast_role.clone(),
                        anchor_position,
                    });
                }
            }
        }
    }

    traces.sort_by(|left, right| {
        (
            left.added_outcast_position,
            normalize_role(&left.added_outcast_role),
            left.original_position,
            left.anchor_position,
        )
            .cmp(&(
                right.added_outcast_position,
                normalize_role(&right.added_outcast_role),
                right.original_position,
                right.anchor_position,
            ))
    });
    traces.dedup();
    traces
}

fn added_outcast_matches_final_position(
    state: &GameState,
    position: u8,
    added_role: &str,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
) -> bool {
    let normalized = normalize_role(added_role);
    if normalized == "doppelganger" {
        return doppelganger_position == Some(position) && drunk_position != Some(position);
    }
    if normalized == "drunk" {
        return drunk_position == Some(position) && doppelganger_position != Some(position);
    }
    if doppelganger_position == Some(position) || drunk_position == Some(position) {
        return false;
    }

    match state.card_at(position) {
        Some(card) => {
            is_state_outcast_role(&card.apparent_role, state)
                && normalize_role(&card.apparent_role) == normalized
        }
        None => true,
    }
}

#[allow(clippy::too_many_arguments)]
fn final_board_has_another_true_role(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    puppet_position: Option<u8>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    added_position: u8,
    added_role: &str,
) -> bool {
    let wanted = normalize_role(added_role);
    (1..=state.n_cards).any(|position| {
        if position == added_position
            || full_evil.contains_key(&position)
            || puppet_position == Some(position)
        {
            return false;
        }
        let role = if doppelganger_position == Some(position) {
            Some("Doppelganger")
        } else if drunk_position == Some(position) {
            Some("Drunk")
        } else {
            state.card_at(position).map(|card| card.apparent_role.as_str())
        };
        role.is_some_and(|role| {
            is_state_outcast_role(role, state) && normalize_role(role) == wanted
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn can_be_final_outcast_anchor(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    puppet_position: Option<u8>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    added_position: u8,
    position: u8,
) -> bool {
    if full_evil.contains_key(&position) || puppet_position == Some(position) {
        return false;
    }
    if position == added_position
        || doppelganger_position == Some(position)
        || drunk_position == Some(position)
    {
        return true;
    }
    match state.card_at(position) {
        Some(card) => is_state_outcast_role(&card.apparent_role, state),
        // An unrevealed good card can be a natural Outcast anchor when the HUD
        // and pool budgets below leave such a slot available.
        None => true,
    }
}

#[allow(clippy::too_many_arguments)]
fn take_outcast_role(pool: &mut HashMap<String, usize>, role: &str) -> bool {
    let Some(count) = pool.get_mut(role) else {
        return false;
    };
    if *count == 0 {
        return false;
    }
    *count -= 1;
    true
}

fn add_fixed_outcast(
    fixed: &mut HashMap<u8, String>,
    position: u8,
    role: String,
) -> bool {
    match fixed.get(&position) {
        Some(existing) => existing == &role,
        None => {
            fixed.insert(position, role);
            true
        }
    }
}

fn natural_outcast_hypothesis_allows(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    puppet_position: Option<u8>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    trace: Option<&RawChancellorTrace>,
    plague_doctor_acts: Option<bool>,
) -> bool {
    natural_outcast_hypothesis_allows_with_required_villagers(
        state,
        full_evil,
        puppet_position,
        doppelganger_position,
        drunk_position,
        trace,
        plague_doctor_acts,
        &HashSet::new(),
        &HashMap::new(),
        &HashSet::new(),
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn natural_outcast_hypothesis_allows_with_required_villagers(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    puppet_position: Option<u8>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    trace: Option<&RawChancellorTrace>,
    plague_doctor_acts: Option<bool>,
    required_villagers: &HashSet<u8>,
    required_outcast_roles: &HashMap<u8, String>,
    forbidden_outcast_role_positions: &HashSet<u8>,
    forbidden_outcast_role: Option<&str>,
    exact_ordinary_outcasts: Option<&HashSet<u8>>,
) -> bool {
    let doppelganger_role = normalize_role("Doppelganger");
    let drunk_role = normalize_role("Drunk");
    let pd_role = normalize_role("Plague Doctor");
    let forbidden_outcast_role = forbidden_outcast_role.map(normalize_role);

    // Pf is authoritative and already includes Chancellor's generated role.
    // Subtract exactly that occurrence to recover the natural pre-Chancellor
    // multiset P0; no Baa/availability hypothesis may add another role.
    let mut pool: HashMap<String, usize> = HashMap::new();
    for role in &state.deck.outcasts {
        *pool.entry(normalize_role(role)).or_insert(0) += 1;
    }
    let generated_position = trace.map(|trace| trace.added_outcast_position);
    let generated_role = trace.map(|trace| normalize_role(&trace.added_outcast_role));
    if let Some(role) = generated_role.as_deref() {
        if !take_outcast_role(&mut pool, role) {
            return false;
        }
    }

    let excluded = |position: u8| {
        full_evil.contains_key(&position)
            || puppet_position == Some(position)
            || generated_position == Some(position)
    };
    if required_villagers
        .iter()
        .any(|position| *position == 0 || *position > state.n_cards || excluded(*position))
    {
        return false;
    }
    if required_outcast_roles.iter().any(|(position, _)| {
        *position == 0
            || *position > state.n_cards
            || excluded(*position)
            || required_villagers.contains(position)
            || doppelganger_position == Some(*position)
            || drunk_position == Some(*position)
    }) {
        return false;
    }
    if required_outcast_roles.iter().any(|(position, role)| {
        forbidden_outcast_role.as_deref() == Some(normalize_role(role).as_str())
            && forbidden_outcast_role_positions.contains(position)
    }) {
        return false;
    }
    if exact_ordinary_outcasts.is_some_and(|positions| {
        required_outcast_roles
            .keys()
            .any(|position| !positions.contains(position))
            || positions.iter().any(|position| {
                *position == 0
                    || *position > state.n_cards
                    || excluded(*position)
                    || required_villagers.contains(position)
                    || doppelganger_position == Some(*position)
                    || drunk_position == Some(*position)
            })
    }) {
        return false;
    }
    let can_host_special = |position: u8, special: &str| {
        if excluded(position) || required_villagers.contains(&position) {
            return false;
        }
        if let Some(role) = state.executed_good_roles.get(&position) {
            return normalize_role(role) == special;
        }
        match state.card_at(position) {
            None => true,
            Some(card) if is_state_outcast_role(&card.apparent_role, state) => {
                normalize_role(&card.apparent_role) == special
            }
            Some(card) => is_state_villager_role(&card.apparent_role, state),
        }
    };

    if doppelganger_position.is_some() && doppelganger_position == drunk_position {
        return false;
    }
    let mut fixed: HashMap<u8, String> = HashMap::new();
    for (special, position) in [
        (doppelganger_role.as_str(), doppelganger_position),
        (drunk_role.as_str(), drunk_position),
    ] {
        let generated_special = generated_role.as_deref() == Some(special);
        match (generated_special, position) {
            (true, Some(position)) if Some(position) == generated_position => {}
            (true, _) => return false,
            (false, Some(position)) if Some(position) == generated_position => return false,
            (false, Some(position)) => {
                if !can_host_special(position, special)
                    || !add_fixed_outcast(&mut fixed, position, special.to_string())
                {
                    return false;
                }
            }
            (false, None) => {}
        }
    }

    for (&position, role) in required_outcast_roles {
        if !add_fixed_outcast(&mut fixed, position, normalize_role(role)) {
            return false;
        }
    }

    // Exact killed-good roles supersede apparent identities. Otherwise every
    // revealed serialized Outcast is a fixed natural role, except at replaced
    // evil/Puppet/generated positions.
    for position in 1..=state.n_cards {
        if excluded(position) {
            continue;
        }
        let observed = if let Some(role) = state.executed_good_roles.get(&position) {
            is_state_outcast_role(role, state).then(|| normalize_role(role))
        } else {
            state.card_at(position).and_then(|card| {
                is_state_outcast_role(&card.apparent_role, state)
                    .then(|| normalize_role(&card.apparent_role))
            })
        };
        if let Some(role) = observed {
            if required_villagers.contains(&position) {
                return false;
            }
            if forbidden_outcast_role.as_deref() == Some(role.as_str())
                && forbidden_outcast_role_positions.contains(&position)
            {
                return false;
            }
            if !add_fixed_outcast(&mut fixed, position, role) {
                return false;
            }
        }
    }

    // Unknown natural identities need actual hidden, unconstrained good seats.
    let anonymous_hosts: HashSet<u8> = (1..=state.n_cards)
        .filter(|position| !excluded(*position))
        .filter(|position| !required_villagers.contains(position))
        .filter(|position| !fixed.contains_key(position))
        .filter(|position| state.card_at(*position).is_none())
        .filter(|position| !state.executed_good_roles.contains_key(position))
        .collect();
    let mut required_anonymous_positions = HashSet::new();
    if let Some(trace) = trace {
        if trace.anchor_position != trace.added_outcast_position
            && !fixed.contains_key(&trace.anchor_position)
        {
            if !anonymous_hosts.contains(&trace.anchor_position) {
                return false;
            }
            required_anonymous_positions.insert(trace.anchor_position);
        }
    }

    let fixed_ordinary_positions: HashSet<u8> = fixed
        .iter()
        .filter(|(_, role)| !matches!(role.as_str(), "doppelganger" | "drunk"))
        .map(|(&position, _)| position)
        .collect();
    let exact_anonymous_count = if let Some(exact) = exact_ordinary_outcasts {
        if !fixed_ordinary_positions.is_subset(exact)
            || !required_anonymous_positions.is_subset(exact)
            || exact.iter().any(|position| {
                !fixed_ordinary_positions.contains(position) && !anonymous_hosts.contains(position)
            })
        {
            return false;
        }
        Some(exact.len() - fixed_ordinary_positions.len())
    } else {
        None
    };

    for role in fixed.values() {
        if !take_outcast_role(&mut pool, role) {
            return false;
        }
    }
    let fixed_has_pd = fixed.values().any(|role| role == &pd_role);

    let mut role_only = Vec::new();
    match plague_doctor_acts {
        None => {}
        Some(false) => {
            if state.pd_corruption_target.is_some()
                || generated_role.as_ref() == Some(&pd_role)
                || fixed_has_pd
            {
                return false;
            }
        }
        Some(true) => {
            if generated_role.as_ref() != Some(&pd_role) && !fixed_has_pd {
                role_only.push(pd_role.clone());
            }
        }
    }
    for role in &role_only {
        if !take_outcast_role(&mut pool, role) {
            return false;
        }
    }

    // Dopp/Drunk can only be the explicitly represented natural/generated
    // identity above. Once PD behavior is concrete, PD has likewise already
    // been accounted for and cannot silently occupy another filler slot.
    let mut forbidden_filler = HashSet::from([
        doppelganger_role,
        drunk_role,
    ]);
    if plague_doctor_acts.is_some() {
        forbidden_filler.insert(pd_role);
    }

    // Doppelganger and Drunk consume real identities from P0, but native
    // registers both disguisers in the Villager HUD count rather than the
    // Outcast count. They therefore constrain the pool and physical host
    // without consuming one of `board_outcast_count`'s natural O slots.
    let fixed_count = fixed_ordinary_positions.len();
    let anchor_count = required_anonymous_positions.len();
    let role_only_count = role_only.len();
    let minimum_anonymous = anchor_count.max(role_only_count);
    let anonymous_needed = if let (Some(exact), Some(exact_anonymous_count)) =
        (exact_ordinary_outcasts, exact_anonymous_count)
    {
        match (state.board_outcast_count, state.board_count_provenance) {
            (Some(count), BoardCountProvenance::TrustedPreStart)
                if exact.len() != count as usize =>
            {
                return false;
            }
            (Some(count), BoardCountProvenance::LegacyUnknown) if exact.len() > count as usize => {
                return false;
            }
            _ => {}
        }
        if exact_anonymous_count < minimum_anonymous {
            return false;
        }
        exact_anonymous_count
    } else {
        match (state.board_outcast_count, state.board_count_provenance) {
            (Some(count), BoardCountProvenance::TrustedPreStart) => {
                let count = count as usize;
                if count < fixed_count {
                    return false;
                }
                let remaining = count - fixed_count;
                if remaining < minimum_anonymous {
                    return false;
                }
                remaining
            }
            (Some(count), BoardCountProvenance::LegacyUnknown) => {
                // Old sessions did not retain whether `no` came from the HUD,
                // Baa-obscured deck view, or a post-hoc transcription. Preserve it
                // as a safe ceiling, but do not invent anonymous identities merely
                // to make an unproven historical value exact.
                if fixed_count + minimum_anonymous > count as usize {
                    return false;
                }
                minimum_anonymous
            }
            (None, _) => minimum_anonymous,
        }
    };
    if anonymous_needed > anonymous_hosts.len() {
        return false;
    }
    let filler_needed = anonymous_needed - role_only_count;
    let filler_capacity: usize = pool
        .iter()
        .filter(|(role, _)| !forbidden_filler.contains(*role))
        .map(|(_, count)| *count)
        .sum();
    if filler_capacity < filler_needed {
        return false;
    }

    if forbidden_outcast_role_positions.is_empty() {
        return true;
    }

    // Public negative evidence can exclude one particular ordinary Outcast
    // identity from physical seats that remain grouped in the scenario. The
    // exact O header may still force that role into a filler slot, so preserve
    // the seat restriction instead of treating the remaining pool as an
    // unpositioned bag. Rambler callbacks and Scout/Hunter's registered-Wretch
    // distances share this allocator.
    let Some(forbidden_outcast_role) = forbidden_outcast_role.as_deref() else {
        return false;
    };
    let available_forbidden_role_fillers = pool
        .get(forbidden_outcast_role)
        .copied()
        .unwrap_or(0);
    let other_filler_capacity = filler_capacity - available_forbidden_role_fillers;
    let minimum_forbidden_role_fillers = filler_needed.saturating_sub(other_filler_capacity);
    if minimum_forbidden_role_fillers == 0 {
        return true;
    }

    let maximum_allowed_selected = if let Some(exact) = exact_ordinary_outcasts {
        exact
            .iter()
            .filter(|position| !fixed_ordinary_positions.contains(position))
            .filter(|position| !forbidden_outcast_role_positions.contains(position))
            .count()
    } else {
        let required_allowed = required_anonymous_positions
            .iter()
            .filter(|position| !forbidden_outcast_role_positions.contains(position))
            .count();
        let optional_slots = anonymous_needed - required_anonymous_positions.len();
        let optional_allowed = anonymous_hosts
            .iter()
            .filter(|position| !required_anonymous_positions.contains(position))
            .filter(|position| !forbidden_outcast_role_positions.contains(position))
            .count();
        required_allowed + optional_slots.min(optional_allowed)
    };
    let selected_forbidden = anonymous_needed - maximum_allowed_selected;
    // A role-only PD identity can occupy a forbidden seat first. Any role-only
    // identities left after that consume allowed seats before filler roles are
    // assigned.
    let allowed_consumed_by_role_only = role_only_count.saturating_sub(selected_forbidden);
    let maximum_allowed_filler_seats =
        maximum_allowed_selected.saturating_sub(allowed_consumed_by_role_only);
    minimum_forbidden_role_fillers <= maximum_allowed_filler_seats
}

/// Whether one of the ordinary-good identity assignments grouped into a
/// scenario can put a particular natural Outcast role at every listed
/// still-anonymous physical seat simultaneously.
///
/// Scenario generation intentionally collapses Villager/ordinary-Outcast
/// placements once they have identical represented Start consequences. Public
/// role evidence can nevertheless require or exclude a particular
/// hidden identity at specific seats (for example Rambler callbacks or
/// Scout/Hunter registered-Wretch distances). Re-run the same joint
/// multiset/header checks here rather than validating each existential against
/// the same single pool occurrence.
pub(crate) fn scenario_allows_anonymous_natural_outcast_role_assignments(
    positions: &HashSet<u8>,
    role: &str,
    forbidden_role_positions: &HashSet<u8>,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if positions.is_empty() && forbidden_role_positions.is_empty() {
        return true;
    }
    if !is_state_outcast_role(role, state) || is_hud_villager_outcast(role) {
        return positions.is_empty();
    }
    if forbidden_role_positions
        .iter()
        .any(|position| *position == 0 || *position > state.n_cards)
    {
        return false;
    }
    if positions.iter().any(|position| {
        *position == 0
            || *position > state.n_cards
            || state.card_at(*position).is_some()
            || state.executed_good_roles.contains_key(position)
            || state.confirmed_evil.contains(position)
            || scenario.is_evil(*position)
            || scenario.doppelganger_position == Some(*position)
            || scenario.drunk_position == Some(*position)
            || scenario.chancellor_added_outcast_position() == Some(*position)
            || scenario.shaman_trace.as_ref().is_some_and(|trace| {
                trace.source_position == *position || trace.target_position == *position
            })
    }) {
        return false;
    }

    let mut required_outcast_roles: HashMap<u8, String> = positions
        .iter()
        .map(|position| (*position, role.to_string()))
        .collect();
    if let Some(shaman) = scenario.shaman_trace.as_ref() {
        // A copied ordinary-Outcast role consumes a natural pool occurrence
        // only when the source itself is an ordinary Good identity. Evil,
        // Drunk, and Doppelganger sources copy their bluff; a Chancellor-added
        // source consumes the already-subtracted generated occurrence.
        let source = shaman.source_position;
        if is_state_outcast_role(&shaman.copied_role, state)
            && !is_hud_villager_outcast(&shaman.copied_role)
            && !scenario.is_evil(source)
            && scenario.doppelganger_position != Some(source)
            && scenario.drunk_position != Some(source)
            && scenario.chancellor_added_outcast_position() != Some(source)
        {
            required_outcast_roles.insert(source, shaman.copied_role.clone());
        }
    }
    let required_villagers = HashSet::new();
    // A known Start target proves that the Plague Doctor actor existed and
    // consumed its natural/generated Outcast identity. Scenarios intentionally
    // collapse unknown cured target histories, so absence remains conservative.
    let plague_doctor_acts = (state.pd_corruption_target.is_some()
        || scenario.pd_corrupted.is_some())
    .then_some(true);
    let try_trace = |trace: Option<&RawChancellorTrace>| {
        natural_outcast_hypothesis_allows_with_required_villagers(
            state,
            &scenario.evil_positions,
            scenario.puppet_position,
            scenario.doppelganger_position,
            scenario.drunk_position,
            trace,
            plague_doctor_acts,
            &required_villagers,
            &required_outcast_roles,
            forbidden_role_positions,
            Some(role),
            None,
        )
    };

    let Some(trace) = scenario.chancellor_trace.as_ref() else {
        return try_trace(None);
    };
    let mut final_chancellors = scenario
        .evil_positions
        .iter()
        .filter(|(_, role)| normalize_role(role) == "chancellor")
        .map(|(&position, _)| position);
    let final_chancellor_position = final_chancellors.next().filter(|_| {
        final_chancellors.next().is_none()
    });
    let original_positions = if trace.original_positions.is_empty() {
        vec![0]
    } else {
        trace.original_positions.clone()
    };
    // Old serialized traces predate anchor provenance. Using the generated
    // position as a non-consuming stand-in preserves their conservative
    // behavior; current traces always retain the selected anchor candidates.
    let anchor_positions = if trace.affected_anchor_positions.is_empty() {
        vec![trace.added_outcast_position]
    } else {
        trace.affected_anchor_positions.clone()
    };

    original_positions.iter().any(|&original_position| {
        anchor_positions.iter().any(|&anchor_position| {
            let raw = RawChancellorTrace {
                original_position,
                added_outcast_position: trace.added_outcast_position,
                added_outcast_role: trace.added_outcast_role.clone(),
                anchor_position,
            };
            // The first target was a real Villager before Chancellor replaced
            // it, so it cannot simultaneously host one of these required
            // natural Outcast identities. Grouped traces remain valid when any other
            // retained native history satisfies all requested assignments.
            if final_chancellor_position.is_some_and(|final_position| {
                positions.contains(&raw.original_villager_position(final_position))
            }) {
                return false;
            }
            try_trace(Some(&raw))
        })
    })
}

pub(crate) fn scenario_allows_anonymous_natural_outcast_role_at(
    position: u8,
    role: &str,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    scenario_allows_anonymous_natural_outcast_role_assignments(
        &HashSet::from([position]),
        role,
        &HashSet::new(),
        scenario,
        state,
    )
}

/// Whether one complete assignment of the still-grouped ordinary Good seats
/// can expose the requested live Bishop type surfaces simultaneously.
///
/// `villagers`, `outcasts`, and `wretches` are disjoint and complete for the
/// anonymous seats supplied by the caller. Wretch consumes a natural Outcast
/// identity but projects through its live Minion register-as; `outcasts`
/// therefore means a natural non-Wretch Outcast. Reuse the scenario builder's
/// exact multiset/header allocator so several current observations cannot each
/// spend the same hidden identity independently.
pub(crate) fn scenario_allows_anonymous_good_type_assignment(
    villagers: &HashSet<u8>,
    outcasts: &HashSet<u8>,
    wretches: &HashSet<u8>,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if !villagers.is_disjoint(outcasts)
        || !villagers.is_disjoint(wretches)
        || !outcasts.is_disjoint(wretches)
    {
        return false;
    }

    let requested: HashSet<u8> = villagers
        .iter()
        .chain(outcasts.iter())
        .chain(wretches.iter())
        .copied()
        .collect();
    if requested.iter().any(|position| {
        *position == 0
            || *position > state.n_cards
            || state.card_at(*position).is_some()
            || state.executed_good_roles.contains_key(position)
            || state.confirmed_evil.contains(position)
            || scenario.is_evil(*position)
            || scenario.doppelganger_position == Some(*position)
            || scenario.drunk_position == Some(*position)
            || scenario.chancellor_added_outcast_position() == Some(*position)
            || scenario.shaman_trace.as_ref().is_some_and(|trace| {
                trace.source_position == *position || trace.target_position == *position
            })
    }) {
        return false;
    }

    let mut required_outcast_roles: HashMap<u8, String> = wretches
        .iter()
        .map(|position| (*position, "Wretch".to_string()))
        .collect();
    if let Some(shaman) = scenario.shaman_trace.as_ref() {
        let source = shaman.source_position;
        if is_state_outcast_role(&shaman.copied_role, state)
            && !is_hud_villager_outcast(&shaman.copied_role)
            && !scenario.is_evil(source)
            && scenario.doppelganger_position != Some(source)
            && scenario.drunk_position != Some(source)
            && scenario.chancellor_added_outcast_position() != Some(source)
        {
            required_outcast_roles.insert(source, shaman.copied_role.clone());
        }
    }

    let plague_doctor_acts = (state.pd_corruption_target.is_some()
        || scenario.pd_corrupted.is_some())
    .then_some(true);
    let try_trace = |trace: Option<&RawChancellorTrace>| {
        let generated_position = trace.map(|value| value.added_outcast_position);
        let excluded = |position: u8| {
            scenario.evil_positions.contains_key(&position)
                || scenario.puppet_position == Some(position)
                || generated_position == Some(position)
        };

        let mut exact_ordinary_outcasts: HashSet<u8> = outcasts
            .iter()
            .chain(wretches.iter())
            .copied()
            .collect();
        for position in 1..=state.n_cards {
            if excluded(position)
                || scenario.doppelganger_position == Some(position)
                || scenario.drunk_position == Some(position)
            {
                continue;
            }
            let observed = state
                .executed_good_roles
                .get(&position)
                .map(String::as_str)
                .or_else(|| state.card_at(position).map(|card| card.apparent_role.as_str()));
            if observed.is_some_and(|role| {
                is_state_outcast_role(role, state) && !is_hud_villager_outcast(role)
            }) {
                exact_ordinary_outcasts.insert(position);
            }
        }
        if let Some(trace) = trace {
            if trace.anchor_position != trace.added_outcast_position
                && !excluded(trace.anchor_position)
                && scenario.doppelganger_position != Some(trace.anchor_position)
                && scenario.drunk_position != Some(trace.anchor_position)
            {
                exact_ordinary_outcasts.insert(trace.anchor_position);
            }
        }
        if let Some(TwinTrace {
            outcome:
                crate::types::TwinStartOutcome::Swap {
                    neighbor_position,
                    neighbor_pre_swap_role,
                    ..
                },
            ..
        }) = scenario.twin_trace.as_ref()
        {
            if is_state_outcast_role(neighbor_pre_swap_role, state)
                && !is_hud_villager_outcast(neighbor_pre_swap_role)
                && !excluded(*neighbor_position)
                && scenario.doppelganger_position != Some(*neighbor_position)
                && scenario.drunk_position != Some(*neighbor_position)
            {
                exact_ordinary_outcasts.insert(*neighbor_position);
            }
        }
        exact_ordinary_outcasts.extend(required_outcast_roles.keys().copied());

        natural_outcast_hypothesis_allows_with_required_villagers(
            state,
            &scenario.evil_positions,
            scenario.puppet_position,
            scenario.doppelganger_position,
            scenario.drunk_position,
            trace,
            plague_doctor_acts,
            villagers,
            &required_outcast_roles,
            outcasts,
            Some("Wretch"),
            Some(&exact_ordinary_outcasts),
        )
    };

    let Some(trace) = scenario.chancellor_trace.as_ref() else {
        return try_trace(None);
    };
    let mut final_chancellors = scenario
        .evil_positions
        .iter()
        .filter(|(_, role)| normalize_role(role) == "chancellor")
        .map(|(&position, _)| position);
    let final_chancellor_position = final_chancellors.next().filter(|_| {
        final_chancellors.next().is_none()
    });
    let original_positions = if trace.original_positions.is_empty() {
        vec![0]
    } else {
        trace.original_positions.clone()
    };
    let anchor_positions = if trace.affected_anchor_positions.is_empty() {
        vec![trace.added_outcast_position]
    } else {
        trace.affected_anchor_positions.clone()
    };

    original_positions.iter().any(|&original_position| {
        anchor_positions.iter().any(|&anchor_position| {
            let raw = RawChancellorTrace {
                original_position,
                added_outcast_position: trace.added_outcast_position,
                added_outcast_role: trace.added_outcast_role.clone(),
                anchor_position,
            };
            if final_chancellor_position.is_some_and(|final_position| {
                outcasts
                    .iter()
                    .chain(wretches.iter())
                    .any(|position| *position == raw.original_villager_position(final_position))
            }) {
                return false;
            }
            try_trace(Some(&raw))
        })
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitialAlchemistConstraint {
    Never,
    Maybe,
    Required,
}

fn puppet_overlays_stable_twin(
    full_evil: &HashMap<u8, String>,
    puppet_position: Option<u8>,
) -> bool {
    puppet_position.is_some_and(|position| {
        full_evil
            .get(&position)
            .is_some_and(|role| normalize_role(role) == "twinminion")
    })
}

fn exact_twin_puppet_source_positions(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    puppet_position: Option<u8>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    chancellor_trace: Option<&RawChancellorTrace>,
) -> Vec<u8> {
    if !puppet_overlays_stable_twin(full_evil, puppet_position) {
        return Vec::new();
    }
    let generated_outcast = chancellor_trace.map(|trace| trace.added_outcast_position);
    let eligible = |position: u8| {
        position > 0
            && position <= state.n_cards
            && Some(position) != puppet_position
            && !full_evil.contains_key(&position)
            && Some(position) != doppelganger_position
            && Some(position) != drunk_position
            && Some(position) != generated_outcast
    };

    let mut positions: Vec<u8> = state
        .executed_current_roles
        .iter()
        .chain(state.executed_good_roles.iter())
        .filter_map(|(&position, role)| {
            (normalize_role(role) == "twinminion" && eligible(position))
                .then_some(position)
        })
        .collect();
    positions.extend(state.cards.iter().filter_map(|card| {
        (normalize_role(&card.apparent_role) == "twinminion" && eligible(card.position))
            .then_some(card.position)
    }));
    positions.extend(state.slayer_results.iter().filter_map(|result| {
        (result.killed
            && result.was_evil != Some(true)
            && result
                .revealed_role
                .as_deref()
                .is_some_and(|role| normalize_role(role) == "twinminion")
            && eligible(result.target_pos))
        .then_some(result.target_pos)
    }));
    positions.sort_unstable();
    positions.dedup();
    positions
}

#[allow(clippy::too_many_arguments)]
fn twin_puppet_overlay_start_context_variants(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    chancellor_trace: Option<&RawChancellorTrace>,
    base: StartCorruptionContext,
) -> Vec<StartCorruptionContext> {
    if !puppet_overlays_stable_twin(full_evil, puppet_position) {
        return vec![base];
    }
    if !full_evil.values().any(|role| {
        role_faction_in_state(role, state) == Some(Faction::Demon)
    }) {
        // Twin cannot move Villager data onto its stable body without a Demon
        // selection, so Puppeteer cannot produce this overlap.
        return Vec::new();
    }

    let overlap_position = puppet_position.expect("overlap helper requires Puppet");
    let exact_sources = exact_twin_puppet_source_positions(
        state,
        full_evil,
        puppet_position,
        doppelganger_position,
        drunk_position,
        chancellor_trace,
    );
    let mut sources: Vec<u8> = if exact_sources.is_empty() {
        base.real_villagers_before_puppet
            .iter()
            .copied()
            .filter(|position| *position != overlap_position)
            .collect()
    } else {
        exact_sources
    };
    sources.retain(|position| base.real_villagers_before_puppet.contains(position));
    sources.sort_unstable();
    sources.dedup();

    let alchemist_capacity = state
        .deck
        .villagers
        .iter()
        .filter(|role| normalize_role(role) == "alchemist")
        .count();
    let overlay_alchemist = puppet_displayed_alchemist_constraint(overlap_position, state);
    let mut contexts = Vec::new();
    let mut seen = HashSet::new();
    for source_position in sources {
        let mut context = base.clone();

        // Before Twin, the stable Twin body is a Minion and `source` is
        // the real Villager whose data will move. After Twin and Puppet,
        // the body is Puppet while `source` carries Twin data, so neither
        // participates in Plague Doctor or Shaman's Villager scan.
        context
            .real_villagers_before_puppet
            .remove(&overlap_position);
        context
            .real_villagers_before_puppet
            .insert(source_position);
        context
            .registered_villagers_at_pd_call
            .remove(&overlap_position);
        context
            .registered_villagers_at_pd_call
            .remove(&source_position);

        // Init-time resistance stays on the physical source if it began as
        // Alchemist. The stable Twin body never ran a Villager Init, and
        // the source no longer owns Alchemist data at the later global
        // Alchemist Start slot.
        context
            .corruption_resistant_at_init
            .remove(&overlap_position);
        context
            .messed_up_resistant_at_init
            .remove(&overlap_position);
        context.true_alchemist_positions.retain(|position| {
            *position != overlap_position && *position != source_position
        });

        let source_was_alchemist = context
            .corruption_resistant_at_init
            .contains(&source_position);
        let mut variants = Vec::new();
        match overlay_alchemist {
            InitialAlchemistConstraint::Required
                if source_was_alchemist
                    || context.corruption_resistant_at_init.len() < alchemist_capacity =>
            {
                context
                    .corruption_resistant_at_init
                    .insert(source_position);
                variants.push(context);
            }
            InitialAlchemistConstraint::Required => {}
            InitialAlchemistConstraint::Never if !source_was_alchemist => {
                variants.push(context);
            }
            InitialAlchemistConstraint::Never => {}
            InitialAlchemistConstraint::Maybe if source_was_alchemist => {
                variants.push(context);
            }
            InitialAlchemistConstraint::Maybe => {
                let can_assign_alchemist = context.corruption_resistant_at_init.len()
                    < alchemist_capacity;
                variants.push(context.clone());
                if can_assign_alchemist {
                    context
                        .corruption_resistant_at_init
                        .insert(source_position);
                    variants.push(context);
                }
            }
        }
        for variant in variants {
            let key = start_context_key(&variant);
            if seen.insert(key) {
                contexts.push(variant);
            }
        }
    }
    contexts
}

#[allow(clippy::too_many_arguments)]
fn build_chancellor_start_context_variants(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    final_chancellor_position: u8,
    trace: &RawChancellorTrace,
    plague_doctor_acts: bool,
) -> Vec<StartCorruptionContext> {
    let twin_puppet_overlap = puppet_overlays_stable_twin(full_evil, puppet_position);
    let initial_alchemist_variants = enumerate_initial_alchemist_positions(
        state,
        full_evil,
        doppelganger_position,
        drunk_position,
        puppet_position,
        final_chancellor_position,
        trace,
    );

    let mut contexts = Vec::new();
    let mut seen = HashSet::new();
    for initial_alchemists in initial_alchemist_variants {
        let replaced_villager = trace.original_villager_position(final_chancellor_position);

        let mut alchemists_before_puppet = HashSet::new();
        for &initial_position in &initial_alchemists {
            if initial_position == replaced_villager {
                continue;
            }
            let actor_position = if initial_position == final_chancellor_position
                && trace.original_position != final_chancellor_position
            {
                trace.original_position
            } else {
                initial_position
            };
            alchemists_before_puppet.insert(actor_position);
        }

        let mut true_alchemist_positions: Vec<u8> = alchemists_before_puppet
            .iter()
            .copied()
            .filter(|position| puppet_position != Some(*position))
            .collect();
        true_alchemist_positions.sort_unstable();

        let mut real_villagers_before_puppet = HashSet::new();
        for position in 1..=state.n_cards {
            let is_real_villager = if puppet_position == Some(position)
                && !twin_puppet_overlap
            {
                true
            } else if position == final_chancellor_position
                || position == trace.added_outcast_position
                || full_evil.contains_key(&position)
                || doppelganger_position == Some(position)
                || drunk_position == Some(position)
            {
                false
            } else if alchemists_before_puppet.contains(&position) {
                true
            } else if let Some(card) = state.card_at(position) {
                is_state_villager_role(&card.apparent_role, state)
            } else {
                trace_unrevealed_must_be_villager(
                    position,
                    state,
                    full_evil,
                    doppelganger_position,
                    drunk_position,
                    trace,
                )
            };
            if is_real_villager {
                real_villagers_before_puppet.insert(position);
            }
        }

        let mut registered_villagers_at_pd_call = real_villagers_before_puppet.clone();
        if let Some(puppet_position) = puppet_position {
            registered_villagers_at_pd_call.remove(&puppet_position);
        }

        let context = StartCorruptionContext {
            real_villagers_before_puppet,
            registered_villagers_at_pd_call,
            corruption_resistant_at_init: initial_alchemists,
            messed_up_resistant_at_init: HashSet::new(),
            true_alchemist_positions,
            initial_messed_up_by_evil: HashSet::from([trace.anchor_position]),
            drunk_position,
            puppet_position,
            plague_doctor_acts,
            shaman_trace: None,
        };
        let key = start_context_key(&context);
        if seen.insert(key) {
            contexts.push(context);
        }
    }
    contexts
}

#[allow(clippy::too_many_arguments)]
fn enumerate_initial_alchemist_positions(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    final_chancellor_position: u8,
    trace: &RawChancellorTrace,
) -> Vec<HashSet<u8>> {
    let available_count = state
        .deck
        .villagers
        .iter()
        .filter(|role| normalize_role(role) == "alchemist")
        .count();

    let replaced_villager = trace.original_villager_position(final_chancellor_position);
    let mut required = Vec::new();
    let mut maybe = Vec::new();
    for position in 1..=state.n_cards {
        let constraint = initial_alchemist_constraint(
            position,
            state,
            full_evil,
            doppelganger_position,
            drunk_position,
            puppet_position,
            final_chancellor_position,
            trace,
            replaced_villager,
        );
        match constraint {
            InitialAlchemistConstraint::Never => {}
            InitialAlchemistConstraint::Maybe => maybe.push(position),
            InitialAlchemistConstraint::Required => required.push(position),
        }
    }

    required.sort_unstable();
    required.dedup();
    maybe.sort_unstable();
    maybe.dedup();
    maybe.retain(|position| !required.contains(position));

    if required.len() > available_count {
        return Vec::new();
    }
    let remaining = available_count - required.len();
    let mut variants = Vec::new();
    for count in 0..=remaining.min(maybe.len()) {
        for selected in combinations_of(&maybe, count) {
            let mut positions: HashSet<u8> = required.iter().copied().collect();
            positions.extend(selected);
            variants.push(positions);
        }
    }
    if variants.is_empty() && required.is_empty() {
        variants.push(HashSet::new());
    }
    variants
}

#[allow(clippy::too_many_arguments)]
fn initial_alchemist_constraint(
    position: u8,
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    final_chancellor_position: u8,
    trace: &RawChancellorTrace,
    replaced_villager: u8,
) -> InitialAlchemistConstraint {
    // c was Chancellor during the universal Init pass.
    if position == trace.original_position {
        return InitialAlchemistConstraint::Never;
    }
    // The first Baron target was a real Villager whose precise role is erased.
    if position == replaced_villager {
        return InitialAlchemistConstraint::Maybe;
    }

    // When Chancellor moves, pre-swap f's data moves to c. If c is later made
    // Puppet that source role is another erased real-Villager identity.
    if position == final_chancellor_position
        && trace.original_position != final_chancellor_position
    {
        if puppet_position == Some(trace.original_position)
            && !puppet_overlays_stable_twin(full_evil, puppet_position)
        {
            return puppet_displayed_alchemist_constraint(
                trace.original_position,
                state,
            );
        }
        return final_position_alchemist_constraint(
            trace.original_position,
            state,
            full_evil,
            doppelganger_position,
            drunk_position,
            trace.added_outcast_position,
        );
    }

    if puppet_position == Some(position)
        && !puppet_overlays_stable_twin(full_evil, puppet_position)
    {
        return puppet_displayed_alchemist_constraint(position, state);
    }
    final_position_alchemist_constraint(
        position,
        state,
        full_evil,
        doppelganger_position,
        drunk_position,
        trace.added_outcast_position,
    )
}

fn puppet_displayed_alchemist_constraint(
    position: u8,
    state: &GameState,
) -> InitialAlchemistConstraint {
    match state.card_at(position) {
        Some(card) if normalize_role(&card.apparent_role) == "alchemist" => {
            InitialAlchemistConstraint::Required
        }
        Some(_) => InitialAlchemistConstraint::Never,
        None => InitialAlchemistConstraint::Maybe,
    }
}

fn final_position_alchemist_constraint(
    position: u8,
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    added_outcast_position: u8,
) -> InitialAlchemistConstraint {
    if position == added_outcast_position
        || full_evil.contains_key(&position)
        || doppelganger_position == Some(position)
        || drunk_position == Some(position)
    {
        return InitialAlchemistConstraint::Never;
    }
    match state.card_at(position) {
        Some(card) if normalize_role(&card.apparent_role) == "alchemist" => {
            InitialAlchemistConstraint::Required
        }
        Some(_) => InitialAlchemistConstraint::Never,
        None => InitialAlchemistConstraint::Maybe,
    }
}

#[allow(clippy::too_many_arguments)]
fn trace_unrevealed_must_be_villager(
    position: u8,
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    trace: &RawChancellorTrace,
) -> bool {
    let Some(board_outcasts) = state.board_outcast_count else {
        return false;
    };
    if trace.anchor_position == position
        && trace.anchor_position != trace.added_outcast_position
        && doppelganger_position != Some(position)
        && drunk_position != Some(position)
    {
        return false;
    }
    let mut occupied = HashSet::new();
    for candidate in 1..=state.n_cards {
        if candidate == position {
            continue;
        }
        if is_known_natural_ordinary_outcast(
            candidate,
            state,
            full_evil,
            doppelganger_position,
            drunk_position,
            Some(trace.added_outcast_position),
        ) {
            occupied.insert(candidate);
        }
    }
    if trace.anchor_position != position
        && trace.anchor_position != trace.added_outcast_position
        && !full_evil.contains_key(&trace.anchor_position)
        && doppelganger_position != Some(trace.anchor_position)
        && drunk_position != Some(trace.anchor_position)
    {
        // An unrevealed anchor admitted by the exact multiset helper is a
        // required natural ordinary Outcast even before its role is visible.
        occupied.insert(trace.anchor_position);
    }
    occupied.len() >= board_outcasts as usize
}

/// Resolve hidden ordinary-good seats into complete Villager/Outcast faction
/// assignments before any Start actor selects a Villager. The exact natural
/// Outcast set is only an internal history variable; contexts that later
/// produce the same represented outcome are collapsed by the scenario key.
#[allow(clippy::too_many_arguments)]
fn hidden_faction_start_context_variants(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    chancellor_trace: Option<&RawChancellorTrace>,
    plague_doctor_acts: bool,
    base: StartCorruptionContext,
) -> Vec<StartCorruptionContext> {
    let generated_position = chancellor_trace.map(|trace| trace.added_outcast_position);
    let generated_role = chancellor_trace.map(|trace| normalize_role(&trace.added_outcast_role));
    let excluded = |position: u8| {
        full_evil.contains_key(&position)
            || puppet_position == Some(position)
            || generated_position == Some(position)
            || doppelganger_position == Some(position)
            || drunk_position == Some(position)
    };

    let mut fixed_outcasts: HashSet<u8> = (1..=state.n_cards)
        .filter(|position| {
            is_known_natural_ordinary_outcast(
                *position,
                state,
                full_evil,
                doppelganger_position,
                drunk_position,
                generated_position,
            )
        })
        .collect();
    if let Some(trace) = chancellor_trace {
        if trace.anchor_position != trace.added_outcast_position && !excluded(trace.anchor_position)
        {
            // A hidden anchor is a concrete natural ordinary Outcast even
            // though its precise role has not been revealed.
            fixed_outcasts.insert(trace.anchor_position);
        }
    }

    let mut fixed_villagers: HashSet<u8> = base
        .real_villagers_before_puppet
        .iter()
        .chain(base.registered_villagers_at_pd_call.iter())
        .copied()
        .filter(|position| !excluded(*position))
        .collect();
    for position in 1..=state.n_cards {
        if excluded(position) {
            continue;
        }
        let observed_role = state
            .executed_good_roles
            .get(&position)
            .map(String::as_str)
            .or_else(|| {
                state
                    .card_at(position)
                    .map(|card| card.apparent_role.as_str())
            });
        if observed_role.is_some_and(|role| is_state_villager_role(role, state)) {
            fixed_villagers.insert(position);
        }
    }
    if let Some(trace) = chancellor_trace {
        if trace.original_position != trace.added_outcast_position
            && !excluded(trace.original_position)
        {
            // When Chancellor moves, the selected Villager identity moves
            // back to the original Chancellor seat.
            fixed_villagers.insert(trace.original_position);
        }
    }
    fixed_villagers.extend(exact_twin_puppet_source_positions(
        state,
        full_evil,
        puppet_position,
        doppelganger_position,
        drunk_position,
        chancellor_trace,
    ));

    if !fixed_outcasts.is_disjoint(&fixed_villagers) {
        return Vec::new();
    }

    let ambiguous: Vec<u8> = (1..=state.n_cards)
        .filter(|position| !excluded(*position))
        .filter(|position| !fixed_outcasts.contains(position))
        .filter(|position| !fixed_villagers.contains(position))
        .filter(|position| state.card_at(*position).is_none())
        .filter(|position| !state.executed_good_roles.contains_key(position))
        .collect();

    // Pf includes Chancellor's generated identity. Removing that occurrence
    // yields a cheap upper bound on natural ordinary-Outcast assignments and
    // avoids enumerating irrelevant powerset layers when no HUD count exists.
    let mut natural_outcast_limit = state
        .deck
        .outcasts
        .iter()
        .filter(|role| !is_hud_villager_outcast(role))
        .count();
    if generated_role
        .as_deref()
        .is_some_and(|role| !matches!(role, "doppelganger" | "drunk"))
    {
        natural_outcast_limit = natural_outcast_limit.saturating_sub(1);
    }
    let fixed_count = fixed_outcasts.len();
    let max_total = match (state.board_outcast_count, state.board_count_provenance) {
        (Some(count), BoardCountProvenance::TrustedPreStart) => count as usize,
        (Some(count), BoardCountProvenance::LegacyUnknown) => {
            natural_outcast_limit.min(count as usize)
        }
        (None, _) => natural_outcast_limit,
    };
    if fixed_count > max_total {
        return Vec::new();
    }
    let selected_counts: Vec<usize> =
        match (state.board_outcast_count, state.board_count_provenance) {
            (Some(count), BoardCountProvenance::TrustedPreStart) => {
                let selected = count as usize - fixed_count;
                (selected <= ambiguous.len())
                    .then_some(vec![selected])
                    .unwrap_or_default()
            }
            _ => (0..=(max_total - fixed_count).min(ambiguous.len())).collect(),
        };

    let mut variants = Vec::new();
    let mut seen = HashSet::new();
    for selected_count in selected_counts {
        for selected_outcasts in combinations_of(&ambiguous, selected_count) {
            let mut exact_outcasts = fixed_outcasts.clone();
            exact_outcasts.extend(selected_outcasts.iter().copied());
            let mut required_villagers = fixed_villagers.clone();
            required_villagers.extend(
                ambiguous
                    .iter()
                    .copied()
                    .filter(|position| !exact_outcasts.contains(position)),
            );

            if !natural_outcast_hypothesis_allows_with_required_villagers(
                state,
                full_evil,
                puppet_position,
                doppelganger_position,
                drunk_position,
                chancellor_trace,
                Some(plague_doctor_acts),
                &required_villagers,
                &HashMap::new(),
                &HashSet::new(),
                None,
                Some(&exact_outcasts),
            ) {
                continue;
            }

            let mut context = base.clone();
            context
                .real_villagers_before_puppet
                .extend(required_villagers.iter().copied());
            context
                .registered_villagers_at_pd_call
                .extend(required_villagers.iter().copied());
            let key = start_context_key(&context);
            if seen.insert(key) {
                variants.push(context);
            }
        }
    }
    variants
}

/// Expand the native Shaman Start action into ordered source/target identity
/// histories. The source remains unchanged; the target's former Villager
/// identity is erased and replaced with the source's current identity.
#[allow(clippy::too_many_arguments)]
fn shaman_start_context_variants(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    chancellor_trace: Option<&RawChancellorTrace>,
    plague_doctor_acts: bool,
    base: StartCorruptionContext,
) -> Vec<StartCorruptionContext> {
    let faction_bases = hidden_faction_start_context_variants(
        state,
        full_evil,
        doppelganger_position,
        drunk_position,
        puppet_position,
        chancellor_trace,
        plague_doctor_acts,
        base,
    );
    let faction_bases: Vec<StartCorruptionContext> = faction_bases
        .into_iter()
        .flat_map(|context| {
            twin_puppet_overlay_start_context_variants(
                state,
                full_evil,
                doppelganger_position,
                drunk_position,
                puppet_position,
                chancellor_trace,
                context,
            )
        })
        .collect();
    let shaman_acts = full_evil
        .values()
        .any(|role| normalize_role(role) == "shaman");
    if !shaman_acts {
        return faction_bases;
    }

    let deck_roles = unique_deck_villager_roles(state);
    if deck_roles.is_empty() {
        return Vec::new();
    }

    let mut variants = Vec::new();
    let mut seen = HashSet::new();
    for pair_base in faction_bases {
        let mut eligible: Vec<u8> = pair_base
            .registered_villagers_at_pd_call
            .iter()
            .copied()
            .collect();
        eligible.sort_unstable();
        eligible.dedup();
        if eligible.len() < 2 {
            // Native code has no count guard and cannot complete the second
            // draw for this concrete faction assignment.
            continue;
        }
        for &source_position in &eligible {
            for &target_position in &eligible {
                if source_position == target_position {
                    continue;
                }

                let source_role = observed_final_villager_role(source_position, state);
                let target_role = observed_final_villager_role(target_position, state);
                let copied_roles: Vec<&String> = deck_roles
                    .iter()
                    .filter(|role| {
                        source_role
                            .as_deref()
                            .is_none_or(|known| {
                                normalize_role(known) == "baker"
                                    || normalize_role(known) == normalize_role(role)
                            })
                            && target_role
                                .as_deref()
                                .is_none_or(|known| {
                                    normalize_role(known) == "baker"
                                        || normalize_role(known) == normalize_role(role)
                                })
                    })
                    .collect();

                for copied_role in copied_roles {
                    // InitWithNoReset preserves runtimeData. Alchemist stays
                    // distinct for its resistance. Enlightened differs from
                    // null-runtime roles only when copied Baker later attempts
                    // the BakerRuntimeData cast.
                    let mut previous_role_classes:
                        Vec<(BakerPreservedRuntimeClass, Vec<String>)> = Vec::new();
                    for target_previous_role in &deck_roles {
                        let exact_trace = ShamanTrace {
                            source_position,
                            target_position,
                            copied_role: copied_role.clone(),
                            target_previous_roles: vec![target_previous_role.clone()],
                        };
                        if !shaman_initial_role_counts_fit(&exact_trace, state, &pair_base) {
                            continue;
                        }

                        let runtime_class = shaman_erased_role_class(
                            copied_role,
                            target_previous_role,
                        );
                        if let Some((_, roles)) = previous_role_classes
                            .iter_mut()
                            .find(|(class, _)| *class == runtime_class)
                        {
                            roles.push(target_previous_role.clone());
                        } else {
                            previous_role_classes
                                .push((runtime_class, vec![target_previous_role.clone()]));
                        }
                    }

                    for (_, mut target_previous_roles) in previous_role_classes {
                        target_previous_roles.sort_by_key(|role| normalize_role(role));
                        let trace = ShamanTrace {
                            source_position,
                            target_position,
                            copied_role: copied_role.clone(),
                            target_previous_roles,
                        };
                        let context = apply_shaman_trace_to_context(pair_base.clone(), trace);
                        let key = start_context_key(&context);
                        if seen.insert(key) {
                            variants.push(context);
                        }
                    }
                }
            }
        }
    }
    variants
}

fn unique_deck_villager_roles(state: &GameState) -> Vec<String> {
    let mut roles = Vec::new();
    let mut seen = HashSet::new();
    for role in &state.deck.villagers {
        if seen.insert(normalize_role(role)) {
            roles.push(role.clone());
        }
    }
    roles
}

fn observed_final_villager_role(position: u8, state: &GameState) -> Option<String> {
    state
        .executed_good_roles
        .get(&position)
        .map(String::as_str)
        .or_else(|| {
            state
                .card_at(position)
                .map(|card| card.apparent_role.as_str())
        })
        .filter(|role| is_state_villager_role(role, state))
        .map(str::to_string)
}

/// Cheap pre-pruning for erased-role alternatives. A visible final Baker may
/// have replaced any hidden Villager after Shaman's Start, so its initial
/// identity is deferred to the dedicated Baker history validator.
fn shaman_initial_role_counts_fit(
    trace: &ShamanTrace,
    state: &GameState,
    context: &StartCorruptionContext,
) -> bool {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut source_seen = false;

    for &position in &context.registered_villagers_at_pd_call {
        let role = if position == trace.target_position {
            trace
                .target_previous_roles
                .first()
                .expect("exact Shaman pre-pruning trace has one prior role")
                .as_str()
        } else if let Some(observed) = observed_final_villager_role(position, state) {
            if position == trace.source_position {
                source_seen = true;
            }
            if normalize_role(&observed) == "baker" {
                if position == trace.source_position
                    && normalize_role(&trace.copied_role) == "baker"
                {
                    *counts.entry("baker".to_string()).or_insert(0) += 1;
                }
                continue;
            }
            // The trace generator has already required the source observation
            // to equal copied_role.
            *counts.entry(normalize_role(&observed)).or_insert(0) += 1;
            continue;
        } else {
            continue;
        };
        *counts.entry(normalize_role(role)).or_insert(0) += 1;
    }

    if !source_seen {
        *counts
            .entry(normalize_role(&trace.copied_role))
            .or_insert(0) += 1;
    }

    let mut available: HashMap<String, usize> = HashMap::new();
    for role in &state.deck.villagers {
        *available.entry(normalize_role(role)).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .all(|(role, count)| count <= available.get(&role).copied().unwrap_or(0))
}

fn apply_shaman_trace_to_context(
    mut context: StartCorruptionContext,
    trace: ShamanTrace,
) -> StartCorruptionContext {
    let source_is_alchemist = normalize_role(&trace.copied_role) == "alchemist";
    let target_was_alchemist = trace
        .target_previous_roles
        .iter()
        .any(|role| normalize_role(role) == "alchemist");

    // These sets describe Init-time resistance and actors still eligible at
    // the later all-Alchemist Start slot. InitWithNoReset preserves the
    // target's former resistance but its immediate copied Start consumes the
    // target's one-shot guard, so it never acts again in the global scan.
    context
        .corruption_resistant_at_init
        .remove(&trace.source_position);
    context
        .corruption_resistant_at_init
        .remove(&trace.target_position);
    context.true_alchemist_positions.retain(|position| {
        *position != trace.source_position && *position != trace.target_position
    });

    if source_is_alchemist {
        context
            .corruption_resistant_at_init
            .insert(trace.source_position);
        context.true_alchemist_positions.push(trace.source_position);
    }
    if target_was_alchemist {
        context
            .corruption_resistant_at_init
            .insert(trace.target_position);
    }
    context.true_alchemist_positions.sort_unstable();
    context.true_alchemist_positions.dedup();
    context.shaman_trace = Some(trace);
    context
}

fn sorted_positions(values: &HashSet<u8>) -> Vec<u8> {
    let mut result: Vec<u8> = values.iter().copied().collect();
    result.sort_unstable();
    result
}

fn start_context_key(context: &StartCorruptionContext) -> StartContextSemanticKey {
    let mut alchemists = context.true_alchemist_positions.clone();
    alchemists.sort_unstable();
    StartContextSemanticKey {
        real_villagers_before_puppet: sorted_positions(
            &context.real_villagers_before_puppet,
        ),
        registered_villagers_at_pd_call: sorted_positions(
            &context.registered_villagers_at_pd_call,
        ),
        corruption_resistant_at_init: sorted_positions(
            &context.corruption_resistant_at_init,
        ),
        messed_up_resistant_at_init: sorted_positions(
            &context.messed_up_resistant_at_init,
        ),
        true_alchemists: alchemists,
        initial_messed_up_by_evil: sorted_positions(
            &context.initial_messed_up_by_evil,
        ),
        drunk_position: context.drunk_position,
        puppet_position: context.puppet_position,
        plague_doctor_acts: context.plague_doctor_acts,
        shaman_trace: context.shaman_trace.clone(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_start_corruption_context(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    chancellor_conversion: Option<u8>,
    extra_true_alchemists: &[u8],
    plague_doctor_acts: bool,
) -> StartCorruptionContext {
    let extra_alchemists: HashSet<u8> = extra_true_alchemists.iter().copied().collect();
    let twin_puppet_overlap = puppet_overlays_stable_twin(full_evil, puppet_position);
    let mut real_villagers_before_puppet = HashSet::new();

    for position in 1..=state.n_cards {
        let is_real_villager = if Some(position) == chancellor_conversion {
            false // Chancellor has already replaced this data with an Outcast.
        } else if Some(position) == puppet_position && !twin_puppet_overlap {
            true // Puppeteer selects a real Villager and replaces it later.
        } else if full_evil.contains_key(&position)
            || Some(position) == doppelganger_position
            || Some(position) == drunk_position
        {
            false
        } else if extra_alchemists.contains(&position) {
            true
        } else if let Some(card) = state.card_at(position) {
            is_state_villager_role(&card.apparent_role, state)
        } else {
            unrevealed_must_be_villager(
                position, full_evil, state, doppelganger_position, drunk_position,
            )
        };
        if is_real_villager {
            real_villagers_before_puppet.insert(position);
        }
    }

    let mut corruption_resistant_at_init = extra_alchemists.clone();
    let mut true_alchemist_positions: Vec<u8> = extra_true_alchemists.to_vec();
    for card in &state.cards {
        if normalize_role(&card.apparent_role) != "alchemist"
            || Some(card.position) == doppelganger_position
            || Some(card.position) == drunk_position
        {
            continue;
        }

        // A future Puppet or Chancellor target already ran its original
        // Alchemist Init hook, so resistance survives the role replacement.
        if !full_evil.contains_key(&card.position)
            || (Some(card.position) == puppet_position && !twin_puppet_overlap)
        {
            corruption_resistant_at_init.insert(card.position);
        }
        if !full_evil.contains_key(&card.position)
            && Some(card.position) != puppet_position
            && Some(card.position) != chancellor_conversion
        {
            true_alchemist_positions.push(card.position);
        }
    }
    true_alchemist_positions.sort_unstable();
    true_alchemist_positions.dedup();

    // Init clears registerAs, and delayed internal Reveal has not resumed yet.
    // PD's registered-type predicate therefore matches post-conversion dataRef.
    let mut registered_villagers_at_pd_call = real_villagers_before_puppet.clone();
    if let Some(puppet_position) = puppet_position {
        registered_villagers_at_pd_call.remove(&puppet_position);
    }

    StartCorruptionContext {
        real_villagers_before_puppet,
        registered_villagers_at_pd_call,
        corruption_resistant_at_init,
        messed_up_resistant_at_init: HashSet::new(),
        true_alchemist_positions,
        initial_messed_up_by_evil: HashSet::new(),
        drunk_position,
        puppet_position,
        plague_doctor_acts,
        shaman_trace: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn plague_doctor_act_variants(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    chancellor_trace: Option<&RawChancellorTrace>,
) -> Vec<bool> {
    let mut variants = Vec::new();
    for acts in [false, true] {
        if state.pd_corruption_target.is_some() && !acts {
            continue;
        }
        if natural_outcast_hypothesis_allows(
            state,
            full_evil,
            puppet_position,
            doppelganger_position,
            drunk_position,
            chancellor_trace,
            Some(acts),
        ) {
            variants.push(acts);
        }
    }
    variants.sort_unstable();
    variants.dedup();
    variants
}

#[allow(clippy::too_many_arguments)]
fn night_killed_alchemist_variants(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    chancellor_conversion: Option<u8>,
) -> Vec<Vec<u8>> {
    let twin_puppet_overlap = puppet_overlays_stable_twin(full_evil, puppet_position);
    let deck_alchemists = state.deck.villagers.iter()
        .filter(|role| normalize_role(role) == "alchemist").count();
    if state.night_kills.is_empty() || deck_alchemists == 0 {
        return vec![Vec::new()];
    }

    let known_alchemists = state.cards.iter().filter(|card| {
        normalize_role(&card.apparent_role) == "alchemist"
            && Some(card.position) != doppelganger_position
            && Some(card.position) != drunk_position
            && (!full_evil.contains_key(&card.position)
                || (Some(card.position) == puppet_position && !twin_puppet_overlap))
    }).count();
    let missing_alchemists = deck_alchemists.saturating_sub(known_alchemists);
    if missing_alchemists == 0 { return vec![Vec::new()]; }

    let revealed: HashSet<u8> = state.cards.iter().map(|card| card.position).collect();
    let hidden_night_kills: Vec<u8> = state.night_kills.iter().copied()
        .filter(|position| {
            !revealed.contains(position)
                && !full_evil.contains_key(position)
                && Some(*position) != doppelganger_position
                && Some(*position) != drunk_position
                && Some(*position) != puppet_position
                && Some(*position) != chancellor_conversion
        }).collect();

    let mut variants = vec![Vec::new()];
    for count in 1..=missing_alchemists.min(hidden_night_kills.len()) {
        variants.extend(combinations_of(&hidden_night_kills, count));
    }
    variants
}

// ── Placement generation ──

/// Candidate occurrences seen by Puppeteer's native circular-neighbour scan
/// for one exact pre-Puppet faction branch. The boolean is true when the
/// observed role proves that the occurrence is a non-Saint Villager. The first
/// known Saint occurrence is removed exactly once; on a two-card board the
/// duplicated second occurrence remains convertible.
fn native_puppeteer_candidate_occurrences(
    puppeteer_position: u8,
    real_villagers_before_puppet: &HashSet<u8>,
    state: &GameState,
) -> Vec<(u8, bool)> {
    if state.n_cards <= 1 {
        return Vec::new();
    }
    let mut removed_first_saint = false;
    let mut candidates = Vec::new();
    for position in adjacent_positions(puppeteer_position, state.n_cards) {
        if position == puppeteer_position {
            continue;
        }
        if !real_villagers_before_puppet.contains(&position) {
            continue;
        }
        let observed_role = state
            .executed_good_roles
            .get(&position)
            .map(String::as_str)
            .or_else(|| {
                state
                    .card_at(position)
                    .map(|card| card.apparent_role.as_str())
            });
        if observed_role.is_some_and(|role| !is_state_villager_role(role, state)) {
            continue;
        }
        if observed_role.is_some_and(|role| normalize_role(role) == "saint")
            && !removed_first_saint
        {
            removed_first_saint = true;
            continue;
        }
        candidates.push((position, observed_role.is_some()));
    }
    candidates
}

fn placement_matches_trusted_evil_faction_counts(
    placement: &HashMap<u8, String>,
    state: &GameState,
) -> bool {
    let (Some(expected_minions), Some(expected_demons)) =
        (state.board_minion_count, state.board_demon_count)
    else {
        return true;
    };
    let mut minions = 0usize;
    let mut demons = 0usize;
    for role in placement
        .values()
        .chain(state.executed_evil_roles.values())
    {
        if normalize_role(role) == "puppet" {
            continue;
        }
        match role_faction_in_state(role, state) {
            Some(Faction::Minion) => minions += 1,
            Some(Faction::Demon) => demons += 1,
            _ => return false,
        }
    }
    minions == expected_minions as usize && demons == expected_demons as usize
}

fn placement_has_valid_puppeteer_structure(
    placement: &HashMap<u8, String>,
    puppet_position: Option<u8>,
    state: &GameState,
) -> bool {
    let mut puppeteers: Vec<u8> = placement
        .iter()
        .chain(state.executed_evil_roles.iter())
        .filter_map(|(&position, role)| {
            (normalize_role(role) == "puppeteer").then_some(position)
        })
        .collect();
    puppeteers.sort_unstable();
    puppeteers.dedup();
    let mut explicit_puppets: Vec<u8> = placement
        .iter()
        .chain(state.executed_evil_roles.iter())
        .filter_map(|(&position, role)| {
            (normalize_role(role) == "puppet").then_some(position)
        })
        .collect();
    explicit_puppets.sort_unstable();
    explicit_puppets.dedup();

    if puppeteers.len() > 1 || explicit_puppets.len() > 1 {
        // Duplicate Puppeteer ordering/provenance is not represented yet.
        return false;
    }

    let mut exact_dead_twin_puppets: Vec<u8> = placement
        .iter()
        .chain(state.executed_evil_roles.iter())
        .filter_map(|(&position, role)| {
            (normalize_role(role) == "twinminion"
                && state
                    .executed_current_roles
                    .get(&position)
                    .is_some_and(|current| normalize_role(current) == "puppet"))
            .then_some(position)
        })
        .collect();
    exact_dead_twin_puppets.sort_unstable();
    exact_dead_twin_puppets.dedup();
    if exact_dead_twin_puppets.len() > 1
        || exact_dead_twin_puppets
            .first()
            .is_some_and(|&position| puppet_position != Some(position))
    {
        return false;
    }

    let Some(&puppeteer_position) = puppeteers.first() else {
        return explicit_puppets.is_empty() && puppet_position.is_none();
    };

    let stable_role_at = |position: u8| {
        placement
            .get(&position)
            .or_else(|| state.executed_evil_roles.get(&position))
            .map(String::as_str)
    };
    let explicit_puppet_position = explicit_puppets.first().copied();
    match (explicit_puppet_position, puppet_position) {
        (Some(explicit), Some(represented)) if explicit == represented => {}
        (Some(_), _) => return false,
        (None, Some(represented)) => {
            if represented == puppeteer_position
                || !stable_role_at(represented)
                    .is_some_and(|role| normalize_role(role) == "twinminion")
                || state
                    .executed_current_roles
                    .get(&represented)
                    .is_some_and(|role| normalize_role(role) != "puppet")
            {
                return false;
            }
        }
        (None, None) => {}
    }

    let has_twin = placement
        .values()
        .chain(state.executed_evil_roles.values())
        .any(|role| normalize_role(role) == "twinminion");
    if has_twin {
        // Twin acts before Puppeteer and can move its current data. The legacy
        // mixed-Twin path does not yet retain that post-swap actor position, so
        // adjacency to the stable origin cannot be enforced soundly here.
        return true;
    }
    puppet_position.is_none_or(|represented| {
        adjacent_positions(puppeteer_position, state.n_cards).contains(&represented)
    })
}

fn start_context_matches_native_puppeteer_conversion(
    full_evil: &HashMap<u8, String>,
    puppet_position: Option<u8>,
    context: &StartCorruptionContext,
    state: &GameState,
) -> bool {
    if full_evil
        .values()
        .any(|role| normalize_role(role) == "twinminion")
    {
        // The ordered Twin slice deliberately excludes Puppeteer. Its legacy
        // fallback has no exact post-Twin actor/type map, so retain both
        // conversion outcomes until the general ordered replay is complete.
        return true;
    }
    let mut puppeteers: Vec<u8> = full_evil
        .iter()
        .filter_map(|(&position, role)| {
            (normalize_role(role) == "puppeteer").then_some(position)
        })
        .collect();
    puppeteers.sort_unstable();
    puppeteers.dedup();
    if puppeteers.len() > 1 {
        return false;
    }
    let Some(&puppeteer_position) = puppeteers.first() else {
        return puppet_position.is_none();
    };
    let raw_candidates: Vec<u8> = adjacent_positions(puppeteer_position, state.n_cards)
        .into_iter()
        .filter(|position| *position != puppeteer_position)
        .filter(|position| context.real_villagers_before_puppet.contains(position))
        .collect();
    let shaman_unstable = full_evil
        .values()
        .any(|role| normalize_role(role) == "shaman");
    let baker_unstable = !state.cards.is_empty()
        && state
            .deck
            .villagers
            .iter()
            .any(|role| normalize_role(role) == "baker");
    let identity_unstable = shaman_unstable || baker_unstable;

    match puppet_position {
        Some(position) => {
            if !raw_candidates.contains(&position) {
                return false;
            }
            let observed_role = state
                .card_at(position)
                .map(|card| card.apparent_role.as_str());
            if observed_role.is_some_and(|role| !is_state_villager_role(role, state)) {
                return false;
            }
            if identity_unstable {
                return true;
            }
            native_puppeteer_candidate_occurrences(
                puppeteer_position,
                &context.real_villagers_before_puppet,
                state,
            )
            .iter()
            .any(|(candidate, _)| *candidate == position)
        }
        None => {
            if raw_candidates.is_empty() {
                return true;
            }
            // Native removes at most one Saint occurrence. Two distinct
            // neighbours, or the duplicated neighbour on a two-card board,
            // therefore always leave a mandatory conversion candidate.
            if raw_candidates.len() >= 2 {
                return false;
            }
            if identity_unstable {
                // Shaman/Baker can make the final visible Villager role newer
                // than Puppeteer's scan. A sole candidate may have been Saint.
                return true;
            }
            let position = raw_candidates[0];
            let observed_role = state
                .executed_good_roles
                .get(&position)
                .map(String::as_str)
                .or_else(|| {
                    state
                        .card_at(position)
                        .map(|card| card.apparent_role.as_str())
                });
            match observed_role {
                Some(role) if normalize_role(role) == "saint" => true,
                Some(role) if is_state_villager_role(role, state) => false,
                Some(_) => false,
                None => state
                    .deck
                    .villagers
                    .iter()
                    .any(|role| normalize_role(role) == "saint"),
            }
        }
    }
}

fn possible_total_evil_counts(state: &GameState, puppeteer_authored: bool) -> Vec<usize> {
    let mut totals = vec![state.n_evil as usize];
    // Pre-provenance archives used both count conventions: some saved the
    // authored Evil total before Puppeteer generated a Puppet, while current
    // trusted HUD captures include that Puppet. Retain both meanings only for
    // those ambiguous historical states.
    if puppeteer_authored && state.board_count_provenance == BoardCountProvenance::LegacyUnknown {
        totals.push(state.n_evil as usize + 1);
    }
    totals.sort_unstable();
    totals.dedup();
    totals
}

fn placement_matches_evil_total_provenance(
    placement: &HashMap<u8, String>,
    state: &GameState,
) -> bool {
    let total = placement.len() + state.executed_evil_roles.len();
    if total == state.n_evil as usize {
        return true;
    }
    if state.board_count_provenance != BoardCountProvenance::LegacyUnknown
        || total != state.n_evil as usize + 1
    {
        return false;
    }
    let roles: Vec<&str> = placement
        .values()
        .chain(state.executed_evil_roles.values())
        .map(String::as_str)
        .collect();
    roles
        .iter()
        .any(|role| normalize_role(role) == "puppeteer")
        && roles
            .iter()
            .any(|role| normalize_role(role) == "puppet")
}

fn generate_evil_placements(state: &GameState) -> Vec<EvilPlacement> {
    let n = state.n_cards;
    let mut evil_roles: Vec<String> = state.deck.evil_roles();

    let puppet_in_deck = evil_roles
        .iter()
        .any(|role| normalize_role(role) == "puppeteer");
    if puppet_in_deck {
        if let Some(idx) = evil_roles
            .iter()
            .position(|role| normalize_role(role) == "puppet")
        {
            evil_roles.remove(idx);
        }
    }

    // Remove executed evil roles
    let mut remaining = evil_roles.clone();
    for (_pos, role) in &state.executed_evil_roles {
        let norm = normalize_role(role);
        if let Some(idx) = remaining.iter().position(|r| normalize_role(r) == norm) {
            remaining.remove(idx);
        }
        // Puppet isn't in evil_roles — silently skip
    }

    // Publicly untyped deaths are expanded into exact construction clones by
    // `build_scenarios` before this private generator is reached.
    let n_executed_evil = state.executed_evil_roles.len();
    let expected_remaining_counts: HashSet<usize> =
        possible_total_evil_counts(state, puppet_in_deck)
            .into_iter()
            .filter_map(|total| total.checked_sub(n_executed_evil))
            .collect();
    if expected_remaining_counts.is_empty() {
        return Vec::new();
    }

    // Trusted `n_evil` is the HUD total and includes a generated Puppet.
    // LegacyUnknown can additionally preserve the older authored-only count.
    // Enumerate an authored subset one smaller while Puppet can still be
    // created; the final allowed-size and Start-context predicates choose the
    // branch.
    let generated_puppet_already_recorded = state
        .executed_evil_roles
        .values()
        .any(|role| normalize_role(role) == "puppet");
    let mut authored_remaining_sizes = Vec::new();
    for &expected_remaining in &expected_remaining_counts {
        authored_remaining_sizes.push(expected_remaining);
        if puppet_in_deck && !generated_puppet_already_recorded && expected_remaining > 0 {
            authored_remaining_sizes.push(expected_remaining - 1);
        }
    }
    authored_remaining_sizes.sort_unstable();
    authored_remaining_sizes.dedup();
    let mut possible_remaining_lists = Vec::new();
    let mut seen_remaining_lists = HashSet::new();
    for authored_count in authored_remaining_sizes {
        for roles in evil_role_subsets_fn(&remaining, state, authored_count) {
            let mut key: Vec<String> = roles.iter().map(|role| normalize_role(role)).collect();
            key.sort_unstable();
            if seen_remaining_lists.insert(key) {
                possible_remaining_lists.push(roles);
            }
        }
    }

    let night_kills_set: HashSet<u8> = state.night_kills.iter().copied().collect();
    let player_executed: HashSet<u8> = state.executed.iter()
        .filter(|p| !night_kills_set.contains(p))
        .copied().collect();
    let confirmed_good_set: HashSet<u8> = state.confirmed_good.iter().copied().collect();
    let exact_dead_evil_positions: HashSet<u8> = state
        .executed_evil_roles
        .keys()
        .copied()
        .filter(|position| state.executed.contains(position))
        .collect();
    let available: Vec<u8> = (1..=n)
        .filter(|p| {
            !player_executed.contains(p)
                && !exact_dead_evil_positions.contains(p)
                && !confirmed_good_set.contains(p)
        })
        .collect();

    // Check Puppeteer/Puppet execution status
    let puppeteer_executed_pos: Option<u8> = state.executed_evil_roles.iter()
        .find(|(_, role)| normalize_role(role) == "puppeteer")
        .map(|(&position, _)| position);
    let puppet_executed_pos: Option<u8> = state.executed_evil_roles.iter()
        .find(|(_, role)| normalize_role(role) == "puppet")
        .map(|(&position, _)| position);
    let mut all_placements: Vec<HashMap<u8, String>> = Vec::new();
    let mut seen_placements: HashSet<Vec<(u8, String)>> = HashSet::new();

    let mut add_placement = |p: &HashMap<u8, String>| {
        let explicit_puppet = p
            .iter()
            .chain(state.executed_evil_roles.iter())
            .find_map(|(&position, role)| {
                (normalize_role(role) == "puppet").then_some(position)
            });
        let has_twin = p
            .values()
            .chain(state.executed_evil_roles.values())
            .any(|role| normalize_role(role) == "twinminion");
        if expected_remaining_counts.contains(&p.len())
            && placement_matches_evil_total_provenance(p, state)
            && placement_matches_trusted_evil_faction_counts(p, state)
            && (has_twin
                || placement_has_valid_puppeteer_structure(p, explicit_puppet, state))
        {
            let mut key: Vec<(u8, String)> = p.iter().map(|(&k, v)| (k, v.clone())).collect();
            key.sort_by_key(|(k, _)| *k);
            if seen_placements.insert(key) {
                all_placements.push(p.clone());
            }
        }
    };

    for evil_roles in &possible_remaining_lists {
        let has_puppeteer = evil_roles
            .iter()
            .any(|role| normalize_role(role) == "puppeteer");
        let twin_can_move_puppeteer = evil_roles
            .iter()
            .chain(state.executed_evil_roles.values())
            .any(|role| normalize_role(role) == "twinminion");

        // Case: Puppeteer executed, Puppet still alive
        let puppet_still_alive = puppeteer_executed_pos.is_some()
            && !has_puppeteer
            && !state
                .executed_evil_roles
                .values()
                .any(|role| normalize_role(role) == "puppet");

        if puppet_still_alive {
            let pep = puppeteer_executed_pos.unwrap();
            let puppet_cands: Vec<u8> = if twin_can_move_puppeteer {
                available.clone()
            } else {
                adjacent_positions(pep, n)
                    .into_iter()
                    .filter(|position| available.contains(position))
                    .collect()
            };
            // Case 1: Puppet at a possible post-Twin neighbour.
            for &pp in &puppet_cands {
                let remaining_avail: Vec<u8> = available.iter().filter(|&&p| p != pp).copied().collect();
                if evil_roles.is_empty() {
                    let mut p = HashMap::new();
                    p.insert(pp, "Puppet".to_string());
                    add_placement(&p);
                } else {
                    for combo in combinations_of(&remaining_avail, evil_roles.len()) {
                        for perm in permutations_of(evil_roles) {
                            let mut p = HashMap::new();
                            p.insert(pp, "Puppet".to_string());
                            for (i, &pos) in combo.iter().enumerate() {
                                p.insert(pos, perm[i].clone());
                            }
                            add_placement(&p);
                        }
                    }
                }
            }
            for combo in combinations_of(&available, evil_roles.len()) {
                for perm in permutations_of(evil_roles) {
                    let mut p = HashMap::new();
                    for (i, &pos) in combo.iter().enumerate() {
                        p.insert(pos, perm[i].clone());
                    }
                    add_placement(&p);
                }
            }
            continue;
        }

        // Case: Puppet executed, Puppeteer must occupy a possible pre-action
        // actor position. Mixed Twin fallback conservatively keeps every seat.
        if puppet_executed_pos.is_some() && has_puppeteer {
            let pxp = puppet_executed_pos.unwrap();
            let base_evil: Vec<String> = evil_roles
                .iter()
                .filter(|role| normalize_role(role) != "puppeteer")
                .cloned()
                .collect();
            let puppeteer_cands: Vec<u8> = if twin_can_move_puppeteer {
                available.clone()
            } else {
                adjacent_positions(pxp, n).into_iter()
                    .filter(|position| available.contains(position))
                    .collect()
            };
            for &pp_pos in &puppeteer_cands {
                let rem: Vec<u8> = available.iter().filter(|&&p| p != pp_pos).copied().collect();
                if base_evil.is_empty() {
                    let mut p = HashMap::new();
                    p.insert(pp_pos, "Puppeteer".to_string());
                    add_placement(&p);
                } else {
                    for combo in combinations_of(&rem, base_evil.len()) {
                        for perm in permutations_of(&base_evil) {
                            let mut p = HashMap::new();
                            p.insert(pp_pos, "Puppeteer".to_string());
                            for (i, &pos) in combo.iter().enumerate() {
                                p.insert(pos, perm[i].clone());
                            }
                            add_placement(&p);
                        }
                    }
                }
            }
            continue;
        }

        // Case: Puppeteer present (needs Puppet slot)
        if has_puppeteer {
            let base_evil: Vec<String> = evil_roles
                .iter()
                .filter(|role| normalize_role(role) != "puppeteer")
                .cloned()
                .collect();
            for &puppeteer_pos in &available {
                let actual_puppet_cands: Vec<u8> = if twin_can_move_puppeteer {
                    available
                        .iter()
                        .copied()
                        .filter(|position| *position != puppeteer_pos)
                        .collect()
                } else {
                    adjacent_positions(puppeteer_pos, n)
                        .into_iter()
                        .filter(|position| {
                            *position != puppeteer_pos && available.contains(position)
                        })
                        .collect()
                };

                // Case 1: Puppet created
                for &puppet_p in &actual_puppet_cands {
                    let rem: Vec<u8> = available.iter()
                        .filter(|&&p| p != puppeteer_pos && p != puppet_p)
                        .copied().collect();
                    if base_evil.is_empty() {
                        let mut p = HashMap::new();
                        p.insert(puppeteer_pos, "Puppeteer".to_string());
                        p.insert(puppet_p, "Puppet".to_string());
                        add_placement(&p);
                    } else {
                        for combo in combinations_of(&rem, base_evil.len()) {
                            for perm in permutations_of(&base_evil) {
                                let mut p = HashMap::new();
                                p.insert(puppeteer_pos, "Puppeteer".to_string());
                                p.insert(puppet_p, "Puppet".to_string());
                                for (i, &pos) in combo.iter().enumerate() {
                                    p.insert(pos, perm[i].clone());
                                }
                                add_placement(&p);
                            }
                        }
                    }
                }
                let rem: Vec<u8> = available.iter()
                    .filter(|&&p| p != puppeteer_pos).copied().collect();
                if base_evil.is_empty() {
                    let mut p = HashMap::new();
                    p.insert(puppeteer_pos, "Puppeteer".to_string());
                    add_placement(&p);
                } else {
                    for combo in combinations_of(&rem, base_evil.len()) {
                        for perm in permutations_of(&base_evil) {
                            let mut p = HashMap::new();
                            p.insert(puppeteer_pos, "Puppeteer".to_string());
                            for (i, &pos) in combo.iter().enumerate() {
                                p.insert(pos, perm[i].clone());
                            }
                            add_placement(&p);
                        }
                    }
                }
            }
            continue;
        }

        // No Puppeteer — the authored multiset was selected above.
        for combo in combinations_of(&available, evil_roles.len()) {
            for perm in permutations_of(evil_roles) {
                let mut p = HashMap::new();
                for (i, &pos) in combo.iter().enumerate() {
                    p.insert(pos, perm[i].clone());
                }
                add_placement(&p);
            }
        }
    }

    let mut generated = Vec::new();
    let mut seen_generated = HashSet::new();
    for roles in all_placements {
        let explicit_puppet = roles
            .iter()
            .chain(state.executed_evil_roles.iter())
            .find_map(|(&position, role)| {
                (normalize_role(role) == "puppet").then_some(position)
            });
        let mut add_generated = |puppet_position: Option<u8>| {
            if !placement_has_valid_puppeteer_structure(&roles, puppet_position, state) {
                return;
            }
            let mut key: Vec<(u8, String)> = roles
                .iter()
                .map(|(&position, role)| (position, normalize_role(role)))
                .collect();
            key.sort_unstable();
            if seen_generated.insert((key, puppet_position)) {
                generated.push(EvilPlacement {
                    roles: roles.clone(),
                    puppet_position,
                });
            }
        };
        add_generated(explicit_puppet);

        if explicit_puppet.is_none() {
            let combined_roles = roles
                .iter()
                .chain(state.executed_evil_roles.iter());
            let has_puppeteer = combined_roles
                .clone()
                .any(|(_, role)| normalize_role(role) == "puppeteer");
            if has_puppeteer {
                let mut twin_origins: Vec<u8> = roles
                    .iter()
                    .chain(state.executed_evil_roles.iter())
                    .filter_map(|(&position, role)| {
                        (normalize_role(role) == "twinminion").then_some(position)
                    })
                    .collect();
                twin_origins.sort_unstable();
                twin_origins.dedup();
                for twin_origin in twin_origins {
                    // Twin may leave Villager current data on its stable
                    // runtime-Evil body. Later Puppeteer can replace that data
                    // with Puppet without adding a new Evil physical seat, so
                    // keep the generated role as a separate overlay.
                    add_generated(Some(twin_origin));
                }
            }
        }
    }
    generated
}

// ── Helper functions ──

fn apply_placement_constraints(placement: &HashMap<u8, String>, state: &GameState) -> bool {
    let n = state.n_cards;
    for (&pos, role) in placement {
        if role == "Chancellor" {
            let adj = adjacent_positions(pos, n);
            if !adj.iter().any(|a| !placement.contains_key(a)) {
                return false;
            }
        }
    }
    for &pos in &state.confirmed_evil {
        if !state.executed.contains(&pos) && !placement.contains_key(&pos) {
            return false;
        }
    }
    true
}

fn unrevealed_must_be_villager(
    pos: u8, evil_positions: &HashMap<u8, String>, state: &GameState,
    doppelganger_pos: Option<u8>, drunk_pos: Option<u8>,
) -> bool {
    // This helper is used only by the ordinary/no-trace Start path. A
    // Chancellor merely being available in the pool does not add an Outcast;
    // the concrete trace path projects its generated identity separately.
    if let Some(max_outcasts) = state.board_outcast_count {
        let occupied = (1..=state.n_cards)
            .filter(|position| *position != pos)
            .filter(|position| {
                is_known_natural_ordinary_outcast(
                    *position,
                    state,
                    evil_positions,
                    doppelganger_pos,
                    drunk_pos,
                    None,
                )
            })
            .count();
        return occupied >= max_outcasts as usize;
    }

    // Without a header count, retain the conservative total-pool fallback.
    // Here the represented Dopp/Drunk identities do consume pool identities.
    let max_outcasts = state.deck.outcasts.len();
    let mut occupied = HashSet::new();
    for card in &state.cards {
        if evil_positions.contains_key(&card.position) || card.position == pos {
            continue;
        }
        if is_state_outcast_role(&card.apparent_role, state) {
            occupied.insert(card.position);
        }
    }
    for special in [doppelganger_pos, drunk_pos].into_iter().flatten() {
        if special != pos && !evil_positions.contains_key(&special) {
            occupied.insert(special);
        }
    }
    occupied.len() >= max_outcasts
}

fn hidden_outcast_presence_flags(
    role_name: &str,
    state: &GameState,
    _chancellor_present: bool,
) -> (bool, bool) {
    let normalized_role = normalize_role(role_name);
    let role_visible = state
        .deck
        .outcasts
        .iter()
        .any(|role| normalize_role(role) == normalized_role);
    if !role_visible {
        return (false, true);
    }
    // Dopp/Drunk register as Villagers in the HUD, and Chancellor may generate
    // either identity after those base counts are captured. The exact shared
    // multiset/host check below decides whether each candidate is feasible;
    // the header alone cannot require or exclude the special identity.
    (true, true)
}

fn evil_role_subsets_fn(evil_roles: &[String], state: &GameState, expected_remaining: usize) -> Vec<Vec<String>> {
    let minion_pool: Vec<String> = evil_roles.iter()
        .filter(|role| role_faction_in_state(role, state) == Some(Faction::Minion))
        .cloned().collect();
    let demon_pool: Vec<String> = evil_roles.iter()
        .filter(|role| role_faction_in_state(role, state) == Some(Faction::Demon))
        .cloned().collect();

    let mut bm = state.board_minion_count.map(|x| x as i32);
    let mut bd = state.board_demon_count.map(|x| x as i32);

    let mp = minion_pool;
    let dp = demon_pool;
    // `evil_roles` is already pruned by every exact stable-origin branch.
    // Board faction quotas still describe the whole board, so subtract each
    // authored dead role directly instead of trying to find it a second time
    // in the pruned pool. Generated Puppet consumes no authored occurrence.
    for role in state.executed_evil_roles.values() {
        if normalize_role(role) == "puppet" {
            continue;
        }
        match role_faction_in_state(role, state) {
            Some(Faction::Minion) => {
                if let Some(ref mut count) = bm {
                    *count -= 1;
                }
            }
            Some(Faction::Demon) => {
                if let Some(ref mut count) = bd {
                    *count -= 1;
                }
            }
            _ => {}
        }
    }

    if let (Some(bm_v), Some(bd_v)) = (bm, bd) {
        if bm_v < 0 || bd_v < 0 {
            return Vec::new();
        }
        let bm_pick = bm_v as usize;
        let bd_pick = bd_v as usize;
        if bm_pick + bd_pick != expected_remaining {
            return Vec::new();
        }
        let mut subsets = Vec::new();
        for m_combo in combinations_of_strings(&mp, bm_pick) {
            for d_combo in combinations_of_strings(&dp, bd_pick) {
                let mut subset: Vec<String> = m_combo.clone();
                subset.extend(d_combo);
                if subset.len() == expected_remaining {
                    subsets.push(subset);
                }
            }
        }
        subsets
    } else {
        combinations_of_strings(evil_roles, expected_remaining)
    }
}

// ── Combinatorial helpers ──

fn combinations_of(items: &[u8], k: usize) -> Vec<Vec<u8>> {
    if k == 0 { return vec![vec![]]; }
    if k > items.len() { return vec![]; }
    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        for mut rest in combinations_of(&items[i + 1..], k - 1) {
            rest.insert(0, item);
            result.push(rest);
        }
    }
    result
}

fn combinations_of_strings(items: &[String], k: usize) -> Vec<Vec<String>> {
    if k == 0 { return vec![vec![]]; }
    if k > items.len() { return vec![]; }
    let mut result = Vec::new();
    for (i, item) in items.iter().enumerate() {
        for mut rest in combinations_of_strings(&items[i + 1..], k - 1) {
            rest.insert(0, item.clone());
            result.push(rest);
        }
    }
    result
}

fn combinations_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    let items: Vec<usize> = (0..n).collect();
    if k == 0 { return vec![vec![]]; }
    if k > n { return vec![]; }
    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        for mut rest in combinations_indices_inner(&items[i + 1..], k - 1) {
            rest.insert(0, item);
            result.push(rest);
        }
    }
    result
}

fn combinations_indices_inner(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 { return vec![vec![]]; }
    if k > items.len() { return vec![]; }
    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        for mut rest in combinations_indices_inner(&items[i + 1..], k - 1) {
            rest.insert(0, item);
            result.push(rest);
        }
    }
    result
}

/// Distinct permutations (handles duplicates).
fn permutations_of(roles: &[String]) -> Vec<Vec<String>> {
    if roles.is_empty() { return vec![vec![]]; }
    let mut seen: HashSet<Vec<String>> = HashSet::new();
    let mut result = Vec::new();
    for (i, role) in roles.iter().enumerate() {
        let rest: Vec<String> = roles.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, r)| r.clone()).collect();
        for perm in permutations_of(&rest) {
            let mut full = vec![role.clone()];
            full.extend(perm);
            if seen.insert(full.clone()) {
                result.push(full);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CardInfo, TwinNeighborSide, TwinStartOutcome};

    fn card(position: u8, apparent_role: &str) -> CardInfo {
        CardInfo {
            position,
            apparent_role: apparent_role.to_string(),
            ..CardInfo::default()
        }
    }

    fn chancellor_at(position: u8) -> HashMap<u8, String> {
        HashMap::from([(position, "Chancellor".to_string())])
    }

    fn raw_trace(added_position: u8, added_role: &str, anchor_position: u8) -> RawChancellorTrace {
        RawChancellorTrace {
            original_position: 1,
            added_outcast_position: added_position,
            added_outcast_role: added_role.to_string(),
            anchor_position,
        }
    }

    fn safe_twin_state() -> GameState {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.deck.minions = vec!["Twin Minion".to_string(), "Witch".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state
    }

    fn exact_three_evil_twin_state() -> GameState {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.n_evil = 3;
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string(), "Lilis".to_string()];
        state.executed = vec![1, 2, 3];
        state.executed_evil_roles = HashMap::from([
            (1, "Twin Minion".to_string()),
            (2, "Pooka".to_string()),
            (3, "Lilis".to_string()),
        ]);
        state
    }

    #[test]
    fn untyped_executed_roles_branch_before_start_and_preserve_the_multiset() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.n_evil = 2;
        state.deck.villagers = vec![
            "Baker".to_string(),
            "Baker".to_string(),
            "Baker".to_string(),
        ];
        state.deck.minions = vec!["Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.cards = vec![card(2, "Baker"), card(3, "Baker"), card(4, "Baker")];
        state.executed = vec![1];
        state.confirmed_evil = vec![1];

        let scenarios = build_scenarios(&state);
        assert!(!scenarios.is_empty());
        assert!(scenarios.iter().all(|scenario| {
            !scenario
                .evil_positions
                .values()
                .any(|role| normalize_role(role) == "unknown")
                && scenario.evil_positions.len() == 2
                && scenario
                    .evil_positions
                    .values()
                    .filter(|role| normalize_role(role) == "minion")
                    .count()
                    == 1
                && scenario
                    .evil_positions
                    .values()
                    .filter(|role| normalize_role(role) == "pooka")
                    .count()
                    == 1
        }));
        assert!(scenarios.iter().any(|scenario| {
            scenario
                .evil_positions
                .get(&1)
                .is_some_and(|role| normalize_role(role) == "minion")
        }));
        assert!(scenarios.iter().any(|scenario| {
            scenario
                .evil_positions
                .get(&1)
                .is_some_and(|role| normalize_role(role) == "pooka")
                && scenario.corrupted == HashSet::from([2, 4])
        }));
        assert!(state.executed_evil_roles.is_empty());
    }

    #[test]
    fn duplicate_role_multiset_branches_without_factorial_duplicates() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.n_evil = 3;
        state.deck.minions = vec!["Witch".to_string(), "Witch".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.executed = vec![1, 2];
        state.confirmed_evil = vec![1, 2];

        let branches = branch_untyped_executed_evil_roles(&state);
        let assignments: HashSet<(String, String)> = branches
            .iter()
            .map(|branch| {
                (
                    normalize_role(branch.get(&1).unwrap()),
                    normalize_role(branch.get(&2).unwrap()),
                )
            })
            .collect();
        assert_eq!(branches.len(), 3);
        assert_eq!(
            assignments,
            HashSet::from([
                ("witch".to_string(), "witch".to_string()),
                ("witch".to_string(), "pooka".to_string()),
                ("pooka".to_string(), "witch".to_string()),
            ]),
        );
    }

    #[test]
    fn exact_night_killed_evil_seat_cannot_receive_a_second_placement() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.n_evil = 2;
        state.deck.minions = vec!["Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.executed = vec![1];
        state.night_kills = vec![1];
        state.confirmed_evil = vec![1];
        state
            .executed_evil_roles
            .insert(1, "Minion".to_string());

        let scenarios = build_scenarios(&state);
        assert_eq!(scenarios.len(), 2);
        assert!(scenarios.iter().all(|scenario| {
            scenario.evil_positions.get(&1) == Some(&"Minion".to_string())
                && scenario.evil_positions.len() == 2
        }));
    }

    #[test]
    fn inferred_puppet_and_puppeteer_branches_keep_creator_adjacency() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.n_evil = 2;
        state.deck.minions = vec!["Puppeteer".to_string()];
        state.executed = vec![1];
        state.confirmed_evil = vec![1];

        let scenarios = build_scenarios(&state);
        assert!(!scenarios.is_empty());
        assert!(scenarios.iter().all(|scenario| {
            let puppeteer = scenario
                .evil_positions
                .iter()
                .find_map(|(&position, role)| {
                    (normalize_role(role) == "puppeteer").then_some(position)
                })
                .unwrap();
            let puppet = scenario.puppet_position.unwrap();
            adjacent_positions(puppeteer, state.n_cards).contains(&puppet)
        }));
        assert!(scenarios.iter().any(|scenario| {
            scenario
                .evil_positions
                .get(&1)
                .is_some_and(|role| normalize_role(role) == "puppeteer")
        }));
        assert!(scenarios
            .iter()
            .any(|scenario| scenario.puppet_position == Some(1)));

        state.executed = vec![1, 3];
        state.confirmed_evil = vec![1, 3];
        state.executed_evil_roles = HashMap::from([
            (1, "Puppeteer".to_string()),
            (3, "Puppet".to_string()),
        ]);
        assert!(build_scenarios(&state).is_empty());
    }

    #[test]
    fn untyped_twin_death_reconstructs_the_ordered_trace_slice() {
        let mut state = exact_three_evil_twin_state();
        state.confirmed_evil = vec![1, 2, 3];
        state.executed_evil_roles.remove(&1);

        let scenarios = build_scenarios(&state);
        assert_eq!(scenarios.len(), 4);
        assert!(scenarios.iter().all(|scenario| {
            scenario
                .evil_positions
                .get(&1)
                .is_some_and(|role| normalize_role(role) == "twinminion")
                && scenario.twin_trace.is_some()
        }));
    }

    #[test]
    fn exact_role_outside_the_authored_multiset_is_rejected() {
        let mut state = GameState::default();
        state.n_cards = 2;
        state.n_evil = 2;
        state.deck.minions = vec!["Witch".to_string()];
        state.deck.demons = vec!["Pooka".to_string(), "Lilis".to_string()];
        state.board_minion_count = Some(1);
        state.board_demon_count = Some(1);
        state.executed = vec![1];
        state.confirmed_evil = vec![1];
        state
            .executed_evil_roles
            .insert(1, "Shaman".to_string());

        assert!(build_scenarios(&state).is_empty());
    }

    #[test]
    fn puppeteer_candidates_follow_villager_and_one_saint_removal() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.deck.villagers = vec!["Baker".to_string(), "Saint".to_string()];
        state.deck.outcasts = vec!["Plague Doctor".to_string()];
        state.cards = vec![card(2, "Baker"), card(3, "Plague Doctor")];
        let mut real_villagers = HashSet::from([2]);
        assert_eq!(
            native_puppeteer_candidate_occurrences(1, &real_villagers, &state),
            vec![(2, true)],
        );

        state.cards[0].apparent_role = "Saint".to_string();
        assert!(native_puppeteer_candidate_occurrences(1, &real_villagers, &state).is_empty());

        state.n_cards = 2;
        state.cards = vec![card(2, "Saint")];
        real_villagers = HashSet::from([2]);
        assert_eq!(
            native_puppeteer_candidate_occurrences(1, &real_villagers, &state),
            vec![(2, true)],
        );
    }

    #[test]
    fn known_villager_makes_puppeteer_conversion_mandatory() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.n_evil = 2;
        state.deck.villagers = vec!["Scout".to_string()];
        state.deck.outcasts = vec!["Plague Doctor".to_string()];
        state.deck.minions = vec!["Puppeteer".to_string()];
        state.cards = vec![card(2, "Scout"), card(3, "Plague Doctor")];

        let scenarios = build_scenarios(&state);
        let puppeteer_at_one: Vec<_> = scenarios
            .iter()
            .filter(|scenario| {
                scenario
                    .evil_positions
                    .get(&1)
                    .is_some_and(|role| normalize_role(role) == "puppeteer")
            })
            .collect();
        assert!(!puppeteer_at_one.is_empty());
        assert!(puppeteer_at_one
            .iter()
            .all(|scenario| scenario.puppet_position == Some(2)),
            "unexpected Puppeteer worlds: {:?}",
            puppeteer_at_one
                .iter()
                .map(|scenario| (
                    scenario.puppet_position,
                    scenario.doppelganger_position,
                    scenario.drunk_position,
                    scenario.evil_positions.clone(),
                ))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn hidden_drunk_neighbor_allows_native_puppeteer_no_op() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.n_evil = 1;
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        state.deck.villagers = vec!["Baker".to_string()];
        state.deck.outcasts = vec!["Drunk".to_string(), "Plague Doctor".to_string()];
        state.deck.minions = vec!["Puppeteer".to_string()];
        state.cards = vec![card(2, "Baker"), card(3, "Plague Doctor")];

        assert!(build_scenarios(&state).iter().any(|scenario| {
            scenario
                .evil_positions
                .get(&1)
                .is_some_and(|role| normalize_role(role) == "puppeteer")
                && scenario.puppet_position.is_none()
                && scenario.drunk_position == Some(2)
        }));
    }

    #[test]
    fn puppeteer_removes_at_most_one_unknown_saint_occurrence() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.deck.villagers = vec!["Saint".to_string(), "Scout".to_string()];
        let full_evil = HashMap::from([(1, "Puppeteer".to_string())]);
        let context = StartCorruptionContext {
            real_villagers_before_puppet: HashSet::from([2, 3]),
            ..StartCorruptionContext::default()
        };
        assert!(!start_context_matches_native_puppeteer_conversion(
            &full_evil,
            None,
            &context,
            &state,
        ));

        state.n_cards = 2;
        let context = StartCorruptionContext {
            real_villagers_before_puppet: HashSet::from([2]),
            ..StartCorruptionContext::default()
        };
        assert!(!start_context_matches_native_puppeteer_conversion(
            &full_evil,
            None,
            &context,
            &state,
        ));
    }

    #[test]
    fn later_shaman_and_baker_writers_keep_a_sole_saint_no_op_branch() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.deck.villagers = vec!["Saint".to_string(), "Baker".to_string()];
        state.cards = vec![card(2, "Baker"), card(3, "Baker")];
        let context = StartCorruptionContext {
            real_villagers_before_puppet: HashSet::from([2, 3]),
            ..StartCorruptionContext::default()
        };

        let with_shaman = HashMap::from([
            (1, "Puppeteer".to_string()),
            (4, "Shaman".to_string()),
        ]);
        assert!(start_context_matches_native_puppeteer_conversion(
            &with_shaman,
            None,
            &context,
            &state,
        ));

        let after_baker = HashMap::from([
            (1, "Puppeteer".to_string()),
            (4, "Pooka".to_string()),
        ]);
        assert!(start_context_matches_native_puppeteer_conversion(
            &after_baker,
            None,
            &context,
            &state,
        ));

        state.deck.villagers = vec!["Scout".to_string(), "Witness".to_string()];
        state.cards = vec![card(2, "Scout"), card(3, "Witness")];
        assert!(!start_context_matches_native_puppeteer_conversion(
            &after_baker,
            None,
            &context,
            &state,
        ));
    }

    #[test]
    fn mixed_twin_puppeteer_defers_post_swap_adjacency_and_conversion() {
        let mut state = GameState {
            n_cards: 5,
            n_evil: 4,
            board_count_provenance: BoardCountProvenance::TrustedPreStart,
            ..GameState::default()
        };
        state.deck.minions = vec!["Puppeteer".to_string(), "Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        let placement = HashMap::from([
            (1, "Puppeteer".to_string()),
            (2, "Twin Minion".to_string()),
            (3, "Puppet".to_string()),
            (5, "Pooka".to_string()),
        ]);
        assert!(placement_has_valid_puppeteer_structure(
            &placement,
            Some(3),
            &state,
        ));
        assert!(start_context_matches_native_puppeteer_conversion(
            &placement,
            Some(3),
            &StartCorruptionContext::default(),
            &state,
        ));
        assert!(generate_evil_placements(&state).iter().any(|candidate| {
            placement.iter().all(|(position, role)| {
                candidate
                    .get(position)
                    .is_some_and(|actual| normalize_role(actual) == normalize_role(role))
                })
        }));

        let mut overlap_state = state.clone();
        overlap_state.n_evil = 3;
        let overlap_worlds = generate_evil_placements(&overlap_state);
        assert!(overlap_worlds.iter().any(|candidate| {
            candidate.puppet_position == Some(2)
                && candidate
                    .get(&1)
                    .is_some_and(|role| normalize_role(role) == "puppeteer")
                && candidate
                    .get(&2)
                    .is_some_and(|role| normalize_role(role) == "twinminion")
                && candidate
                    .get(&5)
                    .is_some_and(|role| normalize_role(role) == "pooka")
                && candidate
                    .values()
                    .all(|role| normalize_role(role) != "puppet")
        }));
        assert!(build_scenarios(&overlap_state).iter().any(|scenario| {
            scenario.puppet_position == Some(2)
                && scenario
                    .evil_positions
                    .get(&2)
                    .is_some_and(|role| normalize_role(role) == "twinminion")
        }));
        assert!(!placement_has_valid_puppeteer_structure(
            &HashMap::from([
                (1, "Puppeteer".to_string()),
                (2, "Twin Minion".to_string()),
                (5, "Pooka".to_string()),
            ]),
            Some(1),
            &overlap_state,
        ));
        assert!(!placement_has_valid_puppeteer_structure(
            &HashMap::from([
                (1, "Puppeteer".to_string()),
                (2, "Twin Minion".to_string()),
                (5, "Pooka".to_string()),
            ]),
            Some(5),
            &overlap_state,
        ));

        state.executed = vec![1];
        state.confirmed_evil = vec![1];
        state.executed_evil_roles = HashMap::from([(1, "Puppeteer".to_string())]);
        assert!(generate_evil_placements(&state).iter().any(|candidate| {
            candidate
                .get(&2)
                .is_some_and(|role| normalize_role(role) == "twinminion")
                && candidate
                    .get(&3)
                    .is_some_and(|role| normalize_role(role) == "puppet")
                && candidate
                    .get(&5)
                    .is_some_and(|role| normalize_role(role) == "pooka")
        }));

        state.executed = vec![3];
        state.confirmed_evil = vec![3];
        state.executed_evil_roles = HashMap::from([(3, "Puppet".to_string())]);
        assert!(generate_evil_placements(&state).iter().any(|candidate| {
            candidate
                .get(&1)
                .is_some_and(|role| normalize_role(role) == "puppeteer")
                && candidate
                    .get(&2)
                    .is_some_and(|role| normalize_role(role) == "twinminion")
                && candidate
                    .get(&5)
                    .is_some_and(|role| normalize_role(role) == "pooka")
        }));

        state.executed = vec![1, 3];
        state.confirmed_evil = state.executed.clone();
        state.executed_evil_roles = HashMap::from([
            (1, "Puppeteer".to_string()),
            (3, "Puppet".to_string()),
        ]);
        assert!(generate_evil_placements(&state).iter().any(|candidate| {
            candidate
                .get(&2)
                .is_some_and(|role| normalize_role(role) == "twinminion")
                && candidate
                    .get(&5)
                    .is_some_and(|role| normalize_role(role) == "pooka")
        }));

        state.executed = vec![1, 2, 3, 5];
        state.confirmed_evil = state.executed.clone();
        state.executed_evil_roles = placement.clone();
        let all_dead = generate_evil_placements(&state);
        assert_eq!(all_dead.len(), 1);
        assert!(all_dead[0].roles.is_empty());
        assert_eq!(all_dead[0].puppet_position, Some(3));
        assert!(!build_scenarios(&state).is_empty());

        let without_twin = HashMap::from([
            (1, "Puppeteer".to_string()),
            (3, "Puppet".to_string()),
        ]);
        state.executed.clear();
        state.confirmed_evil.clear();
        state.executed_evil_roles.clear();
        assert!(!placement_has_valid_puppeteer_structure(
            &without_twin,
            Some(3),
            &state,
        ));
    }

    #[test]
    fn twin_body_puppet_overlay_survives_each_stable_death_state() {
        let stable = HashMap::from([
            (1, "Puppeteer".to_string()),
            (2, "Twin Minion".to_string()),
            (5, "Pooka".to_string()),
        ]);
        let assert_world = |dead: &[u8]| {
            let mut state = GameState {
                n_cards: 5,
                n_evil: 3,
                board_count_provenance: BoardCountProvenance::TrustedPreStart,
                ..GameState::default()
            };
            state.deck.minions = vec!["Puppeteer".to_string(), "Twin Minion".to_string()];
            state.deck.demons = vec!["Pooka".to_string()];
            state.executed = dead.to_vec();
            state.confirmed_evil = dead.to_vec();
            state.executed_evil_roles = stable
                .iter()
                .filter(|(position, _)| dead.contains(position))
                .map(|(&position, role)| (position, role.clone()))
                .collect();
            if dead.contains(&2) {
                state.executed_current_roles.insert(2, "Puppet".to_string());
            }

            let placements = generate_evil_placements(&state);
            assert!(placements.iter().any(|candidate| {
                candidate.puppet_position == Some(2)
                    && stable.iter().all(|(&position, expected)| {
                        candidate
                            .get(&position)
                            .or_else(|| state.executed_evil_roles.get(&position))
                            .is_some_and(|actual| {
                                normalize_role(actual) == normalize_role(expected)
                            })
                    })
            }), "missing Twin/Puppet overlap with dead positions {dead:?}");
            assert!(build_scenarios(&state).iter().any(|scenario| {
                scenario.puppet_position == Some(2)
                    && scenario
                        .evil_positions
                        .get(&2)
                        .is_some_and(|role| normalize_role(role) == "twinminion")
            }), "scenario lost Twin/Puppet overlap with dead positions {dead:?}");
        };

        assert_world(&[]);
        assert_world(&[1]);
        assert_world(&[2]);
        assert_world(&[1, 2]);
        assert_world(&[1, 2, 5]);
    }

    #[test]
    fn exact_dead_twin_current_role_selects_puppet_overlay() {
        let make_state = |observed: &str| {
            let mut state = GameState {
                n_cards: 5,
                n_evil: 3,
                board_count_provenance: BoardCountProvenance::TrustedPreStart,
                executed: vec![2],
                confirmed_evil: vec![2],
                executed_evil_roles: HashMap::from([(2, "Twin Minion".to_string())]),
                executed_current_roles: HashMap::from([(2, observed.to_string())]),
                ..GameState::default()
            };
            state.deck.minions = vec!["Puppeteer".to_string(), "Twin Minion".to_string()];
            state.deck.demons = vec!["Pooka".to_string()];
            state
        };

        let puppet = generate_evil_placements(&make_state("Puppet"));
        assert!(!puppet.is_empty());
        assert!(puppet
            .iter()
            .all(|candidate| candidate.puppet_position == Some(2)));

        let mut oversized_pool = make_state("Puppet");
        oversized_pool.n_evil = 2;
        oversized_pool.deck.minions = vec![
            "Puppeteer".to_string(),
            "Twin Minion".to_string(),
            "Witch".to_string(),
        ];
        let selected = generate_evil_placements(&oversized_pool);
        assert!(!selected.is_empty());
        assert!(selected.iter().all(|candidate| {
            candidate.puppet_position == Some(2)
                && candidate
                    .values()
                    .chain(oversized_pool.executed_evil_roles.values())
                    .any(|role| normalize_role(role) == "puppeteer")
        }));

        let non_puppet = generate_evil_placements(&make_state("Scout"));
        assert!(!non_puppet.is_empty());
        assert!(non_puppet
            .iter()
            .all(|candidate| candidate.puppet_position != Some(2)));
    }

    #[test]
    fn twin_body_puppet_source_leaves_the_post_twin_pd_villager_pool() {
        let state = GameState::from_json(&serde_json::json!({
            "n_cards": 6,
            "n_evil": 3,
            "board_count_provenance": "trusted_pre_start",
            "board_villager_count": 1,
            "board_outcast_count": 2,
            "board_minion_count": 2,
            "board_demon_count": 1,
            "deck": {
                "villagers": ["Scout"],
                "outcasts": ["Plague Doctor", "Bombardier"],
                "minions": ["Puppeteer", "Twin Minion"],
                "demons": ["Lilis"]
            },
            "cards": [
                {"position": 1, "apparent_role": "Scout"},
                {"position": 2, "apparent_role": "Scout"},
                {"position": 3, "apparent_role": "Plague Doctor"},
                {"position": 4, "apparent_role": "Scout"},
                {"position": 5, "apparent_role": "Scout"},
                {"position": 6, "apparent_role": "Bombardier"}
            ],
            "executed": [1, 2, 4, 5],
            "confirmed_evil": [1, 2, 4],
            "confirmed_good": [5],
            "executed_evil_roles": {
                "1": "Puppeteer",
                "2": "Twin Minion",
                "4": "Lilis"
            },
            "executed_current_roles": {
                "1": "Puppeteer",
                "2": "Puppet",
                "4": "Lilis",
                "5": "Twin Minion"
            },
            "executed_good_roles": {"5": "Twin Minion"},
            "executed_good_corrupted": {"5": false}
        }))
        .unwrap();

        let result = crate::solver::solve(&state);
        assert!(result.n_surviving > 0);
        assert!(result.surviving_scenarios.iter().any(|scenario| {
            scenario.puppet_position == Some(2)
                && scenario
                    .evil_positions
                    .get(&2)
                    .is_some_and(|role| normalize_role(role) == "twinminion")
                && !scenario.corrupted.contains(&5)
        }));
    }

    #[test]
    fn twin_body_puppet_alchemist_source_respects_pool_capacity() {
        let mut state = GameState {
            n_cards: 5,
            ..GameState::default()
        };
        state.cards = vec![card(2, "Alchemist")];
        state.deck.villagers = vec!["Alchemist".to_string()];
        state.deck.minions = vec!["Puppeteer".to_string(), "Twin Minion".to_string()];
        state.deck.demons = vec!["Lilis".to_string()];
        state
            .executed_current_roles
            .insert(3, "Twin Minion".to_string());
        let full_evil = HashMap::from([
            (1, "Puppeteer".to_string()),
            (2, "Twin Minion".to_string()),
            (5, "Lilis".to_string()),
        ]);
        let mut base = StartCorruptionContext {
            real_villagers_before_puppet: HashSet::from([2, 3, 4]),
            registered_villagers_at_pd_call: HashSet::from([3, 4]),
            corruption_resistant_at_init: HashSet::from([4]),
            true_alchemist_positions: vec![4],
            puppet_position: Some(2),
            ..StartCorruptionContext::default()
        };

        assert!(twin_puppet_overlay_start_context_variants(
            &state,
            &full_evil,
            None,
            None,
            Some(2),
            None,
            base.clone(),
        )
        .is_empty());

        base.corruption_resistant_at_init.clear();
        base.true_alchemist_positions.clear();
        let contexts = twin_puppet_overlay_start_context_variants(
            &state,
            &full_evil,
            None,
            None,
            Some(2),
            None,
            base,
        );
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].corruption_resistant_at_init, HashSet::from([3]));
        assert!(contexts[0].true_alchemist_positions.is_empty());
    }

    #[test]
    fn legacy_puppeteer_count_may_exclude_generated_puppet() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.n_evil = 3;
        state.deck.minions = vec!["Puppeteer".to_string(), "Witch".to_string()];
        state.deck.demons = vec!["Lilis".to_string()];
        state.executed = vec![1, 2, 3, 4];
        state.confirmed_evil = state.executed.clone();
        state.executed_evil_roles = HashMap::from([
            (1, "Puppeteer".to_string()),
            (2, "Puppet".to_string()),
            (3, "Witch".to_string()),
            (4, "Lilis".to_string()),
        ]);

        let legacy = generate_evil_placements(&state);
        assert_eq!(legacy.len(), 1);
        assert!(legacy[0].roles.is_empty());
        assert_eq!(legacy[0].puppet_position, Some(2));
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        assert!(generate_evil_placements(&state).is_empty());
    }

    #[test]
    fn trusted_faction_counts_cover_puppeteer_special_placements() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.n_evil = 3;
        state.deck.minions = vec!["Puppeteer".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.board_minion_count = Some(0);
        state.board_demon_count = Some(2);
        assert!(generate_evil_placements(&state).is_empty());

        state.board_minion_count = Some(1);
        state.board_demon_count = Some(1);
        assert!(!generate_evil_placements(&state).is_empty());
    }

    #[test]
    fn apparent_villagers_that_are_authored_evils_do_not_force_a_puppet() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.n_evil = 3;
        state.deck.villagers = vec!["Baker".to_string()];
        state.deck.minions = vec!["Puppeteer".to_string(), "Witch".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.cards = vec![card(1, "Baker"), card(2, "Baker"), card(3, "Baker")];

        assert!(generate_evil_placements(&state).iter().any(|placement| {
            placement.len() == 3
                && placement
                    .values()
                    .all(|role| normalize_role(role) != "puppet")
                && placement
                    .values()
                    .any(|role| normalize_role(role) == "puppeteer")
        }));
    }

    #[test]
    fn oversized_puppeteer_pool_selects_a_valid_authored_subset() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.n_evil = 3;
        state.deck.minions = vec!["Puppeteer".to_string(), "Witch".to_string()];
        state.deck.demons = vec!["Pooka".to_string(), "Lilis".to_string()];
        state.board_minion_count = Some(1);
        state.board_demon_count = Some(1);

        assert!(generate_evil_placements(&state).iter().any(|placement| {
            placement.len() == 3
                && placement
                    .values()
                    .any(|role| normalize_role(role) == "puppeteer")
                && placement
                    .values()
                    .any(|role| normalize_role(role) == "puppet")
                && placement.values().any(|role| {
                    matches!(normalize_role(role).as_str(), "pooka" | "lilis")
                })
        }));
    }

    #[test]
    fn trusted_faction_counts_subtract_already_pruned_executed_roles() {
        let mut state = GameState::default();
        state.deck.minions = vec!["Witch".to_string(), "Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.board_minion_count = Some(1);
        state.board_demon_count = Some(1);
        state
            .executed_evil_roles
            .insert(1, "Witch".to_string());
        let remaining = vec!["Minion".to_string(), "Pooka".to_string()];

        assert_eq!(
            evil_role_subsets_fn(&remaining, &state, 1),
            vec![vec!["Pooka".to_string()]],
        );
    }

    #[test]
    fn trusted_faction_counts_are_not_bypassed_by_an_equal_sized_pool() {
        let mut state = GameState::default();
        state.n_cards = 2;
        state.n_evil = 2;
        state.deck.minions = vec!["Witch".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.board_minion_count = Some(0);
        state.board_demon_count = Some(2);

        assert!(generate_evil_placements(&state).is_empty());
    }

    #[test]
    fn trusted_zero_remaining_counts_reject_an_inconsistent_dead_multiset() {
        let mut state = GameState::default();
        state.n_cards = 1;
        state.n_evil = 1;
        state.deck.minions = vec!["Witch".to_string()];
        state.board_minion_count = Some(0);
        state.board_demon_count = Some(1);
        state.executed = vec![1];
        state.confirmed_evil = vec![1];
        state
            .executed_evil_roles
            .insert(1, "Witch".to_string());

        assert!(generate_evil_placements(&state).is_empty());
    }

    #[test]
    fn ordered_three_card_two_demon_slice_retains_all_four_traces_and_self_swaps() {
        let scenarios = build_scenarios(&exact_three_evil_twin_state());

        assert_eq!(scenarios.len(), 4);
        let indices: Vec<u8> = scenarios.iter().map(|scenario| {
            match &scenario.twin_trace.as_ref().unwrap().outcome {
                TwinStartOutcome::Swap { demon_occurrence_index, .. } => *demon_occurrence_index,
                TwinStartOutcome::NoDemon => unreachable!(),
            }
        }).collect();
        assert_eq!(indices, vec![0, 0, 1, 1]);
        assert_eq!(scenarios.iter().filter(|scenario| matches!(
            scenario.twin_trace.as_ref().map(|trace| &trace.outcome),
            Some(TwinStartOutcome::Swap { neighbor_position: 1, .. })
        )).count(), 2);
        assert!(scenarios.iter().any(|scenario| matches!(
            scenario.twin_trace.as_ref().map(|trace| &trace.outcome),
            Some(TwinStartOutcome::Swap {
                demon_anchor_position: 3,
                neighbor_side: TwinNeighborSide::Previous,
                neighbor_position: 1,
                neighbor_pre_swap_role,
                ..
            }) if neighbor_pre_swap_role == "Twin Minion"
        )));
        assert!(scenarios.iter().all(|scenario| scenario.corrupted.is_empty()));
    }

    #[test]
    fn incomplete_endpoint_aborts_the_ordered_state_and_preserves_legacy_worlds() {
        let mut state = safe_twin_state();
        state.n_cards = 4;
        state.n_evil = 3;

        assert!(build_scenarios_with_start_mode(&state, true).is_none());
        let scenarios = build_scenarios(&state);
        assert!(!scenarios.is_empty());
        assert!(scenarios.iter().all(|scenario| scenario.twin_trace.is_none()));
    }

    #[test]
    fn semantic_key_keeps_explicit_no_demon_distinct_from_legacy_none() {
        let key = |twin_trace| ScenarioSemanticKey {
            corrupted: Vec::new(),
            messed_up_by_evil: Vec::new(),
            pd_target: None,
            alchemist_counts: Vec::new(),
            doppelganger_position: None,
            drunk_position: None,
            chancellor_added: None,
            shaman_trace: None,
            twin_trace,
        };
        let legacy = key(None);
        let explicit = key(Some(TwinTrace {
            actor_position: 1,
            outcome: TwinStartOutcome::NoDemon,
        }));

        assert_ne!(legacy, explicit);
        assert_eq!(HashSet::from([legacy, explicit]).len(), 2);
    }

    #[test]
    fn ordered_twin_slice_gates_every_deferred_current_data_mutator() {
        for gated_role in [
            "Chancellor",
            "Puppeteer",
            "Puppet",
            "Shaman",
            "Plague Doctor",
            "Alchemist",
            "Doppelganger",
            "Drunk",
        ] {
            let mut state = safe_twin_state();
            state.deck.villagers.push(gated_role.to_string());
            assert!(
                !supports_ordered_twin_start_slice(&state),
                "{gated_role} must retain the legacy one-shot path"
            );
        }

        let mut unknown_role = safe_twin_state();
        unknown_role.deck.villagers.push("Future Oracle".to_string());
        assert!(!supports_ordered_twin_start_slice(&unknown_role));

        let state = safe_twin_state();
        let roles = HashMap::from([
            (1, "Twin Minion".to_string()),
            (2, "Unknown".to_string()),
            (3, "Pooka".to_string()),
        ]);
        assert!(exact_pre_twin_structural_roles(&state, &roles).is_none());

        let mut gated_state = exact_three_evil_twin_state();
        gated_state.deck.villagers.push("Alchemist".to_string());
        let legacy = build_scenarios(&gated_state);
        assert!(!legacy.is_empty());
        assert!(legacy.iter().all(|scenario| scenario.twin_trace.is_none()));
    }

    #[test]
    fn no_demon_ordered_helper_emits_one_explicit_trace() {
        let mut state = GameState::default();
        state.n_cards = 2;
        state.deck.minions = vec!["Twin Minion".to_string(), "Witch".to_string()];
        let roles = HashMap::from([
            (1, "Twin Minion".to_string()),
            (2, "Witch".to_string()),
        ]);

        let (width, outcomes) = enumerate_ordered_twin_start_outcomes(
            &state,
            &roles,
            &StartCorruptionContext::default(),
        ).unwrap();

        assert_eq!(width, 1);
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(outcomes[0].1.outcome, TwinStartOutcome::NoDemon));
    }

    #[test]
    fn poisoner_status_branches_cross_every_twin_trace() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.deck.villagers = vec!["Scout".to_string(), "Baker".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string(), "Poisoner".to_string()];
        state.deck.demons = vec!["Lilis".to_string()];
        let current_roles = HashMap::from([
            (1, "Twin Minion".to_string()),
            (2, "Lilis".to_string()),
            (3, "Scout".to_string()),
            (4, "Poisoner".to_string()),
            (5, "Baker".to_string()),
        ]);
        let real_villagers = HashSet::from([3, 5]);
        let context = StartCorruptionContext {
            real_villagers_before_puppet: real_villagers.clone(),
            registered_villagers_at_pd_call: real_villagers,
            ..StartCorruptionContext::default()
        };

        let (width, outcomes) =
            enumerate_ordered_twin_start_outcomes(&state, &current_roles, &context).unwrap();

        assert_eq!(width, 2);
        assert_eq!(outcomes.len(), 4);
        for target in [3, 5] {
            let branch_traces: HashSet<TwinTrace> = outcomes
                .iter()
                .filter(|(outcome, _)| outcome.corrupted == HashSet::from([target]))
                .map(|(_, trace)| trace.clone())
                .collect();
            assert_eq!(branch_traces.len(), 2);
        }
    }

    #[test]
    fn shaman_variants_bind_the_visible_duplicate_to_an_erased_deck_role() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.deck.villagers = vec!["Scout".to_string(), "Witness".to_string()];
        state.cards = vec![card(2, "Scout"), card(3, "Scout")];
        let real = HashSet::from([2, 3]);
        let context = StartCorruptionContext {
            real_villagers_before_puppet: real.clone(),
            registered_villagers_at_pd_call: real,
            ..StartCorruptionContext::default()
        };
        let evil = HashMap::from([(1, "Shaman".to_string())]);

        let variants = shaman_start_context_variants(
            &state, &evil, None, None, None, None, false, context,
        );

        assert_eq!(variants.len(), 2);
        let traces: HashSet<(u8, u8, String, Vec<String>)> = variants
            .into_iter()
            .map(|context| {
                let trace = context.shaman_trace.expect("Shaman world carries a trace");
                (
                    trace.source_position,
                    trace.target_position,
                    normalize_role(&trace.copied_role),
                    trace
                        .target_previous_roles
                        .iter()
                        .map(|role| normalize_role(role))
                        .collect(),
                )
            })
            .collect();
        assert_eq!(
            traces,
            HashSet::from([
                (2, 3, "scout".to_string(), vec!["witness".to_string()]),
                (3, 2, "scout".to_string(), vec!["witness".to_string()]),
            ])
        );
    }

    #[test]
    fn shaman_candidates_exclude_puppet_and_chancellor_generated_outcast() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.deck.villagers = vec!["Scout".to_string(), "Witness".to_string()];
        state.deck.outcasts = vec!["Bombardier".to_string()];
        state.cards = vec![
            card(2, "Scout"),
            card(3, "Scout"),
            card(4, "Scout"),
            card(5, "Bombardier"),
        ];
        let evil = HashMap::from([(1, "Shaman".to_string()), (4, "Puppet".to_string())]);
        let context =
            build_start_corruption_context(&state, &evil, None, None, Some(4), Some(5), &[], false);

        let variants = shaman_start_context_variants(
            &state, &evil, None, None, Some(4), None, false, context,
        );

        assert!(!variants.is_empty());
        assert!(variants.iter().all(|context| {
            let trace = context.shaman_trace.as_ref().unwrap();
            [trace.source_position, trace.target_position]
                .into_iter()
                .all(|position| matches!(position, 2 | 3))
        }));
    }

    #[test]
    fn shaman_can_select_a_hidden_villager_from_mixed_hidden_factions() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.deck.villagers = vec!["Scout".to_string(), "Witness".to_string()];
        state.deck.outcasts = vec!["Plague Doctor".to_string()];
        state.deck.minions = vec!["Shaman".to_string()];
        state.cards = vec![card(2, "Scout")];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        let known = HashSet::from([2]);
        let context = StartCorruptionContext {
            real_villagers_before_puppet: known.clone(),
            registered_villagers_at_pd_call: known,
            plague_doctor_acts: true,
            ..StartCorruptionContext::default()
        };
        let evil = HashMap::from([(1, "Shaman".to_string())]);

        let variants = shaman_start_context_variants(
            &state, &evil, None, None, None, None, true, context,
        );
        let endpoint_sets: HashSet<Vec<u8>> = variants
            .iter()
            .filter_map(|context| context.shaman_trace.as_ref())
            .map(|trace| {
                let mut endpoints = vec![trace.source_position, trace.target_position];
                endpoints.sort_unstable();
                endpoints
            })
            .collect();

        assert!(endpoint_sets.contains(&vec![2, 3]));
        assert!(endpoint_sets.contains(&vec![2, 4]));
        assert!(
            !endpoint_sets.contains(&vec![3, 4]),
            "the exact Outcast budget requires one of the two hidden seats"
        );

        let hidden_three_context = variants
            .iter()
            .find(|context| {
                context.shaman_trace.as_ref().is_some_and(|trace| {
                    HashSet::from([trace.source_position, trace.target_position])
                        == HashSet::from([2, 3])
                })
            })
            .expect("one branch assigns hidden #3 as Shaman-eligible Villager");
        assert!(hidden_three_context
            .registered_villagers_at_pd_call
            .contains(&3));
        let outcomes = enumerate_start_corruption(
            state.n_cards,
            &evil,
            hidden_three_context,
            Some(3),
        );
        assert!(outcomes.iter().any(|outcome| {
            outcome.pd_target == Some(3) && outcome.corrupted.contains(&3)
        }));
    }

    #[test]
    fn plague_doctor_can_select_each_feasible_hidden_villager() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.deck.villagers = vec!["Scout".to_string(), "Witness".to_string()];
        state.deck.outcasts = vec!["Plague Doctor".to_string()];
        state.deck.minions = vec!["Minion".to_string()];
        state.cards = vec![card(2, "Scout")];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        let evil = HashMap::from([(1, "Minion".to_string())]);
        let base = build_start_corruption_context(
            &state, &evil, None, None, None, None, &[], true,
        );

        // The old saturated-only context knew #2 was a Villager but omitted
        // both feasible hidden assignments (#3 V/#4 O and #3 O/#4 V).
        assert_eq!(base.registered_villagers_at_pd_call, HashSet::from([2]));
        let contexts = shaman_start_context_variants(
            &state, &evil, None, None, None, None, true, base,
        );
        let faction_sets: HashSet<Vec<u8>> = contexts
            .iter()
            .map(|context| sorted_positions(&context.registered_villagers_at_pd_call))
            .collect();
        assert_eq!(faction_sets, HashSet::from([vec![2, 3], vec![2, 4]]));

        let pd_targets: HashSet<u8> = contexts
            .iter()
            .flat_map(|context| {
                enumerate_start_corruption(state.n_cards, &evil, context, None)
            })
            .filter_map(|outcome| outcome.pd_target)
            .collect();
        assert_eq!(pd_targets, HashSet::from([2, 3, 4]));
    }

    #[test]
    fn shaman_alchemist_context_keeps_only_source_as_a_later_actor() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.deck.villagers = vec!["Alchemist".to_string(), "Witness".to_string()];
        state.cards = vec![card(2, "Alchemist"), card(3, "Alchemist")];
        let real = HashSet::from([2, 3]);
        let context = StartCorruptionContext {
            real_villagers_before_puppet: real.clone(),
            registered_villagers_at_pd_call: real,
            corruption_resistant_at_init: HashSet::from([2, 3]),
            true_alchemist_positions: vec![2, 3],
            ..StartCorruptionContext::default()
        };
        let evil = HashMap::from([(1, "Shaman".to_string())]);

        let variants = shaman_start_context_variants(
            &state, &evil, None, None, None, None, false, context,
        );

        assert_eq!(variants.len(), 2);
        for context in variants {
            let trace = context.shaman_trace.as_ref().unwrap();
            assert_eq!(
                context.true_alchemist_positions,
                vec![trace.source_position]
            );
            assert!(context
                .corruption_resistant_at_init
                .contains(&trace.source_position));
            assert!(!context
                .corruption_resistant_at_init
                .contains(&trace.target_position));
            assert_eq!(
                trace
                    .target_previous_roles
                    .iter()
                    .map(|role| normalize_role(role))
                    .collect::<Vec<_>>(),
                vec!["witness".to_string()]
            );
        }
    }

    #[test]
    fn shaman_groups_solver_equivalent_erased_roles_without_losing_candidates() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.deck.villagers = vec![
            "Scout".to_string(),
            "Witness".to_string(),
            "Judge".to_string(),
        ];
        let real = HashSet::from([2, 3]);
        let context = StartCorruptionContext {
            real_villagers_before_puppet: real.clone(),
            registered_villagers_at_pd_call: real,
            ..StartCorruptionContext::default()
        };
        let evil = HashMap::from([(1, "Shaman".to_string())]);

        let variants = shaman_start_context_variants(
            &state, &evil, None, None, None, None, false, context,
        );
        let scout_trace = variants
            .iter()
            .filter_map(|context| context.shaman_trace.as_ref())
            .find(|trace| {
                trace.source_position == 2
                    && trace.target_position == 3
                    && normalize_role(&trace.copied_role) == "scout"
            })
            .expect("hidden Scout copy has one grouped non-Alchemist history");

        assert_eq!(
            scout_trace
                .target_previous_roles
                .iter()
                .map(|role| normalize_role(role))
                .collect::<HashSet<_>>(),
            HashSet::from(["witness".to_string(), "judge".to_string()])
        );
    }

    #[test]
    fn shaman_splits_only_solver_visible_preserved_runtime_classes() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.deck.villagers = vec![
            "Scout".to_string(),
            "Witness".to_string(),
            "Judge".to_string(),
            "Baker".to_string(),
            "Alchemist".to_string(),
            "Enlightened".to_string(),
        ];
        let real = HashSet::from([2, 3]);
        let context = StartCorruptionContext {
            real_villagers_before_puppet: real.clone(),
            registered_villagers_at_pd_call: real,
            ..StartCorruptionContext::default()
        };
        let evil = HashMap::from([(1, "Shaman".to_string())]);

        let variants = shaman_start_context_variants(
            &state, &evil, None, None, None, None, false, context,
        );
        let traces: Vec<&ShamanTrace> = variants
            .iter()
            .filter_map(|context| context.shaman_trace.as_ref())
            .filter(|trace| {
                trace.source_position == 2
                    && trace.target_position == 3
                    && normalize_role(&trace.copied_role) == "scout"
            })
            .collect();

        assert_eq!(traces.len(), 2);
        let classes: HashSet<Vec<String>> = traces
            .iter()
            .map(|trace| {
                trace
                    .target_previous_roles
                    .iter()
                    .map(|role| normalize_role(role))
                    .collect()
            })
            .collect();
        assert_eq!(
            classes,
            HashSet::from([
                vec!["alchemist".to_string()],
                vec![
                    "baker".to_string(),
                    "enlightened".to_string(),
                    "judge".to_string(),
                    "witness".to_string(),
                ],
            ]),
        );

        let baker_traces: Vec<&ShamanTrace> = variants
            .iter()
            .filter_map(|context| context.shaman_trace.as_ref())
            .filter(|trace| {
                trace.source_position == 2
                    && trace.target_position == 3
                    && normalize_role(&trace.copied_role) == "baker"
            })
            .collect();
        assert_eq!(baker_traces.len(), 3);
        let baker_classes: HashSet<Vec<String>> = baker_traces
            .iter()
            .map(|trace| {
                trace
                    .target_previous_roles
                    .iter()
                    .map(|role| normalize_role(role))
                    .collect()
            })
            .collect();
        assert_eq!(
            baker_classes,
            HashSet::from([
                vec!["alchemist".to_string()],
                vec!["enlightened".to_string()],
                vec![
                    "judge".to_string(),
                    "scout".to_string(),
                    "witness".to_string(),
                ],
            ]),
        );
    }

    #[test]
    fn pd_presence_uses_the_natural_pool_left_by_each_trace() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.cards = vec![card(1, "Baker"), card(2, "Wretch")];
        state.deck.outcasts = vec!["Plague Doctor".to_string(), "Wretch".to_string()];
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        let full_evil = chancellor_at(1);
        let trace = raw_trace(2, "Wretch", 2);

        state.board_outcast_count = Some(0);
        assert_eq!(
            plague_doctor_act_variants(
                &state, &full_evil, None, None, None, Some(&trace),
            ),
            vec![false],
        );

        state.board_outcast_count = Some(1);
        assert_eq!(
            plague_doctor_act_variants(
                &state, &full_evil, None, None, None, Some(&trace),
            ),
            vec![true],
        );
    }

    #[test]
    fn natural_outcast_hypotheses_share_one_multiset_budget() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.board_outcast_count = Some(1);
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        state.deck.outcasts = vec!["Doppelganger".to_string(), "Drunk".to_string()];

        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            None,
            None,
            None,
            None,
        ));
        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            Some(2),
            None,
            None,
            None,
        ));
        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            None,
            Some(3),
            None,
            None,
        ));
        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            Some(2),
            Some(3),
            None,
            None,
        ));

        let generated_doppelganger = raw_trace(2, "Doppelganger", 2);
        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &chancellor_at(1),
            None,
            None,
            Some(3),
            Some(&generated_doppelganger),
            None,
        ));
        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &chancellor_at(1),
            None,
            Some(2),
            Some(3),
            Some(&generated_doppelganger),
            None,
        ));

        state.board_outcast_count = Some(0);
        assert!(natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            Some(2),
            Some(3),
            None,
            None,
        ));

        state.board_outcast_count = Some(1);
        state.deck.outcasts = vec![
            "Wretch".to_string(),
            "Doppelganger".to_string(),
            "Drunk".to_string(),
        ];
        let generated_wretch = raw_trace(2, "Wretch", 2);
        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &chancellor_at(1),
            None,
            Some(3),
            Some(4),
            Some(&generated_wretch),
            None,
        ));

        state.board_outcast_count = Some(0);
        state.deck.outcasts = vec!["Wretch".to_string()];
        let anchor_is_generated = raw_trace(2, "Wretch", 2);
        let natural_anchor = raw_trace(2, "Wretch", 3);
        assert!(natural_outcast_hypothesis_allows(
            &state,
            &chancellor_at(1),
            None,
            None,
            None,
            Some(&anchor_is_generated),
            None,
        ));
        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &chancellor_at(1),
            None,
            None,
            None,
            Some(&natural_anchor),
            None,
        ));
    }

    #[test]
    fn disguised_outcasts_do_not_consume_trusted_hud_outcast_slots() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.board_outcast_count = Some(1);
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        state.deck.outcasts = vec!["Bombardier".to_string(), "Drunk".to_string()];
        state.cards = vec![card(1, "Bombardier"), card(2, "Knitter")];
        assert!(natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            None,
            Some(2),
            None,
            None,
        ));

        state.deck.outcasts = vec![
            "Plague Doctor".to_string(),
            "Doppelganger".to_string(),
        ];
        state.cards = vec![card(1, "Plague Doctor"), card(2, "Knitter")];
        assert!(natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            Some(2),
            None,
            None,
            None,
        ));
    }

    #[test]
    fn legacy_unknown_header_count_is_a_ceiling_not_an_equality() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.board_outcast_count = Some(2);
        state.deck.outcasts = vec!["Bombardier".to_string(), "Wretch".to_string()];
        state.cards = vec![
            card(1, "Bombardier"),
            card(2, "Knitter"),
            card(3, "Baker"),
        ];

        assert!(natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            None,
            None,
            None,
            None,
        ));
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            None,
            None,
            None,
            None,
        ));
    }

    #[test]
    fn pd_role_only_fact_needs_a_real_anonymous_host() {
        let mut state = GameState::default();
        state.n_cards = 3;
        state.board_outcast_count = Some(1);
        state.deck.outcasts = vec!["Plague Doctor".to_string()];
        state.cards = vec![card(1, "Baker"), card(2, "Baker"), card(3, "Baker")];

        assert!(!natural_outcast_hypothesis_allows(
            &state,
            &HashMap::new(),
            None,
            None,
            None,
            None,
            Some(true),
        ));
    }

    #[test]
    fn hidden_villager_saturation_ignores_natural_drunk_hud_slot() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.board_outcast_count = Some(1);
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        state.deck.outcasts = vec!["Bombardier".to_string(), "Drunk".to_string()];
        let evil = HashMap::from([(1, "Pooka".to_string())]);

        assert!(!unrevealed_must_be_villager(
            2,
            &evil,
            &state,
            None,
            Some(3),
        ));
        state.cards.push(card(4, "Bombardier"));
        assert!(unrevealed_must_be_villager(
            2,
            &evil,
            &state,
            None,
            Some(3),
        ));
    }

    #[test]
    fn trace_saturation_excludes_generated_role_and_natural_drunk() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.board_outcast_count = Some(1);
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        state.deck.outcasts = vec![
            "Bombardier".to_string(),
            "Drunk".to_string(),
            "Wretch".to_string(),
        ];
        let evil = chancellor_at(1);
        let generated_anchor = raw_trace(2, "Bombardier", 2);

        assert!(!trace_unrevealed_must_be_villager(
            4,
            &state,
            &evil,
            None,
            Some(3),
            &generated_anchor,
        ));
        state.cards.push(card(5, "Wretch"));
        assert!(trace_unrevealed_must_be_villager(
            4,
            &state,
            &evil,
            None,
            Some(3),
            &generated_anchor,
        ));

        state.cards.clear();
        let anonymous_natural_anchor = raw_trace(2, "Bombardier", 4);
        assert!(trace_unrevealed_must_be_villager(
            5,
            &state,
            &evil,
            None,
            Some(3),
            &anonymous_natural_anchor,
        ));
        assert!(!trace_unrevealed_must_be_villager(
            4,
            &state,
            &evil,
            None,
            Some(3),
            &anonymous_natural_anchor,
        ));
    }

    #[test]
    fn undealt_chancellor_does_not_add_an_outcast_slot() {
        let mut state = GameState::default();
        state.n_cards = 2;
        state.board_outcast_count = Some(0);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.minions = vec!["Chancellor".to_string()];

        assert!(unrevealed_must_be_villager(
            1,
            &HashMap::new(),
            &state,
            None,
            None,
        ));
    }

    #[test]
    fn chancellor_added_outcast_is_not_restricted_to_adjacent_cards() {
        let mut state = GameState::default();
        state.n_cards = 6;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Plague Doctor"),
            card(3, "Baker"),
            card(4, "Bombardier"),
            card(5, "Baker"),
            card(6, "Baker"),
        ];
        state.board_outcast_count = Some(1);
        state.deck.outcasts = vec!["Plague Doctor".to_string(), "Bombardier".to_string()];

        let candidates = enumerate_raw_chancellor_traces(
            &state,
            &chancellor_at(1),
            None,
            None,
            None,
            1,
        );

        assert!(candidates.iter().any(|trace| trace.added_outcast_position == 2));
        assert!(candidates.iter().any(|trace| trace.added_outcast_position == 4));
        assert!(!candidates.iter().any(|trace| trace.added_outcast_position == 3));
    }

    #[test]
    fn hidden_doppelganger_can_be_the_added_outcast_identity() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Plague Doctor"),
            card(3, "Baker"),
            card(4, "Architect"),
            card(5, "Baker"),
        ];
        state.board_outcast_count = Some(1);
        state.deck.outcasts = vec!["Plague Doctor".to_string(), "Doppelganger".to_string()];

        let candidates = enumerate_raw_chancellor_traces(
            &state,
            &chancellor_at(1),
            None,
            Some(4),
            None,
            1,
        );

        assert!(candidates.iter().any(|trace| {
            trace.added_outcast_position == 4
                && normalize_role(&trace.added_outcast_role) == "doppelganger"
        }));
    }

    #[test]
    fn final_chancellor_requires_a_real_outcast_neighbor() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Baker"),
            card(3, "Bombardier"),
            card(4, "Baker"),
            card(5, "Baker"),
        ];
        state.board_outcast_count = Some(0);
        state.deck.outcasts = vec!["Bombardier".to_string()];

        let candidates = enumerate_raw_chancellor_traces(
            &state,
            &chancellor_at(1),
            None,
            None,
            None,
            1,
        );

        assert!(candidates.is_empty());
    }

    #[test]
    fn chancellor_generated_plague_doctor_still_acts_later_in_start() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.n_evil = 1;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Plague Doctor"),
            card(3, "Baker"),
            card(4, "Architect"),
        ];
        state.deck.villagers = vec!["Baker".to_string(), "Architect".to_string()];
        state.deck.outcasts = vec!["Plague Doctor".to_string()];
        state.deck.minions = vec!["Chancellor".to_string()];
        state.board_villager_count = Some(3);
        state.board_outcast_count = Some(0);

        let generated_pd_worlds: Vec<Scenario> = build_scenarios(&state)
            .into_iter()
            .filter(|scenario| {
                scenario.evil_positions.get(&1).map(String::as_str) == Some("Chancellor")
                    && scenario.chancellor_added_outcast_position() == Some(2)
            })
            .collect();

        assert!(!generated_pd_worlds.is_empty());
        assert!(generated_pd_worlds
            .iter()
            .all(|scenario| scenario.pd_corrupted.is_none()
                && scenario.corrupted.len() == 1));
    }

    #[test]
    fn equivalent_original_chancellor_seats_do_not_multiply_world_weight() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.n_evil = 1;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Plague Doctor"),
            card(3, "Baker"),
            card(4, "Architect"),
        ];
        state.deck.villagers = vec!["Baker".to_string(), "Architect".to_string()];
        state.deck.outcasts = vec!["Plague Doctor".to_string()];
        state.deck.minions = vec!["Chancellor".to_string()];
        state.board_villager_count = Some(3);
        state.board_outcast_count = Some(0);

        let worlds: Vec<Scenario> = build_scenarios(&state)
            .into_iter()
            .filter(|scenario| {
                scenario.evil_positions.get(&1).map(String::as_str) == Some("Chancellor")
                    && scenario.chancellor_added_outcast_position() == Some(2)
            })
            .collect();

        // The generated Plague Doctor has two observable random targets. The
        // three compatible original Chancellor seats are history aliases of
        // each outcome, not six weighted solver worlds.
        assert_eq!(worlds.len(), 2);
        let corrupted_targets: HashSet<Vec<u8>> = worlds
            .iter()
            .map(|scenario| sorted_positions(&scenario.corrupted))
            .collect();
        assert_eq!(corrupted_targets, HashSet::from([vec![3], vec![4]]));
        assert!(worlds
            .iter()
            .all(|scenario| scenario.pd_corrupted.is_none()));
        for scenario in worlds {
            assert_eq!(scenario.chancellor_original_villager_positions(), vec![2]);
            let trace = scenario.chancellor_trace.expect("new worlds carry a trace");
            assert_eq!(trace.original_positions, vec![1, 3, 4]);
            assert_eq!(trace.added_outcast_role, "Plague Doctor");
            assert_eq!(trace.affected_anchor_positions, vec![2]);
            assert_eq!(scenario.chancellor_conversion, Some(2));
        }
    }

    #[test]
    fn converged_chancellor_histories_merge_sorted_anchor_candidates() {
        let mut original_positions = vec![4, 2];
        let mut affected_anchor_positions = vec![5, 3];

        // These helpers run only after complete scenario semantic keys collide:
        // source provenance is not stored by status 50, so equivalent histories
        // stay one solver world rather than gaining probability mass.
        merge_position_candidates(&mut original_positions, &[2, 1]);
        merge_position_candidates(&mut affected_anchor_positions, &[5, 4, 3]);

        assert_eq!(original_positions, vec![1, 2, 4]);
        assert_eq!(affected_anchor_positions, vec![3, 4, 5]);
    }

    #[test]
    fn cured_hidden_pd_targets_collapse_but_a_known_target_is_preserved() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.n_evil = 1;
        state.confirmed_evil = vec![1];
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Plague Doctor"),
            card(3, "Alchemist"),
            card(4, "Scout"),
            card(5, "Witness"),
        ];
        state.deck.villagers = vec![
            "Alchemist".to_string(),
            "Scout".to_string(),
            "Witness".to_string(),
        ];
        state.deck.outcasts = vec!["Plague Doctor".to_string()];
        state.deck.minions = vec!["Minion".to_string()];
        state.board_villager_count = Some(3);
        state.board_outcast_count = Some(1);
        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;

        let evil = HashMap::from([(1, "Minion".to_string())]);
        let context = build_start_corruption_context(
            &state, &evil, None, None, None, None, &[], true,
        );
        let hidden_histories = enumerate_start_corruption(
            state.n_cards, &evil, &context, None,
        );
        assert_eq!(
            hidden_histories
                .iter()
                .filter_map(|outcome| outcome.pd_target)
                .collect::<HashSet<_>>(),
            HashSet::from([4, 5]),
        );
        assert!(hidden_histories.iter().all(|outcome| {
            outcome.corrupted.is_empty()
                && outcome.alchemist_counts.get(&3) == Some(&1)
        }));

        let hidden_target_worlds: Vec<Scenario> = build_scenarios(&state)
            .into_iter()
            .filter(|scenario| {
                scenario.evil_positions.get(&1).map(String::as_str) == Some("Minion")
            })
            .collect();
        assert_eq!(hidden_target_worlds.len(), 1);
        assert!(hidden_target_worlds[0].corrupted.is_empty());
        assert_eq!(hidden_target_worlds[0].alchemist_cures.get(&3), Some(&1));
        assert_eq!(hidden_target_worlds[0].pd_corrupted, None);

        state.pd_corruption_target = Some(4);
        let known_target_worlds: Vec<Scenario> = build_scenarios(&state)
            .into_iter()
            .filter(|scenario| {
                scenario.evil_positions.get(&1).map(String::as_str) == Some("Minion")
            })
            .collect();
        assert_eq!(known_target_worlds.len(), 1);
        assert!(known_target_worlds[0].corrupted.is_empty());
        assert_eq!(known_target_worlds[0].alchemist_cures.get(&3), Some(&1));
        assert_eq!(known_target_worlds[0].pd_corrupted, Some(4));
    }

    #[test]
    fn puppeteer_target_cannot_be_chancellors_added_outcast() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Bombardier"),
            card(3, "Baker"),
            card(4, "Baker"),
            card(5, "Baker"),
        ];
        state.deck.outcasts = vec!["Bombardier".to_string()];
        state.board_outcast_count = Some(0);
        let full_evil = HashMap::from([
            (1, "Chancellor".to_string()),
            (4, "Puppet".to_string()),
        ]);

        let traces = enumerate_raw_chancellor_traces(
            &state,
            &full_evil,
            Some(4),
            None,
            None,
            1,
        );

        assert!(!traces.is_empty());
        assert!(traces
            .iter()
            .all(|trace| trace.added_outcast_position != 4));
    }

    #[test]
    fn baa_cannot_invent_a_role_absent_from_authoritative_pool() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Wretch"),
            card(4, "Baker"),
            card(5, "Baker"),
        ];
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.demons = vec!["Baa".to_string()];
        state.board_outcast_count = Some(1);
        let full_evil = HashMap::from([
            (1, "Chancellor".to_string()),
            (5, "Baa".to_string()),
        ]);

        let traces = enumerate_raw_chancellor_traces(
            &state,
            &full_evil,
            None,
            None,
            Some(3),
            1,
        );

        assert!(!traces.iter().any(|trace| {
            normalize_role(&trace.added_outcast_role) == "drunk"
        }));
        assert_eq!(
            hidden_outcast_presence_flags("Drunk", &state, true),
            (false, true),
        );
    }

    #[test]
    fn authoritative_pool_is_independent_of_whether_baa_was_placed() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Wretch"),
            card(4, "Baker"),
            card(5, "Baker"),
        ];
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.demons = vec!["Baa".to_string()];
        state.board_outcast_count = Some(1);

        let traces = enumerate_raw_chancellor_traces(
            &state,
            &chancellor_at(1),
            None,
            None,
            Some(3),
            1,
        );

        assert!(!traces.iter().any(|trace| {
            normalize_role(&trace.added_outcast_role) == "drunk"
        }));
    }

    #[test]
    fn absent_doppelganger_and_drunk_cannot_supply_chancellor_role() {
        let mut state = GameState::default();
        state.n_cards = 6;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Wretch"),
            card(3, "Baker"),
            card(4, "Baker"),
            card(5, "Baker"),
            card(6, "Baker"),
        ];
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.demons = vec!["Baa".to_string()];
        state.board_outcast_count = Some(2);
        let full_evil = HashMap::from([
            (1, "Chancellor".to_string()),
            (6, "Baa".to_string()),
        ]);

        let traces = enumerate_raw_chancellor_traces(
            &state,
            &full_evil,
            None,
            Some(3),
            Some(4),
            1,
        );

        assert!(traces.is_empty());
    }

    #[test]
    fn alchemist_data_moves_but_its_resistance_stays_on_physical_f() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Bombardier"),
            card(3, "Alchemist"),
            card(4, "Baker"),
            card(5, "Baker"),
        ];
        state.deck.villagers = vec!["Alchemist".to_string()];
        state.deck.outcasts = vec!["Bombardier".to_string()];
        let full_evil = chancellor_at(1);
        let trace = RawChancellorTrace {
            original_position: 3,
            added_outcast_position: 2,
            added_outcast_role: "Bombardier".to_string(),
            anchor_position: 2,
        };

        let contexts = build_chancellor_start_context_variants(
            &state,
            &full_evil,
            None,
            None,
            None,
            1,
            &trace,
            false,
        );

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].corruption_resistant_at_init, HashSet::from([1]));
        assert_eq!(contexts[0].true_alchemist_positions, vec![3]);
        assert!(!contexts[0].corruption_resistant_at_init.contains(&3));
    }

    #[test]
    fn puppet_former_displayed_identity_determines_alchemist_resistance() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.cards = vec![card(2, "Alchemist"), card(3, "Baker")];
        let trace = raw_trace(4, "Wretch", 4);
        let full_evil = HashMap::from([
            (1, "Chancellor".to_string()),
            (2, "Puppet".to_string()),
        ]);

        assert_eq!(
            initial_alchemist_constraint(
                2, &state, &full_evil, None, None, Some(2), 1, &trace, 4,
            ),
            InitialAlchemistConstraint::Required,
        );
        assert_eq!(
            initial_alchemist_constraint(
                3, &state, &full_evil, None, None, Some(3), 1, &trace, 4,
            ),
            InitialAlchemistConstraint::Never,
        );
        state.cards.retain(|card| card.position != 2);
        assert_eq!(
            initial_alchemist_constraint(
                2, &state, &full_evil, None, None, Some(2), 1, &trace, 4,
            ),
            InitialAlchemistConstraint::Maybe,
        );
    }

    #[test]
    fn erased_alchemist_resistance_can_block_generated_drunk_at_v() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Baker"),
            card(3, "Baker"),
            card(4, "Baker"),
            card(5, "Baker"),
        ];
        state.deck.villagers = vec!["Alchemist".to_string()];
        state.deck.outcasts = vec!["Drunk".to_string()];
        let full_evil = chancellor_at(1);
        let trace = RawChancellorTrace {
            original_position: 1,
            added_outcast_position: 2,
            added_outcast_role: "Drunk".to_string(),
            anchor_position: 2,
        };

        let contexts = build_chancellor_start_context_variants(
            &state,
            &full_evil,
            None,
            Some(2),
            None,
            1,
            &trace,
            false,
        );
        let resistant = contexts
            .iter()
            .find(|context| context.corruption_resistant_at_init.contains(&2))
            .expect("the erased Villager may have been Alchemist");

        assert!(resistant.true_alchemist_positions.is_empty());
        let outcomes = enumerate_start_corruption(5, &full_evil, resistant, None);
        assert_eq!(outcomes.len(), 1);
        assert!(!outcomes[0].corrupted.contains(&2));
        assert!(outcomes[0].messed_up_by_evil.contains(&2));
    }

    #[test]
    fn c_equals_a_keeps_erased_f_resistance_on_f_not_generated_drunk() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Baker"),
            card(3, "Baker"),
            card(4, "Baker"),
            card(5, "Wretch"),
        ];
        state.deck.villagers = vec!["Alchemist".to_string()];
        state.deck.outcasts = vec!["Drunk".to_string(), "Wretch".to_string()];
        let full_evil = chancellor_at(1);
        let trace = RawChancellorTrace {
            original_position: 2,
            added_outcast_position: 2,
            added_outcast_role: "Drunk".to_string(),
            anchor_position: 5,
        };

        let contexts = build_chancellor_start_context_variants(
            &state,
            &full_evil,
            None,
            Some(2),
            None,
            1,
            &trace,
            false,
        );
        let erased_f_alchemist = contexts
            .iter()
            .find(|context| context.corruption_resistant_at_init.contains(&1))
            .expect("the first Villager target f may have been Alchemist");

        assert!(!erased_f_alchemist
            .corruption_resistant_at_init
            .contains(&2));
        assert!(erased_f_alchemist.true_alchemist_positions.is_empty());
        let outcomes = enumerate_start_corruption(5, &full_evil, erased_f_alchemist, None);
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].corrupted.contains(&2));
        assert!(outcomes[0].messed_up_by_evil.contains(&5));
    }

    #[test]
    fn asc74_v1_true_chancellor_trace_survives_generation() {
        let mut state = GameState::default();
        state.n_cards = 8;
        state.n_evil = 2;
        state.cards = vec![
            card(1, "Poet"),
            card(2, "Rambler"),
            card(3, "Confessor"),
            card(4, "Empress"),
            card(5, "Bishop"),
            card(6, "Bombardier"),
            card(8, "Plague_Doctor"),
        ];
        state.executed = vec![7];
        state.night_kills = vec![7];
        state.board_villager_count = Some(5);
        state.board_outcast_count = Some(1);
        state.deck.villagers = vec![
            "Bishop".to_string(),
            "Judge".to_string(),
            "Empress".to_string(),
            "Confessor".to_string(),
            "Poet".to_string(),
            "Rambler".to_string(),
        ];
        state.deck.outcasts = vec![
            "Wretch".to_string(),
            "Plague Doctor".to_string(),
            "Bombardier".to_string(),
        ];
        state.deck.minions = vec!["Chancellor".to_string()];
        state.deck.demons = vec!["Lilis".to_string()];

        let full_evil = HashMap::from([
            (4, "Lilis".to_string()),
            (5, "Chancellor".to_string()),
        ]);
        let traces = enumerate_raw_chancellor_traces(
            &state,
            &full_evil,
            None,
            None,
            None,
            5,
        );
        assert!(added_outcast_matches_final_position(
            &state,
            6,
            "Bombardier",
            None,
            None,
        ));
        assert!(!final_board_has_another_true_role(
            &state,
            &full_evil,
            None,
            None,
            None,
            6,
            "Bombardier",
        ));
        assert!(can_be_final_outcast_anchor(
            &state,
            &full_evil,
            None,
            None,
            None,
            6,
            6,
        ));
        let trace = RawChancellorTrace {
            original_position: 5,
            added_outcast_position: 6,
            added_outcast_role: "Bombardier".to_string(),
            anchor_position: 6,
        };
        assert!(natural_outcast_hypothesis_allows(
            &state,
            &full_evil,
            None,
            None,
            None,
            Some(&trace),
            None,
        ));
        assert!(traces.iter().any(|trace| {
            trace.added_outcast_position == 6
                && trace.anchor_position == 6
                && normalize_role(&trace.added_outcast_role) == "bombardier"
        }));

        assert!(build_scenarios(&state).into_iter().any(|scenario| {
            scenario.evil_positions == full_evil
                && scenario.chancellor_added_outcast_position() == Some(6)
                && scenario.chancellor_added_outcast_role() == Some("Bombardier")
        }));
    }
}
