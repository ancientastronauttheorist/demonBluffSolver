/// Scenario generation: enumerate all possible evil placements and build
/// full scenarios with corruption variants.

use std::collections::{HashMap, HashSet};
use crate::corruption::{enumerate_start_corruption, StartCorruptionContext};
use crate::geometry::adjacent_positions;
use crate::knowledge_base::{
    get_card, normalize_role, shaman_erased_role_class,
    BakerPreservedRuntimeClass, Faction,
};
use crate::twin::enumerate_twin_traces;
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
    let placements = generate_evil_placements(state);
    let mut scenarios = Vec::new();
    let n = state.n_cards;

    for placement in &placements {
        if !apply_placement_constraints(placement, state) {
            continue;
        }

        // Find Puppet position
        let mut puppet_pos: Option<u8> = placement.iter()
            .find(|(_, r)| r.as_str() == "Puppet")
            .map(|(&p, _)| p);
        if puppet_pos.is_none() {
            puppet_pos = state.executed_evil_roles.iter()
                .find(|(_, r)| r.as_str() == "Puppet")
                .map(|(&p, _)| p);
        }

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

        // `full_evil` is the final post-Start role map. Chancellor's original
        // physical seat is a hidden history variable, not the final evil seat.
        let final_chancellor_positions: Vec<u8> = full_evil.iter()
            .filter(|(_, role)| role.as_str() == "Chancellor")
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
                            ) {
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
                    let outcomes = enumerate_start_corruption(
                        n,
                        &full_evil,
                        &pending.context,
                        state.pd_corruption_target,
                    );
                    for outcome in outcomes {
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
                            twin_trace: None,
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

    populate_safe_twin_traces(state, scenarios)
}

/// Populate exact initial Twin swap events only where their pre-swap identities
/// have structural provenance independent of post-Start presentation.
///
/// This is deliberately an atomic postprocessing step. Every already-deduped
/// base scenario must provide the same number of complete native outcomes; one
/// incomplete or unsafe world leaves the entire vector unchanged. Consequently
/// validators still see uniform clones of their prior worlds and no incomplete
/// event is mistaken for a generated no-op. Presence is not yet a capability
/// signal for later Start effects, validators, or live play.
fn populate_safe_twin_traces(state: &GameState, scenarios: Vec<Scenario>) -> Vec<Scenario> {
    if scenarios.is_empty()
        || state.n_cards == 0
        || !state
            .deck
            .minions
            .iter()
            .any(|role| normalize_role(role) == "twinminion")
        || state.deck.minions.iter().any(|role| {
            matches!(
                normalize_role(role).as_str(),
                "puppeteer" | "puppet" | "shaman" | "chancellor"
            )
        })
        || state
            .deck
            .outcasts
            .iter()
            .any(|role| {
                matches!(
                    normalize_role(role).as_str(),
                    "plaguedoctor" | "doppelganger" | "drunk"
                )
            })
        || state
            .deck
            .villagers
            .iter()
            .any(|role| normalize_role(role) == "alchemist")
        || scenarios.iter().any(|scenario| scenario.twin_trace.is_some())
    {
        return scenarios;
    }

    // Initial ManageCharacters construction and the ordinary Start scanner use
    // descending displayed-ID order. At Start every physical card is alive, so
    // the alive circular ring has the same order even when this saved state now
    // contains later deaths.
    let current_order: Vec<u8> = (1..=state.n_cards).rev().collect();
    let alive_order = &current_order;
    let mut all_traces: Vec<Vec<TwinTrace>> = Vec::with_capacity(scenarios.len());
    let mut common_width = None;

    for scenario in &scenarios {
        let Some(current_roles) = exact_pre_twin_structural_roles(state, scenario) else {
            return scenarios;
        };
        if !current_roles
            .values()
            .any(|role| normalize_role(role) == "twinminion")
        {
            return scenarios;
        }

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

        // The pure enumerator intentionally skips malformed/missing neighbor
        // paths. Check structural completeness first so such a skip can never
        // turn a partial set into a serialized exact trace set.
        for demon_position in &demon_positions {
            let anchor_index = current_order
                .iter()
                .position(|position| position == demon_position)
                .expect("Demon position came from current_order");
            let previous = current_order
                [(anchor_index + current_order.len() - 1) % current_order.len()];
            let next = current_order[(anchor_index + 1) % current_order.len()];
            if !current_roles.contains_key(&previous) || !current_roles.contains_key(&next) {
                return scenarios;
            }
        }

        let traces = enumerate_twin_traces(&current_roles, &current_order, alive_order);
        let expected_width = if demon_positions.is_empty() {
            1
        } else {
            demon_positions.len() * 2
        };
        if traces.is_empty() || traces.len() != expected_width {
            return scenarios;
        }
        if common_width.is_some_and(|width| width != traces.len()) {
            return scenarios;
        }
        common_width = Some(traces.len());
        all_traces.push(traces);
    }

    let width = common_width.expect("nonempty scenarios produced a trace width");
    let mut expanded = Vec::with_capacity(scenarios.len() * width);
    for (scenario, traces) in scenarios.into_iter().zip(all_traces) {
        for twin_trace in traces {
            let mut traced = scenario.clone();
            traced.twin_trace = Some(twin_trace);
            expanded.push(traced);
        }
    }
    expanded
}

/// Exact structural current-role facts at Twin's ordered Start slot.
///
/// Final apparent card roles are intentionally excluded: on Twin boards they
/// can be post-swap bluffs or presentations and cannot identify the former
/// neighbor data. This first slice therefore admits only exact, non-Unknown
/// `evil_positions`; generated/hidden Good identities remain presentation-
/// derived in the existing scenario builder and are rejected wholesale.
fn exact_pre_twin_structural_roles(
    state: &GameState,
    scenario: &Scenario,
) -> Option<HashMap<u8, String>> {
    let unsafe_role = |role: &str| {
        matches!(
            normalize_role(role).as_str(),
            "unknown"
                | "none"
                | "?"
                | "puppeteer"
                | "puppet"
                | "shaman"
                | "chancellor"
                | "doppelganger"
                | "drunk"
        )
    };
    if scenario.puppet_position.is_some()
        || scenario.shaman_trace.is_some()
        || scenario.pd_corrupted.is_some()
        || !scenario.alchemist_cures.is_empty()
        || scenario.chancellor_trace.is_some()
        || scenario.chancellor_conversion.is_some()
        || scenario.doppelganger_position.is_some()
        || scenario.drunk_position.is_some()
    {
        return None;
    }

    let mut roles: HashMap<u8, String> = HashMap::new();
    let mut insert = |position: u8, role: &str| -> bool {
        if position == 0 || position > state.n_cards || role.trim().is_empty() || unsafe_role(role) {
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

    for (&position, role) in &scenario.evil_positions {
        if !insert(position, role) {
            return None;
        }
    }

    Some(roles)
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
    forbidden_rambler_positions: &HashSet<u8>,
    exact_ordinary_outcasts: Option<&HashSet<u8>>,
) -> bool {
    let doppelganger_role = normalize_role("Doppelganger");
    let drunk_role = normalize_role("Drunk");
    let pd_role = normalize_role("Plague Doctor");

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
        normalize_role(role) == "rambler" && forbidden_rambler_positions.contains(position)
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
            if role == "rambler" && forbidden_rambler_positions.contains(&position) {
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

    // Normal adjacent clues under the current Rambler2 capture contract prove
    // that a matching callback was not installed at the neighboring source.
    // When ordinary hidden identities are collapsed, the natural Rambler can
    // still be forced into one of those seats by the exact O header. Preserve
    // the physical-seat restriction instead of treating the remaining pool as
    // an unpositioned bag.
    let rambler_role = normalize_role("Rambler");
    let available_rambler_fillers = pool.get(&rambler_role).copied().unwrap_or(0);
    let non_rambler_filler_capacity = filler_capacity - available_rambler_fillers;
    let minimum_rambler_fillers = filler_needed.saturating_sub(non_rambler_filler_capacity);
    if minimum_rambler_fillers == 0 {
        return true;
    }

    let maximum_allowed_selected = if let Some(exact) = exact_ordinary_outcasts {
        exact
            .iter()
            .filter(|position| !fixed_ordinary_positions.contains(position))
            .filter(|position| !forbidden_rambler_positions.contains(position))
            .count()
    } else {
        let required_allowed = required_anonymous_positions
            .iter()
            .filter(|position| !forbidden_rambler_positions.contains(position))
            .count();
        let optional_slots = anonymous_needed - required_anonymous_positions.len();
        let optional_allowed = anonymous_hosts
            .iter()
            .filter(|position| !required_anonymous_positions.contains(position))
            .filter(|position| !forbidden_rambler_positions.contains(position))
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
    minimum_rambler_fillers <= maximum_allowed_filler_seats
}

/// Whether one of the ordinary-good identity assignments grouped into a
/// scenario can put a particular natural Outcast role at every listed
/// still-anonymous physical seat simultaneously.
///
/// Scenario generation intentionally collapses Villager/ordinary-Outcast
/// placements once they have identical represented Start consequences. Public
/// Rambler interference can nevertheless prove that hidden seats had real
/// Rambler data. Re-run the same joint multiset/header checks here rather than
/// validating each existential against the same single pool occurrence.
pub(crate) fn scenario_allows_anonymous_natural_outcast_role_assignments(
    positions: &HashSet<u8>,
    role: &str,
    forbidden_rambler_positions: &HashSet<u8>,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if positions.is_empty() && forbidden_rambler_positions.is_empty() {
        return true;
    }
    if !is_state_outcast_role(role, state) || is_hud_villager_outcast(role) {
        return positions.is_empty();
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
            forbidden_rambler_positions,
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
            // it, so it cannot simultaneously host one of these natural
            // Rambler identities. Grouped traces remain valid when any other
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitialAlchemistConstraint {
    Never,
    Maybe,
    Required,
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
            let is_real_villager = if puppet_position == Some(position) {
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
        if puppet_position == Some(trace.original_position) {
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

    if puppet_position == Some(position) {
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
    let mut real_villagers_before_puppet = HashSet::new();

    for position in 1..=state.n_cards {
        let is_real_villager = if Some(position) == chancellor_conversion {
            false // Chancellor has already replaced this data with an Outcast.
        } else if Some(position) == puppet_position {
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
        if !full_evil.contains_key(&card.position) || Some(card.position) == puppet_position {
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
                || Some(card.position) == puppet_position)
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

fn generate_evil_placements(state: &GameState) -> Vec<HashMap<u8, String>> {
    let n = state.n_cards;
    let mut evil_roles: Vec<String> = state.deck.evil_roles();

    let puppet_in_deck = evil_roles.iter().any(|r| r == "Puppeteer");
    if puppet_in_deck {
        if let Some(idx) = evil_roles.iter().position(|r| r == "Puppet") {
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

    // Unknown executed evils
    let executed_evil_without_role: Vec<u8> = state.confirmed_evil.iter()
        .filter(|&&p| state.executed.contains(&p) && !state.executed_evil_roles.contains_key(&p))
        .copied().collect();

    let n_to_remove = executed_evil_without_role.len().min(remaining.len());
    let possible_remaining_lists = if n_to_remove > 0 {
        let mut seen_keys: HashSet<Vec<String>> = HashSet::new();
        let mut lists = Vec::new();
        for removal in combinations_indices(remaining.len(), n_to_remove) {
            let kept: Vec<String> = remaining.iter().enumerate()
                .filter(|(i, _)| !removal.contains(i))
                .map(|(_, r)| r.clone()).collect();
            let mut key = kept.clone();
            key.sort();
            if seen_keys.insert(key) {
                lists.push(kept);
            }
        }
        lists
    } else {
        vec![remaining]
    };

    let n_executed_evil = state.executed_evil_roles.len() + executed_evil_without_role.len();
    let expected_remaining = state.n_evil as i32 - n_executed_evil as i32;

    let valid_sizes: HashSet<usize> = if puppet_in_deck {
        [expected_remaining.max(0) as usize, (expected_remaining + 1).max(0) as usize].into()
    } else {
        [expected_remaining.max(0) as usize].into()
    };

    let night_kills_set: HashSet<u8> = state.night_kills.iter().copied().collect();
    let player_executed: HashSet<u8> = state.executed.iter()
        .filter(|p| !night_kills_set.contains(p))
        .copied().collect();
    let confirmed_good_set: HashSet<u8> = state.confirmed_good.iter().copied().collect();
    let available: Vec<u8> = (1..=n)
        .filter(|p| !player_executed.contains(p) && !confirmed_good_set.contains(p))
        .collect();

    // Check Puppeteer/Puppet execution status
    let puppeteer_executed_pos: Option<u8> = state.executed_evil_roles.iter()
        .find(|(_, r)| r.as_str() == "Puppeteer").map(|(&p, _)| p);
    let puppet_executed_pos: Option<u8> = state.executed_evil_roles.iter()
        .find(|(_, r)| r.as_str() == "Puppet").map(|(&p, _)| p);

    let mut all_placements: Vec<HashMap<u8, String>> = Vec::new();
    let mut seen_placements: HashSet<Vec<(u8, String)>> = HashSet::new();

    let mut add_placement = |p: &HashMap<u8, String>| {
        if valid_sizes.contains(&p.len()) {
            let mut key: Vec<(u8, String)> = p.iter().map(|(&k, v)| (k, v.clone())).collect();
            key.sort_by_key(|(k, _)| *k);
            if seen_placements.insert(key) {
                all_placements.push(p.clone());
            }
        }
    };

    for evil_roles in &possible_remaining_lists {
        let has_puppeteer = evil_roles.iter().any(|r| r == "Puppeteer");

        // Case: Puppeteer executed, Puppet still alive
        let puppet_still_alive = puppeteer_executed_pos.is_some()
            && !has_puppeteer
            && !state.executed_evil_roles.values().any(|r| r == "Puppet");

        if puppet_still_alive {
            let pep = puppeteer_executed_pos.unwrap();
            let puppet_cands: Vec<u8> = adjacent_positions(pep, n).into_iter()
                .filter(|a| available.contains(a)).collect();
            // Case 1: Puppet at adjacent
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
            // Case 2: No Puppet
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

        // Case: Puppet executed, Puppeteer must be adjacent
        if puppet_executed_pos.is_some() && has_puppeteer {
            let pxp = puppet_executed_pos.unwrap();
            let base_evil: Vec<String> = evil_roles.iter().filter(|r| r.as_str() != "Puppeteer").cloned().collect();
            let puppeteer_cands: Vec<u8> = adjacent_positions(pxp, n).into_iter()
                .filter(|a| available.contains(a)).collect();
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
            let base_evil: Vec<String> = evil_roles.iter().filter(|r| r.as_str() != "Puppeteer").cloned().collect();
            for &puppeteer_pos in &available {
                let adj = adjacent_positions(puppeteer_pos, n);
                let puppet_cands: Vec<u8> = adj.iter()
                    .filter(|&&a| available.contains(&a) && a != puppeteer_pos)
                    .copied().collect();

                // Filter to Villager-or-unknown
                let mut villager_or_unknown = Vec::new();
                let mut has_known_villager = false;
                for &pc in &puppet_cands {
                    if let Some(card) = state.card_at(pc) {
                        if is_state_villager_role(&card.apparent_role, state) {
                            villager_or_unknown.push(pc);
                            has_known_villager = true;
                        }
                    } else {
                        villager_or_unknown.push(pc);
                    }
                }
                let actual_puppet_cands = if has_known_villager { &villager_or_unknown } else { &puppet_cands };

                // Case 1: Puppet created
                for &puppet_p in actual_puppet_cands {
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
                // Case 2: Puppet NOT created
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

        // No Puppeteer — straightforward combinations
        let n_evil_count = evil_roles.len() as i32;
        let evil_role_subsets: Vec<Vec<String>> = if n_evil_count > expected_remaining && expected_remaining > 0 {
            evil_role_subsets_fn(evil_roles, state, expected_remaining as usize)
        } else if n_evil_count == expected_remaining {
            vec![evil_roles.clone()]
        } else if !puppet_in_deck && n_evil_count != expected_remaining {
            continue; // Skip invalid branch
        } else {
            if evil_roles.is_empty() { vec![] } else { vec![evil_roles.clone()] }
        };

        if expected_remaining > 0 {
            for role_set in &evil_role_subsets {
                for combo in combinations_of(&available, role_set.len()) {
                    for perm in permutations_of(role_set) {
                        let mut p = HashMap::new();
                        for (i, &pos) in combo.iter().enumerate() {
                            p.insert(pos, perm[i].clone());
                        }
                        add_placement(&p);
                    }
                }
            }
        } else {
            add_placement(&HashMap::new());
        }
    }

    all_placements
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
        .filter(|r| get_card(r).map_or(false, |c| c.faction == Faction::Minion))
        .cloned().collect();
    let demon_pool: Vec<String> = evil_roles.iter()
        .filter(|r| get_card(r).map_or(false, |c| c.faction == Faction::Demon))
        .cloned().collect();

    let mut bm = state.board_minion_count.map(|x| x as i32);
    let mut bd = state.board_demon_count.map(|x| x as i32);

    let mut mp = minion_pool.clone();
    let mut dp = demon_pool.clone();
    for (_pos, role) in &state.executed_evil_roles {
        let norm = normalize_role(role);
        if let Some(idx) = mp.iter().position(|r| normalize_role(r) == norm) {
            if let Some(ref mut b) = bm { *b -= 1; }
            mp.remove(idx);
        } else if let Some(idx) = dp.iter().position(|r| normalize_role(r) == norm) {
            if let Some(ref mut b) = bd { *b -= 1; }
            dp.remove(idx);
        }
    }

    if let (Some(bm_v), Some(bd_v)) = (bm, bd) {
        let bm_pick = bm_v.max(0) as usize;
        let bd_pick = bd_v.max(0) as usize;
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
        if subsets.is_empty() {
            vec![evil_roles[..expected_remaining.min(evil_roles.len())].to_vec()]
        } else {
            subsets
        }
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

    fn structurally_complete_twin_scenario() -> Scenario {
        Scenario {
            evil_positions: HashMap::from([
                (1, "Twin Minion".to_string()),
                (2, "Witch".to_string()),
                (3, "Pooka".to_string()),
            ]),
            ..Scenario::default()
        }
    }

    fn scenario_without_twin_trace(scenario: &Scenario) -> serde_json::Value {
        let mut scenario = scenario.clone();
        scenario.twin_trace = None;
        serde_json::to_value(scenario).unwrap()
    }

    #[test]
    fn safe_twin_postprocessing_expands_every_base_world_uniformly() {
        let state = safe_twin_state();
        let first = structurally_complete_twin_scenario();
        let mut second = first.clone();
        second.corrupted.insert(2);
        let originals = vec![first, second];

        let expanded = populate_safe_twin_traces(&state, originals.clone());

        assert_eq!(expanded.len(), 4);
        for (base_index, pair) in expanded.chunks_exact(2).enumerate() {
            assert_eq!(
                scenario_without_twin_trace(&pair[0]),
                scenario_without_twin_trace(&originals[base_index])
            );
            assert_eq!(
                scenario_without_twin_trace(&pair[1]),
                scenario_without_twin_trace(&originals[base_index])
            );
            assert!(matches!(
                pair[0].twin_trace.as_ref().map(|trace| &trace.outcome),
                Some(TwinStartOutcome::Swap {
                    demon_occurrence_index: 0,
                    demon_anchor_position: 3,
                    neighbor_side: TwinNeighborSide::Previous,
                    neighbor_position: 1,
                    neighbor_pre_swap_role,
                }) if neighbor_pre_swap_role == "Twin Minion"
            ));
            assert!(matches!(
                pair[1].twin_trace.as_ref().map(|trace| &trace.outcome),
                Some(TwinStartOutcome::Swap {
                    demon_occurrence_index: 0,
                    demon_anchor_position: 3,
                    neighbor_side: TwinNeighborSide::Next,
                    neighbor_position: 2,
                    neighbor_pre_swap_role,
                }) if neighbor_pre_swap_role == "Witch"
            ));
        }
    }

    #[test]
    fn one_unknown_endpoint_keeps_every_base_world_untraced_and_ignores_appearance() {
        let mut state = safe_twin_state();
        state.cards = vec![card(2, "Witch")];
        let complete = structurally_complete_twin_scenario();
        let mut incomplete = complete.clone();
        incomplete.evil_positions.remove(&2);

        let structural = exact_pre_twin_structural_roles(&state, &incomplete).unwrap();
        assert!(!structural.contains_key(&2));

        let originals = vec![complete, incomplete];
        let unchanged = populate_safe_twin_traces(&state, originals.clone());
        assert_eq!(unchanged.len(), originals.len());
        assert!(unchanged.iter().all(|scenario| scenario.twin_trace.is_none()));
        assert_eq!(
            unchanged
                .iter()
                .map(scenario_without_twin_trace)
                .collect::<Vec<_>>(),
            originals
                .iter()
                .map(scenario_without_twin_trace)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn all_structural_multi_demon_world_preserves_pool_indices() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.deck.minions = vec![
            "Twin Minion".to_string(),
            "Witch".to_string(),
            "Poisoner".to_string(),
        ];
        state.deck.demons = vec!["Pooka".to_string(), "Lilis".to_string()];
        let scenario = Scenario {
            evil_positions: HashMap::from([
                (1, "Twin Minion".to_string()),
                (2, "Witch".to_string()),
                (3, "Pooka".to_string()),
                (4, "Poisoner".to_string()),
                (5, "Lilis".to_string()),
            ]),
            ..Scenario::default()
        };

        let expanded = populate_safe_twin_traces(&state, vec![scenario]);
        assert_eq!(expanded.len(), 4);
        let indices: Vec<u8> = expanded
            .iter()
            .map(|scenario| match &scenario.twin_trace.as_ref().unwrap().outcome {
                TwinStartOutcome::Swap {
                    demon_occurrence_index,
                    ..
                } => *demon_occurrence_index,
                TwinStartOutcome::NoDemon => unreachable!(),
            })
            .collect();
        assert_eq!(indices, vec![0, 0, 1, 1]);
        assert!(expanded.iter().any(|scenario| matches!(
            scenario.twin_trace.as_ref().map(|trace| &trace.outcome),
            Some(TwinStartOutcome::Swap {
                neighbor_position: 4,
                neighbor_pre_swap_role,
                ..
            }) if neighbor_pre_swap_role == "Poisoner"
        )));
    }

    #[test]
    fn later_identity_mutators_or_unknown_evil_keep_the_atomic_slice_untraced() {
        let base_state = safe_twin_state();
        let base_scenario = structurally_complete_twin_scenario();

        let mut unsafe_states = Vec::new();
        let mut puppeteer = base_state.clone();
        puppeteer.deck.minions.push("Puppeteer".to_string());
        unsafe_states.push(puppeteer);
        let mut shaman = base_state.clone();
        shaman.deck.minions.push("Shaman".to_string());
        unsafe_states.push(shaman);
        let mut chancellor = base_state.clone();
        chancellor.deck.minions.push("Chancellor".to_string());
        unsafe_states.push(chancellor);
        let mut plague_doctor = base_state.clone();
        plague_doctor
            .deck
            .outcasts
            .push("Plague Doctor".to_string());
        unsafe_states.push(plague_doctor);
        let mut alchemist = base_state.clone();
        alchemist.deck.villagers.push("Alchemist".to_string());
        unsafe_states.push(alchemist);
        let mut doppelganger = base_state.clone();
        doppelganger.deck.outcasts.push("Doppelganger".to_string());
        unsafe_states.push(doppelganger);
        let mut drunk = base_state.clone();
        drunk.deck.outcasts.push("Drunk".to_string());
        unsafe_states.push(drunk);

        for state in unsafe_states {
            let result = populate_safe_twin_traces(&state, vec![base_scenario.clone()]);
            assert_eq!(result.len(), 1);
            assert!(result[0].twin_trace.is_none());
        }

        let mut unknown = base_scenario;
        unknown.evil_positions.insert(2, "Unknown".to_string());
        let result = populate_safe_twin_traces(&base_state, vec![unknown]);
        assert_eq!(result.len(), 1);
        assert!(result[0].twin_trace.is_none());
    }

    #[test]
    fn presentation_derived_identity_fields_are_rejected_without_pool_hints() {
        let state = safe_twin_state();
        let base = structurally_complete_twin_scenario();

        let mut variants = Vec::new();
        let mut doppelganger = base.clone();
        doppelganger.doppelganger_position = Some(2);
        variants.push(doppelganger);
        let mut drunk = base.clone();
        drunk.drunk_position = Some(2);
        variants.push(drunk);
        let mut chancellor = base;
        chancellor.chancellor_trace = Some(ChancellorTrace {
            original_positions: vec![2],
            added_outcast_position: 2,
            added_outcast_role: "Bombardier".to_string(),
            affected_anchor_positions: vec![1],
        });
        chancellor.chancellor_conversion = Some(2);
        variants.push(chancellor);

        for scenario in variants {
            let result = populate_safe_twin_traces(&state, vec![scenario]);
            assert_eq!(result.len(), 1);
            assert!(result[0].twin_trace.is_none());
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
