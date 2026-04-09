"""Strategy/planning layer for Demon Bluff solver.

Uses surviving scenarios from the constraint solver to recommend actions:
which card to reveal, which ability to use (and on which targets), when to execute.
Decisions are driven by Shannon entropy -- actions that split scenarios most evenly
give the most information.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from knowledge_base import get_card, Role, CARDS_BY_NAME
from solver import (
    GameState, SolverResult, Scenario, TruthStatus,
    truth_status, scenario_is_evil, effective_alignment,
    get_card_at, adjacent_positions, Alignment,
    EXECUTION_IMMUNE_ROLES,
    circle_distance, circle_direction,
)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Action:
    action_type: str  # "execute", "reveal", "use_ability", "win", "error"
    position: Optional[int] = None
    targets: Optional[list[int]] = None
    ability_name: Optional[str] = None
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)
    _ability_recs: Optional[list] = field(default_factory=lambda: None, repr=False)  # cached for display


@dataclass
class RevealRecommendation:
    position: int
    entropy: float        # fingerprint-based entropy (for position ranking)
    p_evil: float
    binary_entropy: float = 0.0  # simple evil/good entropy (for ability comparison)
    reasoning: str = ""


@dataclass
class AbilityRecommendation:
    position: int
    ability_name: str
    targets: list[int]
    score: float  # higher = better (entropy or negative expected posterior)
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)


# ============================================================
# Helpers
# ============================================================

def _shannon_entropy(counts: list[int]) -> float:
    """Shannon entropy from partition counts. Higher = more informative."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def evil_probabilities(state: GameState, result: SolverResult) -> dict[int, float]:
    """Per-position probability of being evil across surviving scenarios."""
    if result.n_surviving == 0:
        return {}
    probs = {}
    for pos in range(1, state.n_cards + 1):
        if pos in state.executed:
            continue
        count = sum(1 for s in result.surviving_scenarios
                    if scenario_is_evil(pos, s))
        probs[pos] = count / result.n_surviving
    return probs


def _unrevealed_positions(state: GameState) -> list[int]:
    """Positions that haven't been revealed yet (no CardInfo) and can still be flipped."""
    revealed = {c.position for c in state.cards}
    dead = set(state.executed) | set(state.night_kills)
    return [p for p in range(1, state.n_cards + 1)
            if p not in revealed and p not in dead]


def _revealed_fraction(state: GameState) -> float:
    """Fraction of non-executed cards that have been revealed."""
    total = state.n_cards - len(state.executed)
    if total <= 0:
        return 1.0
    revealed = len([c for c in state.cards if c.position not in state.executed])
    return revealed / total


def _ability_timing_factor(state: GameState) -> float:
    """Scaling factor for ability scores based on how much info we have.

    Returns 0.0-1.0. Low when few cards revealed (save abilities for later),
    high when most cards revealed (use them now before it's too late).

    The curve ramps up after ~40% revealed, reaching full value at ~70%.
    Slayer is exempt from this (handled separately -- always good to use early
    if target is high-probability evil).
    """
    frac = _revealed_fraction(state)
    # Smooth ramp: 0 at 0%, ~0.15 at 25%, ~0.5 at 40%, ~0.85 at 55%, 1.0 at 70%+
    if frac >= 0.7:
        return 1.0
    if frac <= 0.1:
        return 0.05
    # Sigmoid-like: map [0.1, 0.7] -> [0.05, 1.0]
    t = (frac - 0.1) / 0.6  # 0 to 1
    return 0.05 + 0.95 * (t * t * (3 - 2 * t))  # smoothstep


def _apply_timing(rec: Optional[AbilityRecommendation], timing: float,
                   state: GameState, recommendations: list[AbilityRecommendation]):
    """Apply timing factor to an ability recommendation and append if valid."""
    if rec:
        rec.score *= timing
        rec.reasoning += f" | timing x{timing:.2f}"
        if timing < 0.5:
            rec.warnings.append(f"Only {_revealed_fraction(state):.0%} revealed — consider waiting")
        recommendations.append(rec)


def _remaining_evil_bounds(state: GameState, result: SolverResult) -> tuple[int, int]:
    """Return min/max evil characters still alive across surviving scenarios."""
    if not result.surviving_scenarios:
        return (0, 0)

    executed_set = set(state.executed)
    counts = []
    for scenario in result.surviving_scenarios:
        count = sum(1 for p in scenario.evil_positions if p not in executed_set)
        if scenario.puppet_position and scenario.puppet_position not in executed_set:
            count += 1
        counts.append(count)

    return (min(counts), max(counts))


def _witch_might_be_alive(state: GameState, result: SolverResult) -> bool:
    """Check if Witch could be alive (non-executed) in any surviving scenario."""
    for s in result.surviving_scenarios:
        for pos, role in s.evil_positions.items():
            if role == "Witch" and pos not in state.executed:
                return True
    return False


def _corruption_risk(pos: int, result: SolverResult) -> float:
    """Probability that position is corrupted across surviving scenarios."""
    if result.n_surviving == 0:
        return 0.0
    count = sum(1 for s in result.surviving_scenarios if pos in s.corrupted)
    return count / result.n_surviving


def _execution_reveal_outcome(
    pos: int,
    scenario: Scenario,
    state: GameState,
) -> tuple[str, bool, bool]:
    """Observed outcome if `pos` is executed in a scenario.

    Returns `(revealed_role, was_evil, was_corrupted)` using the real revealed
    role when a hidden outcast flips over.
    """
    if pos in scenario.evil_positions:
        return (scenario.evil_positions[pos], True, False)

    if pos == scenario.puppet_position:
        return ("Puppet", True, False)

    if pos == scenario.drunk_position:
        return ("Drunk", False, True)

    if pos == scenario.doppelganger_position:
        return ("Doppelganger", False, pos in scenario.corrupted)

    card = get_card_at(pos, state)
    role = card.apparent_role if card else "Unknown"
    return (role, False, pos in scenario.corrupted)


def _find_forced_execution(
    state: GameState,
    result: SolverResult,
    candidate_positions: list[int],
) -> Optional[int]:
    """Return an execution that guarantees a win across all reveal branches.

    This is a shallow endgame planner for the "all revealed / no more info"
    state. It explores execution outcomes by exact revealed role and only
    returns a position when every branch still has a forced execution-only win
    under the current HP budget.
    """
    scenarios = result.surviving_scenarios
    if not scenarios or not candidate_positions:
        return None

    probs = evil_probabilities(state, result)
    ordered_candidates = sorted(
        candidate_positions,
        key=lambda p: (-probs.get(p, 0.0), p in result.bombardier_positions, p),
    )
    all_positions = tuple(range(1, state.n_cards + 1))
    memo: dict[tuple[tuple[int, ...], tuple[int, ...], int], tuple[bool, Optional[int]]] = {}

    def all_evils_gone(indices: tuple[int, ...], executed_now: frozenset[int]) -> bool:
        for idx in indices:
            scenario = scenarios[idx]
            for pos in all_positions:
                if pos in state.executed or pos in executed_now:
                    continue
                if scenario_is_evil(pos, scenario):
                    return False
        return True

    def can_force(indices: tuple[int, ...], executed_now: frozenset[int], hp: int
                  ) -> tuple[bool, Optional[int]]:
        key = (indices, executed_now, hp)
        if key in memo:
            return memo[key]

        if all_evils_gone(indices, executed_now):
            memo[key] = (True, None)
            return memo[key]

        available = [
            pos for pos in ordered_candidates
            if pos not in executed_now and pos not in state.executed
        ]
        if not available:
            memo[key] = (False, None)
            return memo[key]

        for pos in available:
            branches: dict[tuple[str, bool, bool], list[int]] = {}
            for idx in indices:
                outcome = _execution_reveal_outcome(pos, scenarios[idx], state)
                branches.setdefault(outcome, []).append(idx)

            branch_ok = True
            for (role, was_evil, was_corrupted), branch_indices in branches.items():
                if was_evil:
                    next_hp = hp
                elif role == "Bombardier":
                    # Executing a good Bombardier = instant game loss
                    branch_ok = False
                    break
                elif role in EXECUTION_IMMUNE_ROLES and not was_corrupted:
                    # Execution blocked by immunity — no HP cost, confirms good
                    next_hp = hp
                elif role == "Drunk":
                    # Drunk execution costs 2 HP, but Drunk-as-Knight costs 6 HP (wiki)
                    card_at = next((c for c in state.cards if c.position == pos), None)
                    if card_at and card_at.apparent_role == "Knight":
                        next_hp = hp - 6
                    else:
                        next_hp = hp - 2
                else:
                    next_hp = hp - state.wrong_exec_cost
                if next_hp <= 0:
                    branch_ok = False
                    break

                can_win, _ = can_force(
                    tuple(sorted(branch_indices)),
                    executed_now | {pos},
                    next_hp,
                )
                if not can_win:
                    branch_ok = False
                    break

            if branch_ok:
                memo[key] = (True, pos)
                return memo[key]

        memo[key] = (False, None)
        return memo[key]

    success, pos = can_force(tuple(range(len(scenarios))), frozenset(), state.hp)
    if success:
        return pos
    return None


def _forced_execution_reasoning(
    pos: int,
    state: GameState,
    result: SolverResult,
) -> str:
    """Explain why an execution is safe under lookahead."""
    branches: dict[tuple[str, bool, bool], int] = {}
    for scenario in result.surviving_scenarios:
        outcome = _execution_reveal_outcome(pos, scenario, state)
        branches[outcome] = branches.get(outcome, 0) + 1

    parts = []
    total = max(1, result.n_surviving)
    for (role, was_evil, was_corrupted), count in sorted(
        branches.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]),
    ):
        label = f"{'evil' if was_evil else 'good'} {role}"
        if not was_evil and role in EXECUTION_IMMUNE_ROLES and not was_corrupted:
            label += " (immune, 0 HP)"
        elif was_corrupted and not was_evil:
            label += " (corrupted)"
        parts.append(f"{count / total:.0%} {label}")

    summary = ", ".join(parts[:3])
    return (f"Execution lookahead: #{pos} guarantees a win across all reveal branches "
            f"with current HP budget ({summary}).")


def _could_have_active_ability(pos: int, state: GameState, result: SolverResult) -> bool:
    """Check if an unrevealed position might have an active ability."""
    # Active ability roles from knowledge base
    active_roles = {c.name for c in CARDS_BY_NAME.values() if c.activated_ability}
    # Roles already placed at revealed positions
    placed_roles = {c.apparent_role for c in state.cards}
    # Unplaced active roles
    all_deck_roles = state.deck.villagers + state.deck.outcasts
    unplaced_active = [r for r in all_deck_roles
                       if r in active_roles and r not in placed_roles]
    if not unplaced_active or result.n_surviving == 0:
        return False

    return any(not scenario_is_evil(pos, scenario)
               for scenario in result.surviving_scenarios)


def _lie_probability(pos: int, state: GameState, result: SolverResult) -> float:
    """Probability that a position would present false information."""
    if result.n_surviving == 0:
        return 0.0
    lying = sum(
        1 for scenario in result.surviving_scenarios
        if truth_status(pos, scenario, state) == TruthStatus.LYING
    )
    return lying / result.n_surviving


def _ambiguity_score(prob: float) -> float:
    """How close a probability is to a 50/50 split."""
    return max(0.0, 1.0 - abs(0.5 - prob) * 2.0)


def _has_unused_followup(state: GameState, used_abilities: list[int]) -> bool:
    """Check whether Judge or Slayer can still follow an FT result."""
    for card in state.cards:
        if card.position in state.executed or card.position in used_abilities:
            continue
        role = card.apparent_role.replace("_", " ")
        if role in {"Judge", "Slayer"}:
            return True
    return False


def _fortune_teller_followup_bonus(
    targets: list[int],
    state: GameState,
    result: SolverResult,
    used_abilities: list[int],
) -> float:
    """Late-game FT tie-breaker favoring cleaner Judge/Slayer follow-up lines."""
    if result.n_surviving == 0 or _revealed_fraction(state) < 0.75:
        return 0.0
    if not _has_unused_followup(state, used_abilities):
        return 0.0

    probs = evil_probabilities(state, result)

    def control_value(pos: int) -> float:
        return _lie_probability(pos, state, result) * (1.0 - probs.get(pos, 0.0))

    def suspect_value(pos: int) -> float:
        return _ambiguity_score(probs.get(pos, 0.0))

    a, b = targets
    return max(
        control_value(a) * suspect_value(b),
        control_value(b) * suspect_value(a),
    )


# ============================================================
# Ground Truth Simulators (per ability)
# ============================================================

def _ft_ground_truth(targets: list[int], scenario: Scenario, state: GameState) -> bool:
    """Fortune Teller: any target evil?"""
    return any(effective_alignment(t, scenario, state) == Alignment.EVIL
               for t in targets)


def _jester_ground_truth(targets: list[int], scenario: Scenario, state: GameState) -> int:
    """Jester: count evil among targets."""
    return sum(1 for t in targets
               if effective_alignment(t, scenario, state) == Alignment.EVIL)


def _judge_ground_truth(target: int, scenario: Scenario, state: GameState) -> bool:
    """Judge: is target lying?"""
    return truth_status(target, scenario, state) == TruthStatus.LYING


def _dreamer_ground_truth(target: int, scenario: Scenario, state: GameState) -> str:
    """Dreamer: if target is evil, return evil role name. Else 'any_evil'."""
    if scenario_is_evil(target, scenario):
        return scenario.evil_positions.get(target, "Puppet")
    return "any_evil"


def _druid_ground_truth(targets: list[int], scenario: Scenario, state: GameState) -> str:
    """Druid: find outcast among targets, or 'none'."""
    for t in targets:
        if scenario_is_evil(t, scenario):
            continue
        card = get_card_at(t, state)
        if card:
            card_def = get_card(card.apparent_role)
            if card_def and card_def.role == Role.OUTCAST and card.apparent_role != "Wretch":
                return card.apparent_role
    return "none"


def _slayer_ground_truth(target: int, scenario: Scenario) -> bool:
    """Slayer: is target evil?"""
    return scenario_is_evil(target, scenario)


def _pd_ground_truth(target: int, scenario: Scenario, state: GameState) -> tuple:
    """Plague Doctor: (is_corrupted, evil_char_pos or None)."""
    is_corrupted = target in scenario.corrupted
    evil_pos = None
    if is_corrupted:
        # Learn an evil character
        evil_positions = [p for p in scenario.evil_positions if p not in state.executed]
        if evil_positions:
            evil_pos = min(evil_positions)  # deterministic regardless of dict ordering
    return (is_corrupted, evil_pos)


# ============================================================
# Ability Recommenders
# ============================================================

def _recommend_boolean_ability(
    ability_name: str,
    ability_pos: int,
    ground_truth_fn,
    candidate_targets: list[list[int]],
    state: GameState,
    result: SolverResult,
    tie_break_bonus_fn=None,
    bonus_weight: float = 0.0,
    tie_break_margin: float = 0.0,
    used_abilities: Optional[list[int]] = None,
) -> Optional[AbilityRecommendation]:
    """Recommend targets for a deterministic boolean-outcome ability."""
    best_primary = -1.0
    scored_candidates = []

    for targets in candidate_targets:
        true_count = 0
        false_count = 0
        for s in result.surviving_scenarios:
            truth = truth_status(ability_pos, s, state)
            real = ground_truth_fn(targets, s, state)
            observed = real if truth == TruthStatus.TRUTHFUL else (not real)
            if observed:
                true_count += 1
            else:
                false_count += 1
        ent = _shannon_entropy([true_count, false_count])
        normalized_targets = targets if isinstance(targets, list) else [targets]
        bonus = 0.0
        if tie_break_bonus_fn is not None:
            bonus = tie_break_bonus_fn(
                normalized_targets, state, result, used_abilities or []
            )
        ranking_score = ent + bonus_weight * bonus
        best_primary = max(best_primary, ranking_score)
        scored_candidates.append((ranking_score, ent, bonus, normalized_targets))

    if not scored_candidates:
        return None

    shortlist = [
        candidate
        for candidate in scored_candidates
        if candidate[0] >= best_primary - tie_break_margin
    ]
    _, best_entropy, best_bonus, best_targets = max(
        shortlist,
        key=lambda item: (item[0], item[1], item[2]),
    )

    corr = _corruption_risk(ability_pos, result)
    adjusted = best_entropy * (1 - 0.5 * corr)
    warnings = []
    if corr > 0:
        warnings.append(f"Corruption risk: {corr:.0%}")

    reasoning = f"Entropy {best_entropy:.3f} (adjusted {adjusted:.3f})"
    if best_bonus > 0:
        reasoning += f" | follow-up bonus {best_bonus:.3f}"

    return AbilityRecommendation(
        position=ability_pos,
        ability_name=ability_name,
        targets=best_targets,
        score=adjusted,
        reasoning=reasoning,
        warnings=warnings,
    )


def _recommend_judge(
    ability_pos: int,
    state: GameState,
    result: SolverResult,
    judge_targets: list[int],
) -> Optional[AbilityRecommendation]:
    """Recommend a Judge target using compatible posterior size.

    A corrupted Judge does not produce a clean inversion; its observed result is
    effectively unconstrained, so entropy over disjoint branches is the wrong
    metric here.
    """
    scenario_count = len(result.surviving_scenarios)
    if scenario_count == 0:
        return None

    best_target = None
    best_expected_posterior = float('inf')

    for target in judge_targets:
        compatible = {True: 0, False: 0}

        for scenario in result.surviving_scenarios:
            actual = _judge_ground_truth(target, scenario, state)
            if ability_pos in scenario.corrupted:
                compatible[True] += 1
                compatible[False] += 1
                continue

            truth = truth_status(ability_pos, scenario, state)
            observed = actual if truth == TruthStatus.TRUTHFUL else (not actual)
            compatible[observed] += 1

        total_weight = sum(compatible.values())
        if total_weight == 0:
            continue

        expected_posterior = sum(c * c for c in compatible.values()) / total_weight
        if expected_posterior < best_expected_posterior:
            best_expected_posterior = expected_posterior
            best_target = target

    if best_target is None:
        return None

    corr = _corruption_risk(ability_pos, result)
    adjusted = best_expected_posterior * (1 + 0.5 * corr)
    info_gain = 0.0
    if adjusted > 0:
        info_gain = max(0.0, math.log2(scenario_count) - math.log2(adjusted))

    warnings = []
    if corr > 0:
        warnings.append(
            f"Corruption risk: {corr:.0%} -- corrupted Judge results are unreliable"
        )

    return AbilityRecommendation(
        position=ability_pos,
        ability_name="Judge",
        targets=[best_target],
        score=info_gain,
        reasoning=(
            f"Expected posterior {best_expected_posterior:.1f} scenarios "
            f"(adjusted {adjusted:.1f}, info gain {info_gain:.3f} bits)"
        ),
        warnings=warnings,
    )


def _recommend_count_ability(
    ability_name: str,
    ability_pos: int,
    ground_truth_fn,
    candidate_targets: list[list[int]],
    max_count: int,
    state: GameState,
    result: SolverResult,
) -> Optional[AbilityRecommendation]:
    """Recommend targets for a count-outcome ability (Jester).
    Uses expected-posterior-size metric since lying makes multiple values compatible."""
    best_targets = None
    best_expected_posterior = float('inf')
    scenario_count = len(result.surviving_scenarios)

    for targets in candidate_targets:
        # For each possible observed value, count compatible scenarios
        compatible = {v: 0 for v in range(max_count + 1)}

        for s in result.surviving_scenarios:
            truth = truth_status(ability_pos, s, state)
            real = ground_truth_fn(targets, s, state)
            if truth == TruthStatus.TRUTHFUL:
                compatible[real] += 1
            else:
                # Lying: observed != real, so this scenario is compatible with any v != real
                for v in range(max_count + 1):
                    if v != real:
                        compatible[v] += 1

        # Expected posterior size: weighted average of compatible counts
        total_weight = sum(compatible.values())
        if total_weight == 0:
            continue
        expected_posterior = sum(c * c for c in compatible.values()) / total_weight
        if expected_posterior < best_expected_posterior:
            best_expected_posterior = expected_posterior
            best_targets = targets

    if best_targets is None:
        return None

    corr = _corruption_risk(ability_pos, result)
    adjusted = best_expected_posterior * (1 + 0.5 * corr)
    info_gain = 0.0
    if scenario_count > 0 and adjusted > 0:
        info_gain = max(0.0, math.log2(scenario_count) - math.log2(adjusted))
    warnings = []
    if corr > 0:
        warnings.append(f"Corruption risk: {corr:.0%}")

    return AbilityRecommendation(
        position=ability_pos,
        ability_name=ability_name,
        targets=best_targets,
        score=info_gain,
        reasoning=(
            f"Expected posterior {best_expected_posterior:.1f} scenarios "
            f"(adjusted {adjusted:.1f}, info gain {info_gain:.3f} bits)"
        ),
        warnings=warnings,
    )


def _recommend_partition_ability(
    ability_name: str,
    ability_pos: int,
    ground_truth_fn,
    candidate_targets: list,
    state: GameState,
    result: SolverResult,
) -> Optional[AbilityRecommendation]:
    """Recommend targets for a multi-valued partition ability (Dreamer, Druid, PD)."""
    best_targets = None
    best_entropy = -1.0

    for targets in candidate_targets:
        partition: dict[str, int] = {}
        for s in result.surviving_scenarios:
            truth = truth_status(ability_pos, s, state)
            real = ground_truth_fn(targets, s, state)

            if truth == TruthStatus.TRUTHFUL:
                key = str(real)
            else:
                key = f"lie_{real}"
            partition[key] = partition.get(key, 0) + 1

        ent = _shannon_entropy(list(partition.values()))
        if ent > best_entropy:
            best_entropy = ent
            best_targets = targets if isinstance(targets, list) else [targets]

    if best_targets is None:
        return None

    corr = _corruption_risk(ability_pos, result)
    adjusted = best_entropy * (1 - 0.5 * corr)
    warnings = []
    if corr > 0:
        warnings.append(f"Corruption risk: {corr:.0%}")

    return AbilityRecommendation(
        position=ability_pos,
        ability_name=ability_name,
        targets=best_targets,
        score=adjusted,
        reasoning=f"Entropy {best_entropy:.3f} (adjusted {adjusted:.3f})",
        warnings=warnings,
    )


def _wretch_kill_probability(target: int, state: GameState, result: SolverResult) -> float:
    """Probability that Slayer targeting this position kills a Wretch (not truly evil).

    Wretch registers as evil for abilities, so Slayer kills it — but the game
    treats it as a wrong execution (costs wrong_exec_cost HP).
    """
    if result.n_surviving == 0:
        return 0.0
    count = 0
    for s in result.surviving_scenarios:
        if not scenario_is_evil(target, s):
            # Target is good in this scenario — check if it's Wretch
            card = get_card_at(target, state)
            if card and card.apparent_role == "Wretch":
                count += 1
    return count / result.n_surviving


def _recommend_slayer(
    ability_pos: int,
    state: GameState,
    result: SolverResult,
) -> Optional[AbilityRecommendation]:
    """Slayer: pick target with highest evil probability. Not entropy-based."""
    probs = evil_probabilities(state, result)
    # Filter out Bombardier (auto-lose if Slayer kills Good Bombardier)
    # and execution-immune roles (Knight can't die)
    dangerous = set(result.bombardier_positions)
    immune = {c.position for c in state.cards if c.apparent_role in EXECUTION_IMMUNE_ROLES}
    candidates = [p for p in range(1, state.n_cards + 1)
                  if p not in state.executed and p != ability_pos
                  and p not in dangerous and p not in immune]
    if not candidates:
        return None

    # Score each candidate accounting for Wretch HP penalty
    corr = _corruption_risk(ability_pos, result)
    best_pos = None
    best_score = -1
    best_prob = 0
    best_wretch = 0
    for pos in candidates:
        prob = probs.get(pos, 0)
        wretch_prob = _wretch_kill_probability(pos, state, result)

        # Base score: true evil probability (successful kill)
        # Penalty: Wretch kill costs wrong_exec_cost HP
        if wretch_prob > 0 and state.hp <= state.wrong_exec_cost:
            # Killing Wretch would be fatal — skip this target entirely
            score = prob - wretch_prob  # Only count if truly evil, not Wretch
        else:
            # Penalize proportionally: Wretch kill wastes HP but isn't fatal
            score = prob - wretch_prob * 0.5  # Wretch kill is costly but not catastrophic
        score *= (1 - corr)  # Corrupted Slayer = ability disabled

        if score > best_score:
            best_score = score
            best_pos = pos
            best_prob = prob
            best_wretch = wretch_prob

    if best_pos is None or best_score <= 0:
        return None

    adjusted = best_score
    warnings = []
    if corr > 0:
        warnings.append(f"Corruption risk: {corr:.0%} -- Slayer ability disabled if corrupted")
    if best_wretch > 0:
        warnings.append(f"Wretch kill risk: {best_wretch:.0%} -- costs {state.wrong_exec_cost} HP")

    return AbilityRecommendation(
        position=ability_pos,
        ability_name="Slayer",
        targets=[best_pos],
        score=adjusted,
        reasoning=f"Target #{best_pos} is {best_prob:.0%} evil (adjusted {adjusted:.2f})",
        warnings=warnings,
    )


def recommend_abilities(
    state: GameState,
    result: SolverResult,
    used_abilities: list[int],
) -> list[AbilityRecommendation]:
    """Find all available active abilities and recommend optimal targets."""
    if result.n_surviving == 0:
        return []

    timing = _ability_timing_factor(state)
    recommendations = []
    available = [p for p in range(1, state.n_cards + 1) if p not in state.executed]

    # Build sets of useless targets
    # Poet targets are useless for Judge (random info = meaningless "lying" check)
    poet_positions = {c.position for c in state.cards if c.apparent_role.replace("_", " ") == "Poet"}
    # Bombardier targets are dangerous for Slayer (auto-lose if killed)
    bombardier_safe = set(result.bombardier_positions)
    # Execution-immune targets are useless for Slayer
    immune_positions = {c.position for c in state.cards
                        if c.apparent_role in EXECUTION_IMMUNE_ROLES}

    for card in state.cards:
        pos = card.position
        if pos in state.executed or pos in used_abilities:
            continue

        role = card.apparent_role.replace("_", " ")
        card_def = get_card(role)
        if not card_def or not card_def.activated_ability:
            continue

        # Build candidate target lists (exclude self and executed)
        others = [p for p in available if p != pos]

        if role == "Fortune Teller" and len(others) >= 2:
            candidates = [list(c) for c in combinations(others, 2)]
            rec = _recommend_boolean_ability(
                "Fortune Teller", pos,
                _ft_ground_truth, candidates, state, result,
                tie_break_bonus_fn=_fortune_teller_followup_bonus,
                bonus_weight=0.25,
                used_abilities=used_abilities)
            _apply_timing(rec, timing, state, recommendations)

        elif role == "Jester" and len(others) >= 3:
            candidates = [list(c) for c in combinations(others, 3)]
            rec = _recommend_count_ability(
                "Jester", pos, _jester_ground_truth,
                candidates, 3, state, result)
            _apply_timing(rec, timing, state, recommendations)

        elif role == "Judge":
            # Filter out Poets — their info is random, so Judge result is meaningless
            judge_targets = [t for t in others if t not in poet_positions]
            if not judge_targets:
                continue
            rec = _recommend_judge(pos, state, result, judge_targets)
            if rec and rec.targets and rec.targets[0] in poet_positions:
                rec.warnings.append("WARNING: Target is a Poet (random info) — Judge result meaningless!")
            _apply_timing(rec, timing, state, recommendations)

        elif role == "Dreamer":
            candidates = others
            rec = _recommend_partition_ability(
                "Dreamer", pos,
                _dreamer_ground_truth, candidates, state, result)
            _apply_timing(rec, timing, state, recommendations)

        elif role == "Druid" and len(others) >= 3:
            candidates = [list(c) for c in combinations(others, 3)]
            rec = _recommend_partition_ability(
                "Druid", pos,
                _druid_ground_truth, candidates, state, result)
            _apply_timing(rec, timing, state, recommendations)

        elif role == "Slayer":
            # Slayer is exempt from timing penalty — killing evil early is always good
            rec = _recommend_slayer(pos, state, result)
            if rec:
                recommendations.append(rec)

        elif role == "Plague Doctor":
            candidates = others
            rec = _recommend_partition_ability(
                "Plague Doctor", pos,
                lambda t, s, st: str(_pd_ground_truth(t, s, st)),
                candidates, state, result)
            _apply_timing(rec, timing, state, recommendations)

    return recommendations


# ============================================================
# Reveal Recommendation
# ============================================================

def _compute_position_fingerprint(
    pos: int, scenario: Scenario, state: GameState,
) -> tuple:
    """Compute an observation fingerprint for flipping position pos in scenario.

    The fingerprint captures what the player would learn -- two scenarios with
    different fingerprints will produce observably different card info for at
    least one possible card role. Entropy over fingerprints across scenarios
    measures the true information gain from flipping this position.

    Includes: evil/good status, evil role (if evil), corruption status,
    Hunter distance, Enlightened direction, Lover adjacent count,
    Knitter evil pairs (global), Architect side (global), Bard corruption distance.
    """
    n = state.n_cards

    # Evil status
    is_evil = scenario_is_evil(pos, scenario)
    evil_role = None
    if pos in scenario.evil_positions:
        evil_role = scenario.evil_positions[pos]
    elif pos == scenario.puppet_position:
        evil_role = "Puppet"

    # Corruption status (Confessor)
    is_corrupted = pos in scenario.corrupted

    # Build set of effective-evil positions (Wretch counts)
    evil_set = []
    for p in range(1, n + 1):
        if p != pos and effective_alignment(p, scenario, state) == Alignment.EVIL:
            evil_set.append(p)

    if not evil_set:
        return (is_evil, evil_role, is_corrupted, -1, "None", 0, 0, "Equal", -1)

    # Hunter: distance to nearest evil
    dist_nearest = min(circle_distance(pos, ep, n) for ep in evil_set)

    # Enlightened: direction to nearest evil
    closest = [ep for ep in evil_set if circle_distance(pos, ep, n) == dist_nearest]
    if len(closest) >= 2:
        dirs = {circle_direction(pos, ep, n) for ep in closest}
        direction = "Equidistant" if ("CW" in dirs and "CCW" in dirs) else circle_direction(pos, closest[0], n)
    else:
        direction = circle_direction(pos, closest[0], n)

    # Lover: adjacent evil count
    adj = adjacent_positions(pos, n)
    adj_evil = sum(1 for a in adj
                   if effective_alignment(a, scenario, state) == Alignment.EVIL)

    # Knitter: global adjacent evil pairs
    all_evil = set(evil_set)
    if effective_alignment(pos, scenario, state) == Alignment.EVIL:
        all_evil.add(pos)
    pairs = 0
    for p in all_evil:
        for a in adjacent_positions(p, n):
            if a in all_evil and a > p:
                pairs += 1

    # Architect: left/right/equal (board-relative, not position-relative)
    half = n // 2
    both_set = {n}
    left_set = set()
    right_set = set()
    if n % 2 == 0:
        both_set.add(half)
        for i in range(1, half):
            right_set.add(i)
        for i in range(half + 1, n):
            left_set.add(i)
    else:
        for i in range(1, half + 1):
            right_set.add(i)
        for i in range(half + 1, n):
            left_set.add(i)

    left_count = right_count = 0
    for p in all_evil:
        if p in both_set:
            left_count += 1
            right_count += 1
        elif p in left_set:
            left_count += 1
        elif p in right_set:
            right_count += 1
    if left_count > right_count:
        arch_side = "Left"
    elif right_count > left_count:
        arch_side = "Right"
    else:
        arch_side = "Equal"

    # Bard: distance to nearest corrupted
    if scenario.corrupted:
        dist_corrupted = min(circle_distance(pos, c, n) for c in scenario.corrupted)
    else:
        dist_corrupted = -1  # no corrupted

    return (is_evil, evil_role, is_corrupted, dist_nearest, direction,
            adj_evil, pairs, arch_side, dist_corrupted)


def recommend_reveal(
    state: GameState,
    result: SolverResult,
) -> Optional[RevealRecommendation]:
    """Pick the most informative unrevealed position to reveal next.

    Uses fingerprint-based entropy: for each unrevealed position, computes
    what the player would observe in each surviving scenario. Positions
    where observations vary the most (highest entropy) give the most info.
    """
    unrevealed = _unrevealed_positions(state)
    # Filter out blocked positions (Witch)
    blocked = set(getattr(state, 'blocked_positions', []))
    unrevealed = [p for p in unrevealed if p not in blocked]
    if not unrevealed:
        return None

    # Witch edge case: can't reveal last card
    if len(unrevealed) == 1 and _witch_might_be_alive(state, result):
        return None  # Blocked by Witch

    probs = evil_probabilities(state, result)
    best = None
    best_entropy = -1.0

    for pos in unrevealed:
        p = probs.get(pos, 0)

        # Binary entropy (for ability-vs-reveal comparison, same scale as ability scores)
        if p == 0 or p == 1:
            bin_ent = 0.0
        else:
            bin_ent = _shannon_entropy([
                int(p * result.n_surviving),
                int((1 - p) * result.n_surviving)
            ])

        # Fingerprint-based entropy: partition scenarios by observation
        # This captures spatial diversity -- positions where different scenarios
        # would produce observably different card info score higher
        fingerprint_groups: dict[tuple, int] = {}
        for scenario in result.surviving_scenarios:
            fp = _compute_position_fingerprint(pos, scenario, state)
            fingerprint_groups[fp] = fingerprint_groups.get(fp, 0) + 1

        counts = list(fingerprint_groups.values())
        ent = _shannon_entropy(counts)

        # Bonus for positions that might have active abilities
        if _could_have_active_ability(pos, state, result):
            ent += 0.1
            bin_ent += 0.1

        n_outcomes = len(fingerprint_groups)
        if ent > best_entropy:
            best_entropy = ent
            best = RevealRecommendation(
                position=pos, entropy=ent, p_evil=p,
                binary_entropy=bin_ent,
                reasoning=f"#{pos}: {p:.0%} evil, {ent:.3f} bits ({n_outcomes} outcomes)")

    return best


# ============================================================
# Top-Level Entry Point
# ============================================================

def recommend_action(
    state: GameState,
    result: SolverResult,
    used_abilities: list[int],
) -> Action:
    """Recommend the best next action given solver results.

    Priority:
    1. Error -- 0 surviving scenarios
    2. Win -- all evil executed
    3. Execute -- definite evil found (skip Bombardier)
    4. Use ability -- if high info gain
    5. Reveal -- most informative unrevealed position
    6. Witch fallback -- can't reveal, execute best guess
    7. Probability fallback -- all revealed, no certainty
    """
    # Pre-compute evil probabilities (used in knight check, witch fallback, etc.)
    probs = evil_probabilities(state, result)

    # 1. Error
    if result.n_surviving == 0:
        return Action("error", reasoning="No surviving scenarios -- check input data")

    # 2. Win check
    _, max_remaining = _remaining_evil_bounds(state, result)
    if max_remaining == 0:
        return Action("win", reasoning="All evil characters have been executed!")

    # 3. Execute definite evil (skip Bombardier)
    # Note: Knights are NOT blanket-immune -- see "Knight free check" below.
    # Executing an uncertain Knight is a free check (0 HP) when uncorrupted.
    safe_executions = [p for p in result.definite_evil
                       if p not in state.executed
                       and p not in result.bombardier_positions]
    if safe_executions:
        pos = safe_executions[0]
        roles = set()
        for s in result.surviving_scenarios:
            if pos in s.evil_positions:
                roles.add(s.evil_positions[pos])
        return Action(
            "execute", position=pos,
            reasoning=f"#{pos} is evil in ALL {result.n_surviving} scenarios (roles: {roles})")

    # 3.5 Knight free check — executing an uncertain Knight is free info
    # Real Knight (uncorrupted): execution blocked, confirms good, 0 HP cost
    # Evil disguised as Knight: evil dies
    # Corrupted Knight: execution succeeds, costs HP (risky)
    knight_checks = []
    for card in state.cards:
        if (card.apparent_role in EXECUTION_IMMUNE_ROLES
                and card.position not in state.executed
                and card.position not in result.definite_good
                and card.position not in result.definite_evil):
            corr_risk = _corruption_risk(card.position, result)
            evil_prob = probs.get(card.position, 0)
            knight_checks.append((card.position, evil_prob, corr_risk))

    if knight_checks:
        knight_checks.sort(key=lambda x: -x[1])  # highest evil prob first
        kpos, evil_prob, corr_risk = knight_checks[0]
        if corr_risk == 0:
            # Truly free: 0% corruption means execution is either blocked or kills evil
            return Action(
                "execute", position=kpos,
                reasoning=f"Knight free check: #{kpos} is {evil_prob:.0%} evil. "
                          f"If real Knight, execution blocked (confirms good, 0 HP). "
                          f"If evil disguise, evil dies. No corruption risk.")
        elif corr_risk < 0.3:
            # Mostly free: small corruption risk lowers the expected cost
            # Corrupted Knight deals 4 EXTRA damage on top of wrong exec cost
            corrupted_knight_cost = state.wrong_exec_cost + 4
            expected_cost = corr_risk * (1 - evil_prob) * corrupted_knight_cost
            # Never attempt if corrupted Knight would kill us
            if state.hp > corrupted_knight_cost and expected_cost < state.wrong_exec_cost * 0.3:
                return Action(
                    "execute", position=kpos,
                    reasoning=f"Knight check: #{kpos} is {evil_prob:.0%} evil, "
                              f"{corr_risk:.0%} corruption risk. Expected HP cost: "
                              f"{expected_cost:.1f} (corrupted Knight = {corrupted_knight_cost} HP).",
                    warnings=[f"Corruption risk: {corr_risk:.0%} -- corrupted Knight loses immunity + 4 extra damage"])

    # 4. Check available abilities
    ability_recs = recommend_abilities(state, result, used_abilities)
    ability_recs.sort(key=lambda r: r.score, reverse=True)

    def _with_ability_recs(action: Action) -> Action:
        action._ability_recs = ability_recs
        return action

    # 5. Check reveal
    reveal_rec = recommend_reveal(state, result)

    # Choose between ability and reveal based on scores
    best_ability = ability_recs[0] if ability_recs else None
    if best_ability and best_ability.ability_name == "Slayer" and best_ability.score > 0.8:
        # Slayer with high confidence -- use it
        return _with_ability_recs(Action(
            "use_ability", position=best_ability.position,
            targets=best_ability.targets,
            ability_name="Slayer",
            reasoning=best_ability.reasoning,
            warnings=best_ability.warnings))

    if best_ability and reveal_rec:
        # Compare ability info gain vs reveal info gain
        # Use binary_entropy (evil/good split) for comparison -- same 0-1 scale as ability scores.
        # Fingerprint entropy (reveal_rec.entropy) is used for position ranking only.
        if best_ability.score > reveal_rec.binary_entropy and best_ability.score > 0.3:
            return _with_ability_recs(Action(
                "use_ability", position=best_ability.position,
                targets=best_ability.targets,
                ability_name=best_ability.ability_name,
                reasoning=best_ability.reasoning,
                warnings=best_ability.warnings))

    if reveal_rec:
        warnings = []
        if _witch_might_be_alive(state, result):
            n_unrevealed = len(_unrevealed_positions(state))
            if n_unrevealed <= 2:
                warnings.append("Witch may be alive -- be cautious about revealing")
        return _with_ability_recs(Action(
            "reveal", position=reveal_rec.position,
            reasoning=reveal_rec.reasoning,
            warnings=warnings))

    if best_ability:
        return _with_ability_recs(Action(
            "use_ability", position=best_ability.position,
            targets=best_ability.targets,
            ability_name=best_ability.ability_name,
            reasoning=best_ability.reasoning,
            warnings=best_ability.warnings))

    # 6. Witch fallback -- can't reveal, execute by probability
    # HP-aware gating with budget-based confidence thresholds
    wrong_exec_budget = state.hp // state.wrong_exec_cost if state.wrong_exec_cost > 0 else 99

    # 6a. Forced execution lookahead — includes Bombardier/Wretch.
    # The lookahead models Bombardier as instant game loss, so it naturally
    # prefers executing non-Bombardier candidates first.
    all_uncertain = [p for p, prob in probs.items()
                     if prob > 0.0 and p not in state.executed]
    if all_uncertain:
        forced_pos = _find_forced_execution(state, result, all_uncertain)
        if forced_pos is not None:
            warnings = []
            forced_prob = probs.get(forced_pos, 0.0)
            if forced_pos in result.bombardier_positions:
                warnings.append(
                    "Bombardier targeted by lookahead — confirmed safe across all branches")
            if forced_prob < 1.0:
                warnings.append(
                    f"Execution lookahead override -- immediate hit chance is {forced_prob:.0%}, "
                    f"but all reveal branches still lead to a forced win."
                )
            return _with_ability_recs(Action(
                "execute",
                position=forced_pos,
                reasoning=_forced_execution_reasoning(forced_pos, state, result),
                warnings=warnings,
            ))

    # 6b. Probabilistic execution
    # Bombardier candidates excluded from normal probability selection
    bombardier_candidates = {p: probs.get(p, 0) for p in result.bombardier_positions
                            if p not in state.executed and probs.get(p, 0) > 0.0}
    # Exclude Bombardier (instant loss) and Wretch (always wrong exec —
    # abilities see Wretch as evil, inflating evil_probability, but executing
    # Wretch is guaranteed wrong exec penalty with zero upside).
    wretch_positions = {c.position for c in state.cards
                        if c.apparent_role == "Wretch"
                        and c.position not in result.definite_evil}
    active_probs = {p: prob for p, prob in probs.items()
                    if p not in state.executed
                    and p not in result.bombardier_positions
                    and p not in wretch_positions}
    if active_probs:
        best_pos = max(active_probs, key=active_probs.get)
        best_prob = active_probs[best_pos]

        # If Witch is blocking reveals, prefer executing the most likely Witch
        # position -- killing the Witch unblocks the last card reveal
        witch_blocked = (not reveal_rec and _witch_might_be_alive(state, result)
                         and len(_unrevealed_positions(state)) > 0)
        if witch_blocked:
            witch_probs = {}
            for p in active_probs:
                witch_count = sum(1 for s in result.surviving_scenarios
                                 if s.evil_positions.get(p) == "Witch")
                if witch_count > 0:
                    witch_probs[p] = witch_count / result.n_surviving
            if witch_probs:
                best_witch_pos = max(witch_probs, key=witch_probs.get)
                best_witch_prob = witch_probs[best_witch_pos]
                # If a position is both likely evil AND likely Witch, prefer it
                # since killing Witch unblocks reveals for remaining deduction
                if (active_probs.get(best_witch_pos, 0) > 0.5
                        and best_witch_prob > 0.3):
                    best_pos = best_witch_pos
                    best_prob = active_probs[best_pos]

        warnings = [f"Probabilistic execution -- {best_prob:.0%} confident "
                    f"(budget: {wrong_exec_budget} wrong execs)"]
        if witch_blocked:
            warnings.append("Witch is blocking reveals -- killing Witch would unblock last card")

        if wrong_exec_budget == 0:
            warnings.append(f"CRITICAL: HP={state.hp}, wrong exec costs {state.wrong_exec_cost} -- "
                            f"CANNOT afford a mistake! Only execute if certain.")
            if best_prob < 1.0:
                return _with_ability_recs(Action(
                    "error", position=best_pos,
                    reasoning=f"#{best_pos} is {best_prob:.0%} likely evil but HP too low to risk "
                              f"(HP={state.hp}, cost={state.wrong_exec_cost}). Need more info.",
                    warnings=warnings))
        elif wrong_exec_budget == 1:
            # One wrong guess = death. Require high confidence.
            min_threshold = 0.80
            if best_prob < min_threshold:
                if bombardier_candidates:
                    # Bombardier safety: bypass threshold — wrong exec on non-Bombardier
                    # costs HP, wrong exec on Bombardier = instant game loss.
                    warnings.append(
                        f"Bombardier safety: executing #{best_pos} ({best_prob:.0%}) despite "
                        f"low confidence — Bombardier candidate(s) {sorted(bombardier_candidates.keys())} "
                        f"risk instant game loss if executed first.")
                else:
                    warnings.append(f"CAUTION: budget=1, confidence {best_prob:.0%} < {min_threshold:.0%} threshold. "
                                    f"Consider manual override if you have extra information.")
                    return _with_ability_recs(Action(
                        "error", position=best_pos,
                        reasoning=f"#{best_pos} is {best_prob:.0%} likely evil but budget=1 requires "
                                  f">={min_threshold:.0%} confidence (HP={state.hp}, cost={state.wrong_exec_cost}).",
                        warnings=warnings))
        elif best_prob < 0.5:
            warnings.append(f"Low confidence ({best_prob:.0%}) -- consider gathering more info")

        return _with_ability_recs(Action(
            "execute", position=best_pos,
            reasoning=f"No reveals available. #{best_pos} is {best_prob:.0%} likely evil "
                      f"(HP={state.hp}, budget={wrong_exec_budget} wrong execs)",
            warnings=warnings))

    # 6c. Bombardier safety fallback: when all high-probability candidates are
    # Bombardier (excluded from active_probs), prefer a non-Bombardier uncertain
    # position. Wrong exec on non-Bombardier = HP cost; Bombardier = game loss.
    if bombardier_candidates:
        # Include Wretch — wrong exec on Wretch = HP cost, not game loss
        safety_probs = {p: prob for p, prob in probs.items()
                       if p not in state.executed
                       and p not in result.bombardier_positions
                       and prob > 0.0}
        if safety_probs:
            safe_pos = max(safety_probs, key=safety_probs.get)
            safe_prob = safety_probs[safe_pos]
            card = next((c for c in state.cards if c.position == safe_pos), None)
            role_label = card.apparent_role if card else "?"
            bomb_positions = sorted(bombardier_candidates.keys())
            return _with_ability_recs(Action(
                "execute", position=safe_pos,
                reasoning=f"Bombardier safety: #{safe_pos} ({role_label}, {safe_prob:.0%} evil) "
                          f"preferred over Bombardier candidate(s) {bomb_positions}. "
                          f"Wrong exec costs {state.wrong_exec_cost} HP; "
                          f"Bombardier wrong exec = instant game loss.",
                warnings=[f"Bombardier safety play — testing non-Bombardier first "
                         f"(HP={state.hp}, budget={wrong_exec_budget} wrong execs)"]))

    # 7. Shouldn't reach here
    return _with_ability_recs(Action("error", reasoning="No valid action found"))


# ============================================================
# Display
# ============================================================

def print_recommendation(state: GameState, result: SolverResult,
                         used_abilities: list[int]):
    """Print a full strategy recommendation."""
    action = recommend_action(state, result, used_abilities)

    print(f"\n=== STRATEGY RECOMMENDATION ===")
    print(f"  Action: {action.action_type.upper()}", end="")
    if action.position:
        print(f" #{action.position}", end="")
    if action.ability_name:
        print(f" ({action.ability_name})", end="")
    if action.targets:
        print(f" -> targets {['#'+str(t) for t in action.targets]}", end="")
    print()
    print(f"  Reason: {action.reasoning}")
    for w in action.warnings:
        print(f"  WARNING: {w}")

    # Show smart reveal analysis for context
    if action.action_type in ("reveal", "use_ability", "execute"):
        probs = evil_probabilities(state, result)
        unrevealed = _unrevealed_positions(state)
        blocked = set(getattr(state, 'blocked_positions', []))
        if unrevealed:
            # Compute fingerprint entropy for each unrevealed position
            reveal_scores: list[tuple[int, float, float, int]] = []  # (pos, entropy, p_evil, n_outcomes)
            for pos in sorted(unrevealed):
                p = probs.get(pos, 0)
                if pos in blocked or result.n_surviving == 0:
                    reveal_scores.append((pos, 0.0, p, 1))
                    continue
                fp_groups: dict[tuple, int] = {}
                for scenario in result.surviving_scenarios:
                    fp = _compute_position_fingerprint(pos, scenario, state)
                    fp_groups[fp] = fp_groups.get(fp, 0) + 1
                ent = _shannon_entropy(list(fp_groups.values()))
                if _could_have_active_ability(pos, state, result):
                    ent += 0.1
                reveal_scores.append((pos, ent, p, len(fp_groups)))

            reveal_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"\n  Unrevealed positions (ranked by info gain):")
            for pos, ent, p, n_out in reveal_scores:
                marker = " <-- RECOMMEND" if pos == action.position and action.action_type == "reveal" else ""
                blk = " [BLOCKED]" if pos in blocked else ""
                print(f"    #{pos}: {p:.0%} evil, {ent:.2f} bits ({n_out} outcomes){blk}{marker}")

    # Show available abilities (reuse cached recs from recommend_action)
    recs = action._ability_recs if action._ability_recs is not None else recommend_abilities(state, result, used_abilities)
    if recs:
        print(f"\n  Available abilities:")
        recs.sort(key=lambda r: r.score, reverse=True)
        for rec in recs:
            chosen = " <-- RECOMMEND" if (action.action_type == "use_ability"
                                          and action.position == rec.position) else ""
            targets_str = ",".join(f"#{t}" for t in rec.targets)
            print(f"    #{rec.position} {rec.ability_name} -> [{targets_str}] "
                  f"(score {rec.score:.3f}){chosen}")
            for w in rec.warnings:
                print(f"      WARNING: {w}")

    # HP status
    wrong_budget = state.hp // state.wrong_exec_cost if state.wrong_exec_cost > 0 else 999
    print(f"\n  HP: {state.hp}/{10} | Wrong exec cost: {state.wrong_exec_cost} | "
          f"Budget: {wrong_budget} wrong executions")
    print(f"  ({result.n_surviving} surviving scenarios)\n")
    return action
