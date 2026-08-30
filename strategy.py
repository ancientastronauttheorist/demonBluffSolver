"""Strategy/planning layer for Demon Bluff solver.

Uses surviving scenarios from the constraint solver to recommend actions:
which card to reveal, which ability to use (and on which targets), when to execute.
Decisions are driven by Shannon entropy -- actions that split scenarios most evenly
give the most information.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from knowledge_base import execution_cost_for, get_card, Role, CARDS_BY_NAME
from solver import (
    GameState, SolverResult, Scenario, TruthStatus,
    truth_status, truth_appearance_status, scenario_is_evil, effective_alignment,
    effective_role_at,
    slayer_revealed_role,
    get_card_at, adjacent_positions, Alignment,
    EXECUTION_IMMUNE_ROLES,
    circle_distance, circle_direction,
)


# ============================================================
# Tuning knobs (off-by-default experimental flags)
# ============================================================
# When True, before returning a low-score ability recommendation, run
# _find_forced_execution. If it finds a definite-evil position, prefer
# executing that instead (guaranteed kill beats a weak info gather).
# Default OFF so the v2 replay baseline is unaffected. Flip to True for
# experimental tuning runs; compare replay-suite diffs before shipping.
LOOKAHEAD_PREFER_FORCED_OVER_LOW_ABILITY = False
LOW_ABILITY_SCORE_THRESHOLD = 0.30  # info bits


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Action:
    action_type: str  # execute, reveal, use_ability, loss, win, error
    position: Optional[int] = None
    targets: Optional[list[int]] = None
    ability_name: Optional[str] = None
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1, how certain this is the right move
    forced_safe: bool = False  # Lookahead-safe ordinary execution; never bypasses Bombardier
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
    dead = set(state.executed) | set(state.night_kills)
    for pos in range(1, state.n_cards + 1):
        if pos in dead:
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
    """Fraction of living cards that have been revealed."""
    dead = set(state.executed) | set(state.night_kills)
    total = state.n_cards - len(dead)
    if total <= 0:
        return 1.0
    revealed = len([c for c in state.cards if c.position not in dead])
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

    dead = set(state.executed) | set(state.night_kills)
    counts = []
    for scenario in result.surviving_scenarios:
        count = sum(1 for p in scenario.evil_positions if p not in dead)
        if scenario.puppet_position and scenario.puppet_position not in dead:
            count += 1
        counts.append(count)

    return (min(counts), max(counts))


def _witch_might_be_alive(state: GameState, result: SolverResult) -> bool:
    """Check if Witch could be alive in any surviving scenario."""
    dead = set(state.executed) | set(state.night_kills)
    for s in result.surviving_scenarios:
        for pos, role in s.evil_positions.items():
            if role == "Witch" and pos not in dead:
                return True
    return False


def _witch_quota_might_be_active(
    state: GameState,
    result: SolverResult,
) -> bool:
    """Whether the ordinary shared Cipher reveal quota can remain active.

    Duplicate ordinary Witch cards produce one successful Start increment,
    while the death of either real Witch decrements that shared scalar. Thus a
    surviving duplicate does not preserve the ordinary quota after a Witch
    death. Independently repeated Starts can stack, but GameState does not
    represent that history and must not infer it merely from duplicate roles.
    """
    dead = set(state.executed) | set(state.night_kills)
    if any(
        role.lower().replace(" ", "_") == "witch"
        for role in state.executed_evil_roles.values()
    ):
        return False

    for scenario in result.surviving_scenarios:
        witch_positions = {
            pos for pos, role in scenario.evil_positions.items()
            if role == "Witch"
        }
        if witch_positions and not (witch_positions & dead):
            return True
    return False


def _active_witch_blocked_positions(
    state: GameState,
    result: SolverResult,
) -> set[int]:
    """Return markers backed by a possibly active ordinary Witch quota.

    Historical v2/session snapshots can retain a marker after Witch died or
    after its card was later revealed. Such markers are useful provenance but
    must not suppress a legal reveal or appear as a current block.
    """
    if not _witch_quota_might_be_active(state, result):
        return set()
    unrevealed = set(_unrevealed_positions(state))
    return set(getattr(state, "blocked_positions", [])) & unrevealed


def _corruption_risk(pos: int, result: SolverResult, state: GameState) -> float:
    """Probability that a position is unsafe as a clean clue/execution surface.

    Drunk is intrinsically lying and killable even when inherited Alchemist
    resistance prevents its generic Corrupted status bit.
    """
    if result.n_surviving == 0:
        return 0.0
    count = 0
    for scenario in result.surviving_scenarios:
        role = effective_role_at(pos, scenario, state)
        role_key = role.lower().replace(" ", "").replace("_", "") if role else None
        if pos in scenario.corrupted or role_key == "drunk":
            count += 1
    return count / result.n_surviving


def _execution_reveal_outcome(
    pos: int,
    scenario: Scenario,
    state: GameState,
) -> tuple[str, bool, bool, bool]:
    """Observed outcome if `pos` is executed in a scenario.

    Returns `(revealed_role, was_evil, observed_corrupted, active_corrupted)`.
    The split matters for Drunk: bookkeeping reports clean, while an active
    Corrupted status still drives Knight's separate four-damage hook.
    """
    card = get_card_at(pos, state)
    role = effective_role_at(pos, scenario, state)
    if role is None:
        role = card.apparent_role if card else "Unknown"

    was_evil = scenario_is_evil(pos, scenario)
    if was_evil:
        # Runtime alignment determines the correct-execution branch, while a
        # Shaman-copied current dataRef is the role KillAndReveal exposes.
        return (role, True, False, False)

    active_corrupted = pos in scenario.corrupted
    role_key = role.lower().replace(" ", "").replace("_", "")
    observed_corrupted = role_key != "drunk" and active_corrupted
    return (role, False, observed_corrupted, active_corrupted)


def _is_terminal_loss_role(role: Optional[str]) -> bool:
    """Match only canonical public CharacterData Bombardier.

    Managed Saint implements Bombardier, but public CharacterData Saint is a
    distinct role and must not be treated as an alias.
    """
    key = (role or "").strip().replace("_", " ").replace("-", " ").casefold()
    return key == "bombardier"


def _scenario_terminal_loss_position(
    state: GameState,
    scenario: Scenario,
) -> Optional[int]:
    """Return a qualifying already-dead Bombardier in one exact world."""
    night_kills = set(state.night_kills)
    for pos in state.executed:
        if pos in night_kills:
            continue
        # Public KillAndReveal/execution identity is authoritative, including
        # a non-Bombardier negative. Only otherwise infer scenario-effective
        # current data (Shaman/generated roles before bluff appearance).
        role = _public_death_role_at(state, pos)
        if role is None:
            role = effective_role_at(pos, scenario, state)
        if _is_terminal_loss_role(role):
            return pos
    return None


def _public_death_role_at(state: GameState, pos: int) -> Optional[str]:
    """Best authoritative public current-role reveal for one death."""
    for result in reversed(state.slayer_results):
        if result.get("killed") is True and result.get("target_pos") == pos:
            role = slayer_revealed_role(result)
            if role is not None:
                return role
    current_role = state.executed_current_roles.get(pos)
    if current_role is not None:
        return current_role
    return state.executed_good_roles.get(pos)


def _public_terminal_loss_position(state: GameState) -> Optional[int]:
    """Return a terminal death proven without any surviving solver world."""
    night_kills = set(state.night_kills)
    for pos in state.executed:
        if pos in night_kills:
            continue
        role = _public_death_role_at(state, pos)
        if _is_terminal_loss_role(role):
            return pos
    return None


def _has_terminal_role_loss(
    state: GameState,
    result: SolverResult,
) -> bool:
    """Whether public state or any surviving exact world is already terminal."""
    if _is_terminal_loss_role(getattr(state, "terminal_loss_role", None)):
        return True
    if _public_terminal_loss_position(state) is not None:
        return True
    return any(
        _scenario_terminal_loss_position(state, scenario) is not None
        for scenario in result.surviving_scenarios
    )


def _execution_branch_is_protected(
    revealed_role: str,
    apparent_role: str,
    was_evil: bool,
    was_corrupted: bool,
) -> bool:
    """Mirror native Knight/HealthyBluff protection in lookahead branches."""
    if was_evil:
        return False
    if revealed_role == "Knight" and not was_corrupted:
        return True
    return (
        revealed_role in ("Doppelganger", "Doppleganger")
        and apparent_role == "Knight"
        and not was_corrupted
    )


def _execution_observation_key(
    pos: int,
    outcome: tuple[str, bool, bool, bool],
    state: GameState,
) -> tuple:
    """Canonicalize an execution branch to information the player observes."""
    role, was_evil, observed_corrupted, active_corrupted = outcome
    card_at = get_card_at(pos, state)
    apparent_role = card_at.apparent_role if card_at else "Unknown"
    # Current-role death precedes runtime alignment: a Shaman-copied
    # Bombardier on a preserved Evil body is still terminal.
    if _is_terminal_loss_role(role):
        return ("bombardier_loss",)
    if _execution_branch_is_protected(
        role,
        apparent_role,
        was_evil,
        active_corrupted,
    ):
        # A protected true Knight and a protected Doppelganger-as-Knight both
        # remain face-up as Knight. Hidden real identity must not split DFS.
        return ("protected",)
    damage = execution_cost_for(
        role,
        apparent_role=apparent_role,
        was_evil=was_evil,
        was_corrupted=active_corrupted,
        was_killable=True,
        default=state.wrong_exec_cost,
    )
    return ("killed", role, was_evil, observed_corrupted, damage)


def _knight_check_damage_profile(
    pos: int,
    result: SolverResult,
    state: GameState,
) -> tuple[float, float, int]:
    """Return unsafe probability, expected damage, and worst damage.

    An apparent Knight is not necessarily a true Knight: Chancellor can leave
    a generated Outcast at that physical seat.  Derive safety from the same
    native execution observation branches used by forced-execution lookahead,
    rather than treating generic Corrupted status as the only risky outcome.
    """
    if result.n_surviving == 0:
        return (0.0, 0.0, 0)

    unsafe_branches = 0
    total_damage = 0
    worst_damage = 0
    for scenario in result.surviving_scenarios:
        outcome = _execution_reveal_outcome(pos, scenario, state)
        observation = _execution_observation_key(pos, outcome, state)
        if observation[0] == "bombardier_loss":
            # The caller also excludes aggregate Bombardier positions.  Keep
            # this defensive branch unsafe if an incomplete result reaches it.
            return (1.0, math.inf, max(state.hp, 1))
        if observation[0] != "killed":
            continue
        damage = observation[-1]
        if damage > 0:
            unsafe_branches += 1
            total_damage += damage
            worst_damage = max(worst_damage, damage)

    return (
        unsafe_branches / result.n_surviving,
        total_damage / result.n_surviving,
        worst_damage,
    )


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
    already_dead = set(state.executed) | set(state.night_kills)
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
                if pos in already_dead or pos in executed_now:
                    continue
                if scenario_is_evil(pos, scenario):
                    return False
        return True

    def can_force(indices: tuple[int, ...], executed_now: frozenset[int], hp: int
                  ) -> tuple[bool, Optional[int]]:
        key = (indices, executed_now, hp)
        if key in memo:
            return memo[key]

        # Native terminal order is current-role Bombardier death, depleted HP,
        # then the evil-count win condition.
        if (
            _is_terminal_loss_role(getattr(state, "terminal_loss_role", None))
            or _public_terminal_loss_position(state) is not None
            or any(
                _scenario_terminal_loss_position(state, scenarios[idx])
                is not None
                for idx in indices
            )
        ):
            memo[key] = (False, None)
            return memo[key]

        if hp <= 0:
            memo[key] = (False, None)
            return memo[key]

        if all_evils_gone(indices, executed_now):
            memo[key] = (True, None)
            return memo[key]

        available = [
            pos for pos in ordered_candidates
            if pos not in executed_now and pos not in already_dead
        ]
        if not available:
            memo[key] = (False, None)
            return memo[key]

        for pos in available:
            branches: dict[tuple, list[int]] = {}
            for idx in indices:
                outcome = _execution_reveal_outcome(pos, scenarios[idx], state)
                observation = _execution_observation_key(pos, outcome, state)
                branches.setdefault(observation, []).append(idx)

            branch_ok = True
            for observation, branch_indices in branches.items():
                if observation[0] == "bombardier_loss":
                    branch_ok = False
                    break
                if observation[0] == "protected":
                    next_hp = hp
                else:
                    damage = observation[4]
                    next_hp = max(0, hp - damage)
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
    branches: dict[tuple[str, bool, bool, bool], int] = {}
    for scenario in result.surviving_scenarios:
        outcome = _execution_reveal_outcome(pos, scenario, state)
        branches[outcome] = branches.get(outcome, 0) + 1

    parts = []
    total = max(1, result.n_surviving)
    for (role, was_evil, observed_corrupted, active_corrupted), count in sorted(
        branches.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]),
    ):
        label = f"{'evil' if was_evil else 'good'} {role}"
        card_at = get_card_at(pos, state)
        apparent_role = card_at.apparent_role if card_at else "Unknown"
        if _execution_branch_is_protected(
            role,
            apparent_role,
            was_evil,
            active_corrupted,
        ):
            label += " (immune, 0 HP)"
        elif not was_evil:
            damage = execution_cost_for(
                role,
                apparent_role=apparent_role,
                was_corrupted=active_corrupted,
                was_killable=True,
                default=state.wrong_exec_cost,
            )
            status = ", corrupted" if active_corrupted else ""
            label += f" (-{damage} HP{status})"
        parts.append(f"{count / total:.0%} {label}")

    summary = ", ".join(parts[:3])
    return (f"Execution lookahead: #{pos} guarantees a win across all reveal branches "
            f"with current HP budget ({summary}).")


# ============================================================
# E1 — Expected-value scoring
# ============================================================

# Lambda: weight for HP cost penalty in EV computation. Tunable.
_EV_LAMBDA = 0.1


def _compute_ev_ability(ability_score: float) -> float:
    """EV for an ability action: info gain with zero HP cost."""
    return ability_score


def _compute_ev_reveal(binary_entropy: float) -> float:
    """EV for a reveal action: binary entropy with zero HP cost."""
    return binary_entropy


def _compute_ev_execute(
    p_evil: float,
    wrong_exec_cost: int,
    info_gain: float = 0.5,
) -> float:
    """EV for an execution: info gain minus expected HP cost.

    EV = info_gain - lambda * wrong_exec_cost * (1 - p_evil)
    Zero HP cost when target is evil; full penalty when good.
    """
    return info_gain - _EV_LAMBDA * wrong_exec_cost * (1.0 - p_evil)


def _compute_ev_slayer(
    slayer_score: float,
    wrong_exec_cost: int,
    wretch_probability: float,
) -> float:
    """EV for Slayer ability: score minus Wretch penalty.

    Wretch kills cost wrong_exec_cost HP (wrong execution penalty).
    """
    return slayer_score - _EV_LAMBDA * wrong_exec_cost * wretch_probability


# ============================================================
# E2 — Shallow 2-turn lookahead
# ============================================================

_SHALLOW_LOOKAHEAD_MAX_SCENARIOS = 500
_SHALLOW_LOOKAHEAD_TIMEOUT_MS = 100


def _shallow_lookahead(
    state: GameState,
    result: SolverResult,
    candidate_positions: list[int],
) -> Optional[Action]:
    """2-turn lookahead: reveal then force-execute.

    For each unrevealed position, compute fingerprint partitions. For each
    partition (what the player would observe), check if a forced execution
    exists in that sub-problem. Return a 2-step plan if any reveal guarantees
    a forced win on the next step.

    Only runs when n_surviving <= _SHALLOW_LOOKAHEAD_MAX_SCENARIOS.
    Bails out at _SHALLOW_LOOKAHEAD_TIMEOUT_MS wall-clock milliseconds.
    """
    if result.n_surviving > _SHALLOW_LOOKAHEAD_MAX_SCENARIOS:
        return None
    if result.n_surviving == 0 or not candidate_positions:
        return None

    unrevealed = _unrevealed_positions(state)
    blocked = _active_witch_blocked_positions(state, result)
    unrevealed = [p for p in unrevealed if p not in blocked]
    if not unrevealed:
        return None

    scenarios = result.surviving_scenarios
    start_time = time.perf_counter()
    timeout_sec = _SHALLOW_LOOKAHEAD_TIMEOUT_MS / 1000.0

    for reveal_pos in unrevealed:
        # Check timeout
        if time.perf_counter() - start_time > timeout_sec:
            return None

        # Partition scenarios by fingerprint (what player observes)
        partitions: dict[tuple, list[int]] = {}
        for idx, scenario in enumerate(scenarios):
            fp = _compute_position_fingerprint(reveal_pos, scenario, state)
            partitions.setdefault(fp, []).append(idx)

        # For each partition, check if forced execution exists
        all_partitions_have_forced = True
        forced_exec_per_partition: dict[tuple, int] = {}

        for fp_key, indices in partitions.items():
            if time.perf_counter() - start_time > timeout_sec:
                return None

            # Build a sub-result with only this partition's scenarios
            sub_scenarios = [scenarios[i] for i in indices]
            sub_result = SolverResult(
                definite_evil=[],
                definite_good=[],
                bombardier_positions=result.bombardier_positions,
                n_scenarios=len(sub_scenarios),
                n_surviving=len(sub_scenarios),
                surviving_scenarios=sub_scenarios,
                reasoning=[],
            )

            forced_pos = _find_forced_execution(state, sub_result, candidate_positions)
            if forced_pos is not None:
                forced_exec_per_partition[fp_key] = forced_pos
            else:
                all_partitions_have_forced = False
                break

        if all_partitions_have_forced:
            # Found a 2-step plan: reveal reveal_pos, then execute based on outcome
            exec_targets = set(forced_exec_per_partition.values())
            exec_summary = ", ".join(f"#{p}" for p in sorted(exec_targets))
            return Action(
                "reveal",
                position=reveal_pos,
                reasoning=(
                    f"2-turn lookahead: reveal #{reveal_pos}, then forced execution "
                    f"guarantees win (targets: {exec_summary} depending on outcome, "
                    f"{len(partitions)} branches all have forced wins)."
                ),
            )

    return None


# ============================================================
# E4 — 50/50 tiebreaker framework
# ============================================================

def _tiebreak_score(
    pos: int,
    state: GameState,
    result: SolverResult,
) -> tuple[float, float, float]:
    """Tiebreaker score for positions with similar p_evil.

    Returns (corruption_risk_penalty, role_consistency, witch_boost).
    Lower corruption = safer. Fewer distinct evil roles = more predictable.
    Witch likelihood = bonus for unblocking reveals.

    Used as secondary sort key when primary ranking_score is within 0.01 margin.
    """
    # 1. Corruption risk (lower = safer, so negate for "higher is better")
    corr = _corruption_risk(pos, result, state)
    corruption_penalty = -corr  # Higher = less corrupted = safer

    # 2. Role consistency: count distinct evil roles this position could be
    # Fewer distinct roles = more predictable outcome
    evil_roles = set()
    for s in result.surviving_scenarios:
        if pos in s.evil_positions:
            evil_roles.add(s.evil_positions[pos])
        elif pos == s.puppet_position:
            evil_roles.add("Puppet")
    # Normalize: 1 role = best (1.0), many roles = worse (closer to 0.0)
    if evil_roles:
        role_consistency = 1.0 / len(evil_roles)
    else:
        role_consistency = 0.0  # Not evil in any scenario

    # 3. Witch likelihood: boost if Witch is a possible evil role here
    witch_count = sum(1 for s in result.surviving_scenarios
                      if s.evil_positions.get(pos) == "Witch")
    witch_boost = witch_count / max(1, result.n_surviving)

    return (corruption_penalty, role_consistency, witch_boost)


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
    dead = set(state.executed) | set(state.night_kills)
    for card in state.cards:
        if card.position in dead or card.position in used_abilities:
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
    """Judge2: does the target appear to be lying?"""
    return truth_appearance_status(target, scenario, state) == TruthStatus.LYING


def _dreamer_effective_role(target: int, scenario: Scenario, state: GameState) -> str:
    """Best-known real role the shipped Dreamer could name for a target."""
    role = effective_role_at(target, scenario, state)
    return role.replace("_", " ") if role else "Unknown"


def _dreamer_role_identity(role: Optional[str]) -> Optional[str]:
    """Canonicalize the CharacterData identity represented by a role name."""
    if not role:
        return None
    display = " ".join(role.replace("_", " ").split())
    card_def = get_card(display)
    return card_def.name if card_def else display


def _dreamer_real_role(
    position: int,
    scenario: Scenario,
    state: GameState,
) -> str:
    """Return the scenario's real CharacterData role, not registerAs."""
    return _dreamer_role_identity(
        _dreamer_effective_role(position, scenario, state)
    ) or "Unknown"


def _dreamer_known_bluff(
    position: int,
    scenario: Scenario,
    state: GameState,
) -> Optional[str]:
    """Project a live bluff pointer when the scenario abstraction proves one.

    ``Scenario`` has no generic bluff field.  Its Evil, Puppet, Drunk, and
    Doppelganger placements are the cases where a differing visible role is a
    live bluff.  A differing generated-role trace is deliberately not treated
    as one: Dreamer reads ``dataRef``/``bluff`` and ignores ``registerAs``.
    """
    card = get_card_at(position, state)
    if card is None:
        return None

    real = _dreamer_real_role(position, scenario, state)
    apparent = _dreamer_role_identity(card.apparent_role)
    if apparent is None or apparent == real:
        return None

    has_modeled_bluff = (
        scenario_is_evil(position, scenario)
        or position == scenario.doppelganger_position
        or position == scenario.drunk_position
        or position in state.executed_evil_roles
    )
    return apparent if has_modeled_bluff else None


def _dreamer_observation(roles: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Canonical unordered role options; native randomizes display order."""
    return tuple(sorted(roles, key=lambda role: (role.casefold(), role)))


def _dreamer_board_entries(
    scenario: Scenario,
    state: GameState,
) -> list[tuple[str, Optional[str]]]:
    """Return current board entries as their real and known bluff identities."""
    return [
        (
            _dreamer_real_role(position, scenario, state),
            _dreamer_known_bluff(position, scenario, state),
        )
        for position in range(1, state.n_cards + 1)
    ]


def _dreamer_add_probability(
    likelihoods: dict[tuple[str, ...], float],
    roles: list[str] | tuple[str, ...],
    probability: float,
) -> None:
    observation = _dreamer_observation(roles)
    likelihoods[observation] = likelihoods.get(observation, 0.0) + probability


def _dreamer_honest_likelihoods(
    targets: list[int] | tuple[int, int],
    scenario: Scenario,
    state: GameState,
) -> dict[tuple[str, ...], float]:
    """Exact current-build P(role options | scenario) for truthful Dreamer."""
    if len(targets) != 2:
        raise ValueError("Dreamer requires exactly two targets")

    target_positions = list(targets)
    target_roles = [
        _dreamer_real_role(position, scenario, state)
        for position in target_positions
    ]
    if any(role.casefold() == "wretch" for role in target_roles):
        return {("Cabbage",): 1.0}

    likelihoods: dict[tuple[str, ...], float] = {}
    board_entries = _dreamer_board_entries(scenario, state)
    if any(real == "Unknown" for real, _ in board_entries):
        # The scenario abstraction cannot reconstruct an unflipped/blocked
        # card's CharacterData identity. Do not invent a native observation or
        # information score from that incomplete projection.
        return {}

    # Native chooses either selected character as the truthful anchor uniformly.
    for anchor_index, anchor_role in enumerate(target_roles):
        other_position = target_positions[1 - anchor_index]
        other_bluff = _dreamer_known_bluff(other_position, scenario, state)
        if other_bluff is not None and other_bluff != anchor_role:
            _dreamer_add_probability(
                likelihoods, (anchor_role, other_bluff), 0.5
            )
            continue

        # The authored usuallyDisguised pool is empty in the pinned build.  The
        # fallback samples board entries, not unique role identities, and only
        # excludes entries whose dataRef or bluff matches the anchor.  In
        # particular, the other selected target remains eligible.
        eligible_roles = [
            real
            for real, bluff in board_entries
            if real != anchor_role and bluff != anchor_role
        ]
        if not eligible_roles:
            # Native indexes the candidate list without a null fallback. A
            # valid shipped board supplies an entry; an incomplete synthetic
            # projection must not fabricate a one-role result.
            return {}

        branch_probability = 0.5 / len(eligible_roles)
        for fallback_role in eligible_roles:
            _dreamer_add_probability(
                likelihoods,
                (anchor_role, fallback_role),
                branch_probability,
            )

    return likelihoods


def _dreamer_liar_likelihoods(
    targets: list[int] | tuple[int, int],
    scenario: Scenario,
    state: GameState,
) -> dict[tuple[str, ...], float]:
    """Exact current-build P(role options | scenario) for lying Dreamer."""
    if len(targets) != 2:
        raise ValueError("Dreamer requires exactly two targets")

    target_positions = list(targets)
    target_reals = [
        _dreamer_real_role(position, scenario, state)
        for position in target_positions
    ]
    target_bluffs = [
        _dreamer_known_bluff(position, scenario, state)
        for position in target_positions
    ]

    # CharacterPickedDrunk accepts selected bluffs before applying exclusions.
    # This intentionally permits a selected bluff to equal the other target's
    # real identity.  Only duplicate bluff identities are collapsed.
    initial_options: list[str] = []
    for bluff in target_bluffs:
        if bluff is not None and bluff not in initial_options:
            initial_options.append(bluff)

    entries = _dreamer_board_entries(scenario, state)
    if any(real == "Unknown" for real, _ in entries):
        return {}
    identity_pool: list[str] = []
    for identity in [real for real, _ in entries] + [
        bluff for _, bluff in entries if bluff is not None
    ]:
        if identity not in identity_pool:
            identity_pool.append(identity)

    selected_identities = set(target_reals)
    selected_identities.update(
        bluff for bluff in target_bluffs if bluff is not None
    )
    likelihoods: dict[tuple[str, ...], float] = {}
    available_identity_count = sum(
        identity not in selected_identities for identity in identity_pool
    )
    if available_identity_count < 2 - len(initial_options):
        # Native would fault while indexing an empty helper pool. Avoid
        # manufacturing a partial role-pair observation for synthetic states.
        return {}

    def fill(options: list[str], probability: float) -> None:
        if len(options) >= 2:
            _dreamer_add_probability(likelihoods, options[:2], probability)
            return

        # The shipped roster has no usuallyDisguised-authored candidates, so
        # every missing option comes from the unique real-then-bluff identity
        # pool.  Recompute after each draw for uniform sampling without
        # replacement.
        candidates = [
            identity
            for identity in identity_pool
            if identity not in selected_identities and identity not in options
        ]
        if not candidates:
            return

        branch_probability = probability / len(candidates)
        for identity in candidates:
            fill(options + [identity], branch_probability)

    fill(initial_options, 1.0)
    return likelihoods


def _dreamer_observation_likelihoods(
    targets: list[int] | tuple[int, int],
    ability_pos: int,
    scenario: Scenario,
    state: GameState,
) -> dict[tuple[str, ...], float]:
    """Return the native Dreamer observation likelihood for one scenario."""
    if truth_status(ability_pos, scenario, state) == TruthStatus.TRUTHFUL:
        return _dreamer_honest_likelihoods(targets, scenario, state)
    return _dreamer_liar_likelihoods(targets, scenario, state)


def _druid_ground_truth(targets: list[int], scenario: Scenario, state: GameState) -> str:
    """Druid: find outcast among targets, or 'none'."""
    for t in targets:
        if scenario_is_evil(t, scenario):
            continue
        role = effective_role_at(t, scenario, state)
        if role:
            card_def = get_card(role)
            role_key = role.lower().replace(" ", "").replace("_", "")
            if card_def and card_def.role == Role.OUTCAST and role_key != "wretch":
                return role.replace("_", " ")
    return "none"


def _slayer_ground_truth(target: int, scenario: Scenario) -> bool:
    """Slayer: is target evil?"""
    return scenario_is_evil(target, scenario)


def _pd_observation_likelihoods(
    target: int,
    ability_pos: int,
    scenario: Scenario,
    state: GameState,
) -> dict[tuple, float]:
    """Native visible Plague Doctor output distribution for one target.

    The shipped callback reads the target's active Corrupted status directly.
    A truthful check that reports Corrupted uniformly names any runtime/
    registered Evil character; Bluff performs the inverse and uniformly names
    a Good character as Evil. CurrentCharacters retains dead cards. A self-
    check is always displayed as clean, even if the callback drew a hidden
    second reference before formatting.
    """
    if target == ability_pos:
        return {("clean",): 1.0}

    truthful = truth_status(ability_pos, scenario, state) == TruthStatus.TRUTHFUL
    actually_corrupted = target in scenario.corrupted
    reports_corrupted = actually_corrupted if truthful else not actually_corrupted
    if not reports_corrupted:
        return {("clean",): 1.0}

    reveal_alignment = Alignment.EVIL if truthful else Alignment.GOOD
    candidates = [
        position
        for position in range(1, state.n_cards + 1)
        if effective_alignment(position, scenario, state) == reveal_alignment
    ]
    if not candidates:
        # Native has no empty-pool guard. Treat such a synthetic world as having
        # no valid visible observation rather than inventing a fallback.
        return {}

    probability = 1.0 / len(candidates)
    return {
        ("corrupted", position): probability
        for position in candidates
    }


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

    corr = _corruption_risk(ability_pos, result, state)
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
    """Recommend a Judge2 target by deterministic observed-branch entropy."""
    scenario_count = len(result.surviving_scenarios)
    if scenario_count == 0:
        return None

    best_target = None
    best_entropy = -1.0
    best_counts = (0, 0)

    # Native picker accepts every board Character, including the actor and dead
    # cards.  Prefer a non-self target, then the lowest position, on exact ties.
    ordered_targets = sorted(
        set(judge_targets),
        key=lambda target: (target == ability_pos, target),
    )
    for target in ordered_targets:
        observed_counts = {True: 0, False: 0}

        for scenario in result.surviving_scenarios:
            actual = _judge_ground_truth(target, scenario, state)
            truth = truth_status(ability_pos, scenario, state)
            observed = actual if truth == TruthStatus.TRUTHFUL else (not actual)
            observed_counts[observed] += 1

        entropy = _shannon_entropy(
            [observed_counts[False], observed_counts[True]]
        )
        if entropy > best_entropy:
            best_entropy = entropy
            best_target = target
            best_counts = (observed_counts[False], observed_counts[True])

    if best_target is None:
        return None

    return AbilityRecommendation(
        position=ability_pos,
        ability_name="Judge",
        targets=[best_target],
        score=best_entropy,
        reasoning=(
            f"Entropy {best_entropy:.3f} bits over deterministic native Judge "
            f"observations (not lying={best_counts[0]}, lying={best_counts[1]})"
        ),
        warnings=[],
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

    corr = _corruption_risk(ability_pos, result, state)
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


def _information_from_observation_likelihoods(
    likelihoods: list[dict[tuple, float]],
) -> tuple[float, float, int]:
    """Return mutual information for equally likely scenario distributions."""
    scenario_count = len(likelihoods)
    if scenario_count == 0:
        return (0.0, 0.0, 0)

    normalized_likelihoods: list[dict[tuple, float]] = []
    for distribution in likelihoods:
        total = sum(distribution.values())
        if total <= 0:
            return (0.0, math.log2(scenario_count), 0)
        # Native models are normalized. Renormalizing here makes
        # posterior arithmetic robust to floating-point accumulation.
        normalized_likelihoods.append({
            observation: probability / total
            for observation, probability in distribution.items()
        })

    marginal_weights: dict[tuple, float] = {}
    for distribution in normalized_likelihoods:
        for observation, probability in distribution.items():
            marginal_weights[observation] = (
                marginal_weights.get(observation, 0.0) + probability
            )

    expected_posterior_entropy = 0.0
    for observation, likelihood_sum in marginal_weights.items():
        if likelihood_sum <= 0:
            continue
        posterior_entropy = 0.0
        for distribution in normalized_likelihoods:
            posterior = distribution.get(observation, 0.0) / likelihood_sum
            if posterior > 0:
                posterior_entropy -= posterior * math.log2(posterior)
        observation_probability = likelihood_sum / scenario_count
        expected_posterior_entropy += observation_probability * posterior_entropy

    prior_entropy = math.log2(scenario_count)
    mutual_information = max(
        0.0,
        min(prior_entropy, prior_entropy - expected_posterior_entropy),
    )
    return (
        mutual_information,
        expected_posterior_entropy,
        len(marginal_weights),
    )


def _dreamer_information_for_targets(
    targets: list[int] | tuple[int, int],
    ability_pos: int,
    state: GameState,
    scenarios: list[Scenario],
) -> tuple[float, float, int]:
    """Return (mutual information, expected posterior entropy, outcomes)."""
    return _information_from_observation_likelihoods([
        _dreamer_observation_likelihoods(targets, ability_pos, scenario, state)
        for scenario in scenarios
    ])


def _recommend_dreamer_ability(
    ability_pos: int,
    candidate_targets: list[list[int]],
    state: GameState,
    result: SolverResult,
) -> Optional[AbilityRecommendation]:
    """Recommend a Dreamer pair using its stochastic native likelihood."""
    scenarios = result.surviving_scenarios
    if not scenarios or not candidate_targets:
        return None

    best_targets: Optional[tuple[int, int]] = None
    best_information = -1.0
    best_expected_entropy = float("inf")
    best_outcome_count = 0

    for targets in candidate_targets:
        target_key = tuple(sorted(targets))
        if len(target_key) != 2:
            continue
        information, expected_entropy, outcome_count = (
            _dreamer_information_for_targets(
                target_key, ability_pos, state, scenarios
            )
        )
        if outcome_count == 0:
            continue
        if (
            best_targets is None
            or information > best_information + 1e-12
            or (
                math.isclose(
                    information,
                    best_information,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and target_key < best_targets
            )
        ):
            best_targets = target_key
            best_information = information
            best_expected_entropy = expected_entropy
            best_outcome_count = outcome_count

    if best_targets is None:
        return None

    liar_probability = sum(
        truth_status(ability_pos, scenario, state) == TruthStatus.LYING
        for scenario in scenarios
    ) / len(scenarios)
    warnings = []
    if liar_probability > 0:
        warnings.append(
            f"Lying Dreamer path: {liar_probability:.0%} -- included in native likelihood"
        )

    return AbilityRecommendation(
        position=ability_pos,
        ability_name="Dreamer",
        targets=list(best_targets),
        score=best_information,
        reasoning=(
            f"Mutual information {best_information:.3f} bits; "
            f"expected posterior entropy {best_expected_entropy:.3f} bits "
            f"across {best_outcome_count} native role-pair observations"
        ),
        warnings=warnings,
    )


def _pd_information_for_target(
    target: int,
    ability_pos: int,
    state: GameState,
    scenarios: list[Scenario],
) -> tuple[float, float, int]:
    """Return native PD mutual information for one selectable character."""
    return _information_from_observation_likelihoods([
        _pd_observation_likelihoods(target, ability_pos, scenario, state)
        for scenario in scenarios
    ])


def _recommend_pd_ability(
    ability_pos: int,
    candidate_targets: list[int],
    state: GameState,
    result: SolverResult,
) -> Optional[AbilityRecommendation]:
    """Recommend a PD target using its stochastic native reveal callback."""
    scenarios = result.surviving_scenarios
    if not scenarios or not candidate_targets:
        return None

    best_target: Optional[int] = None
    best_information = -1.0
    best_expected_entropy = float("inf")
    best_outcome_count = 0

    for target in candidate_targets:
        information, expected_entropy, outcome_count = _pd_information_for_target(
            target, ability_pos, state, scenarios
        )
        if outcome_count == 0:
            continue
        if best_target is None or information > best_information + 1e-12:
            best_target = target
            best_information = information
            best_expected_entropy = expected_entropy
            best_outcome_count = outcome_count

    if best_target is None:
        return None

    liar_probability = sum(
        truth_status(ability_pos, scenario, state) == TruthStatus.LYING
        for scenario in scenarios
    ) / len(scenarios)
    warnings = []
    if liar_probability > 0:
        warnings.append(
            f"Lying Plague Doctor path: {liar_probability:.0%} -- included in native likelihood"
        )
    if best_target == ability_pos:
        warnings.append("Native self-check always displays Not Corrupted")

    return AbilityRecommendation(
        position=ability_pos,
        ability_name="Plague Doctor",
        targets=[best_target],
        score=best_information,
        reasoning=(
            f"Mutual information {best_information:.3f} bits; "
            f"expected posterior entropy {best_expected_entropy:.3f} bits "
            f"across {best_outcome_count} native PD observations"
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
    """Recommend targets for a deterministic multi-valued partition ability."""
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

    corr = _corruption_risk(ability_pos, result, state)
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
            role = effective_role_at(target, s, state)
            if role and role.lower().replace(" ", "").replace("_", "") == "wretch":
                count += 1
    return count / result.n_surviving


def _slayer_terminal_loss_probability(
    target: int,
    state: GameState,
    result: SolverResult,
) -> float:
    """Chance Slayer kills a runtime-Evil current Bombardier at this seat."""
    if result.n_surviving == 0:
        return 0.0
    losing = sum(
        1
        for scenario in result.surviving_scenarios
        if effective_alignment(target, scenario, state) == Alignment.EVIL
        and _is_terminal_loss_role(effective_role_at(target, scenario, state))
    )
    return losing / result.n_surviving


def _recommend_slayer(
    ability_pos: int,
    state: GameState,
    result: SolverResult,
) -> Optional[AbilityRecommendation]:
    """Slayer: pick target with highest evil probability. Not entropy-based."""
    probs = evil_probabilities(state, result)
    # Native Slayer checks registered alignment before calling KillAndReveal.
    # A Good Bombardier or Knight registers Good and survives harmlessly. An
    # Evil bluffing as either reveals its real Evil role and is safe, but a
    # runtime-Evil Shaman destination whose *current* role is Bombardier invokes
    # the terminal death hook after KillAndReveal.
    dead = set(state.executed) | set(state.night_kills)
    candidates = [
        p
        for p in range(1, state.n_cards + 1)
        if p not in dead and p != ability_pos
    ]
    if not candidates:
        return None

    # Score each candidate accounting for Wretch HP penalty
    corr = _corruption_risk(ability_pos, result, state)
    best_pos = None
    best_score = -1
    best_prob = 0
    best_wretch = 0
    for pos in candidates:
        prob = probs.get(pos, 0)
        wretch_prob = _wretch_kill_probability(pos, state, result)
        terminal_prob = _slayer_terminal_loss_probability(pos, state, result)

        # There is no strategic continuation after this branch. Require every
        # surviving exact world to be free of the current-role death hook.
        if terminal_prob > 0:
            continue

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
    dead_positions = set(state.executed) | set(state.night_kills)
    available = [
        p for p in range(1, state.n_cards + 1)
        if p not in dead_positions
    ]

    for card in state.cards:
        pos = card.position
        if pos in dead_positions or pos in used_abilities:
            continue

        role = card.apparent_role.replace("_", " ")
        card_def = get_card(role)
        if not card_def or not card_def.activated_ability:
            continue

        # Build candidate target lists (exclude self and executed)
        others = [p for p in available if p != pos]

        if role == "Fortune Teller" and state.n_cards >= 2:
            # The shared picker runs before ordinary state/use checks. Every
            # board Character is legal, including self and dead/hidden cards.
            fortune_targets = list(range(1, state.n_cards + 1))
            candidates = [list(c) for c in combinations(fortune_targets, 2)]
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
            # Judge's shared picker accepts every board Character, including
            # self, executed/night-killed cards, and Poet.
            judge_targets = list(range(1, state.n_cards + 1))
            rec = _recommend_judge(pos, state, result, judge_targets)
            _apply_timing(rec, timing, state, recommendations)

        elif role == "Dreamer" and len(others) >= 2:
            candidates = [list(c) for c in combinations(others, 2)]
            rec = _recommend_dreamer_ability(pos, candidates, state, result)
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
            # Native's generic character picker permits self and dead cards.
            # Prefer live non-self targets on equal information, then dead
            # targets, with the always-clean self-check last.
            candidates = (
                [
                    p for p in range(1, state.n_cards + 1)
                    if p != pos and p not in dead_positions
                ]
                + sorted(p for p in dead_positions if p != pos)
                + [pos]
            )
            rec = _recommend_pd_ability(pos, candidates, state, result)
            _apply_timing(rec, timing, state, recommendations)

    return recommendations


# ============================================================
# Reveal Recommendation
# ============================================================

def _witness_observation_support(
    pos: int, scenario: Scenario, state: GameState,
) -> tuple[int, ...]:
    """Return every native Witness result possible in this scenario.

    Witness scans current physical cards without filtering dead or revealed
    cards.  A truthful Witness samples marked cards; a liar samples their exact
    board complement.  The public ``0``/NO result occurs only when the relevant
    sample set is empty.  Chancellor trace anchors are deliberately excluded:
    only a marker that survived into ``messed_up_by_evil`` is observable.
    """
    board_positions = set(range(1, state.n_cards + 1))
    affected = set(scenario.messed_up_by_evil) | set(state.night_kills)
    affected &= board_positions

    if truth_status(pos, scenario, state) == TruthStatus.TRUTHFUL:
        support = affected
    else:
        support = board_positions - affected

    return tuple(sorted(support)) if support else (0,)


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
    Knitter evil pairs (global), Architect side (global), Bard corruption distance,
    and Witness's exact marked/complement result support.
    """
    n = state.n_cards

    # Evil status
    is_evil = scenario_is_evil(pos, scenario)
    evil_role = None
    if pos in scenario.evil_positions:
        evil_role = scenario.evil_positions[pos]
    elif pos == scenario.puppet_position:
        evil_role = "Puppet"

    # Ordinary generated Outcast data is visible when this position flips.
    # Drunk/Doppelganger keep their Villager disguise, so do not leak their
    # hidden identity into a pre-flip information fingerprint.
    generated_role = None
    if (
        scenario.chancellor_trace is not None
        and pos == scenario.chancellor_trace.added_outcast_position
    ):
        role = scenario.chancellor_trace.added_outcast_role
        role_key = role.lower().replace(" ", "").replace("_", "")
        if role_key not in {"drunk", "doppelganger"}:
            generated_role = role_key

    # Active corruption can change copied Confessor/Bard clue surfaces even
    # when the underlying Drunk identity itself remains hidden.
    is_corrupted = pos in scenario.corrupted
    # Do not split worlds on a clue surface that cannot appear in this deck.
    # This preserves the existing hidden Drunk/Doppelganger projection when
    # Witness is absent, while still modeling copied/disguised Witness cards
    # whenever its CharacterData is present.
    witness_in_deck = any(
        role.lower().replace(" ", "").replace("_", "") == "witness"
        for role in state.deck.villagers
    )
    witness_support = (
        _witness_observation_support(pos, scenario, state)
        if witness_in_deck else ()
    )

    # Build set of effective-evil positions (Wretch counts)
    evil_set = []
    for p in range(1, n + 1):
        if p != pos and effective_alignment(p, scenario, state) == Alignment.EVIL:
            evil_set.append(p)

    if not evil_set:
        return (is_evil, evil_role, generated_role, is_corrupted,
                -1, "None", 0, 0, "Equal", -1, witness_support)

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

    return (is_evil, evil_role, generated_role, is_corrupted, dist_nearest, direction,
            adj_evil, pairs, arch_side, dist_corrupted, witness_support)


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
    # A current marker suppresses that public click. Without a marker, probing
    # the final hidden seat is safe and informative even if Witch is possible:
    # the verified miss becomes the observation that establishes the block.
    blocked = _active_witch_blocked_positions(state, result)
    unrevealed = [p for p in unrevealed if p not in blocked]
    if not unrevealed:
        return None

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
    1. Loss -- qualifying Bombardier death, then depleted HP
    2. Error -- 0 surviving scenarios
    3. Win -- all evil executed
    3. Execute -- definite evil found (skip Bombardier)
    3.5. Knight free check
    4. Use ability -- if EV > reveal EV (E1: expected-value scoring)
    5. Reveal -- most informative unrevealed position
    5.5. Forced execution (E5) + 2-turn lookahead (E2)
    6. Probabilistic execution -- HP-aware dynamic thresholds (E3),
       tiebreaker framework (E4) for 50/50 resolution
    7. Bombardier safety fallback
    """
    # Pre-compute evil probabilities (used in knight check, witch fallback, etc.)
    probs = evil_probabilities(state, result)
    dead_positions = set(state.executed) | set(state.night_kills)

    # Native terminal precedence is Bombardier death, HP loss, then
    # all-evils-gone win. Never recommend another action from terminal state.
    if _has_terminal_role_loss(state, result):
        return Action(
            "loss",
            reasoning=(
                "A canonical current-role Bombardier died outside Lilis Night; "
                "native resolution is already a terminal loss."
            ),
        )

    if state.hp <= 0:
        return Action("loss", reasoning="HP is depleted; the game is lost.")

    # Data error
    if result.n_surviving == 0:
        return Action("error", reasoning="No surviving scenarios -- check input data")

    # Win check
    _, max_remaining = _remaining_evil_bounds(state, result)
    if max_remaining == 0:
        return Action("win", reasoning="All evil characters have been executed!")

    # 3. Execute definite evil (skip Bombardier)
    # Note: Knights are NOT blanket-immune -- see "Knight free check" below.
    # Executing an uncertain Knight is a free check (0 HP) when uncorrupted.
    safe_executions = [p for p in result.definite_evil
                       if p not in dead_positions
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
    #
    # Gate: skip Knight check if a non-Knight candidate has >= 65% evil probability.
    # A high-confidence probabilistic target is more valuable than a free Knight check
    # (bug fix: asc27_v5 — Knight at 60% overrode 80% target).
    _KNIGHT_CHECK_THRESHOLD = 0.65
    non_knight_positions = {c.position for c in state.cards
                           if c.apparent_role not in EXECUTION_IMMUNE_ROLES
                           and c.position not in dead_positions}
    best_non_knight_prob = max((probs.get(p, 0) for p in non_knight_positions), default=0)

    knight_checks = []
    for card in state.cards:
        if (card.apparent_role in EXECUTION_IMMUNE_ROLES
                and card.position not in dead_positions
                and card.position not in result.bombardier_positions
                and card.position not in result.definite_good
                and card.position not in result.definite_evil):
            damage_risk, expected_damage, worst_damage = (
                _knight_check_damage_profile(card.position, result, state)
            )
            evil_prob = probs.get(card.position, 0)
            knight_checks.append((
                card.position,
                evil_prob,
                damage_risk,
                expected_damage,
                worst_damage,
            ))

    if knight_checks and best_non_knight_prob < _KNIGHT_CHECK_THRESHOLD:
        # Match the Rust execution chooser: probability first, then the safer
        # native consequence profile, then stable board position. Otherwise an
        # equally likely but damaging first card can hide a truly free check.
        knight_checks.sort(key=lambda x: (-x[1], x[2], x[0]))
        kpos, evil_prob, damage_risk, expected_damage, worst_damage = knight_checks[0]
        if damage_risk == 0:
            # Truly free: every scenario either blocks or kills an Evil.
            return Action(
                "execute", position=kpos,
                reasoning=f"Knight free check: #{kpos} is {evil_prob:.0%} evil. "
                          f"If real Knight, execution blocked (confirms good, 0 HP). "
                          f"If evil disguise, evil dies. No damaging branch.")
        elif damage_risk < 0.3:
            # Mostly free: use exact native branches, including generated
            # Outcasts and the Drunk/Knight extra-damage combination.
            if (state.hp > worst_damage
                    and expected_damage < state.wrong_exec_cost * 0.3):
                return Action(
                    "execute", position=kpos,
                    reasoning=f"Knight check: #{kpos} is {evil_prob:.0%} evil, "
                              f"{damage_risk:.0%} damaging-outcome risk. Expected HP cost: "
                              f"{expected_damage:.1f} (worst branch = {worst_damage} HP).",
                    warnings=[
                        f"Damage risk: {damage_risk:.0%} -- the apparent Knight may "
                        "be Corrupted or a killable generated identity"
                    ])

    # 4. Check available abilities
    ability_recs = recommend_abilities(state, result, used_abilities)
    ability_recs.sort(key=lambda r: r.score, reverse=True)

    def _with_ability_recs(action: Action) -> Action:
        action._ability_recs = ability_recs
        return action

    # 5. Check reveal
    reveal_rec = recommend_reveal(state, result)

    # E1: Choose between ability and reveal using expected-value scoring
    best_ability = ability_recs[0] if ability_recs else None

    # Compute EV for Slayer separately (accounts for Wretch HP penalty)
    if best_ability and best_ability.ability_name == "Slayer":
        slayer_target = best_ability.targets[0] if best_ability.targets else None
        wretch_prob = _wretch_kill_probability(slayer_target, state, result) if slayer_target else 0.0
        slayer_ev = _compute_ev_slayer(best_ability.score, state.wrong_exec_cost, wretch_prob)
        # High-confidence Slayer: use if EV clearly positive and score > 0.8
        if best_ability.score > 0.8 and slayer_ev > 0:
            return _with_ability_recs(Action(
                "use_ability", position=best_ability.position,
                targets=best_ability.targets,
                ability_name="Slayer",
                reasoning=f"{best_ability.reasoning} | EV={slayer_ev:.3f}",
                warnings=best_ability.warnings))

    # Compute EV for best ability (non-Slayer or lower-confidence Slayer)
    best_ability_ev = _compute_ev_ability(best_ability.score) if best_ability else -float('inf')
    # Compute EV for reveal
    best_reveal_ev = _compute_ev_reveal(reveal_rec.binary_entropy) if reveal_rec else -float('inf')

    if best_ability and reveal_rec:
        # E1: Compare using EV framework instead of threshold
        if best_ability_ev > best_reveal_ev:
            return _with_ability_recs(Action(
                "use_ability", position=best_ability.position,
                targets=best_ability.targets,
                ability_name=best_ability.ability_name,
                reasoning=f"{best_ability.reasoning} | EV={best_ability_ev:.3f} > reveal EV={best_reveal_ev:.3f}",
                warnings=best_ability.warnings))

    if reveal_rec:
        warnings = []
        if _witch_quota_might_be_active(state, result):
            n_unrevealed = len(_unrevealed_positions(state))
            if n_unrevealed <= 2:
                warnings.append("Witch reveal quota may still be active -- verify the click")
        return _with_ability_recs(Action(
            "reveal", position=reveal_rec.position,
            reasoning=reveal_rec.reasoning,
            warnings=warnings))

    if best_ability:
        # Optional tuning: if the best ability is weak AND a definite-evil
        # forced execution exists, prefer the guaranteed kill. Gated by
        # LOOKAHEAD_PREFER_FORCED_OVER_LOW_ABILITY (default OFF).
        if (LOOKAHEAD_PREFER_FORCED_OVER_LOW_ABILITY
            and best_ability.score < LOW_ABILITY_SCORE_THRESHOLD
            and result.definite_evil):
            candidate_positions = [p for p in result.definite_evil
                                   if p not in dead_positions]
            if candidate_positions:
                # Run forced-exec lookahead to verify survivability
                forced_pos = _find_forced_execution(state, result, candidate_positions)
                if forced_pos is not None:
                    return _with_ability_recs(Action(
                        "execute", position=forced_pos,
                        reasoning=(f"Tuning override: ability score={best_ability.score:.2f} "
                                   f"below threshold={LOW_ABILITY_SCORE_THRESHOLD:.2f}; "
                                   f"preferring definite-evil forced execution at #{forced_pos}"),
                        warnings=["LOOKAHEAD_PREFER_FORCED_OVER_LOW_ABILITY tuning active"],
                        forced_safe=True,
                    ))
        return _with_ability_recs(Action(
            "use_ability", position=best_ability.position,
            targets=best_ability.targets,
            ability_name=best_ability.ability_name,
            reasoning=best_ability.reasoning,
            warnings=best_ability.warnings))

    # 5.5 E5: Earlier forced execution detection — after abilities/reveals
    # Check for forcing moves before falling through to probabilistic execution.
    # This ensures abilities are considered first (steps 4-5), but forced wins
    # are detected before uncertain probabilistic execution (step 6b).
    wrong_exec_budget = state.hp // state.wrong_exec_cost if state.wrong_exec_cost > 0 else 99

    all_uncertain = [p for p, prob in probs.items()
                     if prob > 0.0 and p not in dead_positions]

    if all_uncertain:
        # 5.5a: Direct forced execution (1-step lookahead)
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
                forced_safe=True,
            ))

        # 5.5b: E2 — Shallow 2-turn lookahead (reveal + forced execution)
        lookahead_action = _shallow_lookahead(state, result, all_uncertain)
        if lookahead_action is not None:
            lookahead_action._ability_recs = ability_recs
            return lookahead_action

    # 6. Witch fallback -- can't reveal, execute by probability
    # HP-aware gating with budget-based confidence thresholds

    # 6b. Probabilistic execution
    # Bombardier candidates excluded from normal probability selection
    bombardier_candidates = {p: probs.get(p, 0) for p in result.bombardier_positions
                            if p not in dead_positions
                            and p not in result.definite_evil
                            and probs.get(p, 0) > 0.0}
    # Exclude a possible current-role Bombardier (instant loss if killed) and
    # Wretch (always wrong exec —
    # abilities see Wretch as evil, inflating evil_probability, but executing
    # Wretch is guaranteed wrong exec penalty with zero upside).
    wretch_positions = {c.position for c in state.cards
                        if c.apparent_role == "Wretch"
                        and c.position not in result.definite_evil}
    active_probs = {p: prob for p, prob in probs.items()
                    if p not in dead_positions
                    and p not in result.bombardier_positions
                    and p not in wretch_positions}
    if active_probs:
        # E4: Sort candidates by (p_evil, tiebreak_score) for stable 50/50 resolution
        # Primary: p_evil (higher = better). Secondary: tiebreak when within 0.01 margin.
        _tiebreak_margin = 0.01

        def _exec_sort_key(p: int) -> tuple:
            prob = active_probs.get(p, 0.0)
            tb = _tiebreak_score(p, state, result)
            return (prob, tb[0], tb[1], tb[2])

        sorted_candidates = sorted(active_probs.keys(), key=_exec_sort_key, reverse=True)
        best_pos = sorted_candidates[0]
        best_prob = active_probs[best_pos]

        # If Witch is blocking reveals, prefer executing the most likely Witch
        # position -- killing the Witch unblocks the last card reveal
        active_witch_blocks = _active_witch_blocked_positions(state, result)
        witch_blocked = bool(active_witch_blocks)
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

        # E3: HP-aware dynamic confidence thresholds
        _, max_remaining = _remaining_evil_bounds(state, result)
        max_remaining_evil = max(1, max_remaining)  # Avoid division by zero

        if wrong_exec_budget == 0:
            min_confidence = 0.95
            warnings.append(f"CRITICAL: HP={state.hp}, wrong exec costs {state.wrong_exec_cost} -- "
                            f"CANNOT afford a mistake! Only execute if certain.")
            if best_prob < min_confidence:
                return _with_ability_recs(Action(
                    "error", position=best_pos,
                    reasoning=f"#{best_pos} is {best_prob:.0%} likely evil but HP too low to risk "
                              f"(HP={state.hp}, cost={state.wrong_exec_cost}, "
                              f"threshold={min_confidence:.0%}). Need more info.",
                    warnings=warnings))
        elif wrong_exec_budget == 1:
            # E3: Dynamic threshold for budget=1
            hp_ratio_b1 = state.hp / state.wrong_exec_cost if state.wrong_exec_cost > 0 else 1
            min_confidence = max(0.75, 0.95 - 0.1 * (hp_ratio_b1))
            if best_prob < min_confidence:
                if bombardier_candidates:
                    # Bombardier safety: bypass the confidence threshold for
                    # the safe candidate. Killing any possible current-role
                    # Bombardier is an instant game loss.
                    warnings.append(
                        f"Bombardier safety: executing #{best_pos} ({best_prob:.0%}) despite "
                        f"low confidence — Bombardier candidate(s) {sorted(bombardier_candidates.keys())} "
                        f"risk instant game loss if executed first.")
                else:
                    warnings.append(f"CAUTION: budget=1, confidence {best_prob:.0%} < {min_confidence:.0%} threshold. "
                                    f"Consider manual override if you have extra information.")
                    return _with_ability_recs(Action(
                        "error", position=best_pos,
                        reasoning=f"#{best_pos} is {best_prob:.0%} likely evil but budget=1 requires "
                                  f">={min_confidence:.0%} confidence (HP={state.hp}, cost={state.wrong_exec_cost}).",
                        warnings=warnings))
        elif wrong_exec_budget >= 2:
            # E3: Dynamic threshold for budget >= 2
            hp_ratio = state.hp / (state.wrong_exec_cost * max_remaining_evil) if state.wrong_exec_cost > 0 else 99
            min_confidence = max(0.4, 0.6 - 0.1 * (hp_ratio - 1))
            if best_prob < min_confidence:
                warnings.append(f"Low confidence ({best_prob:.0%} < {min_confidence:.0%}) -- consider gathering more info")

        return _with_ability_recs(Action(
            "execute", position=best_pos,
            reasoning=f"No reveals available. #{best_pos} is {best_prob:.0%} likely evil "
                      f"(HP={state.hp}, budget={wrong_exec_budget} wrong execs)",
            warnings=warnings))

    # 6c. Bombardier safety fallback: when all high-probability candidates are
    # Bombardier (excluded from active_probs), prefer a non-Bombardier uncertain
    # position. A non-Bombardier mistake costs HP; killing a possible current-
    # role Bombardier ends the game immediately.
    if bombardier_candidates:
        # Include Wretch — wrong exec on Wretch = HP cost, not game loss
        safety_probs = {p: prob for p, prob in probs.items()
                       if p not in dead_positions
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
                          f"killing a current-role Bombardier = instant game loss.",
                warnings=[f"Bombardier safety play — testing non-Bombardier first "
                         f"(HP={state.hp}, budget={wrong_exec_budget} wrong execs)"]))

    # 7. Shouldn't reach here
    return _with_ability_recs(Action("error", reasoning="No valid action found"))


def _compute_confidence(action: Action, state, result, probs: dict) -> float:
    """Compute confidence score (0-1) for an action, based on action type."""
    if action.action_type in {"loss", "win"}:
        return 1.0
    elif action.action_type == "error":
        return 0.0
    elif action.action_type == "execute" and action.position is not None:
        return probs.get(action.position, 0.0)
    elif action.action_type == "reveal" and action.position is not None:
        max_bits = _shannon_entropy([1] * max(result.n_surviving, 2))
        reveal_recs = recommend_reveal(state, result)
        if reveal_recs:
            return min(1.0, reveal_recs.entropy / max(max_bits, 0.01))
        return 0.5
    elif action.action_type == "use_ability":
        if action._ability_recs:
            best = next((a for a in action._ability_recs
                         if a.position == action.position and a.ability_name == action.ability_name), None)
            if best:
                return min(1.0, best.score)
        return 0.5
    return 0.0


# ============================================================
# Display
# ============================================================

def print_recommendation(state: GameState, result: SolverResult,
                         used_abilities: list[int]):
    """Print a full strategy recommendation with confidence score."""
    probs = evil_probabilities(state, result)
    action = recommend_action(state, result, used_abilities)

    # Compute and attach confidence
    action.confidence = _compute_confidence(action, state, result, probs)

    print(f"\n=== STRATEGY RECOMMENDATION ===")
    print(f"  Action: {action.action_type.upper()}", end="")
    if action.position:
        print(f" #{action.position}", end="")
    if action.ability_name:
        print(f" ({action.ability_name})", end="")
    if action.targets:
        print(f" -> targets {['#'+str(t) for t in action.targets]}", end="")
    print(f"  (confidence: {action.confidence:.0%}, {result.n_surviving} scenarios)")
    print(f"  Reason: {action.reasoning}")
    for w in action.warnings:
        print(f"  WARNING: {w}")

    # Show smart reveal analysis for context
    if action.action_type in ("reveal", "use_ability", "execute"):
        probs = evil_probabilities(state, result)
        unrevealed = _unrevealed_positions(state)
        blocked = _active_witch_blocked_positions(state, result)
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
