"""Bridge to the Rust solver binary (crates/solver-cli).

Supports two modes:
- **Daemon mode** (default): Persistent subprocess with --daemon flag.
  Reuses a single process across calls via JSON-line protocol.
- **One-shot mode** (fallback): Spawns a new process per call.

Falls back gracefully if the binary is not found.
"""

import atexit
import json
import os
import subprocess
import threading
import time
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_RUST_BINARY: Optional[str] = None

# Daemon state
_daemon_proc: Optional[subprocess.Popen] = None
_daemon_lock = threading.Lock()
_DAEMON_TIMEOUT = 10  # seconds per solve


class SolverCache:
    """Cache solver results keyed on full game state (minus HP/cost).

    Uses state.to_dict() with HP and wrong_exec_cost removed as the cache key.
    This automatically includes ALL solver-relevant fields. The full JSON string
    is used as the key (zero collision risk).
    """

    def __init__(self):
        self._cache: dict[str, object] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, state_dict: dict, summary_only: bool) -> str:
        d = dict(state_dict)
        d.pop('hp', None)
        d.pop('wrong_exec_cost', None)
        d['__summary_only'] = summary_only
        return json.dumps(d, sort_keys=True)

    def get(self, state_dict: dict, summary_only: bool):
        key = self._key(state_dict, summary_only)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, state_dict: dict, summary_only: bool, result):
        key = self._key(state_dict, summary_only)
        self._cache[key] = result

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> str:
        total = self._hits + self._misses
        rate = f"{self._hits/total:.0%}" if total > 0 else "n/a"
        return f"cache: {self._hits} hits / {self._misses} misses ({rate})"


_solver_cache = SolverCache()


def _find_binary() -> Optional[str]:
    global _RUST_BINARY
    if _RUST_BINARY is not None:
        return _RUST_BINARY

    # Check env var first
    env_path = os.environ.get("DEMON_BLUFF_SOLVER")
    if env_path and os.path.isfile(env_path):
        _RUST_BINARY = env_path
        return env_path

    # Check release then debug
    for profile in ("release", "debug"):
        path = os.path.join(REPO_ROOT, "target", profile, "demon-bluff-solver.exe")
        if os.path.isfile(path):
            if profile == "debug":
                print("  [rust-solver] WARNING: using debug build — run `cargo build --release` for better performance")
            _RUST_BINARY = path
            return path
    return None


def _start_daemon() -> Optional[subprocess.Popen]:
    """Start the daemon subprocess. Returns Popen or None."""
    binary = _find_binary()
    if binary is None:
        return None
    try:
        proc = subprocess.Popen(
            [binary, "--daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",  # CRITICAL: Windows defaults to cp1252
        )
        return proc
    except (FileNotFoundError, OSError) as e:
        print(f"  [rust-solver] daemon start failed: {e}")
        _reset_binary()
        return None


def _ensure_daemon() -> Optional[subprocess.Popen]:
    """Return a live daemon process, starting one if needed."""
    global _daemon_proc
    # Fast path: daemon already alive
    if _daemon_proc is not None and _daemon_proc.poll() is None:
        return _daemon_proc
    # Need to (re)start
    _daemon_proc = _start_daemon()
    return _daemon_proc


def _kill_daemon():
    """Kill the daemon if running."""
    global _daemon_proc
    if _daemon_proc is not None:
        try:
            _daemon_proc.stdin.close()
        except Exception:
            pass
        try:
            _daemon_proc.kill()
            _daemon_proc.wait(timeout=2)
        except Exception:
            pass
        _daemon_proc = None


def _drain_stderr(proc: subprocess.Popen):
    """Non-blocking drain of stderr to prevent pipe buffer overflow."""
    if proc.stderr is None:
        return
    try:
        # On Windows, we can't do non-blocking reads easily.
        # Read whatever is available with a tiny timeout via threading.
        import selectors
        sel = selectors.DefaultSelector()
        sel.register(proc.stderr, selectors.EVENT_READ)
        while sel.select(timeout=0):
            data = proc.stderr.readline()
            if not data:
                break
        sel.close()
    except Exception:
        pass


def _read_line_with_timeout(proc: subprocess.Popen, timeout: float) -> Optional[str]:
    """Read one line from the daemon's stdout with a timeout.

    Returns the line string, or None on timeout/error.
    """
    result = [None]
    error = [None]

    def _reader():
        try:
            line = proc.stdout.readline()
            result[0] = line
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        # Timed out — daemon is stuck, kill it
        return None

    if error[0] is not None:
        return None

    return result[0]


def shutdown_daemon():
    """Shutdown the daemon process. Called at exit."""
    with _daemon_lock:
        _kill_daemon()


# Register cleanup
atexit.register(shutdown_daemon)


def _daemon_solve(state_dict: dict, summary_only: bool = False) -> Optional[dict]:
    """Send a solve request to the daemon. Returns result dict or None on failure."""
    with _daemon_lock:
        proc = _ensure_daemon()
        if proc is None:
            return None

        # Build the request — inject __summary into the state dict if needed
        request = dict(state_dict)
        if summary_only:
            request["__summary"] = True

        request_line = json.dumps(request, separators=(",", ":")) + "\n"

        t0 = time.perf_counter()
        try:
            proc.stdin.write(request_line)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            print(f"  [rust-solver] daemon write error: {e}")
            _kill_daemon()
            return None

        # Drain stderr to prevent buffer overflow
        _drain_stderr(proc)

        # Read response with timeout
        response_line = _read_line_with_timeout(proc, _DAEMON_TIMEOUT)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if response_line is None:
            print("  [rust-solver] daemon TIMEOUT or read error")
            _kill_daemon()
            return None

        if response_line == "":
            # EOF — daemon died
            print("  [rust-solver] daemon EOF (process died)")
            _kill_daemon()
            return None

        try:
            data = json.loads(response_line)
        except json.JSONDecodeError as e:
            print(f"  [rust-solver] daemon JSON parse error: {e}")
            print(f"  [rust-solver]   raw: {response_line[:200]}")
            _kill_daemon()
            return None

        if "error" in data:
            print(f"  [rust-solver] solver error: {data['error']}")
            # Don't kill daemon for solver-level errors (bad input, etc.)
            return None

        data["_elapsed_ms"] = elapsed_ms
        data["_daemon"] = True
        return data


def _oneshot_solve(state_dict: dict, summary_only: bool = False) -> Optional[dict]:
    """Original one-shot mode: spawn a new process per call."""
    binary = _find_binary()
    if binary is None:
        return None

    state_json = json.dumps(state_dict)
    cmd = [binary]
    if summary_only:
        cmd.append("--summary")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            input=state_json,
            capture_output=True,
            text=True,
            encoding="utf-8",  # CRITICAL: Windows defaults to cp1252
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        print("  [rust-solver] TIMEOUT (10s)")
        return None
    except FileNotFoundError:
        print(f"  [rust-solver] binary not found: {binary}")
        _reset_binary()
        return None

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if proc.returncode != 0:
        stderr_preview = proc.stderr[:200] if proc.stderr else ""
        print(f"  [rust-solver] ERROR (exit {proc.returncode}): {stderr_preview}")
        return None

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"  [rust-solver] JSON parse error: {e}")
        return None

    if "error" in data:
        print(f"  [rust-solver] solver error: {data['error']}")
        return None

    data["_elapsed_ms"] = elapsed_ms
    data["_daemon"] = False
    return data


def rust_solve(state_dict: dict, summary_only: bool = False) -> Optional[dict]:
    """Call the Rust solver and return the raw result dict.

    Tries daemon mode first, falls back to one-shot if daemon fails.

    Args:
        state_dict: GameState as a dict (from state.to_dict())
        summary_only: If True, omit surviving_scenarios from output

    Returns:
        Result dict with keys: definite_evil, definite_good, bombardier_positions,
        n_scenarios, n_surviving, and optionally surviving_scenarios.
        Returns None if the binary is not found or fails.
    """
    # Try daemon mode first
    result = _daemon_solve(state_dict, summary_only=summary_only)
    if result is not None:
        return result

    # Fall back to one-shot
    return _oneshot_solve(state_dict, summary_only=summary_only)


def _require_exact_dict(value, name: str, expected_keys: set[str]) -> dict:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact dict")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{name} has invalid keys (missing={missing}, extra={extra})"
        )
    return value


def _require_u8(value, name: str, *, minimum: int = 0, maximum: int = 255) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _require_u16(value, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if not 0 <= value <= 65535:
        raise ValueError(f"{name} must be in [0, 65535]")
    return value


def _require_board_position(value, name: str, n_cards: int) -> int:
    return _require_u8(value, name, minimum=1, maximum=n_cards)


def _require_exact_string(value, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    return value


def _parse_twin_trace(raw_trace, n_cards: int):
    """Parse the tagged Rust Twin trace, preserving legacy absence."""
    if raw_trace is None:
        return None

    from solver import (
        TwinNeighborSide,
        TwinStartKind,
        TwinStartOutcome,
        TwinTrace,
    )

    n_cards = _require_u8(n_cards, "n_cards", minimum=1)
    raw_trace = _require_exact_dict(
        raw_trace,
        "twin_trace",
        {"actor_position", "outcome"},
    )
    actor_position = _require_board_position(
        raw_trace["actor_position"],
        "twin_trace.actor_position",
        n_cards,
    )
    raw_outcome = _require_exact_dict(
        raw_trace["outcome"],
        "twin_trace.outcome",
        {"kind"}
        if type(raw_trace["outcome"]) is dict
        and raw_trace["outcome"].get("kind") == "no_demon"
        else {
            "kind",
            "demon_occurrence_index",
            "demon_anchor_position",
            "neighbor_side",
            "neighbor_position",
            "neighbor_pre_swap_role",
        },
    )
    raw_kind = _require_exact_string(
        raw_outcome["kind"],
        "twin_trace.outcome.kind",
    )
    try:
        kind = TwinStartKind(raw_kind)
    except ValueError as exc:
        raise ValueError(f"unknown Twin outcome kind: {raw_kind!r}") from exc
    if kind is TwinStartKind.NO_DEMON:
        outcome = TwinStartOutcome(kind=kind)
    else:
        raw_side = _require_exact_string(
            raw_outcome["neighbor_side"],
            "twin_trace.outcome.neighbor_side",
        )
        try:
            neighbor_side = TwinNeighborSide(raw_side)
        except ValueError as exc:
            raise ValueError(f"unknown Twin neighbor side: {raw_side!r}") from exc
        neighbor_pre_swap_role = _require_exact_string(
            raw_outcome["neighbor_pre_swap_role"],
            "twin_trace.outcome.neighbor_pre_swap_role",
        )
        if not neighbor_pre_swap_role.strip():
            raise ValueError("Twin neighbor pre-swap role must be nonempty")
        outcome = TwinStartOutcome(
            kind=kind,
            demon_occurrence_index=_require_u8(
                raw_outcome["demon_occurrence_index"],
                "twin_trace.outcome.demon_occurrence_index",
            ),
            demon_anchor_position=_require_board_position(
                raw_outcome["demon_anchor_position"],
                "twin_trace.outcome.demon_anchor_position",
                n_cards,
            ),
            neighbor_side=neighbor_side,
            neighbor_position=_require_board_position(
                raw_outcome["neighbor_position"],
                "twin_trace.outcome.neighbor_position",
                n_cards,
            ),
            neighbor_pre_swap_role=neighbor_pre_swap_role,
        )
    return TwinTrace(
        actor_position=actor_position,
        outcome=outcome,
    )


def _parse_twin_recipient_bluff_trace(raw_trace, n_cards: int):
    """Parse an exact offline Twin-recipient bluff trace."""
    if raw_trace is None:
        return None

    from solver import (
        BluffAcquisitionSource,
        BluffAcquisitionSourceKind,
        TwinRecipientBluffTrace,
    )

    n_cards = _require_u8(n_cards, "n_cards", minimum=1)
    raw_trace = _require_exact_dict(
        raw_trace,
        "twin_recipient_bluff_trace",
        {
            "recipient_position",
            "acquisition_ordinal",
            "bluff_role",
            "source",
        },
    )
    recipient_position = _require_board_position(
        raw_trace["recipient_position"],
        "twin_recipient_bluff_trace.recipient_position",
        n_cards,
    )
    acquisition_ordinal = _require_u16(
        raw_trace["acquisition_ordinal"],
        "twin_recipient_bluff_trace.acquisition_ordinal",
    )
    bluff_role = _require_exact_string(
        raw_trace["bluff_role"],
        "twin_recipient_bluff_trace.bluff_role",
    )
    if not bluff_role.strip():
        raise ValueError("Twin recipient bluff role must be nonempty")

    raw_source = _require_exact_dict(
        raw_trace["source"],
        "twin_recipient_bluff_trace.source",
        {"kind", "occurrence_index"},
    )
    raw_kind = _require_exact_string(
        raw_source["kind"],
        "twin_recipient_bluff_trace.source.kind",
    )
    try:
        source_kind = BluffAcquisitionSourceKind(raw_kind)
    except ValueError as exc:
        raise ValueError(
            f"unknown Twin recipient bluff source kind: {raw_kind!r}"
        ) from exc
    source = BluffAcquisitionSource(
        kind=source_kind,
        occurrence_index=_require_u16(
            raw_source["occurrence_index"],
            "twin_recipient_bluff_trace.source.occurrence_index",
        ),
    )
    return TwinRecipientBluffTrace(
        recipient_position=recipient_position,
        acquisition_ordinal=acquisition_ordinal,
        bluff_role=bluff_role,
        source=source,
    )


def _parse_puppeteer_trace(raw_trace, n_cards: int):
    """Parse the tagged Rust Puppeteer trace, preserving legacy absence."""
    if raw_trace is None:
        return None

    from solver import (
        PuppeteerNeighborSide,
        PuppeteerStartKind,
        PuppeteerStartOutcome,
        PuppeteerTrace,
    )

    n_cards = _require_u8(n_cards, "n_cards", minimum=1)
    raw_trace = _require_exact_dict(
        raw_trace,
        "puppeteer_trace",
        {"actor_position", "outcome"},
    )
    actor_position = _require_board_position(
        raw_trace["actor_position"],
        "puppeteer_trace.actor_position",
        n_cards,
    )
    raw_outcome = _require_exact_dict(
        raw_trace["outcome"],
        "puppeteer_trace.outcome",
        {"kind"}
        if type(raw_trace["outcome"]) is dict
        and raw_trace["outcome"].get("kind") == "no_candidate"
        else {
            "kind",
            "candidate_occurrence_index",
            "neighbor_side",
            "target_position",
            "erased_villager_role",
        },
    )
    raw_kind = _require_exact_string(
        raw_outcome["kind"],
        "puppeteer_trace.outcome.kind",
    )
    try:
        kind = PuppeteerStartKind(raw_kind)
    except ValueError as exc:
        raise ValueError(f"unknown Puppeteer outcome kind: {raw_kind!r}") from exc

    if kind is PuppeteerStartKind.NO_CANDIDATE:
        outcome = PuppeteerStartOutcome(kind=kind)
    else:
        raw_side = _require_exact_string(
            raw_outcome["neighbor_side"],
            "puppeteer_trace.outcome.neighbor_side",
        )
        try:
            neighbor_side = PuppeteerNeighborSide(raw_side)
        except ValueError as exc:
            raise ValueError(
                f"unknown Puppeteer neighbor side: {raw_side!r}"
            ) from exc
        erased_role = _require_exact_string(
            raw_outcome["erased_villager_role"],
            "puppeteer_trace.outcome.erased_villager_role",
        )
        if not erased_role.strip():
            raise ValueError("Puppeteer erased Villager role must be nonempty")
        outcome = PuppeteerStartOutcome(
            kind=kind,
            candidate_occurrence_index=_require_u8(
                raw_outcome["candidate_occurrence_index"],
                "puppeteer_trace.outcome.candidate_occurrence_index",
            ),
            neighbor_side=neighbor_side,
            target_position=_require_board_position(
                raw_outcome["target_position"],
                "puppeteer_trace.outcome.target_position",
                n_cards,
            ),
            erased_villager_role=erased_role,
        )
    return PuppeteerTrace(
        actor_position=actor_position,
        outcome=outcome,
    )


def _parse_pre_twin_current_roles(raw_roles, n_cards: int) -> dict[int, str]:
    """Parse the complete pre-first-writer CharacterData map from Rust."""
    if raw_roles is None:
        return {}
    if type(raw_roles) is not dict:
        raise TypeError("pre_twin_current_roles must be an exact dict")

    n_cards = _require_u8(n_cards, "n_cards", minimum=1)
    parsed: dict[int, str] = {}
    for raw_position, raw_role in raw_roles.items():
        if type(raw_position) is not str or not raw_position.isdecimal():
            raise TypeError(
                "pre_twin_current_roles keys must be decimal position strings"
            )
        position = _require_board_position(
            int(raw_position),
            f"pre_twin_current_roles[{raw_position!r}]",
            n_cards,
        )
        if position in parsed:
            raise ValueError(
                "pre_twin_current_roles contains duplicate normalized positions"
            )
        role = _require_exact_string(
            raw_role,
            f"pre_twin_current_roles[{raw_position!r}]",
        )
        if not role.strip():
            raise ValueError("pre-Twin current roles must be nonempty")
        parsed[position] = role
    if parsed and set(parsed) != set(range(1, n_cards + 1)):
        raise ValueError(
            "pre_twin_current_roles must cover every board position when present"
        )
    return parsed


def rust_solve_to_objects(state, summary_only: bool = False):
    """Call the Rust solver and return Python SolverResult + Scenario objects.

    Uses SolverCache to avoid redundant solves when state hasn't changed.

    Args:
        state: A GameState object (has .to_dict())
        summary_only: If True, surviving_scenarios will be empty

    Returns:
        SolverResult object, or None if Rust solver unavailable.
    """
    from solver import ChancellorTrace, Scenario, ShamanTrace, SolverResult

    state_dict = state.to_dict()

    # Check cache first
    cached = _solver_cache.get(state_dict, summary_only)
    if cached is not None:
        return cached

    data = rust_solve(state_dict, summary_only=summary_only)
    if data is None:
        return None

    scenarios = []
    for s in data.get("surviving_scenarios", []):
        raw_trace = s.get("chancellor_trace")
        trace = None
        if raw_trace is not None:
            trace = ChancellorTrace(
                original_positions=list(raw_trace.get("original_positions", [])),
                added_outcast_position=int(raw_trace["added_outcast_position"]),
                added_outcast_role=str(raw_trace["added_outcast_role"]),
                affected_anchor_positions=sorted({
                    int(position)
                    for position in raw_trace.get("affected_anchor_positions", [])
                }),
            )
        raw_shaman_trace = s.get("shaman_trace")
        shaman_trace = None
        if raw_shaman_trace is not None:
            shaman_trace = ShamanTrace(
                source_position=int(raw_shaman_trace["source_position"]),
                target_position=int(raw_shaman_trace["target_position"]),
                copied_role=str(raw_shaman_trace["copied_role"]),
                target_previous_roles=[
                    str(role)
                    for role in raw_shaman_trace.get("target_previous_roles", [])
                ],
            )
        twin_trace = _parse_twin_trace(
            s.get("twin_trace"),
            state_dict.get("n_cards"),
        )
        twin_recipient_bluff_trace = _parse_twin_recipient_bluff_trace(
            s.get("twin_recipient_bluff_trace"),
            state_dict.get("n_cards"),
        )
        puppeteer_trace = _parse_puppeteer_trace(
            s.get("puppeteer_trace"),
            state_dict.get("n_cards"),
        )
        pre_twin_current_roles = _parse_pre_twin_current_roles(
            s.get("pre_twin_current_roles"),
            state_dict.get("n_cards"),
        )
        scenarios.append(Scenario(
            evil_positions={int(k): v for k, v in s["evil_positions"].items()},
            puppet_position=s.get("puppet_position"),
            corrupted=set(s.get("corrupted", [])),
            pd_corrupted=s.get("pd_corrupted"),
            doppelganger_position=s.get("doppelganger_position"),
            drunk_position=s.get("drunk_position"),
            alchemist_cures={int(k): int(v) for k, v in s.get("alchemist_cures", {}).items()},
            messed_up_by_evil=set(s.get("messed_up_by_evil", [])),
            chancellor_trace=trace,
            chancellor_conversion=s.get("chancellor_conversion"),
            shaman_trace=shaman_trace,
            twin_trace=twin_trace,
            pre_twin_current_roles=pre_twin_current_roles,
            puppeteer_trace=puppeteer_trace,
            twin_recipient_bluff_trace=twin_recipient_bluff_trace,
        ))

    result_obj = SolverResult(
        definite_evil=list(data.get("definite_evil", [])),
        definite_good=list(data.get("definite_good", [])),
        bombardier_positions=list(data.get("bombardier_positions", [])),
        n_scenarios=data.get("n_scenarios", 0),
        n_surviving=data.get("n_surviving", 0),
        surviving_scenarios=scenarios,
        reasoning=data.get("reasoning", []),
    )

    # Cache the result
    _solver_cache.put(state_dict, summary_only, result_obj)

    return result_obj


def clear_solver_cache():
    """Clear the solver cache (call on new game)."""
    _solver_cache.clear()


def solver_cache_stats() -> str:
    """Get cache hit/miss stats."""
    return _solver_cache.stats


def _reset_binary():
    """Reset cached binary path (e.g., after it disappears)."""
    global _RUST_BINARY
    _RUST_BINARY = None
