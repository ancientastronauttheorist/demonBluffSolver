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
