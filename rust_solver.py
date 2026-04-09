"""Bridge to the Rust solver binary (crates/solver-cli).

Calls the Rust solver via subprocess, reconstructs Python Scenario/SolverResult objects.
Falls back gracefully if the binary is not found.
"""

import json
import os
import subprocess
import time
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_RUST_BINARY: Optional[str] = None


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


def rust_solve(state_dict: dict, summary_only: bool = False) -> Optional[dict]:
    """Call the Rust solver and return the raw result dict.

    Args:
        state_dict: GameState as a dict (from state.to_dict())
        summary_only: If True, omit surviving_scenarios from output

    Returns:
        Result dict with keys: definite_evil, definite_good, bombardier_positions,
        n_scenarios, n_surviving, and optionally surviving_scenarios.
        Returns None if the binary is not found or fails.
    """
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
    return data


def rust_solve_to_objects(state, summary_only: bool = False):
    """Call the Rust solver and return Python SolverResult + Scenario objects.

    Args:
        state: A GameState object (has .to_dict())
        summary_only: If True, surviving_scenarios will be empty

    Returns:
        SolverResult object, or None if Rust solver unavailable.
    """
    from solver import Scenario, SolverResult

    data = rust_solve(state.to_dict(), summary_only=summary_only)
    if data is None:
        return None

    scenarios = []
    for s in data.get("surviving_scenarios", []):
        scenarios.append(Scenario(
            evil_positions={int(k): v for k, v in s["evil_positions"].items()},
            puppet_position=s.get("puppet_position"),
            corrupted=set(s.get("corrupted", [])),
            pd_corrupted=s.get("pd_corrupted"),
            doppelganger_position=s.get("doppelganger_position"),
            drunk_position=s.get("drunk_position"),
            alchemist_cures={int(k): int(v) for k, v in s.get("alchemist_cures", {}).items()},
            chancellor_conversion=s.get("chancellor_conversion"),
        ))

    return SolverResult(
        definite_evil=list(data.get("definite_evil", [])),
        definite_good=list(data.get("definite_good", [])),
        bombardier_positions=list(data.get("bombardier_positions", [])),
        n_scenarios=data.get("n_scenarios", 0),
        n_surviving=data.get("n_surviving", 0),
        surviving_scenarios=scenarios,
        reasoning=data.get("reasoning", []),
    )


def _reset_binary():
    """Reset cached binary path (e.g., after it disappears)."""
    global _RUST_BINARY
    _RUST_BINARY = None
