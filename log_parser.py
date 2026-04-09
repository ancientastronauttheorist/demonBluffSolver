"""Player.log parser for Demon Bluff.

Parses %LOCALAPPDATA%Low/UmiArt/Demon Bluff/Player.log to extract game events.

Log format:
  - Blocks: event line -> stack trace lines -> blank line
  - "INIT: <Role>" -- true roles in REVERSE position order (last INIT = pos 1)
  - "ACT: Start/Day/Night/OnDied/OnExecuted" -- phase markers
  - "<Role> acting" -- ability triggered
  - "NANI: DIZZY" -- Confessor clue (via Confessor:GetBluffInfo)
  - "Player info reset" -- village boundary

Limitations:
  - No position numbers in events -- INIT reverse-order is the only position mapping
  - Most roles DON'T log clue content (only Confessor NANI: DIZZY confirmed)
  - ACT: Day doesn't say which card was clicked
  - Best used as a WITNESS (confirms events happened), not a SOURCE OF TRUTH
"""

import os
import re
import sys
import time
from dataclasses import dataclass, field

# --- Log file path ---

def _log_dir():
    """Return the Demon Bluff log directory."""
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        return os.path.normpath(os.path.join(appdata, "..", "LocalLow", "UmiArt", "Demon Bluff"))
    # Fallback: hardcoded
    return os.path.normpath(os.path.expanduser("~/AppData/LocalLow/UmiArt/Demon Bluff"))

LOG_PATH = os.path.join(_log_dir(), "Player.log")
LOG_PREV_PATH = os.path.join(_log_dir(), "Player-prev.log")

# --- Regex patterns ---

RE_INIT = re.compile(r"^INIT:\s+(.+)$")
RE_ACT = re.compile(r"^ACT:\s+(\w+)$")
RE_ACTING = re.compile(r"^(.+?)\s+acting$")
RE_NANI = re.compile(r"^NANI:\s+(.+)$")
RE_RESET = re.compile(r"^Player info reset$")

# Stack trace lines (skip these)
RE_STACK = re.compile(r"^(UnityEngine\.|Sirenix\.|<|Character:|Characters:|RevealCard:|Demon:|Confessor:|Investigator:|Striga:|SoftMask)")

# --- Data classes ---

@dataclass
class LogEvent:
    """A single parsed event from the log."""
    event_type: str   # 'init', 'act', 'acting', 'nani', 'reset'
    role: str         # role name (for init/acting) or ''
    data: str         # phase name (for act), clue text (for nani), or ''
    line_number: int

    def __repr__(self):
        parts = [self.event_type] + [x for x in (self.role, self.data) if x]
        return f"LogEvent({', '.join(parts)}, L{self.line_number})"


@dataclass
class VillageState:
    """Parsed state for one village (between 'Player info reset' boundaries)."""
    positions: dict       # {pos_int: role_name} from INIT entries
    n_cards: int          # number of cards (len of positions)
    events: list          # list of LogEvent
    start_line: int       # line number of the 'Player info reset'

    def get_true_roles(self) -> dict:
        """Return {position: role} mapping."""
        return dict(self.positions)

    def summary(self) -> str:
        lines = [f"Village ({self.n_cards} cards, starting L{self.start_line}):"]
        for pos in sorted(self.positions):
            lines.append(f"  #{pos}: {self.positions[pos]}")
        phase_events = [e for e in self.events if e.event_type in ('act', 'acting', 'nani')]
        if phase_events:
            lines.append(f"  Events: {len(phase_events)} phase/ability events")
        return "\n".join(lines)


# --- Parser ---

class LogParser:
    """Parse Demon Bluff Player.log files."""

    def __init__(self, log_path=None):
        self.log_path = log_path or LOG_PATH

    def _read_lines(self, path=None) -> list:
        """Read log file with universal newlines, return (line_text, line_number) pairs."""
        path = path or self.log_path
        if not os.path.exists(path):
            return []
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        # Normalize mixed line endings
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        result = []
        for i, line in enumerate(raw.split("\n"), 1):
            stripped = line.strip()
            if stripped:
                result.append((stripped, i))
        return result

    def _parse_events(self, lines) -> list:
        """Parse (text, line_number) pairs into LogEvent list."""
        events = []
        for text, lno in lines:
            m = RE_RESET.match(text)
            if m:
                events.append(LogEvent("reset", "", "", lno))
                continue

            m = RE_INIT.match(text)
            if m:
                events.append(LogEvent("init", m.group(1).strip(), "", lno))
                continue

            m = RE_ACT.match(text)
            if m:
                events.append(LogEvent("act", "", m.group(1).strip(), lno))
                continue

            m = RE_NANI.match(text)
            if m:
                events.append(LogEvent("nani", "", m.group(1).strip(), lno))
                continue

            m = RE_ACTING.match(text)
            if m:
                # Skip stack trace lines that happen to match
                role = m.group(1).strip()
                if not RE_STACK.match(text):
                    events.append(LogEvent("acting", role, "", lno))
                continue

            # Everything else (stack traces, engine init, warnings) -- skip
        return events

    def _build_villages(self, events) -> list:
        """Group events into VillageState objects."""
        villages = []
        current_events = []
        current_inits = []
        current_start = 0

        for ev in events:
            if ev.event_type == "reset":
                # Finalize previous village if it had INIT entries
                if current_inits:
                    villages.append(self._make_village(current_inits, current_events, current_start))
                current_events = []
                current_inits = []
                current_start = ev.line_number
                continue

            current_events.append(ev)
            if ev.event_type == "init":
                current_inits.append(ev.role)

        # Finalize last village
        if current_inits:
            villages.append(self._make_village(current_inits, current_events, current_start))

        return villages

    def _make_village(self, inits, events, start_line) -> VillageState:
        """Build VillageState from INIT role list (reverse order) and events."""
        n = len(inits)
        # INIT entries are in reverse position order: first INIT = last position
        positions = {}
        for i, role in enumerate(inits):
            pos = n - i  # first init -> pos N, last init -> pos 1
            positions[pos] = role
        return VillageState(
            positions=positions,
            n_cards=n,
            events=events,
            start_line=start_line,
        )

    def parse_all(self, path=None) -> list:
        """Parse log file, return list of VillageState (one per village)."""
        lines = self._read_lines(path)
        events = self._parse_events(lines)
        return self._build_villages(events)

    def parse_latest_village(self, path=None) -> VillageState | None:
        """Parse log file, return most recent village (or None)."""
        villages = self.parse_all(path)
        return villages[-1] if villages else None

    def get_true_roles(self, path=None) -> dict:
        """Convenience: return {position: role} from latest village INIT entries."""
        v = self.parse_latest_village(path)
        return v.get_true_roles() if v else {}

    def watch(self, interval=0.5):
        """Tail mode: poll for new events, print as they appear."""
        print(f"Watching {self.log_path} (Ctrl+C to stop)")
        seen_lines = 0
        last_size = 0

        while True:
            try:
                if not os.path.exists(self.log_path):
                    time.sleep(interval)
                    continue

                size = os.path.getsize(self.log_path)
                if size == last_size:
                    time.sleep(interval)
                    continue

                # File changed (or first read) -- re-parse
                if size < last_size:
                    # File was truncated/rotated
                    seen_lines = 0
                    print("--- log rotated ---")
                last_size = size

                lines = self._read_lines()
                events = self._parse_events(lines)
                new_events = events[seen_lines:]
                seen_lines = len(events)

                for ev in new_events:
                    if ev.event_type == "reset":
                        print(f"\n=== New Village (L{ev.line_number}) ===")
                    elif ev.event_type == "init":
                        print(f"  INIT: {ev.role}")
                    elif ev.event_type == "act":
                        print(f"  Phase: {ev.data}")
                    elif ev.event_type == "acting":
                        print(f"  {ev.role} acting")
                    elif ev.event_type == "nani":
                        print(f"  NANI: {ev.data}")

                time.sleep(interval)

            except KeyboardInterrupt:
                print("\nStopped.")
                break


# --- CLI ---

def main():
    watch_mode = "--watch" in sys.argv

    parser = LogParser()

    if watch_mode:
        parser.watch()
        return

    # Default: print latest village info
    print(f"Log: {parser.log_path}")
    if not os.path.exists(parser.log_path):
        print("  File not found.")
        return

    villages = parser.parse_all()
    if not villages:
        print("  No villages found in log.")
        return

    print(f"  Found {len(villages)} village(s)\n")

    # Show all villages briefly, latest in detail
    for i, v in enumerate(villages):
        tag = " (latest)" if i == len(villages) - 1 else ""
        print(f"--- Village {i + 1}{tag} ---")
        print(v.summary())

        # For latest village, also show event timeline
        if i == len(villages) - 1:
            phase_events = [e for e in v.events if e.event_type in ('act', 'acting', 'nani')]
            if phase_events:
                print("\n  Timeline:")
                for e in phase_events:
                    if e.event_type == "act":
                        print(f"    [{e.data}]")
                    elif e.event_type == "acting":
                        print(f"      {e.role} acting")
                    elif e.event_type == "nani":
                        print(f"      NANI: {e.data}")
        print()


if __name__ == "__main__":
    main()
