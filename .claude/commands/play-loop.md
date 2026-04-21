---
description: Play one village under the autonomous ascension grinder protocol. See memory/play_loop_protocol.md.
---

You are running under `/play-loop`. Read the runbook first:

1. `memory/play_loop_protocol.md` — per-village procedure, halt triggers, halt protocol, commit format.
2. `memory/known_recoverable.md` — surprises that do NOT halt.
3. `CLAUDE.md` and `MEMORY.md` — standing rules and memory index.

Then play exactly one village following the protocol. Execute the solver's top pick on every turn — no overrides. Halt on any trigger listed in the protocol (0 scenarios, memory/screenshot mismatch, loss, first-time surprise).

Your final action must be to write the status file at repo root:
- `echo ok > .play_loop_status` on a completed village (win or loss, no halt).
- `echo "halt: <one-line reason>" > .play_loop_status` on halt, after committing the fix on a `halt/<YYYY-MM-DD-HHMM>-<slug>` branch and sending a PushNotification to the user.

Then exit. The wrapper (`scripts/play_loop.sh`) will decide whether to start the next village.

Do not batch villages. Do not play a second village in this session.
