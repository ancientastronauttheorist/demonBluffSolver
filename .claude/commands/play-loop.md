---
description: Play a full ascension (up to 7 winning villages, including retries) under the autonomous ascension grinder protocol. See memory/play_loop_protocol.md.
---

You are running under `/play-loop`. Read the runbook first:

1. `memory/play_loop_protocol.md` — multi-village loop procedure, resume detection, halt triggers, halt protocol, loss-analysis, commit format.
2. `memory/known_recoverable.md` — surprises that do NOT halt.
3. `CLAUDE.md` and `MEMORY.md` — standing rules and memory index.

Then play the current ascension through to completion. One ascension = up to village 7 wins, plus any retries after a loss (a loss resets the village, NOT the ascension). The loop runs across villages in this single session — do NOT exit after one village.

Execute the solver's top pick on every turn — no overrides.

## Session exits in exactly three ways

- **Ascension complete** (village 7 win): emit stdout marker
  `=== ASCENSION_COMPLETE asc=<N> final_hp=<HP> wins=<N> losses=<count> ===`,
  send `PushNotification` with the same summary, then exit.
- **Halt trigger** (0 scenarios / memory-screenshot mismatch / first-time surprise / loss whose analysis surfaces a real solver bug): follow the halt protocol — fix branch, `PushNotification`, emit
  `=== HALT reason="<one-line>" branch=<halt/...> ===`,
  then exit.
- **Hard crash**: unrecoverable exception. No marker emitted; the wrapper treats a missing terminal marker as crash.

## Per-village stdout marker (required)

Emit after every village is committed, before starting the next iteration:
`=== VILLAGE_END village=<N> status=<win|loss> hp=<HP> ascension=<asc_tag> ===`

## Loss handling (mid-ascension)

Losses do not exit the loop. After committing the loss:

1. Spawn an analysis agent per CLAUDE.md rule 5 — full review of critical decisions.
2. If the agent finds a real solver bug → escalate to halt protocol (fix branch + notify + exit).
3. If the loss was unavoidable / solver was correct → commit the "unavoidable" note to `memory/losses_postmortem.md` and continue the loop into the village retry.

Do not silently skip analysis. Do not batch analyses to the end of the ascension.
