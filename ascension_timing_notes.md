# Ascension 54 Timing Notes

Detailed per-step timing for one full ascension run, to identify bottlenecks for future speed improvements.

**Date started:** 2026-04-10
**Run:** Ascension 54

## Methodology
- Record wall-clock time for each phase of each village.
- Note any waits, retries, or manual interventions that ate time.
- After the run, summarize totals and call out the slowest steps.

## Per-Village Log

### Village 1 (9-card, 3 evil + Puppet)
- **Wall-clock window:** 18:17:49 → 18:23:38 (≈ **5m 49s**)
- **Result:** WIN, 10/10 HP perfect, 4 executions all correct
- **Evils:** #3 Puppeteer, #4 Puppet, #5 Poisoner, #7 Baa
- **Phase breakdown (rough):**
  - 18:17 – 18:19: stuck on prior-village dialogs (Village-safe Next + Ascension-Complete Next + Score Continue + intro Close). ~1m30s of pure clicking through dialogs. **Biggest single waste of time so far.**
  - 18:19 – 18:20: deck panel didn't open (icon click toggled?), fell back to memory_reader --deck for pool. Manual `new` + `deck` cmds. ~1m.
  - 18:20: `flip` ran cleanly, ~10s for all 9 cards including memory cross-check.
  - 18:20 – 18:21: `auto_card` entered all 9 cards in one shot. ~0s human time.
  - 18:21 – 18:23: 4 execute cycles (each: `next` → click center → safe_click sword → click target → screenshot → execute cmd). ~25–35s per cycle.
- **Notes / friction points:**
  - **Dialog dismissal between villages is the slowest part.** Each click is ~10–20s round-trip (move, screenshot, verify, click, screenshot to confirm). Three back-to-back dialogs at the start of v1 cost ~90s.
  - btn_next on the Ascension-Complete dialog is at **(1305, 940)**, NOT (1280, 865). Worth recording — `mouse.py click 1280 865` failed twice before I switched to `safe_click btn_next`.
  - The opening `python game_loop.py start` failed all three of its safe_click steps because the game wasn't yet on the main menu (we were still on a previous-village end screen). The `start` macro assumes the game is sitting at the title menu.
  - `auto_card` is *amazing* — saved ~2 minutes of manual `card xxx` typing. 9/9 auto-entered.
  - Each execution cycle has ~5 separate tool calls (next → click center → safe_click sword → click target → screenshot → execute). A combined `execute_solver_pick` macro could collapse this to one call.

## Summary (filled in at end of run)
- Total wall-clock time:
- Slowest step categories:
- Candidate speed improvements:
