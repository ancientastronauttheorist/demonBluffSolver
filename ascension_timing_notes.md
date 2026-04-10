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

### Village 2 (7-card, 2 evil, Slayer kill)
- **Wall-clock window:** 18:25:14 → 18:28:03 (≈ **2m 49s**)
- **Result:** WIN, 10/10 HP perfect, 1 execute + 1 Slayer kill
- **Evils:** #3 Chancellor (slayer-killed), #6 Baa (executed)
- **Phase breakdown (rough):**
  - 18:25:14: clicked V1 victory `Next` → V2 launches directly (no intro dialog this time, no Continue, no Ascension-Complete). One single click between villages! ~10s.
  - 18:25:30 – 18:26:00: deck read via memory_reader, `new` + `deck`. ~30s.
  - 18:26:00 – 18:26:15: `flip` of all 7 cards in ~10s.
  - 18:26:15 – 18:26:45: `auto_card` got 5/7. Two manual entries (Alchemist + Slayer no_info). ~30s.
  - 18:26:45 – 18:27:30: `next` → execute #6 (Baa). One full execute cycle. ~45s.
  - 18:27:30 – 18:28:00: `next` → Slayer kill #3 (Chancellor). Active ability click sequence. ~30s.
- **Notes / friction points:**
  - **Mid-ascension villages are fast.** No intro dialogs to dismiss → just one Next click between V1 and V2.
  - **`auto_card` mis-handles Drunk-disguised-as-Alchemist** because the auto-parser keys off `runtime_data['type']=='cures'`, which is keyed by *true* role (Drunk, no cures). Manual `card alchemist 1 2` was needed. **Fixable**: parser should also fall back to text regex `cured\s+(\d+)` even without runtime data.
  - **`slayer_result` requires the revealed evil_role as 4th arg.** I forgot it the first time and game_over saved an incorrect test case (asc54_v2.json without slayer info). Re-running game_over wrote v2b.json instead of overwriting. **Process tweak**: always run `slayer_result` BEFORE `game_over` for slayer kills.
  - The `game_over` command does NOT auto-run the cargo replay regression. The "Full v2 regression (Rust)" header prints but no test runs. I had to invoke `cargo test --release --test replay` separately. **Possible bug** in game_over post-checklist.
  - Memory reader's `clue` field for some positions is **stale** (e.g., #2 Bombardier showed "Right side is more Evil" — Bombardier has no passive clue). Auto_card correctly skipped these but the noise is confusing. Worth investigating later.
  - Stale clue source: probably `savedAct` from a previous village. Should clear on village transition.

### Village 3 (8-card, 2 evil — adjacent Pooka+Poisoner)
- **Wall-clock window:** 18:29:41 → 18:31:07 (≈ **1m 26s**)
- **Result:** WIN, 10/10 HP perfect, 2 confident executions
- **Evils:** #3 Pooka, #4 Poisoner (adjacent — solver collapsed to 1 scenario immediately!)
- **Phase breakdown (rough):**
  - 18:29:41: `safe_click btn_next` from V2 win → V3 starts. ~5s.
  - 18:29:46 – 18:30:05: deck read + new + deck. ~20s.
  - 18:30:05 – 18:30:20: flip 8 cards. ~15s.
  - 18:30:20: `auto_card` 8/8 in one shot. ~0s manual.
  - 18:30:20 – 18:30:45: `next` → exec #3. ~25s.
  - 18:30:45 – 18:31:07: `next` → exec #4. ~22s.
- **Notes / friction points:**
  - **FASTEST village so far (1m26s).** Why: tight clue set → 1-scenario solver solution; no manual entries; no slayer/active abilities.
  - 8/8 auto_card success rate. The auto-pipeline is very strong on standard passive clues.
  - Each execute cycle is still ~22-25s, dominated by 5 sequential tool calls (next, click center, safe_click sword, click target, screenshot, execute). **A combined `solve_and_execute` macro would shave ~10s/cycle**, saving ~20s on a 2-execute village.

## Summary (filled in at end of run)
- Total wall-clock time:
- Slowest step categories:
- Candidate speed improvements:
