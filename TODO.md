# Solver TODOs

## Wiki Audit: All 40 Cards vs Solver

Research phase COMPLETE. All 40 cards audited against demonbluff.wiki.gg.

### Audit Checklist
- [x] Alchemist — no issues
- [x] Architect — no issues
- [x] Baker — wiki-vs-game discrepancy on "original" (documented, no fix needed)
- [x] Bard — no issues
- [x] Bishop — Wretch type bug + lying constraint gap
- [x] Confessor — no issues
- [x] Dreamer — Wretch "Cabbage" bug
- [x] Druid — no issues
- [x] Empress — self-pointing constraint (low priority)
- [x] Enlightened — no issues
- [x] Fortune Teller — no issues
- [x] Gemcrafter — self-pointing constraint (medium priority)
- [x] Hunter — Wretch bug (critical)
- [x] Jester — no issues
- [x] Judge — no issues
- [x] Knight — no issues
- [x] Knitter — no issues
- [x] Lover — no issues
- [x] Medium — lying target constraint gap
- [x] Oracle — Wretch bug + lying constraint gap
- [x] Poet — copied_role whitelist (low priority)
- [x] Scout — no issues
- [x] Slayer — no issues
- [x] Witness — Lilis kills + Chancellor conversion bugs, 0 test coverage
- [x] Bombardier — no issues
- [x] Doppelganger — no issues
- [x] Drunk — Knight HP edge case
- [x] Plague Doctor — low test coverage (2 cases)
- [x] Wretch — root cause of 4 validator bugs
- [x] Chancellor — outcast count + Witness tracking gaps
- [x] Minion — no issues
- [x] Poisoner — no issues
- [x] Puppet — no issues
- [x] Puppeteer — no issues
- [x] Shaman — guaranteed duplicate not enforced
- [x] Twin Minion — no issues
- [x] Witch — no issues
- [x] Baa — no issues
- [x] Lilis — Knight immunity + Good-priority constraints
- [x] Pooka — no issues

---

## Fixes (prioritized)

### P0: Critical Bugs (wrong scenario filtering)

- [x] **Hunter Wretch bug** — FIXED. Uses `_effective_alignment` now.

- [x] **Oracle Wretch bug** — FIXED. Added Wretch special case (registers as any Minion in deck).

- [x] **Dreamer Wretch "Cabbage" bug** — FIXED. Truthful → require "Cabbage", lying → require != "Cabbage".

- [x] **Bishop Wretch type bug** — FIXED. `_get_position_type` returns "Minion" for Wretch.

- [x] **Witness + Lilis kills** — FIXED. Added `night_kills` to `actually_affected` check.

- [ ] **Witness + Chancellor conversion** — Chancellor-converted positions should count as "affected by evil ability" for Witness. Requires tracking which position Chancellor converted in Scenario.

### P1: Missing Deduction Constraints (improve scenario elimination)

- [x] **Shaman guaranteed duplicate** — FIXED. Requires visible Villager pair when all Good positions revealed (accounts for unrevealed/night-killed positions).

- [x] ~~**Bishop lying = all Villagers**~~ — DROPPED. Wiki claim doesn't match game: corrupted Bishops show mixed types, not all Villagers.

- [x] **Oracle lying = both targets Good** — FIXED. Lying Oracle's targets must both be Good.

- [x] **Medium lying target = Evil/Drunk/Dopp** — FIXED. Lying Medium must point at Evil, Drunk, or Doppelganger.

- [ ] **Lilis can't kill uncorrupted Knight** — Wiki confirms. Not constrained in night-kill validation. Fix: exclude uncorrupted Knight from possible night-kill victims.

- [ ] **Lilis prioritizes Good kills** — Wiki says Lilis prioritizes killing Good characters. Could weight night-kill probabilities or reject scenarios where Lilis kills Evil when Good was available.

### P2: Medium Priority

- [ ] **Gemcrafter self-pointing** — Wiki says Gemcrafter can't point at self unless no other valid Good targets. Could eliminate scenarios where Gemcrafter points at self unnecessarily.

- [ ] **Empress self-pointing** — Wiki says Empress can't include self in targets unless she is a Puppet. Game enforces this at input, but constraint could catch data entry errors.

- [ ] **Drunk-as-Knight HP cost** — Wiki says 6 HP (not flat 2 HP). Strategy lookahead underestimates execution cost for Drunk disguised as Knight.

- [ ] **Chancellor +1 outcast count** — Board outcast count from header doesn't include Chancellor's Villager-to-Outcast conversion. Solver's `<=` bound partially handles this but could be tighter.

- [ ] **Baker reveal_order validation** — `reveal_order` is tracked but unused in Baker validator. Converted Baker must be revealed after original. Could narrow scenarios in Baker-heavy games.

### P3: Low Priority

- [ ] **Poet copied_role whitelist** — Validate that entered `copied_role` is one of the 13 valid Villager abilities (+ Bounty Hunter). Catches data entry errors.

- [ ] **Witness + Shaman tracking** — Wiki says Shaman-affected positions (original + clone) are "affected by evil". Requires Scenario-level tracking.

- [ ] **Doppelganger can't copy Puppet's original role** — Villager turned into Puppet is no longer "in play" as Good Villager. Doppelganger can't disguise as that role.

- [ ] **Puppeteer can't convert Shaman clones** — Wiki says Puppeteer acts before Shaman. Shaman-cloned Villagers can't be Puppet targets.

- [ ] **Shaman clones can't be cured by Alchemist** — Wiki says cloned Villagers retain corruption and can't be cured.

### Test Coverage Gaps

- [ ] **Witness**: 0 test cases with Witness on board. Need regression tests covering corruption, puppeting, and Lilis kills.
- [ ] **Plague Doctor**: Only 2 test cases. Need more coverage for active ability validation.
