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

- [ ] **Hunter Wretch bug** — `_validate_hunter` uses `_is_evil_in_board_state` instead of `_effective_alignment`. Only validator that doesn't count Wretch as Evil. Could reject valid scenarios or accept invalid ones in Hunter+Wretch games.

- [ ] **Oracle Wretch bug** — `_validate_oracle` uses `_known_evil_role()` which returns None for Wretch. Truthful Oracle pointing at Wretch as any Minion role is incorrectly rejected. Fix: add Wretch check (registers as random Evil Minion).

- [ ] **Dreamer Wretch "Cabbage" bug** — `_validate_dreamer` has no special case for Wretch target. Truthful Dreamer targeting Wretch reports "Cabbage" but validator rejects it (Cabbage not in `evil_roles`). Fix: when target is Wretch and truthful, require "Cabbage"; when lying, require != "Cabbage".

- [ ] **Bishop Wretch type bug** — `_get_position_type` returns "Outcast" (real KB role) for Wretch, but Bishop should see Wretch as "Minion" (registers as Evil Minion to abilities). Fix: override return value for Wretch positions.

- [ ] **Witness + Lilis kills** — `_validate_witness` only checks `corrupted` and `puppet_position` for "affected by evil". Wiki says Lilis night-killed chars are also "affected by evil ability". Fix: add `state.night_kills` to `actually_affected` check.

- [ ] **Witness + Chancellor conversion** — Chancellor-converted positions should count as "affected by evil ability" for Witness. Requires tracking which position Chancellor converted in Scenario.

### P1: Missing Deduction Constraints (improve scenario elimination)

- [ ] **Shaman guaranteed duplicate** — When Shaman is on the board, there MUST be exactly 2 identical Villager roles. Solver currently allows but doesn't require duplication. Fix: reject scenarios where Shaman is placed but no villager pair exists.

- [ ] **Bishop lying = all Villagers** — Wiki says lying Bishop always shows 3 Villager types. Validator only checks type mismatch. Fix: enforce that lying Bishop's claimed types are all "Villager".

- [ ] **Oracle lying = both targets Good** — Wiki says lying Oracle can't include two Evil characters. Validator only checks neither is named role. Fix: add constraint that both targets must be Good (via `_effective_alignment`).

- [ ] **Medium lying target = Evil/Drunk/Dopp** — Wiki says lying Medium points at Evil, Drunk, or Doppelganger. Validator allows any wrong-claim target. Fix: restrict lying Medium's target to evil/Drunk/Doppelganger positions.

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
