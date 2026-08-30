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

- [x] **Witness "no one affected" (position=0)** — FIXED. `affected_position=0` means "nobody affected," not position 0. Also excluded PD/Drunk corruption from "affected by Evil" set — only Poisoner/Pooka/Puppeteer/Lilis count.

- [ ] **Witness + Chancellor trace** — the first Villager target should count as
  "affected by evil ability." The existing legacy field now tracks the final
  home of the added Outcast identity; derive the original target from the full
  Chancellor swap trace instead of treating those positions as identical.

### P1: Missing Deduction Constraints (improve scenario elimination)

- [x] **Shaman guaranteed duplicate** — FIXED. Requires visible Villager pair when all Good positions revealed (accounts for unrevealed/night-killed positions).

- [x] ~~**Bishop lying = all Villagers**~~ — DROPPED. Wiki claim doesn't match game: corrupted Bishops show mixed types, not all Villagers.

- [x] **Oracle lying = both targets Good** — FIXED. Lying Oracle's targets must both be Good.

- [x] **Medium lying target = Evil/Drunk/Dopp** — FIXED. Lying Medium must point at Evil, Drunk, or Doppelganger.

- [x] **Lilis can't kill uncorrupted Knight** — FIXED. Night-kill validation rejects scenarios where Knight is forced to an uncorrupted night-killed position. Handles pool > board (Knight might be off-board) and corrupted Knight (loses immunity).

- [x] **Poisoner target enum missing dead positions** — FIXED. Restructured Poisoner target enumeration to run inside Dopp/Drunk loop so `_unrevealed_must_be_villager` has access to dopp_pos/drunk_pos. Dead night-killed positions adjacent to Poisoner were incorrectly excluded.

- [x] **Medium validation for dead/night-killed cards** — FIXED. Medium pointing at a night-killed position no longer causes 0 scenarios. Accept Unknown role as match when position is Good and claimed role is valid.

- [ ] **Lilis prioritizes Good kills** — Wiki says Lilis prioritizes killing Good characters. Could weight night-kill probabilities or reject scenarios where Lilis kills Evil when Good was available.

### P2: Medium Priority

- [x] **Gemcrafter self-pointing** — FIXED. Truthful Gemcrafter can't self-point when other Good targets exist.

- [x] **Empress self-pointing** — FIXED. Can't include self in targets unless Puppet.

- [x] **Drunk-as-Knight HP cost** — FIXED. Strategy lookahead uses 6 HP for Drunk-as-Knight.

- [x] **Chancellor +1 outcast count** — FIXED. When Chancellor is in the deck, allow +1 on board_outcast_count in role count validation, _must_be_villager, and hidden outcast presence checks. Lost asc17_v2 to this — solver rejected true scenario with 2 Good Outcasts when header said O=1.

- [ ] **Baker reveal_order validation** — `reveal_order` is tracked but unused in Baker validator. Converted Baker must be revealed after original. Could narrow scenarios in Baker-heavy games.

- [x] **Baa hidden Outcast in deck view** — NATIVE-VERIFIED. Baa obscures one existing Outcast identity; it does not add a role. `game_loop.py deck` preserves the HUD `no=` value and explains the eye-symbol mismatch.

- [ ] **Endless mode +1 fake Outcast in deck view** — Reportedly added every round (confirmed by dev on Steam), independently of Baa. Native-verify whether this changes gameplay/HUD counts or only deck presentation before adjusting `no=`.

- [x] **Puppeteer targets Villagers only when possible** — FIXED. Placement generator restricts Puppet candidates to Villager (+ unrevealed) positions when a known Villager is adjacent to Puppeteer. Outcasts only eligible when no Villagers are adjacent.

- [ ] **Corrupted Knight = 4 HP damage** — Patch v0.310a: "Knight deals 4 damage if Executed while Corrupted." Solver currently uses `wrong_exec_cost` (5 at Asc4+). Should be 4 specifically for corrupted Knight.

### P3: Low Priority

- [ ] **Poet copied_role whitelist** — Validate that entered `copied_role` is one of the 13 valid Villager abilities (+ Bounty Hunter). Catches data entry errors.

- [ ] **Witness + Shaman tracking** — Wiki says Shaman-affected positions (original + clone) are "affected by evil". Requires Scenario-level tracking.

- [ ] **Doppelganger can't copy Puppet's original role** — Villager turned into Puppet is no longer "in play" as Good Villager. Doppelganger can't disguise as that role.

- [ ] **Puppeteer can't convert Shaman clones** — Wiki says Puppeteer acts before Shaman. Shaman-cloned Villagers can't be Puppet targets.

- [ ] **Shaman clones can't be cured by Alchemist** — Wiki says cloned Villagers retain corruption and can't be cured.

- [x] **Lilis won't kill herself when last revealed** — Confirmed in asc25_v1: Lilis deals 2 HP even with no valid kill target (can't self-kill). No code fix needed — this is a data entry issue (don't assume last card was killed). Documented in CLAUDE.md.

- [x] **Alchemist "0 cured" = always truthful** — Already handled. Evil/corrupted Alchemist has actual_cures=0 and must lie, so claiming 0 is rejected by `claimed != actual`.

### Test Coverage Gaps

- [x] **Witness**: 2 test cases now (asc11_g3_live, asc23_v2). Fixed validator for "no one affected" (position=0) and PD/Drunk corruption exclusion. Still no coverage for Chancellor conversion tracking (see P0).
- [ ] **Plague Doctor**: Only 2 test cases. Need more coverage for active ability validation.
