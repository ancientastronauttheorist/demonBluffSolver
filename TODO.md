# Solver TODOs

## Wiki Audit: All 40 Cards vs Solver

Research phase COMPLETE. All 40 cards audited against demonbluff.wiki.gg.

### Audit Checklist
- [x] Alchemist — no issues
- [x] Architect — no issues
- [x] Baker — full managed `Baker` boundary native-audited; synchronous Day
  conversion, saved runtime identity, exact real/lying clues, candidate filters,
  status composition, small boards, and achievement bookkeeping closed
- [x] Bard — no issues
- [x] Bishop — Wretch type bug + lying constraint gap
- [x] Confessor — no issues
- [x] Dreamer — Wretch "Cabbage" bug
- [x] Druid — no issues
- [x] Empress — self-pointing constraint (low priority)
- [x] Enlightened — no issues
- [x] Fortune Teller — full managed `FortuneTeller` boundary native-audited;
  unrestricted two-target picker, exact registered-alignment truth and lying
  complement, ascending speech/reference order, ResetAfterNight history,
  discarded RNG draw, and both-Evil achievement closed
- [x] Gemcrafter — self-pointing constraint (medium priority)
- [x] Hunter — public asset bound to managed `Tracker`; exact registered-Evil
  circle distance, `N - 1` exhaustion value, half-circle bluff domain, and
  ordered duplicate-preserving acted references are native-audited and covered
- [x] Jester — no issues
- [x] Judge — full managed `Judge2` boundary native-audited; corrupted-actor
  inversion and ResetAfterNight history fixed
- [x] Knight — full managed `Immortal` boundary native-audited; exact
  protection precedence and additional-four execution damage closed
- [x] Knitter — no issues
- [x] Lover — no issues
- [x] Medium — lying target constraint gap
- [x] Oracle — Wretch bug + lying constraint gap
- [x] Poet — public asset bound to managed `Gossip`; exact ordered twelve-role
  provider selector, fresh truthful/bluff draws, Character dispatch, strict
  current provenance schema, legacy compatibility, and exact Scout/Hunter
  provider payloads are native-audited and covered
- [x] Scout — complete managed role boundary native-audited; runtime-Evil
  occurrence selection, register-as/dataRef naming split, duplicate-role
  ambiguity, explicit one-Evil sentence, and strict 1-through-3 bluff domain
  are covered while the archived distance-zero sentinel remains compatible
- [x] Slayer — no issues
- [x] Witness — current-status truth semantics and Chancellor interaction
  native-audited with focused regressions
- [x] Bombardier — full managed `Saint` boundary native-audited; any
  non-Demon death whose current `dataRef.role` is exact `Saint` auto-loses,
  including genuine Shaman/Chancellor identity replacement despite preserved
  Evil alignment, while ordinary bluff, Drunk/Doppel display copies, and
  managed `SaintVillager` do not qualify
- [x] Doppelganger — no issues
- [x] Drunk — Knight HP edge case
- [x] Plague Doctor — 154 active observations plus focused native callback tests
- [x] Rambler — full managed `Rambler2` boundary native-audited; pre-flip
  installation, real/bluff source dispatch, target appearance, persistent
  callbacks, exact history/reference shape, and duplicate ordering closed
- [x] Wretch — root cause of 4 validator bugs
- [x] Chancellor — outcast count + Witness tracking gaps
- [x] Minion — no issues
- [x] Poisoner — full managed `Poisoner` boundary native-audited; exact
  post-Pooka Start slot, previous-then-next real-Villager eligibility, live
  Corrupted/resistance filters, paired-status composition, all-match duplicate
  ordering, dead/small-board behavior, and stale-description legacy surface
  closed with no solver delta
- [x] Puppet — no issues
- [x] Puppeteer — no issues
- [x] Shaman — guaranteed duplicate not enforced
- [x] Twin Minion — full managed `Marionette` boundary native-audited; ordered
  Start swaps current `CharacterData` with one alive neighbour of a selected
  current Demon while preserving physical alignment, status, resistance, and
  runtime data. A gated exact Twin-to-Puppeteer solver slice now preserves the
  complete pre-Twin current-role map, both native traces, relocated actor/target
  identity, and erased Puppet Villager provenance. Broader mixed-writer replay
  remains open; stable Twin/Demon adjacency is not valid.
- [x] Witch — full managed `Cipher` boundary native-audited; global quota,
  last-Hidden predicate, self/dead/Lilis cleanup, and reset semantics closed
- [x] Baa — no issues
- [x] Lilis — full managed `Striga` boundary native-audited; hard Good-first
  selection, repeated Nights, protected no-kill, and duplicate behavior closed
- [x] Pooka — full managed `Pooka` boundary native-audited; deterministic
  two-real-Villager-neighbour Start corruption, independent paired statuses,
  highest-ID duplicate and small-board behavior closed, and the private random
  one-neighbour helper proven unreachable in the shipped flow by native xrefs

---

## Fixes (prioritized)

### P0: Critical Bugs (wrong scenario filtering)

- [x] **Hunter Wretch bug** — FIXED. Uses `_effective_alignment` now.

- [x] **Oracle Wretch bug** — FIXED. Added Wretch special case (registers as any Minion in deck).

- [x] **Dreamer Wretch "Cabbage" bug** — FIXED. Truthful → require "Cabbage", lying → require != "Cabbage".

- [x] **Bishop Wretch type bug** — FIXED. `_get_position_type` returns "Minion" for Wretch.

- [x] **Witness + Lilis kills** — FIXED. Added `night_kills` to `actually_affected` check.

- [x] **Witness "no one affected" (position=0)** — FIXED. `affected_position=0` means "nobody affected," not position 0. Plague Doctor observations and Drunk self-corruption do not create Witness evidence; native Witness reads active `MessedUpByEvil` markers plus successful night-kill markers.

- [x] **Witness + Chancellor trace** — native audit disproved the first-target
  hypothesis. Chancellor reinitializes that Villager without marking it, then
  separately attempts `MessedUpByEvil` on a real-Outcast anchor. Witness reads
  only surviving current markers; grouped trace anchors preserve provenance but
  never create a Witness constraint by themselves.

### P1: Missing Deduction Constraints (improve scenario elimination)

- [x] **Shaman ordered overwrite trace** — FIXED. Enumerates ordered source/target pairs, carries the copied role plus an existential erased-role candidate class, and admits mixed-faction hidden endpoints only when the exact Outcast budget can make them Villagers.

- [ ] **Native branch probability weights** — surviving scenarios are still
  treated as equally likely after semantic deduplication. Preserve native draw
  probabilities and hidden identity multiplicities for PD/Shaman/Chancellor
  histories without reintroducing duplicate logical worlds, so confidence and
  mutual-information scores reflect probability mass rather than world count.

- [x] ~~**Bishop lying = all Villagers**~~ — DROPPED. Wiki claim doesn't match game: corrupted Bishops show mixed types, not all Villagers.

- [x] **Oracle lying = both targets Good** — FIXED. Lying Oracle's targets must both be Good.

- [x] **Medium lying target = Evil/Drunk/Dopp** — FIXED. Lying Medium must point at Evil, Drunk, or Doppelganger.

- [x] **Lilis can't kill uncorrupted Knight** — FIXED. Night-kill validation rejects scenarios where Knight is forced to an uncorrupted night-killed position. Handles pool > board (Knight might be off-board) and corrupted Knight (loses immunity).

- [x] **Poisoner target enum missing dead positions** — FIXED. Restructured Poisoner target enumeration to run inside Dopp/Drunk loop so `_unrevealed_must_be_villager` has access to dopp_pos/drunk_pos. Dead night-killed positions adjacent to Poisoner were incorrectly excluded.

- [x] **Medium validation for dead/night-killed cards** — FIXED. Medium pointing at a night-killed position no longer causes 0 scenarios. Accept Unknown role as match when position is Good and claimed role is valid.

- [x] **Lilis prioritizes Good kills** — NATIVE-CLOSED. This is not weighting:
  Lilis first constructs the eligible registered-Good Hidden pool and samples
  uniformly from it. Only an empty Good pool enables the unaligned fallback.

### P2: Medium Priority

- [x] **Gemcrafter self-pointing** — FIXED. Truthful Gemcrafter can't self-point when other Good targets exist.

- [x] **Empress self-pointing** — FIXED. Can't include self in targets unless Puppet.

- [x] **Drunk-as-Knight HP cost** — FIXED. Strategy lookahead uses 6 HP for Drunk-as-Knight.

- [x] **Chancellor +1 outcast count** — FIXED. When Chancellor is in the deck, allow +1 on board_outcast_count in role count validation, _must_be_villager, and hidden outcast presence checks. Lost asc17_v2 to this — solver rejected true scenario with 2 Good Outcasts when header said O=1.

- [x] **Baker reveal_order validation** — NATIVE-CORRECTED. Baker conversion
  occurs only during a reached Day/user-reveal action; there is no Start
  preseed. A successful click changes Hidden to Alive and synchronously
  completes Baker conversion before the next click. `asc77_v6` predates the
  verified-first-click fix and recorded click attempts rather than actual
  state transitions; a flaked #1 followed by actual order
  `[2,3,4,5,6,7,8,9,10,1]` explains #6 Baker -> #9 Empress -> #1 Judge
  without any preseed.

- [x] **Baa hidden Outcast in deck view** — NATIVE-VERIFIED. Baa obscures one existing Outcast identity; it does not add a role. `game_loop.py deck` preserves the HUD `no=` value and explains the eye-symbol mismatch.

- [ ] **Endless mode +1 fake Outcast in deck view** — Reportedly added every round (confirmed by dev on Steam), independently of Baa. Native-verify whether this changes gameplay/HUD counts or only deck presentation before adjusting `no=`.

- [x] **Puppeteer targets real Villagers** — NATIVE-CORRECTED. The ordered
  Start scan checks the two physical neighbours' real `dataRef` types, removes
  only the first Saint Villager occurrence, and converts a remaining candidate
  mandatorily. If none remains it performs no conversion; Outcasts are never a
  fallback. Conversion checks run after authored Evil placement and hidden
  Drunk, Doppelganger, and Chancellor identity branching, so neither an Evil
  nor an Outcast merely disguised as a Villager can force a fake Puppet.

- [x] **Corrupted Knight = 4 HP damage** — NATIVE-DISPROVEN as a total. The
  authored/native rule is fixed **additional** 4 after ordinary event damage:
  corrupted runtime-Good Knight is 5 + 4 = 9 normally, 4 with NoDamage, and
  Drunk displaying Knight is 2 + 4 = 6. Lilis and Slayer do not run this hook.

### P3: Low Priority

- [x] **Poet copied_role whitelist** — FIXED. Current Poet payloads accept only
  the native twelve-provider pool (Lover, Scout, Oracle, Bounty Hunter, Medium,
  Knitter, Hunter, Enlightened, Empress, Bishop, Gemcrafter, and Bard), enforce
  provider-specific schema and board bounds, and carry explicit
  `poet_variant: public_current` provenance. Unmarked archived payloads retain
  their historical interpretation byte-for-byte.

- [x] **Witness + Shaman tracking** — FIXED. `Scenario.messed_up_by_evil` carries each resistance-aware Shaman marker attempt through the ordered Start pass.

- [x] **Doppelganger can't copy Puppet's original role** — NATIVE-CLOSED.
  Puppeteer finishes conversion before delayed disguise selection. The target's
  real `dataRef` is then non-bluffable Evil Puppet; its saved Villager exists
  only as a display bluff, so both clean and Corrupted Doppelganger reject the
  converted physical card. A separate surviving copy remains independently
  eligible.

- [x] **Puppeteer can't convert Shaman clones** — FIXED. Puppet conversion runs before Shaman; the generated Puppet is removed from Shaman's eligible Villager set and covered by a focused regression.

- [x] **Shaman clone corruption/cure ordering** — FIXED. Corruption initially survives `InitWithNoReset`, but a later Alchemist can cure it; `MessedUpByEvil` remains a separate persistent status.

- [x] **Lilis won't kill herself when last revealed** — NATIVE-CLOSED. Normal
  Start gives the single same-asset Start actor status 60, so Lilis is excluded
  from her own pool and still deals 2 HP when no target exists. Missing/resisted
  status 60 is the bounded self-selection edge. A selected protected victim
  also yields no kill and is not rerolled.

- [x] **Alchemist "0 cured" = always truthful** — Already handled. Evil/corrupted Alchemist has actual_cures=0 and must lie, so claiming 0 is rejected by `claimed != actual`.

### Test Coverage Gaps

- [x] **Witness**: current-status truth/bluff handling, Lilis death markers, and
  Chancellor's separately selected affected anchor now have focused coverage.
  Native audit disproved treating the first Villager replacement as an
  automatic Witness candidate.
- [x] **Plague Doctor**: Full managed `Puzzlemaster` boundary audited; strict
  active-result shape, self/dead targeting, Wretch alignment, ordinary Drunk
  status, random truth/bluff reveal support, and legacy `evil_pos` covered.
- [x] **Judge**: Full managed `Judge2` boundary audited; strict speech/reference
  shape, self/dead/hidden/unused-active targeting, truth-appearance queries,
  deterministic corrupted-actor inversion, and multi-night history covered.
- [x] **Fortune Teller**: Full managed `FortuneTeller` boundary audited;
  self/dead/hidden/unused-active targets, exact two-reference ordering,
  Wretch/Drunk registered-alignment edges, deterministic corrupted-actor
  inversion, ResetAfterNight history, and truthful both-Evil achievement
  documented against all 95 parsed v2 results.
- [x] **Witch**: Full managed `Cipher` boundary audited; exact one-quota Start,
  no stored blocked target, self-block, killed-hidden exclusion, picker/execute
  bypasses, unused-active flip consequence, ordinary/Slayer/Lilis cleanup, and
  between-village reset documented against 67 v2 Witch-deck fixtures.
- [x] **Chancellor + Witness**: Full managed `Baron` and `Witness` boundaries
  audited; ordered anywhere-Villager replacement, independent Outcast anchor,
  exact identity-home equations, duplicate/death persistence, and current-only
  affected-status truth/bluff semantics have focused regressions.
- [x] **Lilis + Knight**: Full managed `Striga` and `Immortal` boundaries
  audited; hard registered-Good selection, repeated/duplicate Night timing,
  protected/no-target fixed damage, real/bluff identity interactions, Slayer
  bypass, and exact 0/4/6/9 HP outcomes have focused regressions.
- [x] **Rambler**: Full managed `Rambler2` boundary audited; pre-flip
  AfterRoundStart installation, actual-source/apparent-target truth, hidden
  persistent callbacks, immediate versus pre-append history, exact one-source
  shut-up references, duplicate/small-board behavior, and constraint-free quote
  references are documented against focused regressions.
- [x] **Poet**: Full managed `Gossip` selector boundary audited; twenty focused
  Python tests (including 150 provider/parser subtests) and eight Rust tests
  cover the exact provider pool, strict current schemas, malformed provenance,
  board bounds, positive-distance sentinels, auto-card placeholder safety,
  unmarked legacy compatibility, and Scout truth/lying inversion.
- [x] **Scout + Hunter**: Full managed `Scout` and `Tracker` boundaries audited;
  current direct and Poet ingestion verify exact text, board bounds, provenance,
  and native reference shape, while Rust validation preserves unmarked legacy
  fixtures, reuses the exact hidden-Outcast allocator for Wretch distance
  support, joins explicit Wretch register-as draws, and enforces
  occurrence-aware Scout and exact Hunter semantics.
