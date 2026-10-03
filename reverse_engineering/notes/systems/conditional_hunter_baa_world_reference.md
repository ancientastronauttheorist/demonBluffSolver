# Conditional Hunter/Baa complete-world reference: development v2

Pinned build `f530404b0f3f_807de4a83df4`, Steam build `23084916`.
[Independent integration comparison](../../../crates/solver-core/tests/small_world_reference.rs).
This is synthetic development evidence for conditional clue/deduction semantics,
not a held-out corpus, generation certificate, likelihood model or completed S2.

## Decision, domain and exit test

Decision blocker: a correct-looking single answer or matching count cannot prove
the solver retains exactly every world consistent with observed Hunter clues.
The reference independently enumerates every Baa seat, scans the physical circle
for Hunter truth, and enumerates the distinct native bluff-distance support. It
does not call production geometry, validators or scenario generation.

Admitted finished setup has four or five distinct actors, physical native board
order `[1,2,...,N]` with those displayed IDs, exactly one Baa, and Hunter current
data on every other actor. The supplied public pool preserves exactly `N-1`
Hunter occurrences and one Baa occurrence; the reference rejects other Hunter
multiplicities. All apparent reveals are Hunter. Finished setup and Hunter
bluff acquisition are explicitly supplied conditions. No Outcasts,
Minions, corruption, register-as changes, identity/status writers, deaths,
abilities or phase changes occur. Each actor has at most one verified reveal.
The ascending native board order is supplied; ordinary Manage-generated board
orientation and live ingestion are not certified by these examples.

[Hunter evidence](../roles/gameplay_roles_scout_hunter.md) establishes self
exclusion, nearest registered-Evil distance or `N-1`, bluff support in
`1..=floor(N/2)` excluding the truthful value, exact singular/plural speech, and
ordered range references including opposite-seat duplicates.
[Baa evidence](../roles/gameplay_role_baa.md) establishes that Start returns
immediately for an empty Outcast pool. Its deck-only effect does not enter this
domain. These native-static rules ground the independently authored reference;
the test does not execute the original clue bodies.

The predeclared finite family covers both board sizes, every Baa seat, every
available bluff value and every physical reveal permutation. All 1,392 complete
histories and their 8,160 prefixes are compared without sampling or timeout.
Independent family enumeration gives 4,376 ambiguous and 3,784 unique prefixes.
Four additional coherent impossible histories (all-one and all-`N-1` for each
size) require complete empty world sets. These are family counters, not numeric
performance targets or population outcome estimates.

Exit test requires complete canonical world equality and identical definite
Good/Evil sets at each prefix. The canonical world is the Baa seat because all
other identities and statuses are fixed in this domain. Every returned Scenario
must equal the default Scenario plus exactly that one Baa assignment, including
all serialized fields; populated residual traces or correlations and duplicate
canonical worlds fail the comparison instead of being discarded.

The first root-coordinated run passed five of seven tests and exposed an input
projection error: the public pool contained only one Hunter name while the
conditional roster contained `N-1` real Hunter occurrences. Production role-count
validation correctly rejected excess revealed Good Hunters. The projection now
preserves the full supplied occurrence multiset; no solver rule or expected
world/history was changed. The original complete family remains intact.

## Admission, sensitivity and limitations

The test-local reference API returns `Unsupported` for unestablished setup,
other board sizes, repeated/out-of-range actors, richer histories, malformed
distance domains, or incoherent raw speech/reference order. Coherent clues
that contradict the model return a complete empty set. The exact current
`hunter_variant: public_current` is used when projecting admitted histories.
Production Hunter's distance payload has no target field, so the reference
projection gate validates raw targets/text first. This does not prove the legacy
solver entry point rejects unsupported histories or malformed capture itself.

Development v2 also sends every family prefix through the production
[strict player-history boundary](player_history_boundary.md) and its
`conditional_initial_day_hunter_baa_v1` adapter. Those synthetic public histories
contain only deck multiplicity, current HUD, initial Day HP/cost, apparent Hunter
position and exact speech. Public targets are empty. Test-local UI-review
registrations are explicitly authored availability contracts, not reviewed
pixels or native scheduling evidence. The projected snapshot preserves
`LegacyUnknown` HUD provenance and excludes offline Plague Doctor/Twin context.

The production-adapter comparison uses the same complete Scenario equality
check and independent enumerator. A separate mutation replaces native-only
ordered references with incoherent duplicates: the raw native validation gate
rejects them, while the public history, planner input, snapshot and resulting
world set remain unchanged. Native-only evidence never becomes a public input
or a prerequisite for planning from the unchanged public sentence.

An additional integration consumes the twenty native-generated sentences from
the [Hunter producer/publication audit](hunter_role_publication.md), covering
twelve truthful actors and eight Baa bluff draws. It constructs separate
conditional synthetic public histories from the generated position/text and
supplied public setup, then compares complete worlds and grades actual Baa
inclusion only in the oracle lane. Native seat order `[4,3,2,1]` reverses this
reference circle; nearest circular distances are preserved, while native
ordered references are excluded from the public payload. The producer's
retained `prior_info` stress record is not relabeled as an initial-Day public
observation. This closes the sentence-to-adapter semantic comparison, not a
complete native/public chronological trace or capture-availability certificate.

Two opposite Baa assignments producing the same legal history must project to
identical solver inputs and retain the same two possible worlds. Privileged
assignment data is used only to construct/grade the synthetic corpus and is
absent from projected solver input. No planner or original RNG is consumed.

The deliberate diagnostic mutation replaces circular truthful distance with
linear distance. Independently enumerated native-support sets disagree on
1,104 complete histories. The test verifies reference-family sensitivity;
production source is not mutated, so this is not a production mutation gate.

Run after shared source freeze:

```text
cargo test --release -p solver-core --test small_world_reference -- --nocapture
```

The earlier corrected v1 run passed all seven tests. The root-coordinated v2
run passed all nine integration tests and all twenty strict-history unit tests.
After adding the native-sentence comparison, all ten integration tests passed.
Complete-world equality held at all 8,160 generated prefixes through both the
legacy fixture projection and production public-history adapter, with 4,376
ambiguous and 3,784 unique results in each comparison. All four coherent
contradiction fixtures were empty through both routes; unsupported inputs,
paired-world input equality, native-reference independence and residual-state
rejection also passed. The reference-only mutation differed on exactly 1,104
complete histories.
Native generation, acquired-bluff provenance, chronology through actual Reveal,
independent held-out families, posterior weights, legal action policy,
calibration, latency/memory and ascension continuations remain separate gates.
