# Conditional Hunter/Baa capture admission

Build scope `f530404b0f3f_807de4a83df4`, Steam build `23084916`. Schema
`player_history_v1`; mechanical adapter domain
`conditional_initial_day_hunter_baa_v1`. See the
[strict history boundary](player_history_boundary.md),
[Hunter native-static audit](../roles/gameplay_roles_scout_hunter.md),
[Baa audit](../roles/gameplay_role_baa.md), and
[conditional complete-world reference](conditional_hunter_baa_world_reference.md).

## Decision and conditional domain

The decision blocker is whether an exact-looking native Hunter clue was
legitimately available at the observation prefix used for deduction. Correct
`ActedInfo` text, successful parsing, an Alive/Revealed memory state or a text
setter request does not prove visible UI exposure. The new adapter interprets
public text once admitted; it does not grant admission or close native Reveal
scheduling/rendering provenance.

The conditional model has four/five distinct numbered physical seats in native
cyclic order `[1,...,N]`, one Baa and Hunter current data on every other actor.
Finished setup and the Baa actor's Hunter bluff acquisition are supplied
conditions. No Outcasts, Minions, corruption, register-as changes,
identity/status writers, deaths, abilities or phase changes occur. That model
is not inferred from the public pool or a domain string. Its native-generation
reachability and exact physical actor identity remain separate validation-lane
obligations, not hidden fields fed to the planner.

The public input requires exactly `N-1` exposed Hunter Villager deck occurrences
and one exposed Baa Demon occurrence; duplicates are retained. The complete
current HUD counts are `N-1` Villagers, zero Outcasts, zero Minions and one
Demon; one Evil remains. Current counts retain `LegacyUnknown` provenance in
the legacy snapshot, never invented `TrustedPreStart`. An initial Day, public
HP and wrong-execution cost are required. A revealed seat is entered once as
apparent Hunter, and its exact speech is preserved. Optional reveal requests
are serial, single-seat, cost-free Day actions with observed completion.

The native Hunter sentence is `I am 1 card away from closest Evil` or
`I am N cards away from closest Evil`. For this conditional domain the parser
admits exactly `1..=floor(board_size/2)` or the `board_size-1` sentinel,
derives the distance only from exact
singular/plural public text and writes `hunter_variant: public_current`.
Leading zeros, wrong singular/plural, case changes, absent words and outside
distances do not become silently normalized native evidence. On five seats,
distance 3 cannot be a native output; it returns `Unsupported` rather than
being treated as a supported-model empty world set.

## Target-reference and review obligations

Hunter speech does not expose numbered target IDs. Its native ActedInfo range
references and their order are memory/native-validation data unless independent
public-source evidence establishes their availability. Public `targets` must
therefore be empty for this adapter. Nonempty references are `Unsupported`,
including the correct range endpoints and opposite-seat duplicate. Distance
plus reviewed physical geometry may justify a separate deterministic derived
reference calculation, but that calculation is not captured native order and
is not inserted into this history payload.

`record_trusted_ui_review` must be called only by a trusted external capture
reviewer after verifying the exact publicly available event at its prefix.
The reviewer must retain build/capture identity, screenshot grounding for
physical positions/apparent role/speech/HUD, full deck exposure and multiplicity,
and actual reveal chronology. Wall-clock screenshots or click attempts alone
cannot establish native reveal order. For a memory transcription, a prior
reviewed reveal of that actor and an exact UI cross-check of the transcription
are both required. The registry binds the entire exact prefix; a reveal from
another history at the same ordinal cannot supply a visibility gate.

The external conditional setup assumptions must be recorded and checked in a
separate fixture/oracle provenance artifact. Pixels cannot generally prove
the absence of hidden corruption or runtime aliases. Neither raw input JSON
nor the domain ID certifies these conditions. A trusted native-availability
contract may supply an offline observation boundary only when explicitly
declared as supplied; it must not be advertised as observed rendered pixels.

## Actual archived specimen and open gate

Read-only review of private archived `screenshots/asc84_v1_after_flip.jpg`
shows apparent Hunter #3 with visible speech `I am 1 card away from closest
Evil`, HP 10, numbered physical geometry and current HUD `[6,1,2,1]`.
Only part of the deck strip is exposed. Native ordered range references are
not explicit in Hunter's speech. The related legacy `asc84_v1.json` records
distance 1 with empty `info_text`, not the exact text capture.

This is a useful partial public-pixel review specimen. It is outside the
four/five-seat Hunter/Baa domain and lacks pinned capture/build binding,
complete deck exposure/multiplicity and actual per-reveal chronology. It is
not registered as an admitted PlayerHistory and supplies no fake certification
of that domain. The separate [archived public-pixel review](asc84_hunter_capture_review.md)
records its reproducible image hashes, reviewed public fields and unresolved
admission gates. The private image remains outside publication.

Exit test for the next integration: freeze a build-bound reviewed public
capture family for the named conditional domain, preserve every actual reveal
prefix and full public deck multiplicity, derive distance only from visible
speech, keep native-only references in the oracle lane, and compare the
projected complete canonical worlds with the independent small reference at
each prefix. Original concrete Hunter/Reveal-to-availability composition,
physical mapping and setup reachability still require evidence; synthetic
passing histories are not those proofs.

The adapter and six new unit cases are development implementation evidence.
No generation certificate, held-out family, posterior weighting, policy result,
CLI player-history solving path, trusted live capture pipeline or live control
is provided. Missing required capture fields return `Incomplete`; unsupported
domain or transition returns `Unsupported`. Neither result asserts a complete
empty possible-world set.
