# Queue-driven retained role publication

`bluff::scheduled_role_publication` connects the existing
[role-publication primitive](character_role_publication.md) to the
[native-tested one-shot queue](unity_wait_queue_projection.md). Its version is
`scheduled_role_publication_native_v1`. The
[native Hunter composition](hunter_scheduled_publication.md) supplies independent
post-click checkpoints for the joined Rust transition.

## Decision and input boundary

The prior explicit replay completed all suspended result instances before
manually resuming speech. That contract could preserve final state while missing
immediate speech storage and intermediate queue readiness. The new API takes
one or two already suspended result instances, their complete queue, exact
queue-label-to-result bindings and up to sixteen explicit drain snapshots.
Legacy resume vectors must be empty and their verification flag false. The
ordinary explicit replay remains available with its original ordering contract.

The publication primitive still validates complete typed storage, references,
captured strings, capacity, unique allocation identities and the first zero
result yield. Each initial result must have exactly one bound queue record with
phase mask `0xA` and a present release slot. Queue order, finite timing values,
monotonic IDs and retained generation are checked by the shared queue kernel.
Initial records and their producer provenance are supplied; this API does not
execute Character.Act, first-yield production, acquisition or a legal click.

Owner responses are per drain and required only when a record passes timing
gates. A missing/null owner erases and releases without callback entry. A
mismatched owner enters the callback but suppresses managed execution. Both
produce explicit discarded continuation records; an empty queue following an
owner rejection does not mean the managed iterator completed. Matched owners
require the declared normal native lifetime contract returning one, inert
release, stable producer clocks and non-reentrant dispatch. Other lifetime
graphs, exceptions, callback mutation and unrelated waits remain unsupported.

## Retained chronology

An eligible result callback appends the captured info, applies the native use
counter change, registers speech and immediately executes its first step.
Speech text and savedAct are written before the result's final picker-hide and
return. If speech yields, its exact float duration `0x3ECCCCCD` enters the same
queue using the supplied producer clock, full signed frame counter plus one
and current drain generation. The duration is promoted to double before adding
the producer time; the consumer time is not substituted for that producer input.

The existing saved-successor traversal determines which inserted waits are
visited. Newly produced records cannot pass that drain's generation gate even
when their deadline and frame threshold are already past. Later speech callbacks
execute Show only after the actual timing and owner gates pass. Equal deadlines
preserve occurrence order. Iterator identities and captured descriptions remain
distinct through two retained results on the same actor.

Each outcome retains exact queue state/trace, managed replay state/events,
pending bindings and discarded instances after every drain. SaveSpeech,
SetText and Show are separate checkpoints. None establishes rendered pixels or
admission to PlayerHistory. This is an offline batch API; native hidden roles and
actor references in its test fixtures never enter live deduction or policy.

## Evidence and reproducibility

[project_scheduled_role_publication.py](../../scripts/project_scheduled_role_publication.py)
expands the existing lossless report codec and verifies the exact source-report
SHA-256 before projecting typed initial storage and allowlisted expected states.
It imports no Rust implementation. The
[fixture](../../fixtures/synthetic/scheduled_role_publication_v1.json) contains
29 native cases and 161 drain checkpoints: twenty real/bluff Hunter cases,
six full-frame/generation cases and three owner rejection cases. This certifies
the declared post-click projection under its supplied services, not its producer
or a complete original setup-to-observation history.

The original four double-Day stress compositions, double-Day same-generation
composition and 254 failed-service prefixes are excluded from this projection.
A separate synthetic two-result integration check exercises overdue inserted
speech visits and generation suppression against the already native-tested
kernel. The old native fixture's initial runtime uses zero remains a declared
stress input; asset abilityUsage zero is not evidence that native Init sets it.

```powershell
python reverse_engineering/scripts/project_scheduled_role_publication.py `
  reverse_engineering/reports/f530404b0f3f_807de4a83df4_hunter_scheduled_publication.json `
  --output reverse_engineering/fixtures/synthetic/scheduled_role_publication_v1.json
cargo test --release -p solver-core --lib scheduled_role_publication
```

The primitive bounds initial retained slots/text to 65,536 and result instances
to two. Sixteen saved drain checkpoints bound aggregate snapshot storage to
roughly one million initial retained slots plus the bounded publication traces;
the adapter does not search or branch over schedules. Invalid later evidence
rejects the complete batch rather than returning a valid-looking prefix.
Independent review reproduced the fixture values and physical bytes exactly.
All six adapter tests and all 963 library tests passed; the release build passed.
The existing 34 simulation results are retained: deduction/action rules were
unchanged by this offline adapter, so that long suite was not repeated.
