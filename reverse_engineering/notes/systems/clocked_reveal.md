# Clock-driven delayed Reveal replay

The offline `bluff::clocked_reveal::replay_clocked_reveal` adapter composes the
[audited clock](unity_clock.md) with the existing native-ordered weighted Reveal
drain. The versioned contract is `clocked_delay_reveal_native_v1`.

The caller supplies an initial clock, acquisition/queue state, ordered clock
operations, an explicit phase mask, and callback owner/lifetime facts. Each
operation either updates the frame clock from a supplied timestamp or selects
the fixed clock. The adapter derives consumer time from the selected public
snapshot and producer time from the retained frame snapshot. It preserves the
complete signed 64-bit frame counter and takes generation from the input queue.
Callbacks cannot independently supply timestamps.

This distinction matters after fixed selection: the consumer may observe fixed
time 1 while newly allocated waits use retained frame time 20. New deadlines
still add the promoted binary32 `0.3f`, and their frame guards use the audited
counter. Chaining the returned queue/registry and clock into another invocation
preserves the existing queue-generation exclusion and occurrence-sensitive
weighted acquisition paths.

The deterministic clock history is returned once alongside all weighted paths;
it neither draws RNG nor duplicates its snapshots for each branch. Every
unsupported branch rejects the complete replay through the existing atomic
fallback. Clock versions and finite initial state are validated even when no
operations are supplied. A maximum of 256 operations bounds retained history.

The caller must explicitly assert that the clock remains stable throughout the
synchronous drain, including writer callbacks. False or absent provenance is
rejected. This is an offline composition contract, not proof of a particular
runtime loop or of clock stability in every game callback. It does not choose
phase masks, infer operating-system timestamps, model callback clock writes,
or resolve the remaining phase-bit-8 and full-interleaving boundary.

Seven Rust tests in `bluff/twin_writer.rs` verify unchanged weighted output with
zero transitions, fixed/retained snapshot separation, successive drains,
suppressed-update and signed-counter rollover, rejection/atomicity, strict
callback deserialization, and the exact 256/257 transition boundary. The final
capacity case also confirms that ineligible waits need no valid owner callback.
The complete 676-test Rust library suite passed. This API does not change live
solver inference or the existing simulation corpus.
