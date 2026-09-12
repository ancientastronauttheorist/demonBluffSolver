# Characters reveal entry wrappers

Pinned build `f530404b0f3f_807de4a83df4`. The authored auditor verifies pinned GameAssembly, Dumper script and dump hashes, exact metadata signatures, complete instruction boundaries, and key native writes/calls. It executes both actual entry points with empty and nonempty global diagnostic boards and a separate local reveal list. **386 differential cases pass**, executing 446 distinct native instructions.

| Declaration | RVA | Exclusive native end |
| --- | --- | --- |
| `void Characters.RevealAllDebug()` | `0x36DA40` | `0x36DF21` |
| `void Characters.RevealAll()` | `0x36DF30` | `0x36E485` |

The complete decode continues through all adjacent native chunks to the next verified managed entry, excluding trailing alignment. This is scoped evidence for the caller behavior below; framework formatting and provider bodies remain explicit gateways.

## Ordering and identity

Both wrappers first enumerate static `Gameplay.CurrentCharacters`, dispose that enumerator, and call `UnityEngine.Debug.Log`. They subsequently load the receiving `Characters.characters` list at +0x20 and enumerate it. The fixture keeps these two list identities distinct. A missing global list fails before the local traversal; a missing local list fails after logging. Logging failure prevents any local reveal action.

For every nonnull local element, `RevealAllDebug` directly calls the exact registered metadata method **`Character.RevealAllReal()` at 0x367E80**. It does not itself change Character state or invoke the state-change delegate. Duplicate references cause repeated calls; a physical null element fails at its position after earlier effects.

`RevealAll` performs these operations for each local element:

1. Copy `state` (+0xE4) to `prevState` (+0xE0), then set `state` to integer 30.
2. If `onStateChange` (+0x180) exists, invoke its method pointer with its target and MethodInfo.
3. Call `Character.RevealAllReal()`.
4. Read the fields as they exist **after** those calls, and swap `prevState` and `state`.
5. Read `onStateChange` again and invoke the current delegate if present.

With preserving callbacks, final state equals its original value and previous state becomes 30. The original previous state is discarded. The native code does not keep an independent saved state for restoration: changing either field in the callback or reveal callee changes the final swap. Clearing or replacing the delegate changes the second invocation. Fixtures use two distinct method/MethodInfo identities and verify the actual selected callback target.

## Nonempty diagnostic traversal

Before any local reveal, each global element is checked for physical null and its signed 32-bit `id` (+0x118) is boxed. The displayed name comes from `dataRef.characterName` (+0x50, then +0x28) when state is Dead (20), Revealed (30), or the `revealed` byte (+0xD8) is nonzero. Otherwise it compares the `bluff` reference (+0x58) with Unity null and selects the real data for a null/destroyed bluff, or the bluff data for a live bluff. The chosen data reference must be physically nonnull, but its name may be null. Values 1 and 255 both satisfy the revealed-byte condition.

The exact pinned format is `{0}: {1}`. Dead adds `, D`; Hidden (5) adds `, H`. A separate Unity nonnull test of `bluff` determines the suffix: absent/destroyed bluff adds `; `, while live bluff uses `, realRole: {0}; ` with the real **data object** as its argument. This latter formatting argument can be null when the initially selected bluff data was valid; there is no intervening physical real-data dereference on that path. The framework/Unity object `ToString` result is an explicit synthetic provider, not a recovered public role label.

The required `acteds` object (+0xA8) supplies `Acted.GetActed()` at 0x35DE70. Its first result is compared to the empty string. If unequal, the wrapper rereads `acteds` and calls GetActed again, concatenating `act: '`, the **second** result, and `'`. Fixtures return different first and second strings to verify the second call. A null first result is unequal to empty and takes the second-call branch; a null second result contributes an empty substring under the declared Concat gateway. Each entry ends in the exact literal `;; \n` (semicolon, semicolon, space, newline).

All nine string literals are verified against pinned script metadata. Framework boxing, formatting and concatenation calls are checked by their exact argument values/order, then supplied declared synthetic results. These gateway bodies are not native string-library audits.

## Failure and validation

Every gateway position on both the cold-type empty-global baseline and a nonempty diagnostic baseline is interrupted before its effect. Additional failure prefixes cover callbacks and reveal calls after mutations and duplicate references. A first-callback failure retains previous=original current and current=30. Reveal failure retains any completed first-callback changes. Second-callback failure occurs after the swap. These are observed prefix snapshots, not modeled managed exception unwinding.

Cases cover empty, singleton, duplicate, null-element and distinct local lists; null global/local list; no delegate and either of two delegates; ordinary and extreme unsigned state bit patterns; callback/reveal mutations to previous/current state and delegate identity; cold Gameplay/Debug/Object type initialization; real/bluff/null/destroyed data; signed ID limits; and changing or null GetActed results. Each event includes all three synthetic characters' states. An independent authored state machine checks exact events, errors and final fields. Normal completion requires the exact return address/stack pointer and preserves eight distinctly seeded nonvolatile integer registers. All other bytes of the synthetic managed object arena are preserved; the three state/delegate fields are separately compared by the state machine.

## Explicit boundaries

Metadata initialization flags are warmed. Gameplay, Debug and Object type initialization, enumeration/disposal, Unity equality, boxing, strings, Acted.GetActed, logging, Character.RevealAllReal, and callback bodies are explicit gateways. Enumeration supplies a stable declared local sequence. Controlled callback mutations are explicit fixture inputs; arbitrary reentrant list mutation is not covered. Native exception unwinding is not executed.

Diagnostic getters and framework gateways preserve the supplied object graph. Arbitrary data/acteds replacement during a provider call, framework allocation internals, managed exception cleanup, and the real data-object ToString implementation remain outside this corpus. The log text is compared only under the declared synthetic formatting/string providers. This scoped wrapper audit does not establish the effects of Character.RevealAllReal or game/UI reveal timing.

Artifacts: `scripts/audit_characters_reveal_entries.py` and `reports/f530404b0f3f_807de4a83df4_characters_reveal_entries.json`.
