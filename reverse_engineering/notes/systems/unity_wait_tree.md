# Native finite-deadline tree ordering

Evidence: `native-static` plus `native-emulated` against UnityPlayer SHA-256
`B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2`.
This narrows the [wait boundary](unity_wait_boundary.md) for finite keys and
unchanged surviving records. It does not yet reconstruct a callback-mutated
consumer traversal or establish the actual times of game continuations.

## Recovered behavior

Insertion at RVA `0x440f00` allocates a `0x60`-byte node and copies the complete
`0x40`-byte wait record to node offset `0x20`. It compares only the double
deadline. For finite keys it descends left when the new deadline is smaller,
and right otherwise, including equality and either sign of zero. Thus new
equal-deadline occurrences are placed after existing equal-deadline occurrences.

The link/balance helper at `0x366cb0` adjusts parent/child links and red/black
colors, preserving in-order record order. Erasure at `0x3eb6d0` removes the
requested node, transplanting its successor when necessary and rebalancing via
`0x3eba60` and `0x3ebac0`. It returns the original requested node. Surviving
records keep their addresses and payloads; the successor's payload is not copied
over the removed record. Minimum, maximum, root and count are maintained.

These routines therefore support stable occurrence order among finite equal
deadlines under insertion and removal, conditional on no other writer changing
the keys or links. This is specific to the fingerprinted engine implementation.
It is not a public Unity ordering guarantee or a claim that all records inserted
during a drain will be visited by that drain's saved-successor traversal.

## Differential native execution

[`audit_unityplayer_wait_tree.py`](../../scripts/audit_unityplayer_wait_tree.py)
maps the pinned PE into Unicorn x86-64 emulated memory. It executes only the five
audited routines; an instruction hook rejects every other native code address
except the allocator entry, which returns synthetic zeroed nodes. No DLL is
loaded for host execution. Each operation has instruction and time bounds.

The audit uses the default `MXCSR 0x1f80` environment and finite deadlines. It
compares the native in-order result with an independent stable sort of surviving
insertion occurrences after every insert and erase. It also verifies all
surviving payload bytes, red/black invariants, parent/sentinel links, extrema
and count. Node identities are deliberately distinct from insertion-order keys.

The deterministic corpus includes ascending and descending insertions, all-equal
keys, mixed signed zeros, positive/negative subnormals and finite extremes,
ordered and shuffled complete erasures, and 128 seeded mixed mutation sequences.
The report records operation counts and distinct executed instructions per
routine; those counts are not a claim of exhaustive branch or input coverage.
The pinned run passed 26,496 operations (13,248 insertions and 13,248 erasures)
across 143 cases, with every operation checked.

Five proprietary-input-free tests check the oracle itself against equal signed
zero occurrences, corrupted payloads, balancing/parent/count/extrema violations,
cycles, empty trees and nonfinite input rejection. Native dependencies are lazy
imports, so ordinary CI does not need Unicorn or the game.

## Reproduce

Install the pinned optional dependency in a private environment:

```powershell
python -m pip install unicorn==2.1.4 pefile
python reverse_engineering/scripts/audit_unityplayer_wait_tree.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_wait_tree.json
python -m unittest discover -s reverse_engineering/tests
```

The [report](../../reports/f530404b0f3f_807de4a83df4_unity_wait_tree.json) contains
authored findings and addresses only. Native input bytes and decompiler output
remain private. The managed coverage denominator and live solver are unchanged.

Validation also passed 778 Python tests, 25 reverse-engineering tests, Python
compilation and diff checks. Rust and simulations were not rerun for this
offline audit-only checkpoint.
