# Character constructor publication

Build `f530404b0f3f_807de4a83df4`. The standalone audit executes the complete
`Character..ctor` caller (`tdi5487.m0076`, RVA `0x3697c0`) in 26 fixtures,
six supplied callback probes and eleven controlled service stops. All 46
decoded instructions execute. The final instruction ends at `0x3698a3`;
the next verified managed entry is `0x3698b0`, with alignment excluded.
Nineteen instruction assertions pin publication, constructor argument setup,
nonzero defaults and the final base-constructor tail transfer.

The inputs are fingerprinted GameAssembly and Dumper metadata on the private
artifact drive. Repository output contains authored semantic states and exact
method metadata, without executable bytes, actor byte dumps or native bodies.

## Ordered writes and physical identities

After resolving its three cold metadata bindings, the caller writes
`pickableUses` (`+0xdc`) to one. It allocates a `List<ActedInfo>`, calls the
shared List constructor with its exact generic MethodInfo, stores the
**allocation identity** to `actedInfos` (`+0x148`), and calls the GC barrier.
It repeats that sequence with a separate physical allocation for `onHoverInfo`
(`+0x150`). Poisoned void-constructor return registers do not replace either
allocation identity.

It then loads the pinned empty managed-string literal, stores it to `savedAct`
(`+0x198`), and calls another GC barrier. The barrier argument separately
reloads the literal slot after the store; normal supplied bindings are inert.
It writes the single `act` byte (`+0x1a1`) to one, restores its stack and saved
registers, and tail-transfers to the folded UnityEngine.MonoBehaviour
constructor entry (`0x1c79770`) with the physical Character receiver and null
MethodInfo. The incoming caller MethodInfo is not forwarded.

The exact Character field declarations are bound to global TypeDefIndex 5487.
The List declaration is bound to TypeDefIndex 1510; Dumper reports zero generic
field offsets there. The list service's explicit storage at items `+0x10`,
count `+0x18` and version `+0x1c` is an authored contract, rather than an
inference from those generic zero offsets. Its constructed lists have count
and version zero and share one supplied empty-array identity. Managed
allocation supplies zeroed unpublished list storage before construction.

Zeroed, patterned and `0xa5` actor fixtures establish exactly which bytes the
caller writes. Every other byte in the authored `0x1b8` actor span remains
unchanged, except declared callback effects. Previously referenced lists keep
their distinct identities, logical contents, count, version and complete
authored storage bytes. The caller replaces references without clearing or
destroying the old lists. This does not establish how Unity allocates or
initializes a scene Character before this constructor is entered.

## Boundaries and callback observations

Each service checks its actual caller arguments. Service returns clobber all
volatile integer registers and XMM0–5, while preserving the Windows x64
nonvolatile register contract. Normal completion verifies all eight integer
nonvolatiles and XMM6–15. The base gateway additionally observes the restored
original stack and nonvolatiles before returning through the original caller
return address. Integer writes, byte writes and pointer stores retain their
native widths.

Six callback probes establish the caller's actual overwrite order:

- A first List-constructor callback can change uses; the caller preserves that
  new count but overwrites a callback-written `actedInfos` with the allocation.
- A first barrier callback can replace the already-published `actedInfos`, and
  the later caller does not restore it.
- A second List-constructor callback can clear `actedInfos` and write
  `onHoverInfo`; only the latter is overwritten by subsequent publication.
- A second barrier callback can replace the empty-literal binding before the
  caller loads and publishes it.
- A final barrier callback can change uses and clear `act`; the caller retains
  the count but subsequently writes `act` to one.
- A base-constructor callback observes every completed caller write, then can
  change uses, saved speech and `act`; no later Character write undoes them.

Eleven controlled stops cover every reached cold metadata, allocation, List
constructor, barrier and base gateway. Each event prefix and final semantic
snapshot equals the corresponding successful baseline snapshot. The complete
actor retention check also runs for stopped prefixes. Null allocation probes
reach the supplied List-constructor service with a null receiver and stop
there; the caller itself provides no intervening null check. This models an
explicit authored service failure, not managed exception unwinding.

The allocation, List constructor, metadata resolver, GC barrier and entire
MonoBehaviour base body remain supplied services. No List backing allocator,
Unity scene loading, scheduler admission or initializer composition is
claimed. The produced identities and defaults are a separate evidence source
for a future explicit join to Init and the refresh routines.

Reproduce with Unicorn 2.1.4 and the private emulation PYTHONPATH:

```powershell
python reverse_engineering/scripts/audit_character_constructor.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_constructor.json
```
