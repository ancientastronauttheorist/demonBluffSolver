# Gameplay role: Twin Minion

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for every method declared by managed
`Marionette` and the complete ordered-Start, dispatch, Demon selection,
alive-adjacency, identity-replacement, delayed-reveal, and integer-RNG boundary
needed to reproduce its shipped behavior. Serialized asset evidence fixes the
public binding and exact Start slot. Native bodies and decompiler output remain
outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_twin_minion.json`](../../targets/gameplay_role_twin_minion.json).
Its read-only baseline and typed Ghidra exports each complete at 22/22
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_twin_minion.json)
passes its regression check: unresolved-type tokens fall from 183 to 47, raw
field-offset accesses from 235 to 40, raw integer type tokens from 166 to 17,
placeholder parameter tokens from 236 to zero, and indirect-call patterns from
13 to zero. Both exports retain two decompiler-error markers; warning markers
increase by one, from 33 to 34.

## Public asset binding and ordered Start slot

The shipped `sharedassets0.assets` `CharacterData` at path ID `21595` is named
`Marionette`, has `characterId` `Marionette_11628408`, and binds its
SerializeReference role to exact managed `Marionette` at TypeDefIndex `5914`
in `Assembly-CSharp`. The canonical public name is **Twin Minion**. Its raw
object SHA-256 is
`E6A3C8B245D7FCA4C2D9B25C1753757B482992FAFE3166EB7DACD1B9953CD9E7`.

The card is an Evil Minion (`characterType == 30`,
`startingAlignment == 20`), is not bluffable, is not usually disguised, and
has `picking == false`. Its `abilityUsage` is enum value zero (`Once`), but it
has no Day picker or player-selected target. It serializes no additional
statuses, tags, or appearance conditions. The exact managed/public passive
description is:

```text
[I sit next to a Demon]
```

The `level0` object at path ID `137026` references Twin Minion's path ID as
zero-based index 5 in `startGameActOrder`, immediately after Witch and before
Puppeteer. Every board card has completed ordinary `Init` before this ordered
pass. Unlike Alchemist, Poisoner, and Plague Doctor, Twin Minion is not an
all-match exception: ordinary first-match scanning runs only the first current
exact-data match. Normal construction makes that the highest displayed-ID
match when duplicate Twin Minion data is present.

## Audited boundary and shared bodies

| Managed identity | RVA | Boundary purpose |
| --- | ---: | --- |
| `Marionette.get_Description` | `0x3E4440` | Exact bracketed passive description |
| `Marionette.GetInfo` | `0x3E4240` | Empty passive information record |
| `Marionette.Act` | `0x3E4090` | Reachable Start-only data swap |
| `Marionette.SitNextToDemon` | `0x3E42A0` | Dormant private duplicate algorithm |
| `Marionette..ctor` | `0x3CFFF0` | Fieldless base-role construction |
| `Characters.ManageCharacters` | `0x36CE30` | Init/Start timing and ordinary duplicate scan |
| `Character.Act` | `0x3645C0` | Truth/lie and real/bluff action dispatch |
| `Character.RoleAct` | `0x368790` | Virtual action routing |
| `CharacterHelper.CheckLying` | `0x397750` | Runtime truth-state input |
| `Role.BluffAct` | `0x3C4CA0` | Inherited forwarding to real `Act` |
| `ActedInfo..ctor` | `0x35D5D0` | Exact empty output shape |
| `Characters.FilterCharacterType(Character)` | `0x36AC30` | Registered-or-real Demon pool |
| `Characters.FilterAliveCharacters` | `0x36A240` | Alive circular ring |
| `Characters.GetAdjacentAliveCharacters` | `0x36C050` | Previous/next alive pair |
| `Character.InitWithNoReset` | `0x365720` | Current-`dataRef` replacement |
| `Character.DelayReveal.MoveNext` | `0x3756B0` | Immediate role clone and delayed reveal |
| `Character.Reveal` | `0x368410` | Register-as, bluff, and presentation refresh |
| `Character.RevealReal` | `0x3682A0` | Final current-`dataRef` presentation when the raw bluff stays null |
| `Character.GetCharacterBluffIfAble` | `0x364C40` | Separate current bluff-or-real accessor; not `Reveal`'s call site |
| `Character.GiveBluff` | `0x365160` | Reveal-time bluff application |
| `Role.GetRegisterAsRole` | `0x3712B0` | Constant-null base register-as identity for Scout and Witness |
| `UnityEngine.Random.Range(Int32)` | `0x1C86600` | Demon and neighbour draws |

The 22 memberships select 22 distinct managed FunctionDefinitions and 22
native RVAs inside this target. `Marionette..ctor` is one of 537 managed
aliases of the fieldless construction body and uses the established
ABI-compatible canonical prototype `Slayer___ctor`. `Role.BluffAct` has two
managed aliases. `Role.GetRegisterAsRole` is one of 323 managed aliases of its
constant-null body and applies the ABI-compatible canonical prototype
`Role__GetBluffIfAble`. The selected integer `UnityEngine.Random.Range` body
likewise has two managed aliases, but the target retains the exact overload
identity and applies the named `UnityEngine_Random__Range_Int32` prototype.
Shared native code is not treated as shared managed identity.

## Start dispatch and fixed output surface

`Marionette.Act` returns immediately for every trigger except `Start` (enum
value 5). A lying real Evil with a null raw `bluffRole` dispatches only its real
role's `BluffAct`; inherited `Role.BluffAct` forwards to the virtual real
`Act`. With a populated raw `bluffRole`, native instead runs the real Evil
`Act` and then the bluff role's `BluffAct`. Twin Minion therefore does not lose
its Start mutation merely because it lies or displays another role.

`Marionette.GetInfo` allocates a fresh `ActedInfo` with an empty description and
a null character-reference list. The Start method never invokes `onActed` and
emits no speech or acted-history record. The role declares no bluff-info or
bluff-action override, picker, reset callback, runtime-data object, status,
achievement request, or role-local field. Its fieldless constructor contributes
only the shared base-role initialization.

The meaningful public information is the fixed bracketed Description shown
above. It describes the **current Twin Minion data identity after Start**, not
a stable original Evil card identity. It does not guarantee an apparent Twin
Minion display after the delayed presentation refresh.

## Demon and neighbour selection

At Start, the action first makes a shallow copy of
`Gameplay.CurrentCharacters`. It filters that copy to exact character type
Demon (`100`) using `(Unity-live registerAs ?? dataRef).characterType`. The
filter preserves source order and repeated references and does not inspect
state, liveness, runtime alignment, status, resistance, bluff, or physical
origin.

An empty Demon pool returns cleanly without consuming RNG. Otherwise the
method consumes one max-exclusive integer draw over all Demon occurrences.
The selected Demon is then looked up in the global **alive-filtered** current
list. That helper preserves global order, skips Dead cards, and returns exactly
the previous then next entries in the resulting circular ring. The action
consumes a second `Random.Range(0, 2)` draw over those two occurrences; it does
not deduplicate equal references.

The normal serialized Start pass occurs before any death, so the selected
Demon is present in the alive ring and the pair has exactly two occurrences.
On a malformed later invocation, a registered-or-real Demon that is already
Dead can enter the first pool but is absent from the alive ring. The resulting
empty adjacency list reaches the unguarded random/get-item failure path rather
than producing a clean no-op.

Executable direct calls inside the live body independently close this chain:
registered-or-real type filtering occurs at call-site RVA `0x3E4172`, the two
integer draws at `0x3E4193` and `0x3E41DA`, alive adjacency at `0x3E41C5`, and
the two replacements at `0x3E420C` and `0x3E421E`.

## Current-dataRef swap

Let `m` be the physical card whose current data is Twin Minion when the ordered
slot finds it, `d` the selected current registered-or-real Demon, and `n` the
selected alive neighbour of `d`. Before mutating either card, the method saves
`n.dataRef`, then performs:

```text
n.InitWithNoReset(m.dataRef, -100)
m.InitWithNoReset(saved n.dataRef, -100)
```

This is a **swap of current `CharacterData`**, not a physical-card move, stable
role-label move, alignment swap, or simple adjacency assertion. At completion:

```text
n.current data = Twin Minion
m.current data = n's former current data
```

The operation preserves both physical cards and their displayed IDs. If `n ==
m`, both calls reinitialize the same card with the same data and the identity
mapping is unchanged, although both RNG draws and both reinitializations still
occur. Otherwise the Twin Minion current-data identity migrates to an alive
neighbour of the chosen Demon.

Most importantly, `InitWithNoReset` does not replace runtime alignment,
statuses, resistances, runtime data, or physical provenance. A Good neighbour
receiving Twin Minion data remains runtime Good; the original runtime-Evil Twin
card remains Evil while receiving the neighbour's former role data. Truth,
execution, status, and current-role-sensitive terminal rules must keep these
identity layers separate.

The swap can also move a role whose ordered Start slot occurs later than Twin
Minion onto `m`. Because `dataRef` changes synchronously and the coroutine
clones the replacement role before its first yield, the later exact-data scan
finds and dispatches that role at its new physical position. A role whose slot
already passed does not receive a second ordered Start merely because it moved.
Twin Minion's own slot does not restart the newly written copy on `n`.

## Offline Start writer kernel

`bluff::twin_writer::replay_twin_start` implements a bounded offline projection
of an already-entered `Marionette.Act(Start)` (`0x3E4090`). Its explicit
`twin_start_writer_native_v1` input carries a v3 Reveal context with no resume
events, the exact global CurrentCharacters occurrence order, body state, and
the entered real/copied Twin action slot. It does not infer dispatch from data
identity or set Character.Act's latch. Callers must establish that method entry;
the enclosing Character.Act dispatch and subsequent copied-slot reread are not
yet composed with this kernel.

The input covers every distinct global body and preserves repeated references.
Its registered identity is explicitly null or live; destroyed register-as
references must first be represented as null by a provenance-aware caller.
Canonical role assets determine registered Demon type, with current Lilis data
as the null-register fallback in this restricted data-role domain. Runtime
alignment and raw bluff do not filter the Demon pool. Both RNG draws retain
occurrence mass, including equal previous/next neighbors. Repeated Demon
references use their first match in the alive list. Any selectable dead Demon
rejects the entire call with empty support, rather than conditioning away its
native failure branch. No Demon gives an identity result with zero RNG draws.

The ordered replacements project `InitWithNoReset` (`0x365720`) and the first
`DelayReveal.MoveNext` entry (`0x3756B0`): reset the Start latch, clear raw bluff,
install current data and immediately cloned action role, increment pending
continuations, clear acted-information count, revealed/killed flags and dead
presentation, set pickable uses to one, and assign previous/current state before
the second replacement. Self-swap performs both calls, so previous state ends
Hidden and the same body gains two continuations. Register-as, copied bluff
role, runtime alignment, statuses/resistance/target and Spy data-role cache
identity remain unchanged. No pool draw or delayed Reveal occurs in this call.

The kernel assumes valid native objects and successful coroutine registration.
State-change subscribers and presentation-side gameplay writers are excluded;
the context explicitly rejects subscribed bodies. UI objects, acted-info
contents, opaque runtimeData and role-object internals are not reconstructed.
Spy cache identity is transported, but its action clone's cache is outside the
modeled trigger surface. The 256-occurrence input cap, retained-entry budget
and checked continuation increments reject atomically without partial outputs.

Eight regressions cover distinct/self swaps, stale registration, registered
Demon filtering and dead mass, duplicate global references, a one-card alive
ring, copied Twin with Spy data, inherited resistance on a moved Drunk's next
Reveal, and provenance/capacity rejection. The moved-Drunk example composes
only the post-writer state with a caller-specified later resume; it does not
claim a complete surrounding Character.Act or scheduler interleaving.

Checkpoint validation passed 564 Rust library tests, 778 Python tests, 13
reverse-engineering tests, release build, formatting and coverage-ledger
integrity. No live/scenario caller changed, so the long simulation suite was
not rerun. This milestone reuses existing native evidence and does not increase
the method coverage counts.

## InitWithNoReset and delayed reveal

Each replacement hides the acted-information object, clears `actedInfos`,
resets the one-shot Start latch, destroys a live dead-character presentation,
clears `bluff` and `revealed`, assigns the new `dataRef`, clears
`killedByDemon`, resets `pickableUses`, preserves the physical ID because the
sentinel is `-100`, sets state to Hidden, refreshes presentation, and starts
`DelayReveal`.

Starting each new coroutine immediately clones the newly assigned role before
its first wait. After the delay, `Character.Reveal` recomputes register-as data
and dispatches Init and AfterRoundStart. Only while the character's `bluff` is
Unity-null does Reveal call the current `dataRef.role.GetBluffIfAble` and apply
the result through `Character.GiveBluff`; later continuations retain a non-null
bluff, while a null selection is retried. The separately selected
`Character.GetCharacterBluffIfAble` helper is not this Reveal call site. There
is no Twin-local immediate copied-Start call like Shaman's. Until a delayed
continuation runs, current `registerAs` and `bluffRole` storage can still
reflect the prior identity even though `dataRef` and the cloned role have
changed.

Ordinary board initialization has already left one initial `DelayReveal`
continuation pending on every card. A distinct-endpoint Twin swap adds one more
continuation to each endpoint, so both have at least two pending Reveal passes.
A self-swap performs both replacements on one card and adds two more, leaving
at least three. There is no cancellation, overlap guard, or sibling-state
recheck: every continuation later recomputes register-as and dispatches
Init/AfterRoundStart, but only a Unity-null `bluff` re-enters role-specific
bluff selection. Later reinitializers can add still more continuations, and
native static evidence does not fix their sibling resume order.

The resulting display is not a stable endpoint locator. In the ordinary
Good-neighbour branch, the runtime-Good card carrying current Minion-type Twin
data can receive a Minion bluff and need not visibly show Twin Minion. The
original runtime-Evil Twin card can instead present the neighbour's former
role. Tooling must carry current data identity directly rather than recover it
from apparent role.

The two reinitializations are sequential. The neighbour's former data is saved
before either call, so the second call receives the intended old value rather
than rereading the already-overwritten neighbour. Failures are not
transactional: a null card/data dependency or missing required acted UI can
leave earlier mutations in place.

## Duplicate and small-board behavior

Only the first current Twin Minion match in normal list order receives the
ordered action, even when the board contains multiple exact Twin data entries.
The action's Demon pool and alive ring use live current data after every earlier
Start mutation. Duplicate candidate occurrences retain probability weight.

For an ordinary two-card Twin-plus-Demon board, the selected Demon's previous
and next alive entries are both the Twin actor. The second draw therefore
chooses the actor through either index, producing the same-card double
reinitialization and no identity relocation. With at least three alive cards,
previous and next are distinct when the global board references are unique;
one may be `m`, yielding a legal no-relocation branch. Dead cards between the
three participants are skipped by the alive-ring helper.

No-Demon input is the only clean early no-op. Null global dependencies,
out-of-range collection state, or a selected Demon absent from the alive ring
follow native failure paths. Those malformed cases are not authored Standard
or Ascension board shapes.

## Dormant SitNextToDemon

Private non-virtual `Marionette.SitNextToDemon` implements the same two-draw
current-data swap without the outer trigger guard, but the live `Act` body
contains the algorithm inline and never calls it. An executable-section
direct-edge scan finds zero static call or jump edges to helper RVA
`0x3E42A0`. Its sole absolute pointer is the ordinary contiguous IL2CPP
method-registration entry at RVA `0x26A5128`.

The reachable virtual `Marionette.Act` likewise has no static direct caller,
as expected for `Character.RoleAct` vtable dispatch; its method-registration
pointer is RVA `0x26A5120`. Public asset binding, the serialized Start slot, and
the audited `Character.Act`/`RoleAct` route establish its reachability. The
private helper owns no callback, picker, serialized binding, or virtual slot.
It is therefore classified unreachable only for the shipped
Standard/Ascension gameplay surface, not as stripped or mechanically
unreflectable code.

## Typed-union accounting

Fifteen of the original target memberships were exact managed-identity overlaps
with the preceding 24-set prefix. The boundary introduced all five declared
Marionette methods; its constructor body was already present under a canonical
folded identity, so the other four methods introduced four native RVAs. This
checkpoint additionally selects `Character.RevealReal` and
`Role.GetRegisterAsRole`; both are new managed definitions, while the latter's
constant-null RVA was already represented by `Role.GetBluffIfAble`.

The current deterministic 41-set union contains 871 memberships, 538 distinct
selected FunctionDefinitions, and 440 unique native RVAs. Its 333 exact
membership overlaps and 98-definition folded/shared-body gap remain explicit.
The rebuilt GDT contains 151,674 datatypes. Twin Minion signature application
and read-only validation both close 22/22 functions and 66 membership-level
parameter-storage locations with zero validation-time program mutations.
Across the whole union, the final read-only pass validates all 871 memberships
and 2,554 parameter-storage locations.

## Corpus, solver, and live implications

A deterministic scan of the 426 checked-in `tests/cases_v2` fixtures finds 79
cases whose authored deck contains Twin Minion plus at least one Demon and
whose stable true-Evil records identify both. Only 21 of those 79 stable Twin
positions are circularly adjacent to a recorded Demon; 58 are not. That is
consistent with the native distinction between stable/original runtime-Evil
identity and the **current data identity** that the Start swap places beside a
Demon. The historical fixtures do not expose a complete current-data swap
trace and cannot independently prove the native selection algorithm.
All 79 fixtures also capture a card on at least one possible neighbour endpoint,
and none carries trusted board-count provenance for this exact slice. The newly
admitted actor-only evidence therefore changes no archived corpus outcome.

Reconstruction, solver, and live tooling should therefore:

- never impose `stable Twin Evil position adjacent to stable Demon position`;
- branch first over registered-or-real Demon occurrences, then over that
  Demon's previous/next alive occurrences;
- swap current role data while preserving physical runtime alignment, statuses,
  resistance, runtime data, and IDs;
- run later ordered-Start roles from their post-swap current positions while
  never replaying already-passed slots;
- permit current Twin Minion data on a runtime-Good physical card and the
  original runtime-Evil card to carry the neighbour's former current role;
- never infer a swapped endpoint from an apparent Twin Minion display, and
  never substitute stable original Demon positions for the current
  registered-or-real Demon pool;
- apply current-`dataRef`-sensitive rules such as Bombardier loss to the moved
  identity, while execution truth continues to follow runtime alignment; and
- emit only the fixed passive Description, with no Twin-local speech, target
  references, reset history, or achievement.

The current Rust solver correctly lacks the tempting but unsound stable
Twin/Demon adjacency predicate. It now represents atomic exact subsets of this
current-data swap: the dedicated Twin-to-Puppeteer slice, plus Twin-to-Shaman
worlds where every possible Twin endpoint is a proven structural non-Villager.
The latter crosses every Twin occurrence trace with every existing Shaman
source/destination trace, including no-Demon and relocated-Shaman outcomes,
because the live Villager pool cannot change. If any possible endpoint is a
Villager or is structurally unknown, the whole ordered state falls back rather
than retaining only the safe RNG branch. Copied Bounty Hunter is also excluded
because its immediate Start alignment mutation remains outside this model.

The first candidate-changing Twin-to-Shaman role-flow slice is now exact for
trusted no-Outcast Scout/Witness pools with exactly Twin, Shaman, and fully
dealt Lilis Demons. It enumerates the complete pre-Twin Villager occurrence
map, applies every Demon/side Twin trace, rebuilds Shaman's live Villager pool
from each post-swap map, and retains both ordered draws in the semantic key.
Duplicate Scout/Witness occurrences deduplicate equivalent assignments without
collapsing native Twin-side or Shaman-direction probability mass. A complete
validator independently reconstructs the map and both reachable traces; an
invalid claimed baseline fails closed, while cap or unsupported inputs trigger
wholesale legacy fallback.

The first public-action checkpoint now admits exactly one direct current-build
Scout or Witness card on the original Twin body. That body remains runtime Evil
and therefore `CheckLying`-positive. Its fresh raw `bluffRole` is null, base
`GetRegisterAsRole` returns null, and the later Villager Reveal preserves the
final Scout/Witness `dataRef` as its presentation. `Character.Act` dispatches
that real Villager data through its concrete `BluffAct`, so the existing lying
validators are exact; the scenario validator independently requires the
presented role to equal the replayed final current role.

A card on the moved neighbour, any second card, reveal ordering, ability or kill
history, Rambler evidence, and terminal-loss evidence still trigger wholesale
fallback. The runtime-Good neighbour carrying Twin data receives an untraced
Minion bluff, so its apparent identity, callback role, and appearance-sensitive
queries remain opaque. General current-data replay, other mixed writers, global
delayed-Reveal ordering, the round duplicate/unique bluff pools, and explicit
moved-neighbour bluff provenance remain native parity gaps; live Twin play stays
conservative outside the supported scenario kernels.
