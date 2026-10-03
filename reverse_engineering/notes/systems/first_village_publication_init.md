# Retained original first-village publication and Init dispatch

The solver dependency is whether the original generated N5 actors can retain
their actual initialization state through native publication and concrete role
Init dispatch. This distinct witness carries the same five physical actors,
role clones, status/list storage, native Manage caller and first waits through
that boundary. Its exit is before ordered Start, not completed acquisition or a
player observation.

The [producer](../../scripts/audit_first_village_publication_init.py) writes
`reports/f530404b0f3f_807de4a83df4_first_village_publication_init.json` for pinned
build `f530404b0f3f_807de4a83df4`. It consumes the frozen
[retained initialization](first_village_initialization.md),
[original generation/pools](first_village_bluff_generation.md) and
[concrete role setup](first_village_role_setup.md) evidence without changing
their source files or report domains.

## Admitted retained witness

Generation row zero and pool row zero retain the original asset order
`21596,21614,21626,21621,21618`: Minion, Confessor, Lover, Hunter and Enlightened.
Their exact source classes are `Minion`, `Confessor`, `Empath`, `Tracker` and
`Shugenja`. Runtime class virtual entries and MethodInfo contexts are a named
pre-entry hydration delta on those existing source classes. Their identities
are retained. No post-Init actor preparation, new runtime role replacement, new
status component, or B's independent enum-list storage is used.

The original primitive pipeline still executes native Character constructors,
original profile/roster generation, both native pool builders, all five
Character.Init calls and actual DelayReveal state-zero first steps. Scene raw
state 20 and previous state 10, empty status/resistance storage, UI services,
CLR/base construction, source-role cloning and synchronous StartCoroutine first
entry remain supplied providers. These values do not certify real scene Init.

At the original `0x36D01E` checkpoint, all five actual Init returns are complete.
Actors are Hidden raw state 5, previous state 20, with IDs `5,4,3,2,1`, one
runtime use, fresh empty history/version 1 and hover/version 0, constructor
empty saved text and act flag 1. Active lists are empty/version 1; resistance
lists are empty/version 0. Five distinct retained role clones and state-one
DelayReveal iterators retain their owners and waits, with exact native float32
wait bits `0x3E99999A`. No acquisition callback is resumed.

## Actual publication and concrete Init

The same original Manage CPU and stack continue past `0x36D01E`; they are not
reseeded by a root invocation. Actual `Gameplay.UpdateCharacters` (`3811B0`),
called at `36D03D`, requests a new `List<Character>`. Its supplied CLR constructor
performs a shallow copy of the original physical board. The native body
publishes that new list through `Gameplay.CurrentCharacters` static `+0x18`.
The old owner board/list remains unchanged, and the new publication retains all
five original actor occurrences in the same order.

The native caller invokes `Character.Act(Init=3)` at `36D0BA` on all five actors.
Actual `Character.Act`, `CharacterHelper.CheckLying`, native status Contains and
`Character.RoleAct` execute. RoleAct allocates and captures its closure/delegate,
performs the native current clone's `onActed +0x28` assignment and calls the
existing class's actual concrete virtual method. Delegate construction and
logging are supplied gateways. Any onActed invocation is rejected by the
declared callback adapter; no Init body should produce information here.

The native call history distinguishes the actual paths:

- Ordinary Evil Minion takes inherited `Role.BluffAct`, which forwards to its
  verified native `Minion.Act` folded `ret 0`.
- Good Confessor enters `Confessor.Act`, its `OnInit` virtual slot and actual
  `CharacterStatuses.AddStatus(status=25, source=actor, target=null)`.
- Empath, Tracker and Shugenja enter their exact metadata memberships of the
  shared Day-only body and return at trigger 3 without a concrete role effect.

The native Confessor writer checks exact resistance and duplicate membership
through supplied int32 CLR list services on the ORIGINAL backing. The declared
clear witness appends only `AppearTruthfull` (25), making actor:1's active list
`[25]`, version **2**. This follows the actual Init-produced version 1; B's
separate supplied post-Init witness began at version zero. Other active lists
remain empty/version 1, and all resistance lists remain empty/version 0. The
native shared target store remains null. Init installs five current callbacks
but leaves every Start latch zero, uses/history/saved text unchanged and all
five earlier iterator/wait bytes intact.

## Exact exit, preservation and stopped services

Each Act return at `36D0BF` verifies caller stack restoration and all eight
integer/ten XMM nonvolatile registers. The native loop then continues; the
normal post-loop branch reaches `36D0F9`. The audit stops before executing that
instruction, `mov r13,[r12+0x28]`, which would read `startGameActOrder`. The live
Manage stack pointer and original return sentinel are checked again. No Start
request, ordered-Start array read, onSetup, Shuffle or later callback is claimed.

After publication, preservation allows only the current runtime clone's native
`onActed +0x28` writes and the exact Confessor active-list count/version/first
int32 insertion. All actor bytes, other list storage, status components,
resistance storage, source assets/classes/roles, role saved fields `+0x30`,
`+0x38` and `+0x40`, first iterators and waits remain protected. Original profile,
source/start/cache/count/current-script/current-roster and native pool
projections are asserted at prepublication, every successful Act return, final
exit and all stopped snapshots.

Every reached service after the prepublication checkpoint is stopped before
its effect, including inherited enumerator and Unity gateways. One phase-aware
service recorder captures raw RCX/RDX/R8/R9, exact caller return and preservation
checks; its ordinals must cover the entire contiguous postpublication service
suffix. Each stopped replay matches that complete chronological event prefix
and exact successful pre-service snapshot. It replays the same native Manage
pipeline from the prepared graph and does not replace initialized actor state.

Original assets are freshly reparsed. Consumed source and report bytes are
fingerprinted and checked unchanged at exit. Semantic checks precede lossless
snapshot pooling, and expanded CLI output must exactly equal the audit return
through a JSON round trip. Full original native instruction bytes stay private.

The successful producer derives five constructors, five primitive Init returns,
five first yields and five ActInit returns. Its 652 Manage services include the
complete contiguous suffix of 69 postpublication services, each with an exact
stopped prefix. The report contains 2,393 visited native instruction addresses,
25 selected postpublication operand pins, 59 method-body memberships, 11 Python
source hashes and 351 pooled authored snapshots. These counts describe this
one clear generated witness and its stopped service suffix, not a factorized
or exhaustive generated-world domain.

## Remaining boundary

This one original clear witness does not widen B's separate sensitivity corpus
or establish world probabilities. Runtime slots, heap/class providers, CLR,
logging and UI operations retain their explicit supplied scope. Ordered Start,
onSetup/Shuffle, native coroutine queue registration/drains, DelayReveal
state-one acquisition completion, Reveal/click/Day publication and pixel or
trusted player-history admission remain unexecuted.

The next join must carry these same actors, their Confessor version-two status,
installed callbacks and five original first waits through the actual ordered
Start caller and then acquisition on the correct native owners. It must retain
the empty-history/one-use initialization chronology rather than supplying the
older finished Hunter stress fixture.

Reproduce with the existing private emulator dependencies:

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_first_village_publication_init.py --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' --output '<distinct-private-full-path>'
```
