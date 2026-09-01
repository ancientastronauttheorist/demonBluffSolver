# Gameplay role: Shaman (managed `Illuzionist`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all four methods declared by the shipped
role and for the previously unaudited `Character.InitWithNoReset` mutation
boundary. Six reused lifecycle, status, selection, and bluff-query identities
close the role's dispatch path. Native bodies and decompiler output remain
outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_shaman.json`](../../targets/gameplay_role_shaman.json).
Its read-only baseline Ghidra export completed at 11/11 functions with no
failures, and the typed export subsequently completed the same 11/11 boundary
without failure.

## Public asset binding and trigger order

The shipped `sharedassets0.assets` `CharacterData` at path ID `21600` is named
`Shaman`, has `characterId` `Shaman_26945607`, and binds its
SerializeReference role to managed `Illuzionist` in `Assembly-CSharp`. Its raw
object SHA-256 is
`D34970DE9C82A2CB90507C075F71902586CB6F9C83266DE439E94A3368A0ACDD`.
The serialized type is Minion (`30`), its starting alignment is Evil (`20`),
and its authored ability says that two same Villager roles exist at game
start.

The `level0` `Characters` component at path ID `137026` references Shaman's
path ID `21600` as zero-based index 9, the tenth entry in
`startGameActOrder`. Shaman therefore runs after Puzzlemaster/Plague Doctor and
before Alchemist in the shipped ordering. Statuses applied by earlier Start
roles can already be present on Shaman's candidates.

The public Witch is a different asset and managed type. Its path ID is `21602`,
its raw object SHA-256 is
`6B4353F67AF9999D0D4BC7E88791B37B5E2436380B6013DF26B7A6F4F04B838D`,
and its SerializeReference role is `Cipher`. Runtime-name normalization must
therefore map `Illuzionist` to Shaman and `Cipher` to Witch; the class name does
not justify mapping `Illuzionist` to Witch.

## Target boundary

| Managed identity | RVA | Boundary role |
| --- | ---: | --- |
| `CharacterStatuses.AddStatus` | `0x363AA0` | Resistance-aware affected marker |
| `Characters.FilterCharacterType(Character)` | `0x36AC30` | Apparent-Villager candidate pool |
| `Character.GetCharacterBluffIfAble` | `0x364C40` | Source bluff-or-real identity |
| `Character.InitWithNoReset` | `0x365720` | Destination identity replacement |
| `Character.DelayReveal.MoveNext` | `0x3756B0` | Role clone and delayed internal reveal |
| `Character.Reveal` | `0x368410` | Internal setup/presentation continuation |
| `Character.Act` | `0x3645C0` | Truth-aware copied-role dispatch |
| `Illuzionist.get_Description` | `0x3DD3E0` | Managed passive description |
| `Illuzionist.GetInfo` | `0x3DD380` | Passive information surface |
| `Illuzionist.Act` | `0x3DD190` | Start clone orchestration |
| `Illuzionist.ctor` | `0x3CFFF0` | Base-only construction |

`Illuzionist` is TypeDefIndex `5915`, declares no fields, and declares exactly
the final four methods in the table. The first seven entries are exact helper
identities needed to distinguish a role copy from a simple duplicate-count
allowance. Six were already covered by subsystem audits;
`Character.InitWithNoReset` is new native-static coverage.

## Ordered source and destination selection

`Illuzionist.Act` returns immediately for every trigger except `Start` (`5`).
At Start it makes a shallow `List<Character>` copy of
`Gameplay.CurrentCharacters`, then passes that copy to the Character overload
of `Characters.FilterCharacterType` with Villager (`10`). The filter allocates
a new list, preserves source order and duplicate occurrences, and compares the
Unity-live `registerAs` record when present; otherwise it compares `dataRef`.
It does not consult the card's current bluff.

The method draws one uniform random index from the entire filtered list. That
entry is the source. It attempts to add `MessedUpByEvil` (`50`) to the source,
using Shaman's `charRef` as `sourceRef` and null as the shared cure target, then
removes one matching source occurrence from the list. It makes a second uniform
draw from the remaining occurrences; that entry is the destination. The
direction matters: the source keeps its identity, while the destination is
overwritten.

There is no filter for real alignment, liveness, existing status, resistance,
or distinct object identity beyond the one list removal. A resistance to
`MessedUpByEvil` independently suppresses either status insertion without
canceling the clone. In an ordinary board list the references are unique, so
selection is an ordered pair without replacement.

Neither draw has a count guard. Fewer than two candidate occurrences reaches a
native failure path. A null source, missing source status container, null
destination, or missing destination status container also fails. Work already
performed is not rolled back: for example, the source can retain
`MessedUpByEvil` if the second draw or later destination mutation fails.

## Copied identity and timing

After selecting the destination, Shaman calls
`source.GetCharacterBluffIfAble()`. That helper returns a Unity-live bluff only
while the source is neither Dead nor Revealed, its separate `revealed` flag is
false, and its bluff reference is live. Every other path returns the source's
real `dataRef`. The apparent-type filter and copied-identity query are therefore
different surfaces: selection is register-as-first, while copying is
bluff-or-real.

In the shipped initial Start pass, however, this generic bluff branch is not
normally reachable. Fresh `Character.Init` calls clear `bluff` and
`registerAs`, and none of the pending `DelayReveal` continuations resumes until
the synchronous ordered Start dispatcher returns. Earlier Chancellor and Twin
`Init`/`InitWithNoReset` writers likewise leave the source's live `bluff`
pointer null at Shaman's slot. An ordinary selected source therefore contributes
its current real Villager data. In particular, public Plague Doctor is an
Outcast and cannot be copied by the shipped pre-Reveal Shaman path merely
because a later presentation could give some physical card a PD bluff. A copied
PD Start remains a valid latent/native composition only for an explicitly
proven nonstandard state with a live eligible source bluff or equivalent
registered/current-data provenance.

Shaman passes the resulting `CharacterData` to
`destination.InitWithNoReset(data, -100)`. It then immediately invokes
`destination.Act(Start)`, and only after that dispatch attempts
`MessedUpByEvil` on the destination. The copied Start action cannot observe the
new destination marker from Shaman because the marker is added afterward.

This behavior is an overwrite, not a swap:

```text
source.data                  -> unchanged
source.MessedUpByEvil        -> attempted before copying

destination.data             -> source bluff-or-real CharacterData
destination prior identity   -> removed from the board assignment
destination copied Start     -> dispatched before its affected marker
destination.MessedUpByEvil   -> attempted after copied Start
```

The deck remains the authored role pool; Shaman changes a live board
assignment. In the normal case this leaves the source identity in place and
adds one board duplicate of it while removing the destination's former board
identity.

## `InitWithNoReset` mutation boundary

`Character.InitWithNoReset` performs the following native mutations in order:

1. hide the acted-information object and clear `actedInfos`;
2. reset the one-shot `characterStartActed` latch;
3. hide and destroy a Unity-live dead-character prefab, then clear its field;
4. clear `bluff` and `revealed`;
5. assign the supplied `CharacterData` to `dataRef`;
6. clear `killedByDemon` and reset `pickableUses` to one;
7. preserve the existing physical ID and number because Shaman passes `-100`;
8. copy the old state to `prevState`, set state to Hidden, and invoke the state
   callback;
9. refresh the card presentation; and
10. start the `DelayReveal` iterator.

The method does **not** reset the destination's alignment, active statuses,
resistances, runtime data, `bluffRole`, or current `registerAs` field. The
subsequent internal Reveal recomputes register-as and bluff choices from the
new real role, but the other preserved values remain consequential. In
particular, a previously Corrupted destination is still lying when the copied
Start dispatch occurs, so `Character.Act` can invoke the copied role's
`BluffAct(Start)` rather than its truthful `Act(Start)`. A stale non-null
`bluffRole` can also receive the corresponding dispatch.

The operation requires live acted UI, acted-info storage, and supplied
`CharacterData`. Missing dependencies reach the native failure path, possibly
after earlier fields were cleared; this is not a transactional replacement.

## Role cloning, dispatch, and internal reveal

`DelayReveal.MoveNext` clones `dataRef.role` into the destination's live `role`
field before yielding `0.3` seconds. Unity starts the iterator through its
first yield before `StartCoroutine` returns, so by the time
`InitWithNoReset` returns to Shaman the new role object is installed. The
immediate `Character.Act(Start)` therefore dispatches the copied role rather
than the destination's stale real role.

`Character.Act` consumes the reset Start latch and applies the destination's
preserved truth state. A truthful destination invokes copied `Act(Start)`; a
lying non-Evil destination invokes copied `BluffAct(Start)`. Any preserved
bluff-role object receives the matching secondary dispatch. This also makes
the explicit immediate call robust to serialized ordering: a copied role whose
ordinary Start slot already passed still acts, while a later attempt is blocked
by the now-consumed Start latch.

When the iterator resumes, it calls the internal `Character.Reveal` setup
routine. That routine recomputes register-as data, obtains a new bluff when
needed, performs `Init` and `AfterRoundStart` role dispatch, and updates the
real-or-bluff presentation. It is not the player's face-down-card flip and does
not change the card state, update reveal order, or invoke the public
`onReveal` callback.

The copied role's own behavior remains a separate per-role boundary. The
dedicated [Baker audit](gameplay_role_baker.md#truth-dispatch-status-composition-and-chain-reachability)
now closes its important composition. Shaman does not synthesize
`BakerRuntimeData` and adds only `MessedUpByEvil`, not
`AlteredCharacter`. A clean copied Baker with null preserved runtime data says
`I am the original Baker` and can create a descendant. Preserved
`BakerRuntimeData` supplies its saved name, while a non-null incompatible
subtype such as `AlchemistRuntimeData` or `EnlightenedRuntimeData` reaches the
ordinary Baker invalid-cast path before output or conversion. A Corrupted
destination dispatches the lying Baker path and does not extend the chain in
current setup because no audited producer gives it `WorkingAbility`.
Later Baker-created descendants, rather than Shaman itself, write the real
identity they replace into fresh Baker runtime data.

A solver trace should preserve Shaman's ordered source, destination, copied
data, viable erased-role identity class, and destination runtime/status
composition instead of representing the effect only as an unconstrained
extra-role allowance.

## Passive and construction surfaces

`Illuzionist.GetInfo` ignores its character argument and returns a fresh
`ActedInfo` with an empty description and null character list. Its managed
description resolves to `[There are 2 Villagers of the same Roles]`; the
public authored ability text remains the player-facing rule surface.

The constructor is the ordinary folded base-role initialization body at
`0x3CFFF0`. That RVA is shared by hundreds of trivial constructors, so the
target retains the exact `Illuzionist.ctor` identity while applying the
established canonical constructor prototype.

## Solver and live-tool consequences

- Native selection is an ordered apparent-Villager source/destination pair,
  not a free choice of any duplicated Villager name.
- One existing board identity is removed whenever the clone succeeds. Counting
  only the new duplicate loses half of the board transformation.
- The copied identity can be the source's live bluff, not necessarily its real
  `dataRef`.
- Destination truth/status state survives the overwrite and controls copied
  Start dispatch before Shaman attempts the destination marker.
- Both source and destination are attempted Witness-style affected markers,
  with resistance checked independently.
- Runtime fallback names must distinguish managed `Illuzionist` (Shaman) from
  managed `Cipher` (Witch).

The Rust solver now carries this boundary as `ShamanTrace`: ordered source and
target positions, the copied role, and a probability-safe class of viable
erased prior roles. Scenario generation branches Shaman after Plague Doctor and
before Alchemist, keeps both marker attempts, and replays a copied Alchemist's
immediate truthful-or-lying Start behavior. Alchemist pre-state remains a
separate trace because its preserved resistance is observable; solver-equivalent
non-Alchemist identities are grouped and deck multiplicity is accepted only if
at least one member reverses to a valid initial multiset. A possible hidden
endpoint is admitted only when the natural-Outcast budget can make it a
Villager, and that endpoint assumption is inserted into the earlier corruption
target pools before Start simulation. The former unconstrained `+1` duplicate
and Villager-header allowances are no longer used. Copied Baker behavior now
follows the linked Baker audit: null runtime means original, compatible Baker
runtime preserves its named history, incompatible non-null runtime can
invalidate the Day surface, and current Corrupted copies do not create
descendants without status 38.

The exact ordered Twin slice can now carry this trace through the earlier Twin
slot when every possible Twin endpoint is a structural non-Villager. Twin may
move the one current Shaman identity onto the original Twin body, but the later
ordinary Shaman scan still dispatches it exactly once and its global Villager
pool is unchanged. Scenario generation therefore preserves the full Cartesian
product of `TwinTrace` and `ShamanTrace`, including the explicit no-Demon path.
It rejects the entire ordered state if even one Twin branch touches a Villager
or unknown endpoint, and continues to exclude copied Bounty Hunter because that
immediate copied-Start alignment writer is not modeled here.

The exact no-Twin Puppeteer slice now composes the other shipped writer order
directly. On trusted no-Outcast boards with exactly one Puppeteer, one Shaman,
fully dealt Lilis Demons, and one deterministic non-Saint Villager neighbour,
the solver enumerates the complete initial Villager occurrence map, applies
Puppet's full replacement, and only then builds Shaman's ordered Villager
source/destination pairs. The erased Villager identity remains Puppeteer
provenance and cannot re-enter Shaman's pool. Both traces and all three
`MessedUpByEvil` attempts are preserved. Ambiguous neighbour surfaces,
preserved-status/runtime-data roles, or enumeration caps fall back atomically.

The pure corruption replay also models the latent ordinary-runtime-Good copied
Plague Doctor callback without exposing it as a normal scenario root. It runs
after every global Plague Doctor and before the destination marker and global
Alchemists, rebuilds the live eligible pool, and preserves a separate copied
target/no-candidate provenance value through later cure convergence. With the
ordinary pre-Reveal null `registerAs`, the overwritten destination drops from
that pool after its real data becomes Outcast; an exact override requires a
caller-proven live Villager `registerAs` pool plus the ordinary Good/no-stale-
bluff dispatch shape. Runtime-Evil or secondary stale-bluff dispatch remains a
separate ordered-replay frontier.

Copied Confessor has a different, shipped-reachable delayed effect. Its
immediate copied Start call is a no-op, but both Shaman endpoints own the
Confessor identity when their internal Reveal dispatches Init. Both the real
and bluff Confessor Init paths attempt status 25 (`AppearTruthfull`). An
exhaustive recovered-source producer check found no shipped resistance for
status 25: Alchemist is the only gameplay `AddResistance` caller and passes
only `Corrupted` (10). The settled status is therefore guaranteed on both
endpoints in current ordinary play, even when Corruption makes their actual
role dispatch lie. Later Baker conversion uses `InitWithNoReset` and preserves
the physical status.

The solver derives this exact fact from `ShamanTrace.copied_role == Confessor`
and endpoint membership instead of adding a second serialized status field.
Judge and Rambler consequently use truthful appearance for either endpoint,
including after a non-Confessor presentation change, while `truth_status`
continues to model actual dispatch independently. Rambler's setup coroutine
order does not add a branch here: every shipped setup target is Hidden, so the
hidden path installs its callback without an appearance query and rechecks
appearance only when the target later acts. A grouped
`target_previous_roles` class that merely contains Confessor remains
insufficient evidence; Twin can move Confessor data onto a physical card after
universal Init and Shaman can overwrite it before that body ever runs
Confessor Init.
