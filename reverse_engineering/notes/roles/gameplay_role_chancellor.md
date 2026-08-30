# Gameplay role: Chancellor (managed `Baron`) and Witness

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for every method declared by the shipped
Chancellor and Witness roles and for the ordered Start, selection, status,
identity-mutation, death, and truth-dispatch helpers needed to close their
observable boundary. Serialized asset evidence establishes both public
bindings, authored text, and Chancellor's exact ordered Start slot. Native
bodies and decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_chancellor.json`](../../targets/gameplay_role_chancellor.json).
Its read-only baseline and typed Ghidra exports each completed at 31/31
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_chancellor.json)
passes its regression check: unresolved-type tokens fall from 237 to 105, raw
field-offset accesses from 294 to 102, placeholder parameter tokens from 239
to zero, and indirect-call patterns from 15 to two. Error-marker counts remain
three on both sides; the typed export adds one nonfatal warning marker.

## Public asset bindings and authored text

The shipped `sharedassets0.assets` `CharacterData` at path ID `21594` is named
`Chancellor`, has the unusual legacy `characterId` `Baron_04539999`, and binds
its SerializeReference role to managed `Baron` in `Assembly-CSharp`. Its raw
object SHA-256 is
`2AC121706A476186E51FAFAF18AE7F8BDBAFEDEBD4EB9946985DE2740071B09C`.
The asset is an Evil Minion (`characterType == 30`,
`startingAlignment == 20`), has `abilityUsage == 0` (`Once`), has no picker,
and is not bluffable or usually disguised. Its roguelike values are 10 points,
a 1.0 multiplier, and one income. `Baron` is TypeDefIndex `5913`, declares no
fields, and declares exactly five methods.

The exact public Chancellor description is:

```text
<b>Game Start:</b>
One Villager becomes an Outcast role.
I sit next to it.

I Lie and Disguise.
```

No Chancellor hint is serialized. The role's separate managed description is:

```text
Add 1 outsider if able. Sits next to an Outsider
```

`Baron.GetInfo` returns a fresh `ActedInfo` with an empty description and a
null character list. Chancellor therefore creates no clue reference, picker
result, or role-local history.

The public Witness is `sharedassets0.assets` path ID `21635`, is named
`Witness`, has `characterId` `Witness_25155076`, and binds managed `Witness` in
`Assembly-CSharp`. Its raw object SHA-256 is
`4743D07B004D6881FD71E271D3DFE74ACB62584AD4526BDBC6876923E8AB8CA4`.
It is a Good Villager (`characterType == 10`, `startingAlignment == 10`). The
asset has `abilityUsage == 0` (`Once`), is bluffable, is not usually
disguised, and has no picker. Its roguelike values are 10 points, a 1.0
multiplier, and zero income. The managed type is TypeDefIndex `5858`, declares
no fields, and declares exactly eight methods.

Witness's exact public description is:

```text
Learn a character that was affected by an Evil ability
```

`Witness.get_Description` returns that same exact string.

Its exact authored hint is:

```text
Can learn:
- Who is the Puppet
- Corrupted character by an Evil
- Villager turned into Outcast by Chancellor
- Cloned character by Shaman
- Who was killed by an Evil
```

The Chancellor line is broader than the current native implementation. Baron
does not mark the first Villager it replaces; it marks a separately selected
Outcast anchor. The hint is therefore not a reliable provenance rule.

Witness's exact English localized result templates are:

```text
NO character was affected by an Evil
{ids} was affected by an Evil
```

For the positive one-card native path, `ConjourInfo` renders the latter as
`#N was affected by an Evil`.

The `level0` `Characters` component at path ID `137026` references Chancellor
as zero-based index zero of `startGameActOrder`. Chancellor runs before every
other ordered Start role, but only after every physical card has completed
ordinary `Character.Init` and `Act(Init)` setup.

## Target boundary

| Managed identity | RVA | Boundary role |
| --- | ---: | --- |
| `Characters.ManageCharacters` | `0x36CE30` | Ordered Start and ordinary duplicate policy |
| `Character.Act` | `0x3645C0` | Start latch and truth-aware dispatch |
| `Character.RoleAct` | `0x368790` | Concrete role dispatch |
| `CharacterHelper.CheckLying` | `0x397750` | Evil-Chancellor and Witness truth choice |
| `Role.BluffAct` | `0x3C4CA0` | Inherited forwarder to concrete `Act` |
| `Character.Init` | `0x365A20` | Full role-data replacement and swap primitive |
| `Character.Kill` | `0x366130` | Death callback and persistence boundary |
| `CharacterStatuses.AddStatus` | `0x363AA0` | Resistance-aware affected marker |
| `CharacterStatuses.Contains` | `0x363C40` | Witness's only affected predicate |
| `Characters.GetAdjacentAliveCharacters` | `0x36C050` | Circular alive-neighbour list |
| `Characters.FilterCharacterType(Character)` | `0x36AC30` | Apparent-Outcast anchor pool |
| `Characters.FilterRealCharacterType(Character)` | `0x36BB40` | Real-Villager board pool |
| `Characters.FilterRealCharacterType(CharacterData)` | `0x36B9C0` | Real-Outcast role pool |
| `Characters.FilterNotInDeckCharactersUnique` | `0x36B1A0` | Prefer absent role data |
| `Characters.FilterAliveCharacters` | `0x36A240` | Exclude Dead first-stage targets |
| `Gameplay.AddScriptCharacter` | `0x37B480` | Add selected Outcast to current script |
| `Gameplay.GetAscensionAllStartingCharacters` | `0x37C3F0` | Preferred role universe |
| `Gameplay.GetAllAscensionCharacters` | `0x37C1A0` | Fallback role universe |
| `Baron.get_Description` | `0x3D41F0` | Managed description |
| `Baron.GetInfo` | `0x3D3FC0` | Empty passive information surface |
| `Baron.Act` | `0x3D3C80` | Start-only role insertion/replacement |
| `Baron.SitNextToOutsider` | `0x3D4020` | Anchor marker and identity relocation |
| `Baron.ctor` | `0x3CFFF0` | Folded base-role construction |
| `Witness.get_Description` | `0x3CFAC0` | Managed description |
| `Witness.ConjourInfo` | `0x3CF2B0` | Exact positive string formatting |
| `Witness.GetInfo` | `0x3CF750` | Truthful current-marker observation |
| `Witness.Act` | `0x3B09F0` | Day truthful callback |
| `Witness.BluffAct` | `0x3B33E0` | Day lying callback |
| `Witness.GetBluffInfo` | `0x3CF380` | Uniform unmarked false claim |
| `Witness.GetMessedCharacters` | `0x3CF8C0` | Physical current-status scan |
| `Witness.ctor` | `0x357920` | Folded base-role construction |

The two constructors, `Role.BluffAct`, `Witness.Act`, and `Witness.BluffAct`
share bodies with other managed identities. The manifest retains the exact
role-facing definitions while applying the established canonical prototypes
where a folded body needs one.

## Ordered Start and inherited Evil dispatch

`Characters.ManageCharacters` completes universal initialization before
walking the serialized Start list. It scans physical cards from highest
displayed ID to lowest for each ordinary entry and stops at the first exact
`CharacterData` match. Only Alchemist, Poisoner, and Puzzlemaster are duplicate
exceptions; Baron is ordinary.

An Evil Chancellor initially satisfies `CheckLying`. This does not suppress
its Start: Baron does not override `BluffAct`, so inherited `Role.BluffAct`
forwards virtually to the same object's concrete `Baron.Act`. Both truthful and
lying dispatch therefore reach the same Start implementation. The per-card
Start latch prevents a second Start call on the same initialized character.

The exact shipped order begins:

```text
Chancellor -> Pooka -> Poisoner -> Drunk -> Witch -> Marionette
-> Puppeteer -> Baa -> Plague Doctor -> Shaman -> Alchemist -> ...
```

Consequently Chancellor runs before every other role-specific Start mutation,
but after OnInit effects such as Alchemist's physical Corrupted resistance.

## Stage one: add a role and replace an anywhere Villager

`Baron.Act` is a no-op unless the trigger is Start (`5`). On Start it:

1. copies `Gameplay.CurrentCharacters` shallowly;
2. concatenates the current ascension starting arrays in exact Demon, Outcast,
   Minion, Villager order;
3. for each CharacterData already in the current script, removes every
   exact-reference-equal occurrence from that copy, then filters the result to
   real Outcast data;
4. if that result is empty, builds the fallback aggregate and filters it to
   real Outcasts;
5. uniformly draws one occurrence `r` and adds that exact Outcast
   `CharacterData` to the current script;
6. filters the board copy to non-Dead cards, then to cards whose real
   `dataRef.characterType` is Villager;
7. uniformly draws one physical target `v`; and
8. calls `v.Init(r, -100)`.

Both filtered lists preserve source order and occurrence multiplicity. Despite
its `Unique` name, `FilterNotInDeckCharactersUnique` does not run a general
distinct operation: duplicate occurrences absent from the script retain their
native random weight. The draws have no role, status, resistance, alignment,
adjacency, or Alchemist exclusion beyond the explicit real-type and non-Dead
predicates. The preferred role pool is therefore the starting-Outcast array
minus every exact CharacterData reference already in the script.

`GetAllAscensionCharacters` has a precise authored-code quirk: it concatenates
all-ascension Townsfolk, Outcast, Minion, and Townsfolk arrays, repeating
Townsfolk and never adding the Demon array. Baron's immediate real-Outcast
filter makes the quirk irrelevant to type eligibility, so the fallback is
exactly the all-ascension Outcast array with its original order and duplicate
occurrences. It does not exclude current-script entries and can reuse an
already-present Outcast identity.

The native implementation obtains the random role item before its effective
nonempty continuation. An empty fallback pool therefore reaches a native
index/get-item failure rather than honoring the authored phrase "if able."
The Villager draw likewise has no empty-list guard. Earlier successful script
mutation is not rolled back if a later step fails.

`Character.Init` with sentinel `-100` preserves the physical card ID and its
resistance collection, but clears active statuses, acted history, runtime data,
the live `bluff` CharacterData, `registerAs`, and reveal/death/activity setup
before installing and cloning `r`. It does not directly clear the separate
`bluffRole` object; the delayed internal reveal path rebuilds the presentation
surface from the new data. Thus this stage moves no `Character` object. It
replaces the role data on the same physical board position.

## Stage two: mark an Outcast and relocate Chancellor

After stage one, Baron always enters `SitNextToOutsider` on the successful
path. It:

1. shallow-copies the current physical board;
2. filters it with the Character overload of `FilterCharacterType` for Outcast;
3. uniformly draws one apparent-Outcast anchor `o`;
4. calls `o.statuses.AddStatus(MessedUpByEvil, c, null)` where `c` is the
   acting physical Chancellor;
5. obtains `o`'s circular previous and next non-Dead neighbours and uniformly
   draws one occurrence `f`;
6. saves `q = f.dataRef`;
7. calls `f.Init(c.dataRef, -100)`; and
8. calls `c.Init(q, -100)`.

This is a real swap of two `CharacterData` identities through full
reinitialization, not a clone and not a swap of physical `Character` objects.
Chancellor data and Evil alignment move to `f`; the original physical card
`c` receives `q`. Resistances remain with physical cards, while active statuses
and histories on `f` and `c` are cleared by their respective `Init` calls.

The apparent-type filter prefers a Unity-live `registerAs` record and otherwise
uses real `dataRef`. At this first ordered Start slot the ordinary board's
register-as setup is clear, so the shipped path selects a current real Outcast,
including the just-generated `v/r`. It does not filter Dead itself. A Dead
anchor is outside normal game-start state; abstractly, it can be marked and
then fails neighbour lookup because it is absent from the alive ring.

`GetAdjacentAliveCharacters` first filters the board to non-Dead entries,
finds the anchor by physical Unity identity, and returns previous then next in
circular order. A two-card alive ring returns the same other card twice; a
one-card ring returns the anchor twice. On an ordinary larger board the draw is
a 50/50 choice of physical sides.

## Exact identity equations and legal overlaps

Define:

- `c`: the original acting Chancellor physical card;
- `v`: the first-stage real-Villager physical target;
- `r`: the added Outcast CharacterData;
- `o`: the second-stage apparent-Outcast physical anchor;
- `f`: the selected alive neighbour and final Chancellor home;
- `a`: the final physical home of `r`; and
- `q`: the old CharacterData at `f` immediately before the swap.

The exact post-action equation and inverse are:

```text
a = if v == f { c } else { v }

if a == c:
    v = f
else:
    v = a
```

Stable constraints are:

```text
v != c
o != c
o != f
a != f
```

The following overlaps are legal and materially distinct:

- `f == v`: the selected Villager becomes Chancellor and original `c` becomes
  the added Outcast `r`, so `a == c`;
- `f == c`: a self-swap is legal when Chancellor is adjacent to `o`; `c`
  remains Chancellor and `a == v`;
- `o == v`: the newly generated Outcast can be selected as the anchor, forcing
  `f != v`; and
- `o == a`: legal exactly on the `f != v` path where the new Outcast remains at
  `v`; it is impossible on `a == c` because `o != c`.

If `f` is an unrelated card, `v` remains `r`, `f` becomes Chancellor, and `c`
receives `q`. The anchor `o` itself is never swapped because it cannot be its
own alive neighbour in the ordinary working board.

No native history field stores `c`, `v`, `o`, `f`, or `a`: Baron has no fields
and emits no acted reference. Only the resulting state remains. The exact `r`
record is present in the script's Outcast list and as `a.dataRef`; Chancellor
data is at `f`; `q` is at `c`; and a successful affected marker remains on
physical `o`. Neither the original source card nor first replacement target is
otherwise tagged, so a clean-room model must carry the ordered trace itself.

## `MessedUpByEvil`, resistance, and Witness provenance

`CharacterStatuses.AddStatus` checks an exact matching resistance before all
other work. If resistance `50` is present, Baron's marker call is a complete
no-op. Otherwise it unique-adds `MessedUpByEvil` (`50`), ignores the supplied
source reference, and writes the supplied cure target to the status container's
single shared `targetCharacter` field. Baron supplies null, so the successful
call clears that shared field even when status `50` was already present. No
source provenance or per-status history is stored.

An exhaustive current-build producer audit found no shipped path that adds
resistance `50`. Alchemist OnInit adds only Corrupted resistance (`40`), and
Chancellor/Shaman preserve but do not transform that enum. Baron's marker is
therefore not resistible in ordinary current-build play, although the generic
status API supports the abstract edge.

Even on that abstract resisted edge, Baron does not branch on insertion
success: the same selected anchor still drives neighbour selection and the
identity swap. Anchor provenance must therefore record the selection whether
or not a current marker survives; Witness consults only the surviving marker.

Most importantly, stage one does **not** mark `v`. Its `Character.Init` clears
any old active marker. Baron independently marks `o` in stage two. A truthful
Witness can attribute Chancellor to `v` only when `v == o` and the generated
Outcast itself was selected as anchor, or when another Evil effect later marks
that same physical card. The old hypothesis that every first Villager
replacement counts as "affected by evil" is native-disproven.

The anchor trace and current Witness marker set are different solver surfaces.
Two Chancellor histories can choose different anchors yet converge to the same
final active status set when other Start roles also mark both cards. Witness
cannot distinguish those histories, while Chancellor relocation probability
and adjacency still require preserving the anchor candidates.

## Witness's exact current-status observation

`Witness.GetMessedCharacters` shallow-copies
`Gameplay.CurrentCharacters`, iterates every physical card in board order, and
keeps exactly those whose current status container `Contains(50)`. It does not
filter by state, alive/dead, hidden/revealed, alignment, role, source, or event
history. It neither clears nor consumes the status.

Truthful `GetInfo` snapshots that list:

- with at least one marked card, it uniformly selects one occurrence and
  returns an `ActedInfo` whose non-null reference list is exactly `[selected]`
  and whose text is `#N was affected by an Evil`;
- with none, it returns exact text `NO character was affected by an Evil` and
  a non-null empty reference list.

`GetBluffInfo` snapshots the marked set, builds the complement across every
current physical board card in board order, and uniformly selects one unmarked
card for the same positive one-reference false claim. Only when **every**
physical card is currently marked does the lying path emit the same NO text
and empty reference list. A lying NO claim is therefore not supported merely
because one or some cards are marked.

`Witness.Act` invokes truthful information only for Day. `Witness.BluffAct`
invokes bluff information only for Day. Other triggers are no-ops. The ordinary
`Character.Act`/`CheckLying` path decides which slot is used; Good clean Witness
is truthful, while a normal Corrupted Witness lies. `MessedUpByEvil` itself is
not a lying status.

Because the predicate is current physical status only:

- a marked card remains eligible after death;
- cards transformed through full `Character.Init` stop qualifying because
  their active statuses are cleared;
- multiple Witness cards do not consume one another's candidates and may name
  the same card; and
- no shipped downstream method can recover which Evil/source added a surviving
  marker.

The delayed demon-kill path sets the target Dead and then attempts
`MessedUpByEvil` (`50`) before death callbacks. Exact-50 resistance could
abstractly reject that insertion, but no current shipped role installs it.
Every successful current-build night kill is therefore visible to Witness,
including after ordinary Alchemist or Chancellor/Shaman relocation histories
that carry only resistance `40`.

## Interaction and ordering consequences

Chancellor runs before Pooka, Poisoner, Drunk, Witch, Puppeteer, Plague Doctor,
Shaman, and Alchemist Start actions. Its full `Init` calls make role identity
move while physical resistance does not:

- if `v` was Alchemist, that role is erased before Alchemist's later Start but
  its already-installed Corrupted resistance stays on physical `v/r`;
- if `f` held Alchemist, Alchemist data moves to `c` without its resistance,
  while physical `f` becomes Chancellor and keeps resistance `40`;
- if `c` receives a Villager `q`, later Puppeteer and Shaman see it as a real or
  apparent Villager candidate and may overwrite/copy it;
- final `f` is real Chancellor and `a` is real Outcast, so neither is an
  ordinary real-Villager candidate for those later mutations; and
- Shaman can later mark either of its eligible endpoints, and its
  `InitWithNoReset` preservation rules compose with the already completed
  Chancellor transformation rather than reversing it.

Puppet and Shaman status insertion does not record source provenance. Witness
therefore consumes only the final union of surviving physical markers, not a
per-role or ordered event log.

## Duplicate, death, and reset behavior

If multiple physical cards share the exact public Chancellor CharacterData,
the ordinary reverse/highest-ID scan calls Start on only the first match. The
others remain Chancellor identities but do not independently add an Outcast or
relocate. This makes a one-Chancellor guard a current corpus/model policy, not a
native invariant. If the acting Chancellor chooses a neighbour already holding
duplicate Chancellor data, saving and swapping identical data is
observationally neutral and both cards remain Chancellor. A distinct
Baron-backed CharacterData absent from `startGameActOrder` receives no ordered
Start merely because its managed role class is Baron.

Witness is not an ordered Start role. Each physical Witness independently
evaluates the live status set on Day; duplicate Witnesses have no local state
and do not consume markers.

Baron declares no fields and does not override `ActOnDied`. `Character.Kill`
dispatches the current real `dataRef.role` death callback, which reaches the
base no-op for Chancellor. Death does not remove the added script CharacterData,
undo either `Init`, swap identities back, or clear `o`'s marker. If Chancellor
data moved to `f`, death behavior follows that final real identity rather than
the original physical `c`.

No Baron-local reset exists. A new board/village reconstructs characters and
reruns `ManageCharacters`; the transformations and current statuses do not
persist as Baron history. During a transformation, `Character.Init` itself
clears the affected physical card's acted/runtime/status history.

## Reconstruction consequences and bounded unknowns

- Bind public Chancellor to managed `Baron`, and keep public Witness bound to
  managed `Witness`.
- Branch the first target over non-Dead real Villagers anywhere on the board;
  do not impose adjacency or status/resistance eligibility.
- Add one selected real-Outcast CharacterData to the script, then preserve its
  final home with the `c/v/f/a` equation rather than a single conversion field.
- Branch the anchor over apparent Outcasts after the replacement, then the
  final Chancellor over the anchor's alive circular neighbours.
- Preserve the selected physical anchor independently from the final current
  `MessedUpByEvil` set. Witness observes only the latter.
- Never mark the first Villager solely because Baron replaced it.
- Treat truthful Witness NO as requiring zero current markers and lying Witness
  NO as requiring every physical board card to be marked.
- Keep dead and night-killed current markers visible to Witness.
- Do not attach source provenance to status `50`; native storage has none.
- Treat duplicate same-asset Chancellor as one ordered action, while retaining
  duplicate identities as a native-representable abstract board edge.

No native semantic uncertainty remains in the shipped working-path selection
order, identity movement, overlap equations, status target, Witness predicate,
truthful/lying zero and positive output shapes, or death persistence. Two edges
remain bounded rather than observed in the current corpus: native empty-pool
paths fail instead of producing a legal no-op, and the generic exact-50
resistance mechanism has no shipped producer under this build fingerprint.
