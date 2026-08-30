# Gameplay role: Rambler (managed `Rambler2`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for every method declared by the shipped
`Rambler2` role, its compiler-generated callback closure, and the setup,
truth-dispatch, adjacency, interference, acted-history, and reveal helpers
needed to close the public role's observable boundary. Serialized asset
evidence establishes the public binding and distinguishes the current
adjacent-character design from stale picker text. Native bodies and decompiler
output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_rambler.json`](../../targets/gameplay_role_rambler.json).
Its read-only baseline and typed Ghidra exports each complete at 36/36
functions with no failures. The body-free
[`quality report`](../../reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_rambler.json)
passes its regression check: unresolved-type tokens fall from 405 to 103, raw
field-offset accesses from 699 to 95, placeholder parameter tokens from 436 to
zero, and indirect-call patterns from 16 to three. Error markers remain zero;
the typed export adds one nonfatal warning marker.

## Public asset binding and authored contract

The shipped `sharedassets0.assets` `CharacterData` at path ID `21607` is named
`Rambler`, has `characterId` `Rambler_57930131`, and binds its
SerializeReference role to managed `Rambler2` in `Assembly-CSharp`. Its raw
object SHA-256 is
`0196744CB9D5587D74F82AADC43F4E7CFD0D93CE7D199BFCF420ADB73F028145`.
It is a Good Outcast (`characterType == 20`, `startingAlignment == 10`), has
`abilityUsage == 10` (`ResetAfterNight`), is bluffable, is not usually
disguised, and has no picker. Its roguelike values are 10 points, a 1.0
multiplier, and one income. Its additional-status, tag, conditional-appearance,
and hint lists are empty.

The exact public description is:

```text
I tell you something really interesting.

Adjacent Truthful characters tell me to shut up instead of sharing their info.
```

The exact authored lying variant is:

```text
Adjacent Liars tell me to shut up instead of sharing their info.
```

The flavor text is `"Started a sentence yesterday. Still going."`. The asset's
long serialized `notes` field describes an older picker/silence design and is
not implemented by `Rambler2`. The public asset also has `picking == false`.
The native implementation and current public description, not that stale notes
field, define the shipped rule.

`Rambler2` is TypeDefIndex `5868`. It declares four fields
(`currentTimesPicked`, `maxTimesPicked`, `savedQote`, and `quotes`) and exactly
14 methods. Its compiler-generated `<>c__DisplayClass9_0` is TypeDefIndex
`5867` and declares exactly two methods. All 16 identities are included in the
target.

## Audited boundary

| Group | Methods | Observable purpose |
| --- | ---: | --- |
| `Rambler2` | 14 | Complete role implementation, quote surface, and stale picker residue |
| `Rambler2.<>c__DisplayClass9_0` | 2 | Persistent hidden-neighbour callback |
| Setup and reveal | 8 | Per-card initialization, delayed internal reveal, AfterRoundStart, and user flip |
| Dispatch and truth | 5 | Real/bluff `Act` choice, pre-act hook, acted dispatch, actual and apparent lying |
| Interference/history | 5 | Immediate coroutine, pre-append mutation, and `ActedInfo` construction |
| Adjacency and setup orchestration | 2 | Exact circular pair and per-physical-card setup |

The target keeps the exact signatures and RVAs for all 36 entries. Two native
bodies are deliberately shared with other managed identities:
`Rambler2.<>c__DisplayClass9_0..ctor` uses the ABI-compatible canonical
`Dreamer..ctor` prototype at `0x357920`, and `Rambler2.IsCharacterLying` shares
its body at `0x3C1590` with the legacy managed `Rambler.IsCharacterLying`
identity. That body delegates to `CharacterHelper.CheckLyingAppearance` at
`0x397630`.

## Installation phase and source dispatch

Rambler installation is not tied to the user's card flip. During board setup,
each physical `Character` receives its role copy in `Character.Init` and then
runs a delayed internal `Character.Reveal`. That method calls
`Character.Act(AfterRoundStart)` (`ETriggerPhase == 7`) while the card's public
state is still Hidden. `Rambler2.Act(7)` installs interference for adjacent
appearance-truthful targets; `Rambler2.BluffAct(7)` installs it for adjacent
appearance-lying targets. This pre-flip installation explains a hidden or
later-killed Rambler already changing an earlier neighbour's result.

The selection of `Act` versus `BluffAct` is based on the source card's current
**actual** `CharacterHelper.CheckLying` result and `Character.Act`'s real/bluff
dispatch, not on the source's displayed role name:

| Physical source state | Dispatched Rambler surface | Neighbours selected |
| --- | --- | --- |
| Clean real public Rambler | real-role `Act` | appearance-truthful |
| Corrupted real Rambler | real-role `BluffAct` | appearance-lying |
| Ordinary lying Evil with Rambler bluff | bluff-role `BluffAct` | appearance-lying |
| HealthyBluff Puppet or clean HealthyBluff Doppelganger with Rambler bluff | bluff-role `Act` | appearance-truthful |
| Drunk or corrupted Doppelganger with Rambler bluff | bluff-role `BluffAct` | appearance-lying |
| Shaman-copied `Rambler2` | whichever real/bluff path the destination's preserved truth, alignment, status, and bluff state dispatch | corresponding truthful or lying set |

More generally, a truthful physical card dispatches `Act` for both its real
role and any bluff role. A lying card dispatches `BluffAct` for its bluff role;
its real role also receives `BluffAct`, except a runtime-Evil card with a
non-null bluff receives real `Act` plus bluff `BluffAct`. Therefore a physical
card whose real role and bluff role are both `Rambler2` can install twice. The
role contains no identity deduplication. That coexistence is not the ordinary
public real-Rambler setup, whose base bluff is null.

The source does not ask whether it *appears* to be Rambler. A Rambler surface
exists precisely when normal `Character.Act` dispatch reaches a `Rambler2`
role object, whether that object is the real role, an Evil bluff, a
HealthyBluff surface, or a role copy left by Shaman.

## Target predicate and circular adjacency

Each target is rechecked through `CharacterHelper.CheckLyingAppearance`, not
through its actual clue truth. In compact form, apparent lying is true for an
explicit AppearLying status, or, absent AppearTruthfull, for Corrupted and for
non-HealthyBluff cards whose runtime alignment is Evil or which currently have
a live bluff. AppearTruthfull suppresses those latter inferred causes;
AppearLying has first precedence.

`Characters.GetAdjacentCharacters` copies the current physical board list and
returns predecessor then successor. It has no Hidden, Revealed, Dead, killed,
alignment, or role filter. The small-board behavior is exact:

- zero cards, or a source absent from the current list: no targets;
- one card: the source appears twice (`[self, self]`);
- two cards: the sole other card appears twice;
- three or more cards: predecessor followed by successor.

Each list entry is processed independently. Duplicate entries therefore
install duplicate callbacks or emit duplicate immediate interference. Every
physical same-asset Rambler independently receives its AfterRoundStart action;
there is no one-asset collapse in this phase.

## Hidden target: persistent pre-append replacement

If an adjacent target is Hidden (`ECharacterState == 5`) when installation
runs, `InterefereCharacter` combines a new closure onto that target's
`Character.onAboutToAct` delegate. It does not alter acted history at install
time. On every later acted result, the closure:

1. re-evaluates the target's current `CheckLyingAppearance` value;
2. requires that value to equal the captured truthful/lying mode;
3. requires the target no longer to be Hidden; and
4. mutates the imminent `ActedInfo` in place.

The replacement description is exactly:

```text
#<Rambler source id>
shut up!
```

The callback replaces the imminent reference list with a fresh one-element
list containing the Rambler source. It does not append a second shut-up record:
`Character.ShowActedDelayed.MoveNext` invokes `onAboutToAct` before adding the
same `ActedInfo` to `actedInfos`, so the final history append contains the
mutated description and exactly `[source]`.

There is no unsubscription in `Rambler2`, no one-shot guard, and no source
liveness or source-role recheck. Source death, reveal, corruption changes, or
role reinitialization do not invalidate a captured callback. The audited
`Character.Init`/`InitWithNoReset` paths do not clear the target delegate, and
the established kill path changes state without destroying the physical
`Character`. A callback therefore persists for the lifetime of that physical
object and rechecks only the **target's** current appearance on each act.

Full `Character.Init` does clear the physical card's native `actedInfos` list,
but it does not clear `onAboutToAct` and does not write `savedAct`.
`InitWithNoReset` also clears acted history without removing the delegate.
Consequently a fresh village cannot inherit an old native ActedInfo entry, even
though stale display text and an old callback can survive object reuse. A live
reader must corroborate `savedAct` against the newest current-list ActedInfo;
an empty freshly initialized list rejects stale text.

Multicast callbacks run in delegate-combine order against the same object. If
several callbacks match, each overwrites the previous description and list, so
the last matching callback determines the single stored source reference.
Within one Rambler action, neighbours are visited predecessor then successor.
The relative resume/registration order of different cards' Unity delay
coroutines is not fixed by this native slice; any model that needs a unique
cross-card last writer must preserve that ordering uncertainty.

## Already non-Hidden target: immediate extra history

If the adjacent target is not Hidden at installation, Rambler checks its
appearance immediately. A match calls
`Character.InterfereActed(shutUpInfo, 0.02f, 3f, false, false)`. The coroutine
waits its short scheduling delay and emits an `Any` acted result with exactly
`[source]`. `clearMemory == false` retains all prior entries, and
`isDelay == false` exits without the timed restore branch. This path therefore
appends a new shut-up history entry; it does not replace or delete the target's
earlier result.

Dead is merely another non-Hidden state here. There is no alive filter, so a
dead matching neighbour can be the immediate target during a later setup or
reinstallation. Conversely, a hidden target may die before acting and retain
its callback; whether it later emits anything depends only on a future action,
not on Rambler cleanup.

The public `InterfereCharacterOnReveal` helper performs the same current
appearance check and in-place description/reference replacement. No current
`Rambler2` native path directly calls it; the shipped hidden branch uses the
captured closure instead.

## User reveal and the constraint-free Day quote

On an ordinary allowed user click, `Character.OnClick` first changes the
character from Hidden to Alive and invokes its state callback. The resulting
`RevealCard.Reveal` call dispatches `Character.Act(Day)`
(`ETriggerPhase == 30`) before `Character.OnReveal`. `Rambler2.Act(30)` and
`Rambler2.BluffAct(30)` are identical: both choose one of 34 fixed flavor
quotes uniformly, store it in the misspelled `savedQote` field, and pass it to
`Role.OnActed`.

`Role.OnActed` has an important narrow gate. When the trigger is not
`OnPicked` (`70`) and both `killedByDemon` and `killedHidden` are false, it
invokes the registered acted delegate regardless of character state, using a
previously saved `ActedInfo` when one exists and otherwise the current record.
Only the `OnPicked`-or-killed branch checks for Hidden and defers the current
record into `savedActInfo`. An ordinary Rambler reveal is already Alive, has no
prior saved role record, and has neither killed flag, so its newly chosen Day
quote is emitted before `Character.OnReveal` performs reveal accounting.

The quote's `ActedInfo` references are **not empty**. They are the exact current
`GetAdjacentCharacters(source)` result: predecessor then successor, including
the `[self, self]` and `[other, other]` duplicate behavior on one- and two-card
boards. The quote conveys no role, alignment, or truth constraint, and the
normal and bluff paths share the same quote pool.

The constructor initializes `currentTimesPicked == 0`,
`maxTimesPicked == 1`, and the 34-string quote list. None of the other 13 role
methods reads or writes either times-picked field, so they impose no current
native limit. Repeated Day actions may choose and store another quote.

## Stale picker surfaces and other role methods

`Rambler2.get_Description` returns the empty string; the serialized public text
is the meaningful description. `GetInfo` and `GetBluffInfo` are unreachable
picker-era residue for the public `picking == false` asset. Both build a
one-reference list containing `CharacterPicker.CurrentPicker`. `GetInfo`
succeeds only when that picker appears lying; `GetBluffInfo` succeeds only
when it appears truthful. In both successful cases the boolean passed to
`ConjourInfo` is true, so the result description is exactly `...` and the
achievement check runs. `ConjourInfo(false, picker)` would instead render
`#<id>\nis NOT Lying`, but no successful current caller in this class passes
false. `CheckAchievementsAndUnlockIfAble` uses achievement key
`Rambler_1_ACHIV_3167`.

`IsCharacterLying` is simply the current appearance predicate and ignores its
extra `charRef` parameter. The debug-only install string is
`NANI: Should Lie: {0}` and has no gameplay effect.

## Solver and live-observation consequences

- A shut-up record constrains the **speaker/target card's appearance truth at
  the time it acted**, paired with the Rambler source mode installed earlier.
  It is not evidence that the source was currently visible, alive, or still
  Rambler when the record appeared.
- Truthful-mode sources replace adjacent appearance-truthful results; bluff-mode
  sources replace adjacent appearance-lying results. Ordinary Evil fake
  Ramblers belong to the latter mode, while HealthyBluff fake Ramblers belong
  to the former.
- One visible event has exactly one Rambler reference even if several callbacks
  ran; the last matching callback wins. Immediate reinstallation can instead
  append another separate event.
- An append-only top-level `(speaker, target)` observation history is faithful
  to repeated acted records and persistent callbacks across solver state.
  Native `actedInfos` itself is chronological only within the current physical
  initialization and is cleared by `Character.Init`. A per-card latest scalar
  is only a compatibility view and must not be treated as the full observation
  history.
- Native Rambler Day emission chooses from the fixed internal 34-quote pool and
  carries exact circular-adjacency references. Safe live capture need not embed
  or enforce that copyrighted string corpus: corroborating the newest current
  `ActedInfo` description against `savedAct` together with those exact refs is
  sufficient to record `quote_observed`. The quote adds no deduction
  constraint.

## Bounded unknowns

The native code fixes delegate-combine behavior and per-source predecessor/
successor enumeration, but not the cross-card ordering in which Unity resumes
separately scheduled `DelayReveal` coroutines. That scheduler order can decide
the final visible source when multiple adjacent Rambler callbacks all match.
The role code also does not define when an entire scene destroys and recreates
physical `Character` objects; persistence is proven across death and
reinitialization of an existing object, not across object destruction.
