# Jester current-build native contract

This note records the clean-room native checkpoint for the shipped public
**Jester**, managed as `Juggler` in build
`f530404b0f3f_807de4a83df4`. It closes all 16 declarations owned by
`Juggler` and `Juggler.<>c`, plus the 16 shared functions needed to explain
the exact-three picker, registered-alignment count, false-count draw, result
history, and lifecycle. It is not a claim that the whole game is decompiled.

Evidence status is **native-static** for executable behavior, **metadata** for
the serialized asset binding, and **behavioral** for the compatibility corpus.
Baseline and typed Ghidra exports remain private. This note, the target
manifest, aggregate typed-quality report, and coverage rows contain no
decompiled bodies or private filesystem paths.

## Public asset binding and managed identity

`Demon Bluff_Data/sharedassets0.assets` contains the shipped public Jester
`CharacterData` at path ID `21622`, absolute file offset `23677616`, size
`4488`, and object SHA-256
`C1050A0977D1908C38FAE20471CC36ACA79EC9AE864E3DEADAE83DAD90F7FC09`.
The containing asset has SHA-256
`E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967`.
Its serialized identity is:

- public name `Jester`;
- character ID `Jester_41367606`;
- managed SerializeReference type `Juggler` (`TypeDefIndex 5902`), with
  compiler-generated helper `Juggler.<>c` (`TypeDefIndex 5901`);
- Good Villager (`type == 10`, `startingAlignment == 10`);
- `abilityUsage == 10`, the `ResetAfterNight` category;
- bluffable, not usually disguised, and picking;
- roguelike values 10 points, multiplier 1.0, and zero income;
- no bundled characters, role achievements, configured statuses, tags, or
  `canAppearIf` records;
- one optional skin reference, path ID `21651`, and no current skin override;
  and
- empty hint and `ifLies` fields.

The exact authored public description is:

```text
<b>Pick 3 characters:</b>
Learn how many of them are Evil
```

The managed `Juggler.get_Description` declaration independently retains a
stale four-target sentence at string-literal RVA `0x2700F50`:

```text
Pick 4 players. Learn how many of them are Evil
```

The executable bodies and public asset agree on exactly three selections, so
the shipped asset and callable path supersede that stale managed sentence.

The normal serialized candidate pool in `Demon Bluff_Data/level0`, path ID
`139347`, contains exactly one file-ID-`2` reference to Jester path ID `21622`,
at pool-object-local offset `564`. The pool object is at file offset
`17578592`, size `1080`, and has SHA-256
`FB9D821AE0A7E3655BEF4A3DD3E544E85B3109258A48DCF68FF0969ACED8D948`.
The containing `level0` file has SHA-256
`B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1`.
Jester is absent from the 15-entry ordered-Start object at path ID `137026`,
file offset `17310672`, size `332`, and object SHA-256
`544328634CD77D551B5864CDC1B643029F3B30BFFC5BB4350DFCF83C66226BB0`.
It is a direct normal-pool active Day role, not an ordered-Start participant.

## Exact callable boundary and shared bodies

`Juggler.<>c` declares five static fields and six methods. `Juggler` declares
no instance fields and ten methods:

| Managed identity | RVA | Role-local purpose |
| --- | ---: | --- |
| `Juggler.<>c..cctor` | `0x3F09B0` | Compiler-helper singleton initialization |
| `Juggler.<>c..ctor` | `0x357920` | Fieldless compiler-helper construction |
| `Juggler.<>c.<CharacterPicked>b__6_0` | `0x3F07B0` | Truth primary sort key: display ID |
| `Juggler.<>c.<CharacterPicked>b__6_1` | `0x392D50` | Truth secondary sort key: `Random.value` |
| `Juggler.<>c.<CharacterPickedDrunk>b__8_0` | `0x3F07B0` | Bluff primary sort key: display ID |
| `Juggler.<>c.<CharacterPickedDrunk>b__8_1` | `0x392D50` | Bluff secondary sort key: `Random.value` |
| `Juggler.get_Description` | `0x3E14F0` | Stale four-player managed description |
| `Juggler.GetInfo` | `0x3E1180` | Passive empty/null result |
| `Juggler.GetBluffInfo` | `0x3E1120` | Passive empty/null result |
| `Juggler.Act` | `0x3DFA30` | Truth Day picker registration |
| `Juggler.StopPick` | `0x3E11E0` | Completion/cancel handler cleanup |
| `Juggler.CharacterPicked` | `0x3E0620` | Truth count, ordering, and result delivery |
| `Juggler.BluffAct` | `0x3DFCA0` | Bluff Day picker registration |
| `Juggler.CharacterPickedDrunk` | `0x3DFF10` | False count, ordering, and result delivery |
| `Juggler.ConjourInfo` | `0x3E0D20` | Singular/plural exact formatter |
| `Juggler..ctor` | `0x3CFFF0` | Fieldless role construction |

The helper constructor, the two integer-identity selectors, the two random
float selectors, and the fieldless role constructor use shared native bodies.
The target assigns canonical ABI-compatible prototypes rather than treating
their unrelated symbol aliases as gameplay identity.

The 32-function target also retains these 16 semantic functions:

| Shared function | RVA | Reason retained |
| --- | ---: | --- |
| `Character.Act` | `0x3645C0` | Truth/bluff and real/raw dispatch order |
| `Character.OnClick` | `0x366270` | Picker-mode click precedence and eligibility |
| `Character.RoleAct` | `0x368790` | Role callback installation |
| `Character.RefreshCharacter` | `0x367970` | ResetAfterNight use restoration |
| `Character.GetRegisterAlignment` | `0x365030` | Register-as-first truth projection |
| `Character.<>c__DisplayClass125_0..ctor` | `0x357920` | Role callback closure allocation |
| `Character.<>c__DisplayClass125_0.<RoleAct>b__0` | `0x377120` | Delayed result scheduling |
| `Character.ShowActedDelayed` | `0x368A50` | Delayed result state construction |
| `Character.<ShowActedDelayed>d__133.MoveNext` | `0x375FE0` | Interference, history, and use consumption |
| `CharacterPicker.CancelPick` | `0x378CC0` | Cancel ordering and callback cleanup |
| `CharacterPicker.StartPickCharacters` | `0x379260` | Exact count, actor, and picker state |
| `CharacterPicker.ClickedCharacter` | `0x378DB0` | Toggle/completion and OnPicked ordering |
| `ActedInfo..ctor` | `0x35D5D0` | Exact description/reference record |
| `Calculator.RemoveNumberAndGetRandomNumberFromList` | `0x396490` | False-count domain and removal |
| `UnityEngine.Random.Range(int,int)` | `0x1C86600` | Uniform false-count index |
| `UnityEngine.Random.value` | `0x1C86710` | Three secondary sort keys |

Framework collection operations, delegate combination/removal, allocator
helpers, and Unity's global PRNG internals remain outside the role target.

## Day picker and target eligibility

Truth and bluff bodies ignore every trigger except Day (`30`). On Day each
performs the same setup in this exact order:

1. `StartPickCharacters(3, actor)`;
2. combine its truth or bluff completion handler onto the global picker; and
3. combine `StopPick` onto the global cancel/finish handler.

The role supplies no target predicate. While the global gameplay state is the
picker state, `Character.OnClick` routes the clicked physical object to
`CharacterPicker.ClickedCharacter` before ordinary reveal, active-ability,
death, or execution routing. Native role eligibility consequently admits:

- the physical actor itself;
- dead or killed cards;
- Hidden and otherwise unrevealed cards that receive a click;
- cards with their own unused active ability; and
- two different physical `Character` objects carrying the same display ID.

Uniqueness is exact `Character` object membership. Clicking the same object a
second time removes it; clicking it again re-adds it at the end. Therefore
three distinct physical objects are required, click chronology can change by
toggling, and a board with fewer than three physical objects cannot complete
normally. The completion test is list-count against requested count; ordinary
single-click growth reaches it at exactly three.

At completion the picker first iterates the selected objects in click order
and dispatches `OnPicked` (`70`) to each object other than the physical actor.
Only after every such target action returns does it invoke Jester's completion
handler. Jester therefore samples display IDs and registered alignments after
those synchronous OnPicked calls. The result handler removes its own
completion and StopPick delegates before scanning. After all completion
delegates return, the picker clears the shared selection list, invokes any
remaining finish handlers, hides the picker, and clears the picker actor.

Cancel instead returns gameplay to the normal state, clears selected objects,
and invokes `StopPick`. `StopPick` removes both possible Jester completion
delegates and itself. A cancelled picker emits no `ActedInfo`, consumes no
Jester completion RNG, does not decrement a use, and leaves prior history
untouched.

## Registered-Evil truth calculation

Truth iterates the three selected objects once. For every object it:

1. evaluates `Character.GetRegisterAlignment()`;
2. increments the count only for exact Evil (`20`); and
3. appends the object's integer display ID to a fresh ID list.

`GetRegisterAlignment` has exact precedence:

```text
live registerAs != null ? registerAs.startingAlignment : Character.alignment
```

It does not inspect displayed `bluff`, `bluffRole`, current `dataRef`, raw role
type, character state, death, visibility, corruption, or runtime data. Those
surfaces can influence upstream truth routing or writers of `registerAs` and
runtime alignment, but the count itself reads only this projection.

Important current-role consequences are:

- natural Wretch has an Evil Minion `registerAs` and counts Evil despite its
  underlying Good Outcast body;
- stable/current Spy can retain a Good Villager `registerAs` and does not
  count Evil despite runtime Evil alignment;
- a Puppeteer-created Puppet has null `registerAs` and runtime Evil alignment,
  so it counts Evil;
- any ordinary runtime-Evil Minion or Demon with null register-as counts Evil,
  regardless of a Jester or other Good disguise;
- ordinary Drunk and Doppelganger retain null register-as and runtime Good
  alignment, so their copied/apparent Jester identity does not make them Evil;
  and
- corruption changes whether a Good Jester actor reaches the truth or bluff
  body, but it does not itself change a selected target's counted alignment.

Jester reads this live value at completion, not at picker activation or target
click. Baker conversion is therefore timing-sensitive in native isolation:
`InitWithNoReset` synchronously changes current data to Baker but preserves an
old Spy register-as until the delayed internal Reveal refreshes it. Ordinary
post-reveal active play is settled after that delay, but Juggler itself has no
wait, reset check, or Baker-specific guard. An artificially immediate picker
completion may still observe the pending register-as surface.

## ID ordering, reference provenance, and exact text

Both completion bodies order the three integer display IDs as:

```text
OrderBy(id).ThenBy(_ => Random.value)
```

Visible IDs are therefore ascending. The secondary selector is evaluated for
all three elements, so each normal completion consumes exactly three
`Random.value` calls. It changes only the relative ordering of equal numeric
IDs, which is not distinguishable in the formatted text.

The `ActedInfo.characters` list is independent of that ID sort. It is a fresh
copy of the global picked-Character list while that list still exists, so its
references remain exact physical objects in click order. Sorting text IDs
does not sort references. Duplicate numeric IDs can consequently map to
different object references, and toggling changes the reference chronology.

`ConjourInfo(id1,id2,id3,count)` selects only between these two literals.
For count `1` it emits:

```text
Among:
#{0}, #{1}, #{2}:
There is {3} Evil
```

For counts `0`, `2`, and `3` it emits:

```text
Among:
#{0}, #{1}, #{2}:
There are {3} Evils
```

There is a colon after `Among`, a colon after the three-ID line, no terminal
punctuation, and exactly two newline boundaries. The singular literal still
formats the supplied integer rather than hardcoding its text, although the
reachable singular branch supplies one. The plural and singular literals are
at RVAs `0x271F1A0` and `0x271F228` respectively.

`GetInfo` and `GetBluffInfo` are passive shells, not the active clue route.
Each returns a fresh `ActedInfo` with the exact empty-string literal and a
final null Character-list pointer.

## Bluff domain, exclusion, probability, and RNG chronology

Bluff performs the same registered-Evil scan and ID collection as truth, so
let `A` be the actual selected count in `{0,1,2,3}`. It then calls
`Calculator.RemoveNumberAndGetRandomNumberFromList(A, 0, 4)`. The helper:

1. constructs the ordered integer list `[0,1,2,3]`;
2. removes the single occurrence equal to `A`;
3. draws one `Random.Range(0, 3)` index; and
4. returns that retained list entry.

The emitted false count is thus uniform over the other three values:

| Actual `A` | Bluff support | Probability per value |
| ---: | --- | ---: |
| 0 | `{1,2,3}` | `1/3` |
| 1 | `{0,2,3}` | `1/3` |
| 2 | `{0,1,3}` | `1/3` |
| 3 | `{0,1,2}` | `1/3` |

There is no role pool, board-count clamp, total-Evil-budget restriction,
collision retry, occurrence weighting, or special zero/three rule. The actual
value is always excluded. The exact completed-path RNG chronology is:

| Path | Float calls | Integer calls |
| --- | ---: | ---: |
| truth | three `Random.value` sort keys | none |
| bluff | three `Random.value` sort keys | one `Range(0,3)` |
| cancel/incomplete picker | none from completion | none |

This records requested call support and order, not Unity's hidden global PRNG
state or independence between calls.

## Result history, Rambler replacement, and lifecycle

The completion handler synchronously invokes the `onActed` callback installed
by `Character.RoleAct`. That callback starts
`Character.ShowActedDelayed(0, result, Day)`. After its zero-duration Unity
yield, a successful nonempty result requires the actor's `act` flag and then:

1. invokes the actor's `onAboutToAct` multicast delegate;
2. appends that same possibly mutated `ActedInfo` to chronological
   `actedInfos`;
3. decrements `pickableUses` once because the trigger is Day;
4. invokes the global acted event and schedules visible text; and
5. disables the active marker when the remaining use count reaches zero.

Current Rambler interference runs at step one. When a matching persistent
Rambler callback fires, it mutates the imminent Jester record in place to the
exact description:

```text
#<Rambler source id>
shut up!
```

and replaces the reference list with a fresh one-element list containing only
that Rambler source. The history append is therefore one interrupted record,
not an original Jester record plus another shut-up record. Native history does
not retain the three Jester picks or count in that `ActedInfo`; a bridge must
not parse its sole reference as a Jester target. Multiple matching callbacks
mutate the same object in combine order, so the last match supplies the final
single source. The use is still consumed because the mutated description is
nonempty. This replacement is the shared `Character`/Rambler callback, not a
Juggler-local formatter: every interrupted speaker receives the same embedded
newline, and native `savedAct` preserves it without whitespace normalization.

Juggler has no local Init, Start, AfterRoundStart, Night, death, execution,
achievement, status, runtime-data, or cleanup branch. The serialized
ResetAfterNight category makes `Character.RefreshCharacter` restore one use
when the projected displayed record has `abilityUsage == 10`. That framework
projection chooses real `dataRef` for Dead/Revealed state or an already
revealed flag, and otherwise a non-null raw bluff; it does not consult
`registerAs`. Reset does not clear `actedInfos`, so repeated nights append
additional chronological Jester results.

`Character.Act` invokes the real role before the raw/bluff role. A truthful
body sends `Act` to both surfaces; a lying non-Evil body sends `BluffAct` to
both; and a lying runtime-Evil body with a bluff sends real `Act` first and
raw/bluff `BluffAct` second. If two role surfaces emit, their callbacks are
queued and appended in that order. The later raw/bluff event remains the
newest history entry and later visible text, while the real event is retained;
Rambler can independently replace either imminent record before its append.

## Poet absence and cross-role identity

Public Poet is managed `Gossip`. Its exact constructor builds twelve provider
roles in this order:

```text
Lover, Scout, Oracle, Bounty Hunter, Medium, Knitter,
Hunter, Enlightened, Empress, Bishop, Gemcrafter, Bard
```

Their managed TypeDefIndices are `5863, 5854, 5893, 5871, 5881, 5855, 5891,
5890, 5883, 5888, 5884, 5896`. `Juggler` TypeDefIndex `5902` is absent. Poet
cannot construct or select Jester, so there is no provider slot, delegated
actor path, Poet-specific target reference order, or provider RNG chronology
for Jester in this build.

Identity and registration remain physical-card surfaces, not names in the
Jester clue. A Baker clue naming Jester reports Baker history and does not run
Juggler. A Shaman destination with a copied Juggler role does run the active
surface but preserves its destination status, alignment, and runtime data. A
Drunk, Doppelganger, Puppet, ordinary Evil disguise, or Spy can display or
hold Jester while its counted target alignment still follows the exact
register-as-first rule above.

## Corpus and compatibility implications

The active `tests/cases_v2` corpus contains 426 fixtures. Jester occurs in 113
deck pools and has 121 apparent records across 100 fixtures: 77 complete
`targets`/`evil_count` payloads, one explicit interrupted payload, and 43
empty payloads. Complete counts are zero 22 times, one 24 times, two 19 times,
and three 12 times. They span board sizes six through ten. Every complete
target list has exactly three unique, ascending, in-board IDs; one active-v2
record self-targets (`asc64_v6`, Jester `#8`, targets `#1,#5,#8`). The explicit
`asc78_v6` interrupted record preserves reconstructed targets but correctly
has no `evil_count`.

The 137 legacy fixtures add 39 Jester deck pools and 38 apparent records
across 32 fixtures: 31 complete and seven empty. Their complete counts are zero
eight times, one eight, two nine, and three six; one legacy record self-targets
(`asc11_g2_live`, Jester `#4`, targets `#1,#4,#5`). Across both corpora all
159 Jester `info_text` fields are blank. Active v2 adds three
Baker-original-Jester records and four Medium clues naming Jester; legacy adds
two Medium clues. Neither corpus has a Poet/Jester record.

The archive protects the historical three-target/count schema, ascending
text-ID convention, self-target legality, full count domain, and interrupted
no-count shape. It does not independently prove current exact text,
click-order references, registered-alignment projection, truth/bluff
provenance, RNG chronology, reset behavior, or real/raw history ordering.
Native evidence closes those current-build points.

## Reproduction and quality gates

The clean-room checkpoint ran these serialized stages against the saved build
projects:

```powershell
python reverse_engineering/scripts/validate_ghidra_targets.py `
  --targets reverse_engineering/targets/gameplay_role_jester.json `
  --script-json <private-current-build-script.json>

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage export-target `
  -TargetSet gameplay_role_jester

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage build-types

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-refresh

powershell -ExecutionPolicy Bypass -File `
  reverse_engineering/scripts/invoke_ghidra.ps1 `
  -GameRoot <private-game-root> -Stage typed-export `
  -TargetSet gameplay_role_jester

python reverse_engineering/scripts/audit_ghidra_type_quality.py `
  --baseline <private-baseline-export> `
  --typed <private-typed-export> `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_typed_quality_gameplay_role_jester.json `
  --check

python reverse_engineering/scripts/build_method_coverage.py `
  --build-manifest reverse_engineering/manifests/builds/f530404b0f3f_807de4a83df4.json `
  --type-index reverse_engineering/symbols/f530404b0f3f_807de4a83df4/assembly_csharp_types.json `
  --dump-cs <private-current-build-dump.cs> `
  --script-json <private-current-build-script.json> `
  --game-assembly <private-current-build-GameAssembly.dll> `
  --check
```

Results are 32/32 baseline, 32/32 typed, 32/32 signature application, and
32/32 read-only signature validation. Typed application imports six
additional reachable datatypes, canonicalizes seven target-level shared
bodies, and the read-only Jester pass validates 91 ABI parameter storages with
zero program mutations.

The typed-quality check passes. Placeholder-parameter tokens fall from 231 to
zero, raw-field-offset accesses from 366 to 222, raw-integer-type tokens from
318 to 223, unresolved-type tokens from 222 to 94, indirect-call patterns from
17 to six, and raw-pointer casts from 516 to 488. Signature-parameter-name
tokens rise from 59 to 240 and typed IL2CPP-type tokens from 43 to 236.
Decompiler errors and warnings remain unchanged at two and 64; the report
records no policy regression.

Adding Jester produces a 40-target-set union with 832 memberships, 528
distinct selected FunctionDefinitions, 304 exact-definition overlap
memberships, and 434 unique native RVAs. Jester contributes 32 memberships:
16 are exact-definition overlaps and 16 are newly selected definitions. Those
16 definitions add ten new native RVAs because six use native bodies already
selected by the union. The union validates 2,433 membership-level ABI
parameter storages.

The rebuilt Assembly-CSharp ledger retains 4,207 method definitions, 3,066
unique native bodies, and 107 shared-body groups. It now contains 521
classification records and 267 evidence records; a second `--check` rebuild
is byte-for-byte clean.

## Required implementation regressions

A solver/reader bridge that claims current Jester support should pin at least:

- exact-three physical-object cardinality, same-object toggling, self, dead,
  hidden, and unused-active target legality;
- OnPicked actions before the live count, sorted clue IDs versus click-order
  `ActedInfo` references, and three float draws including equal display IDs;
- register-as-first truth for natural Wretch, stable Spy, null-registerAs
  Puppet/runtime Evil, and ordinary Good Drunk/Doppelganger;
- exact uniform false support `{0,1,2,3} - {actual}`, one `Range(0,3)`, and no
  total-Evil-budget clamp;
- both exact literals, singular only at one, punctuation/newlines, and all four
  reachable count values;
- Rambler replacement as one shut-up record with exactly the Rambler source,
  no recoverable Jester targets/count, and normal use consumption;
- cancellation, Day-only dispatch, one-use consumption, Night restoration,
  and retained multi-night history;
- real-then-raw two-event ordering and later visible overwrite;
- settled and pending Baker/Spy register-as timing as distinct native states;
- no Jester runtime data and no Poet provider route; and
- the 108 complete archive records plus explicit interrupted fixture as
  compatibility evidence rather than current exact provenance.

## Remaining uncertainty and solver boundaries

- This target proves requested RNG calls and support, not the hidden state or
  independence of Unity's global PRNG.
- Native click routing proves eligibility once a `Character` object receives a
  picker click. UI layout or animation may make a particular hidden object
  practically unclickable in a scene; that is outside role semantics.
- The normal live workflow activates Jester after reveal processing has
  settled Baker's delayed internal Reveal. The executable Juggler boundary
  itself does not enforce that delay, so adversarial immediate activation must
  retain the pending register-as possibility.
- Null global singletons, null picked entries, corrupted picker lists, and
  malformed role callbacks follow native failure paths and were not forced in
  live play.
- Upstream code owns `dataRef`, `registerAs`, physical alignment, corruption,
  truth routing, role replacement, and board construction. This checkpoint
  audits Jester's exact reads of those surfaces, not every writer.
- The mostly textless archive cannot label historical complete results as
  truth or bluff or recover native click-order references.
- This is a build-specific Jester checkpoint, not evidence that every role or
  the whole game has been fully decompiled.
