# Native RoguelikeStandard stage-to-village bridge

This audit replaces the earlier `GameData.IncreaseVillage` gateway with its
actual native body, including its concrete `RoguelikeStandard.MaxLevel` call.
It does not modify the Rust progression contract: that existing replay still
requires its external services to preserve global village state.

## Receiver identity and ordering

`RoguelikeStandard.OnStageCompleted` first applies all its caller-instance
writes: current village increments, stage bonus accrual, round-score reset and
eligible best-village/score updates. Only afterward does it call
`GameData.IncreaseVillage`. Its helper-entry snapshot therefore sees the updated
caller object and the previous static `GameData.CurrentVillage`.

IncreaseVillage chooses behavior from **GameData's static GameMode reference**,
not from the stage caller. The fixture supplies these references separately and
also tests the valid same-instance case. A caller can increment its own
`currentVillage` while the global village remains capped, stays unchanged for
an unrelated/null mode, or increments under another mode's configuration.
The helper does not copy caller `currentVillage` into the static field.

- For a StandardMode-assignable global mode, it reads
  `ProjectContext -> GameData.standardAscensions.Length - 1` and increments
  static CurrentVillage only when its signed current value is below that bound.
- For a RoguelikeStandard-assignable global mode, it calls the concrete
  `RoguelikeStandard.MaxLevel` at `0x3ea1c0`, using the **global mode object** as
  receiver. This is not virtual `GetMaxLevel`, whose folded getter returns zero.
- Other and null global modes do not increment or inspect profile configuration.

Both assignability checks execute native class-depth and type-hierarchy reads.
Fixtures include exact types, assignable synthetic subclasses, unrelated types
at adequate hierarchy depth, and a shorter unrelated hierarchy. The harness
never infers kind from `GetGameMode`'s enum or from the stage caller's identity.

## Rogue profile lookup and failures

The concrete MaxLevel reads the global mode's `currentAscension`, selects that
outer `roguelikeStandardAscensions` entry, and returns its inner
`AscensionsList.ascensions.Length - 1`. An index at or above the outer length
selects the final outer entry. A negative index or empty outer array reaches
the native bounds-failure gateway; null project, GameData, outer array, selected
entry or inner array reaches the native null gateway. Distinct outer entries
have different lengths in fixtures, so substituting the caller's ascension or
using the wrong profile cannot silently pass.

All these helper failures occur after caller score/village writes and before
saving. The helper's failed configuration path preserves the old static village.
A later JSON or SetString failure instead preserves a successful native static
increment as well as caller writes. Missing configuration is irrelevant for
other/null mode branches that never read it.

Caller int32 arithmetic remains independent of helper bounds. For example,
caller `currentVillage` can wrap from INT_MAX to INT_MIN while the static village
increments normally according to its own value. Negative static values are not
clamped. Raw inner/standard array-length boundary fixtures also verify signed
subtraction and comparison; synthetic negative length headers are arithmetic
stress cases, not a claim that the runtime allocator creates such arrays.

## Native evidence and verification

`reverse_engineering/scripts/audit_roguelike_village_bridge.py` pins GameAssembly,
Dumper script and declarations, verifies the four exact native signatures,
checks field identities and twenty instruction relationships, and executes the
stage caller, IncreaseVillage, MaxLevel and immediate save setter in Unicorn
2.1.4. Native code ranges include separately verified null/bounds/cast branch
targets through their complete final call; following trap padding is excluded.

The authored report,
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_village_bridge.json`,
contains 660 cases. Each compares all 128 caller bytes and all 128 distinct
current-mode bytes, the global mode reference, final static village, native
helper-entry state, save state, event ordering and error. Successful returns
check nonvolatile integer registers and the stack. The report records 169
executed instructions and explicitly lists all unvisited sites.

The remaining sites are warmed metadata initialization, redundant class-init
paths bypassed after the caller initializes GameData, and the repeated
assignability cast-failure path that cannot arise with the stable supplied
class hierarchy. No complete instruction-coverage claim is made. Class
initialization remains a gateway that initializes its flag and preserves the
supplied global reference/state; the native GameData static constructor is not
executed. JSON and PlayerPrefs remain gateways. Concurrent mode replacement,
class-hierarchy mutation and arbitrary subclasses with different behavior are
outside this bounded bridge.

With the pinned local reverse-engineering dependencies on `PYTHONPATH`:

```powershell
python reverse_engineering/scripts/audit_roguelike_village_bridge.py 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_roguelike_village_bridge.json
python -m py_compile reverse_engineering/scripts/audit_roguelike_village_bridge.py
```

No Rust changes, live state/save writes, new typed targets or Ghidra mutations
are part of this audit.

## Opt-in Rust bridge

`crates/solver-core/src/roguelike_village_bridge.rs` implements a separate
versioned replay, preserving the original progression API. Its six tests pass,
including all 660 native fixtures, distinct global receiver selection, null and
bounds partial writes, save failures and atomic rejection of unsupported
provenance. Native IncreaseVillage may change the global counter here; only
external class-initialization, JSON and preferences services must preserve it.
