# Complete base GameMode declaration surface

The pinned base `GameMode` declares 21 methods: fifteen abstract virtual
members, four concrete no-op virtual members, `GetSummaryScores`, and its
constructor. The audit checks the complete pinned Dumper class declaration,
all slot identities, and the six concrete ScriptMethod signatures/RVAs.
No new typed target manifest is needed for this supplemental audit.

## Summary getter

`GameMode.GetSummaryScores`, virtual slot 23, has the complete native body
`0x3dd160..0x3dd18c`. It returns the pointer stored in string-literal cell
`0x26df1b8`; pinned ScriptString metadata identifies this literal as the empty
string. It does not call `GetScores`, query a derived mode, inspect any receiver
fields, format a summary, or aggregate scores.

When metadata flag `0x288c64a` is zero, it first calls initialization helper
`0x2b7b40` with the address of that literal cell, then sets the flag to one.
Any nonzero flag skips initialization. The native pointer load occurs after
the optional initialization. Thus successful initialization determines the
returned pointer on the cold path; the warmed path returns the existing cell
value unchanged. A synthetically null warmed cell returns null: identifying
an empty-string literal is not an independent proof of successful runtime
metadata initialization.

Cold failure fixtures stop at the explicit initialization gateway, before the
native flag write, preserving receiver bytes and the initial literal cell.
They do not emulate managed exception unwinding.

## No-op callbacks and constructor

| Base declaration | Virtual slot | Native entry |
| --- | --- | --- |
| OnLoadGame | 7 | `0x33ed50` |
| DeInit | 8 | `0x33ed50` |
| AscensionComplete | 14 | `0x33ed50` |
| AbandonRun | 15 | `0x33ed50` |

All four exact declarations alias the same three-byte `ret 0` instruction,
ending at exclusive `0x33ed53`. They perform no receiver, save or event writes.
The no-op body leaves the incoming return register unchanged; these methods
are void and have no return-value contract. Alias names attached to a shared
native body do not change the declaration's identity.

The base constructor at `0x357920..0x357927` clears its metadata argument
register and tailcalls that same no-op body. It does not initialize receiver
fields. This repeats the earlier constructor evidence in the complete base
surface harness. Derived constructors and overrides remain separate: a no-op
base `DeInit` or `AbandonRun` does not imply a no-op StandardMode override.

## Abstract declarations

The following slots are explicitly abstract, with RVA/offset `-1` in the
pinned Dumper declaration and no matching native `GameMode$$...` ScriptMethod
entry. There is no base native body to emulate:

| Slot | Declaration |
| --- | --- |
| 4 | GetGameMode |
| 5 | Init |
| 6 | LoadGame |
| 9 | GetStartingLevel |
| 10 | GetResetLevel |
| 11 | GetCurrentAscension |
| 12 | GetPreviousAscension |
| 13 | CanResetLevel |
| 16 | OnStageCompleted |
| 17 | UpdateScore |
| 18 | GetScore |
| 19 | GetMaxLevel |
| 20 | GetCurrentLevel |
| 21 | IsLocked |
| 22 | GetScores |

## Verification

`reverse_engineering/scripts/audit_game_mode_base_surface.py` pins GameAssembly,
Dumper `script.json` and `dump.cs` against existing build/extraction manifests.
It verifies exact instruction boundaries, twelve native relationships, all
21 declarations, slot identities and the empty-string literal metadata.
Unicorn 2.1.4 executes 76 cases covering six concrete declarations, null and
synthetic receivers, zero/nonzero metadata flags, null/non-null literal cells,
and cold initialization failure. All twelve decoded native instructions
execute. Every case preserves all 512 synthetic receiver bytes; successful
returns check nonvolatile registers and stack, and all cases check the
metadata flag and literal cell.

The authored report is
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_game_mode_base_surface.json`.
With the pinned local reverse-engineering dependencies on `PYTHONPATH`:

```powershell
python reverse_engineering/scripts/audit_game_mode_base_surface.py 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_game_mode_base_surface.json
python -m py_compile reverse_engineering/scripts/audit_game_mode_base_surface.py
```

This completes the classification of base declarations. It does not establish
all derived-mode overrides, whole-game mode selection or live save behavior.
Metadata initialization is an explicit gateway; native bodies remain private.
