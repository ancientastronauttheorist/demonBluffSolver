# Gameplay role: Wretch (managed `Recluse`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all seven methods in the checked
boundary. The note records authored behavior summaries only; native bodies and
decompiler output remain outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_wretch.json`](../../targets/gameplay_role_wretch.json).
`Recluse` is the current internal managed name for the public Wretch role.
Its read-only baseline Ghidra export completed at 7/7 functions with no
failures.

## Target boundary

| Managed identity | RVA | Boundary role |
| --- | ---: | --- |
| `Recluse.get_Description` | `0x3E8960` | Display description |
| `Recluse.GetInfo` | `0x3E8640` | Real-role passive information |
| `Recluse.GetBluffInfo` | `0x3E85E0` | Bluff passive information |
| `Recluse.Act` | `0x33ED50` | Real-role action surface |
| `Recluse.BluffAct` | `0x33ED50` | Bluff action surface |
| `Recluse.GetRegisterAsRole` | `0x3E86A0` | Apparent-role selection |
| `Recluse.ctor` | `0x3E8840` | Legacy field initialization |

## Passive, action, and constructor surfaces

`GetInfo` and `GetBluffInfo` each return a fresh empty `ActedInfo`. `Act` and
`BluffAct` are both no-ops and intentionally share one trivial native body, so
Wretch has no action-side clue, status change, or picker lifecycle.

The current description says Wretch can register as a Demon, but the selection
method filters and returns **Minion** data. The constructor initializes a legacy
Minion identifier and a one-entry denial-chat list before base construction;
neither field participates in current register-as selection. These strings are
therefore retained data, not the gameplay contract.

## Minion register-as selection

`GetRegisterAsRole` copies the current combined script-character list and
filters the copy to exact `Minion` type. A nonempty filtered list is the primary
candidate pool. When it is empty, the method replaces it with the current
temporary ascension's starting Minion records. It then selects one entry with a
uniform integer index over the candidate count and returns that `CharacterData`.

The supplied `charRef` is unused. The method does not cache its result, add or
remove script entries, mutate the Wretch card, or consult the constructor's
legacy fields. Missing gameplay, character, project-context, game-data, or
temporary-ascension dependencies follow native failure paths. An empty fallback
pool also fails at indexed lookup; there is no null/no-disguise outcome.

## Runtime and registered identity

The selection method only returns data. Internal `Character.Reveal` is the
caller that stores the returned record in the card's `registerAs` field. This
does not replace Wretch's real role or runtime Good alignment.

Other systems can nevertheless observe the registered record. In particular,
`Character.GetRegisterAlignment` prefers `registerAs.startingAlignment`, and
the shipped Minion records are Evil-aligned. Slayer's normal target callback
uses that query, so a Good Wretch takes Slayer's Evil-target kill branch. The
paired [Slayer audit](gameplay_role_slayer.md#joined-wretch-interaction) joins
the exact call surfaces and links the live regression case.

## Metadata cautions

`Recluse.Act` and `Recluse.BluffAct` are distinct managed methods with the same
native RVA. The Wretch target preserves both identities and signatures while
using one canonical applied prototype for that body. Coverage must not collapse
either method into the body's primary symbol label.
