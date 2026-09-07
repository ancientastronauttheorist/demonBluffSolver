# Public Mutant: managed Skinwalker

Build `f530404b0f3f_807de4a83df4`. This public asset is distinct from the
[managed Mutant selector](gameplay_role_managed_mutant.md).

## Asset binding

The aligned [character-asset audit](../systems/character_asset_flags.md) binds
`sharedassets0.assets` path `21592`, object size 680, SHA-256
`A3A703FA630D1A3288E4C031C2EDB4C91C88924F59FC7B957960AAE575470653`,
to managed Skinwalker (TypeDefIndex 5921), derived from Demon (5919).
The public name is Mutant, ID `Mutant_84675843`; serialized type is Demon
(100), starting alignment Good (10), usage Once (0), bluffable true,
usuallyDisguised true, and picking false. There are no configured statuses,
tags, bundled characters, or appearance conditions. Serialized alignment is
an initial value, not a claim that no runtime writer can change it.

Its managed-reference ID `8893039266143666177` connects the role field at
object offset `0x220` to the Skinwalker registry entry at `0x278`. The
class-name string starts at `0x284`. This distinguishes it from the unrelated
managed Mutant class, which applies Mad while acquiring a Good bluff.

## Complete declared boundary and inherited behavior

| Method | RVA | Recovered behavior |
| --- | --- | --- |
| Skinwalker.GetRules | `0x3EBEF0` | Allocate and default-construct a fresh List&lt;SpecialRule&gt;; add no rules |
| Skinwalker.Act | `0x33ED50` | Shared `ret 0`, no trigger-specific action |
| Skinwalker constructor | `0x3CFFF0` | Fieldless shared base-constructor chain |
| Demon description | `0x3D7060` | Empty string |
| Demon.GetInfo | `0x3D6BD0` | Fresh ActedInfo, empty string, null character list |
| Demon.GetBluffInfo | `0x3D6B70` | Same observable empty clue payload |
| Role.GetRegisterAsRole | `0x3712B0` | Null registration override |
| Role.BluffAct | `0x3C4CA0` | Forward trigger/actor through actual Act virtual slot |

The [native auditor](../../scripts/audit_public_mutant.py) verifies both class
boundaries and 30 instruction/metadata relationships. Exact generic metadata
identifies List&lt;SpecialRule&gt; despite the shared constructor's decompiler alias.
Allocation and default-constructor semantics remain runtime contracts; this
audit does not claim native execution of the allocator or generic constructor.

Demon's inherited GetBluffIfAble and kill helpers retain their previously
audited meanings in the acquisition and Lilis/Knight notes. An inert Act does
not by itself establish an entire Demon lifecycle or all callers of those
helpers. No live solver role is introduced from this asset alone.

## Configuration and validation

The [complete asset graph](../systems/ascension_asset_graph.md) finds no direct
reference to this asset in GameData, any of 46 ascension profiles, or any of 12
custom scripts. Dynamic loading and runtime writers remain outside that claim.

The 14-target baseline and typed exports both completed. The rebuilt GDT has
151,699 datatypes and 561 exact FunctionDefinitions across 44 sets, 913 target
memberships, and 455 native RVAs. The read-only pass validated 2,680 parameter
storage locations with zero mutations. Quality checking removed all 28
placeholder parameter tokens and the one indirect-call pattern; one existing
decompiler error marker and seven warnings remain unchanged. Raw pointer
casts increased under explicit typing, so this is not a claim of universally
clean decompiler output. Native instruction checks support the conclusions.
