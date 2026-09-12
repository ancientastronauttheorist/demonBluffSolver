# Saved-roster replacement and same-hand setup

Pinned build `f530404b0f3f_807de4a83df4`. The native audit executes
`Gameplay.ResetSavedCharacters` and `Gameplay.SameHandOut` in 80 fixtures.
Both exact baseline exports completed. `RestartGame`, exported in the same
private batch, belongs to the separate startup audit.

## Saved lists and receiver identity

ResetSavedCharacters first passes the captured ProjectContext instance to
Unity object equality against null. A true result returns without replacing
any list. The fixture also exercises noncanonical high return-register bits:
only AL controls this Boolean branch.

On a false equality result it rereads the static ProjectContext instance,
checks that reference, and captures its GameData field. It then requests
starting characters from that same GameData receiver in order 10, 20, 30, 100.
Each result feeds a new List<CharacterData> copy constructor; only after that
constructor succeeds is the new list stored, followed by its GC barrier.

The destinations are the private **saved** lists at +0x48, +0x50, +0x58 and
+0x60. The public current rosters at +0x28 through +0x40 remain untouched.
Every other receiver byte also stays unchanged in the native caller.

Controlled gateway fixtures distinguish the two identity rules: replacing
ProjectContext with null during the equality service causes the subsequent
reread to fail. Replacing it after the first typed-pool call does not affect
the three remaining requests, which use the already captured GameData.
These mutations are authored tests, not a claim about live callback behavior.

Failure at any typed provider, allocation or copy constructor preserves only
the earlier completed list replacements. Barrier failure includes the current
replacement because the store already happened. Null provider results reach
the explicit copy-constructor collection failure before replacing that field.
Typed selection and actual list-copy internals remain service boundaries.

## Same-hand launch

SameHandOut allocates the exact generated SetupDelay iterator type. Its native
caller executes the shared no-op constructor body, captures the Gameplay
receiver at +0x20, writes iterator state zero at +0x10, and performs a barrier.
It then tail-calls the managed MonoBehaviour.StartCoroutine entry with that
receiver and iterator. The caller does not reset score, rosters or gameplay
fields before forwarding.

Allocation failure creates no iterator; a later barrier or StartCoroutine
failure retains the constructed iterator in this diagnostic fixture. A null
receiver is forwarded to the StartCoroutine gateway too. The audit does not
claim that the real managed wrapper accepts null or that successful gateway
return proves engine scheduling or ownership.

## Validation scope

The script pins GameAssembly and Dumper inputs, verifies exact class fields,
five generic/type metadata bindings and thirteen native instruction
relationships. Native execution includes complete final instructions and the
shared `ret 0` constructor stub. All successful returns verify stack and eight
nonvolatile integer registers; receiver byte comparisons constrain every write.

Metadata/class initialization, Unity equality, typed-pool selection,
allocation/copy, GC barriers and StartCoroutine remain explicit gateways.
No live state or save file is accessed. Run with private Unicorn 2.1.4:

```powershell
python reverse_engineering/scripts/audit_gameplay_roster_reset.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_gameplay_roster_reset.json
```
