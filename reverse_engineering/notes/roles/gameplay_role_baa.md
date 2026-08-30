# Gameplay role: Baa (managed `Imp`)

Build: `f530404b0f3f_807de4a83df4`

Evidence status: **native-static** for all three methods on the shipped role and
the two deck-view helpers that implement its visible effect. The note records
authored behavior summaries only; native bodies and decompiler output remain
outside the repository.

The checked target set is
[`reverse_engineering/targets/gameplay_role_baa.json`](../../targets/gameplay_role_baa.json).
Its read-only baseline and typed Ghidra exports each completed at 5/5 functions
with no failures.

## Public asset binding

The shipped `sharedassets0.assets` `CharacterData` at path ID `21590` is named
`Baa` and binds its SerializeReference role to managed `Imp`. Its object SHA-256
is `F29B3D02CBAF3F4592AEE4BFED202E06474AFF6BEE07C8BF1FCB99196F3B4F7A`.
The authored descriptions say both “Hide 1 Outcast in the deck” and “One fake
Outcast is added to the Deck view.” The apparent tension is resolved by the
native implementation: Baa selects an existing Outcast record and obscures its
identity only in the deck presentation. It does not add a gameplay role.

The `level0` object at path ID `137026` references Baa's path ID `21590` as the
eighth entry (zero-based index 7) in `startGameActOrder`, matching the role's
native `Start` trigger. This binding also corrects the old runtime-name fallback:
managed `Imp` is Baa. The separate public Mutant asset at path ID
`21592` binds managed `Skinwalker` and has object SHA-256
`A3A703FA630D1A3288E4C031C2EDB4C91C88924F59FC7B957960AAE575470653`.

## Target boundary

| Managed identity | RVA | Boundary role |
| --- | ---: | --- |
| `DeckView.AddToObscuredDeckView` | `0x39B0D0` | Store the selected hidden deck entry |
| `DeckView.RemoveObscuredDeckView` | `0x39C250` | Remove it and request a deck refresh |
| `Imp.GetRules` | `0x3DD7B0` | Special-rule surface |
| `Imp.Act` | `0x3DD410` | Start selection and death cleanup |
| `Imp.ctor` | `0x3CFFF0` | Base-only construction |

## Start selection

`Imp.Act` does setup work only for `ETriggerPhase.Start`. It copies the current
script-character pool and filters that copy to exact `Outcast` type. An empty
pool returns without selecting or obscuring anything.

For a nonempty pool, the method first samples one entry uniformly from every
Outcast. It then constructs a second list containing entries whose
`CharacterData.usuallyDisguised` flag is set. If that priority list is
nonempty, a uniform sample from it replaces the first choice; otherwise the
original all-Outcast choice remains. The selected object is stored in
`Imp.blockedOutcast` and passed to `DeckView.AddToObscuredDeckView`.

All 46 shipped core `CharacterData` assets currently have
`usuallyDisguised == false`. The priority branch is therefore latent in this
build, and the reachable selection is uniform across the authored Outcast
pool. The authored “Prioritize hiding Drunks and Doppelgangers” note is not
implemented by the current asset flags.

The method does not remove the selected object from the script pool, clone a
role, change a character type or alignment, alter HUD faction counts, or touch
any board-card state. `state.deck.outcasts` therefore remains the authoritative
full role pool for the solver.

## Deck-view storage and death cleanup

`DeckView.AddToObscuredDeckView` appends the exact selected `CharacterData`
object to the static `DeckView.ObscuredCharacters` list. It does not emit a UI
event itself.

For `ETriggerPhase.OnDied`, `Imp.Act` checks the stored object and passes it to
`DeckView.RemoveObscuredDeckView`. The helper removes that exact object from
the static list and invokes the deck-view update event. The cleanup is tied to
death generally, not specifically to normal execution, so Slayer or another
death path reaches the same role callback. The stored field is not cleared.

No method in this boundary changes a `Character.state` or flips a board card.
The observed post-Baa “reveal” is the formerly obscured identity becoming
visible in the deck strip. The old live hook that searched process memory for a
newly revealed board position could therefore record an unrelated card; it now
reports the deck refresh without mutating reveal order.

## Rule and construction surfaces

`Imp.GetRules` returns a fresh empty `List<SpecialRule>`. Baa's effect is thus
implemented entirely through its ordered role action and the deck-view static
list, not by adding a gameplay `SpecialRule`.

The constructor follows the ordinary folded base-role initialization body and
does not initialize `blockedOutcast` explicitly. Its RVA is shared with
hundreds of trivial constructors, so the target preserves the exact managed
identity while applying the established canonical prototype for that body.

## Solver and live-tool consequences

- The Rust solver's existing full-pool assumption is native-correct: Baa
  obscures presentation only and cannot invent an absent Outcast role.
- Card-vision reconciliation may accept exactly one memory-only Outcast paired
  with an eye-symbol box, while HUD `no=` remains unchanged.
- `memory_reader.clean_name` now maps fallback managed name `Imp[_digits]` to
  Baa. It also maps bare managed `Skinwalker` to Mutant; the current Mutant
  CharacterData already carries the display-derived ID `Mutant_84675843`.
- Baa is marked as having a game-start ability in both knowledge bases.
- Baa death no longer causes the live wrapper to infer or append a board reveal.

The live asc82_v2, asc83_v2, and asc83_v6 notes independently record the hidden
deck-strip identity becoming visible after Baa dies; those observations agree
with the native `OnDied` removal and refresh path.
