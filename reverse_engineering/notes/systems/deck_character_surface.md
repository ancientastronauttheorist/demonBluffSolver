# DeckCharacter data, click, reveal and hover surfaces

Build: `f530404b0f3f_807de4a83df4`.

Producer: `reverse_engineering/scripts/audit_deck_character_surface.py`.
Report: `reverse_engineering/reports/f530404b0f3f_807de4a83df4_deck_character_surface.json`.

The producer executes five complete DeckCharacter bodies and the actual joined
HintInfo constructor in the pinned GameAssembly. Metadata, allocation, class
initialization, reference barriers, event callbacks, List.Contains and the whole
RevealCard.RevealNoAct callee are supplied at explicit entry boundaries. The
report preserves each input, every service-entry snapshot, actual helper entry,
normal result, stopped prefix and final diagnostic bytes.

## Exact entries and complete bounds

| Stable method | Signature | RVA | Exclusive end | Bytes | Instructions |
| --- | --- | --- | --- | --- | --- |
| `tdi5514.m0000` | `CharacterData GetData()` | `36FA10` | `36FA15` | 5 | 2 |
| `tdi5514.m0004` | `void Click()` | `36F9B0` | `36FA05` | 85 | 22 |
| `tdi5514.m0003` | `void Reveal()` | `36FF70` | `36FF8E` | 30 | 9 |
| `tdi5514.m0006` | `void OnHoverExit()` | `36FDC0` | `36FE0C` | 76 | 17 |
| `tdi5514.m0005` | `void OnHover()` | `36FE10` | `36FF65` | 341 | 72 |
| `tdi5800.m0000` | `HintInfo` constructor | `3BC200` | `3BC297` | 151 | 40 |

Exact Dumper signatures, input hashes, class declarations, PE raw backing and
full decode lengths are asserted. GetData is a verified leaf without an unwind
record. OnHover spans three chained unwind ranges, `36FE10..36FEDC`,
`36FEDC..36FF3D`, and `36FF3D..36FF65`. Every method excludes only verified `CC`
alignment before the next managed entry. The decoded terminal traps at `36FF64`
and `36FF8D` belong to the method ranges and are retained as unexecuted paths
after the native null gateway.

The constructor's actual base entry at `33ED50` decodes as `ret 0` and executes.
Its folded aliases receive no coverage promotion. The constructor executes all
40 decoded instructions in the OnHover argument profile; this is bounded join
evidence, not a corpus of arbitrary constructor arguments.

## Pinned storage and channels

DeckCharacter is TypeDefIndex 5514. The exact fields are `Action onClick +20`,
`Character character +28`, `RevealCard revealCard +30`,
`CardInteraction interaction +38`, and private `CharacterData data +40`.

CardsEvents (5522) has static `Action<DeckCharacter> OnDeckCharacterClicked +0`.
UIEvents (5523) has static `Action<HintInfo,Transform> OnShowHint +10` and
`Action OnHideHint +40`. All 21 UIEvents Action fields are retained in the report;
`OnShowCustomHint +90` is a distinct, unused channel. DeckView (5744) has static
`List<CharacterData> ObscuredCharacters +0`. Character (5487) has
`Transform hintPivot +38`.

The caller consumes System.Delegate (419) `invoke_impl +18`, `method +28` and
`method_code +40`. The fixture separately initializes `m_target +20` with an
unused managed-target token. `method_code` is the exact declared IntPtr, not
the managed target. Callback records model only the consumed physical slots;
their class token and unused bytes are diagnostic and make no CLR admission or
multicast implementation claim.

HintInfo (5800) fields are `title +10`, `text +18`, `hints +20`, `flavor +28`,
`Sprite img +30`, and sixteen-byte `Color borderColor +38`.

## Native behavior and ABI

GetData returns the current nullable pointer from owner `+40`, without a service
call or mutation. Click initializes its CardsEvents metadata when needed,
loads the current static event, and returns when it is null. Its callback tail
invocation has `RCX=Action.method_code`, `RDX=DeckCharacter owner`,
`R8=Action.method`. The incoming R9 is not an argument and is recorded as raw
diagnostic bits. The instance `onClick +20` is not read.

OnHoverExit similarly loads UIEvents.OnHideHint and tail invokes with
`RCX=Action.method_code`, `RDX=Action.method`. R8 and R9 are untouched diagnostic
registers. A null channel returns without invoking anything. Mutation during
metadata initialization demonstrates reloading the current static event.

Reveal loads current `revealCard +30`, guards null, and tail calls the supplied
whole `RevealCard.RevealNoAct` at `386920` with exact zero MethodInfo in RDX.
The callee is a separate unaudited game-owned body; neither its visual effects
nor its own native scheduling are reconstructed here.

OnHover performs six cold metadata requests, and conditionally initializes
DeckView using the DWORD at class `+E0`. It reloads the class's static block at
`+B8` after class initialization. A null obscured list enters the native null
gateway. Otherwise List.Contains at shared generic `B55950` receives current
list in RCX, current owner data in RDX, and the exact
`Method$System.Collections.Generic.List<CharacterData>.Contains()` in R8.
Only AL controls the following predicate; higher return bits are poisoned.

Membership false returns. Membership true reloads UIEvents.OnShowHint and
returns if that callback is null. Thus no callback means no allocation and no
Character null check. It captures the Action before allocation and retains that
same Action even when the supplied allocator or a reference barrier replaces
the callback or UIEvents static block.

Allocation returns a fresh diagnostic HintInfo identity. OnHover calls the
actual constructor with hidden text in RDX, null Sprite in R8, empty hints in
R9, empty flavor and title in stack argument slots, a pointer to sixteen zero
Color bytes, and null MethodInfo. At constructor entry these stack arguments
are checked at RSP `+28`, `+30`, `+38`, and `+40`. The exact hidden literal is:

```text
This card is Hidden,
something is blocking it!
```

The hidden literal slot is `26E6C78`; the empty string slot is `26DF1B8`.
Each is resolved from native RIP operands and the pinned ScriptString inventory.
Metadata flags are `288C1E4`, `288C1E5`, and `288C1E6`; zero requests initialization,
whereas nonzero diagnostic `FE` is retained without another request.

The constructor executes the folded base return, then native reference stores
and barriers in text, title, image, hints, flavor order. Native Color copying
follows the final barrier. Snapshot decoding is phase gated: allocator entry
does not decode an unconstructed object; each barrier exposes only the state
then reached. The constructor's void RAX is not an object-return contract.
OnHover uses its preserved allocation identity instead.

After construction, OnHover reloads current Character from owner `+28`; a null
Character now stops with a fully constructed HintInfo retained. It then loads
that Character's current nullable hintPivot and invokes the captured callback
with `RCX=captured Action.method_code`, `RDX=allocated HintInfo`,
`R8=current Character.hintPivot`, `R9=captured Action.method`. Replacements during
allocation and during early or late barriers verify capture versus reload.

All supplied returns poison volatile integer and XMM registers. Native call
sites assert full pointer/MethodInfo widths and exact zero register clearing.
Normal returns verify the stack and all eight integer plus ten XMM nonvolatile
registers. Null method_code remains a valid supplied callback ABI probe; it is
not a claim that a scene would admit that delegate.

## Executed evidence and stopped prefixes

The corpus has 83 individual cases: 62 normal returns and 21 reached native
null gateways. It includes warm/cold method and DeckView class state, full
DWORD initialized values, nullable data/reveal/Character/pivot/channel/list,
unused instance callback aliases, shared pivot identities, and AL values
`00`, `01`, `80`, `FF` with poisoned high return bits. The latter are operand
width probes of supplied results, not managed Boolean admission claims.

Authored service mutations replace or clear data, Character, callback channels
or hintPivot, and swap UI/Deck/Cards static blocks. Each reached mutation has
an exact byte retention allowance. Allocator effects apply only to its fresh
diagnostic buffer. All remaining owner, Character, pivot, RevealCard,
CardInteraction, data, static-event, Action, literal-string and runtime-class
bytes must remain identical. Metadata and literal pointer slots remain stable;
only the exact cold metadata flags may become one.

Two retained seven-call sequences execute GetData, Click, OnHover, OnHoverExit,
Reveal, OnHover, GetData in one diagnostic graph, including alias inputs. Warmed
flags, class state, allocations and callback/reveal request logs persist between
calls rather than being rebuilt.

Seven successful baseline rows generate 50 stopped service prefixes, including
allocation-time callback replacement and barrier-time Character replacement.
Each stop occurs before the selected supplied effect or authored mutation;
its entire event list equals the successful prefix and its full final snapshot
equals the baseline's corresponding service-entry snapshot. Native stores
already reached before reference-barrier entry remain visible. No continuation
or managed unwinding is invented after a stopped service.

There are 30 exact instruction assertions. Across five DeckCharacter bodies
and the HintInfo join, 160 of 162 decoded body instructions execute; the two
unexecuted instructions are the explicitly retained terminal `int3` traps.
The actual folded base plus supplied entries produce 169 total execution
addresses. Every body return path before a supplied null gateway is included.

List.Contains is wholly supplied. The List/array tokens and duplicate identity
bytes are opaque storage and retention diagnostics. The producer does not infer
generic List/array offsets, reconstruct membership from those bytes, or require
supplied predicate outputs to agree with the opaque records. Sentinel buffers
also do not establish actual allocation sizes or valid types of unused fields.

## Remaining scope

Init (`36FA20`) and OnDisable (`36FC00`) register and remove two CardInteraction
event channels and remain a separate family. The shared DeckCharacter
constructor at `33E820` remains unpromoted. This audit does not execute event
registration, the renderer, actual callback bodies, whole List.Contains,
whole RevealNoAct, scene lifecycle/admission, runtime initialization machinery,
GC implementation, scheduler or exception unwinding. It proves the exact
consumed services and chronology of these complete native caller bodies.
