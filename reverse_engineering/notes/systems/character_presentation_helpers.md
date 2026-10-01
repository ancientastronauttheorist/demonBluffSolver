# Character presentation helper callers

Pinned build `f530404b0f3f_807de4a83df4`. The offline audit executes six complete
Character native callers in 311 cases, two retained seven-call sequences and
15 controlled service-stop prefixes. There are 246 normally returning cases
and 65 native null-guard stops. All 130 non-trap caller instructions execute;
the five remaining decoded instructions are terminal traps after null guards.
Eighteen instruction assertions bind stores, operand widths, field reloads,
callback dispatch and tail calls. Every normal call verifies stack balance,
all eight integer nonvolatiles and XMM6–15. The private PE and Dumper artifacts
are checked against their pinned manifest hashes.
Character receivers and supplied class/static storage are valid and nonnull;
the corpus varies optional presentation references, not invalid receiver access.

| Caller | Method ID | Entry |
| --- | --- | --- |
| ShowDisguise | tdi5487.m0058 | 0x369240 |
| HideDisguise | tdi5487.m0059 | 0x365460 |
| ShowHighlight | tdi5487.m0060 | 0x369320 |
| DisableHighlight | tdi5487.m0061 | 0x364B20 |
| ResetRotation | tdi5487.m0032 | 0x367E00 |
| OnHoverOff | tdi5487.m0055 | 0x3674E0 |

The first five entries are decoded through their complete verified unwind
families. OnHoverOff is a verified eight-byte leaf with no unwind record;
its byte store and return are included, and alignment padding is excluded.
Exact Character, UIEvents and Vector3 declarations bind consumed fields.
All 21 physical UIEvents static slots are retained in snapshots, including
the 20 unrelated slots. Metadata bytes and native class-init DWORDs retain
their exact widths; nonzero warm values are preserved. UIEvents/Vector3
class-init words are not consumed or changed by these callers, even when
their supplied static storage is already valid under a cold word.

ShowDisguise captures the current bluff before the supplied Unity null service
and tests only AL. Nonzero AL skips all disguise work; live bluff additionally
requires the raw killedByDemon byte to equal zero. This is an exact byte
comparison, including noncanonical Boolean values. It loads Character.charBluff,
sets showDisguise to one, then checks the loaded view. A missing view thus
retains the newly written flag at its null guard. After supplied AnimateIn,
it reloads the view and bluff for supplied CharacterView.Init. It then reloads
UIEvents.OnShowCharacterDataHint, bluff and hintPivot for callback invocation.
Null callback targets, data or pivots are forwarded as supplied pointers;
the caller adds no null guard for them.

HideDisguise writes showDisguise to zero before checking its view and invoking
supplied AnimateOut. It reads the raw hover byte after AnimateOut returns.
Any nonzero hover byte allows a nonnull hint callback, which receives current
dataRef and hintPivot. Dispatch tail-forwards the delegate's physical target,
code pointer and MethodInfo. An absent callback or zero hover returns without
dispatch. OnHoverOff only clears the hover byte and returns; it hides no UI
and emits no callback.

Authored service mutations demonstrate these native reloads: clearing the view
after AnimateIn stops before Init; changing bluff changes Init and callback
arguments; clearing bluff during its earlier null check still permits Init
and the callback to receive null after a captured live result. Init can clear
the pivot or remove/replace the callback. AnimateOut can clear hover and suppress
the callback, or write a noncanonical nonzero byte and retain dispatch. Supplied
callbacks may alter actor bytes after their exact argument snapshot. These
effects describe fixture services, not implementations of those callbacks.

ShowHighlight and DisableHighlight read Character.highlight, reject a null
pointer and tail-forward to the corresponding supplied CardHighlight method,
with MethodInfo zeroed. Their Character callers change no actor bytes.
ResetRotation requires the supplied icon pointer, calls the Unity transform
getter before its Vector3 metadata initialization, checks the returned transform,
copies exactly 12 bytes from the supplied Vector3.zeroVector static storage and
passes that struct by reference to the supplied eulerAngles setter. Tests retain
the transform/icon alias, a distinct returned transform, and exact signed-zero,
infinity and NaN bits in diagnostic supplied static values. This does not claim
those diagnostics are the game's initialized zeroVector value.

The two retained sequences explicitly invoke ShowDisguise, ShowHighlight,
HideDisguise, OnHoverOff, HideDisguise, DisableHighlight and ResetRotation.
They retain metadata/class state and physical UI aliases, including hintPivot
sharing the icon and a callback target sharing the view. Clearing hover suppresses
the second HideDisguise callback. No scheduler or scene order is inferred.

Every controlled service stop retains the exact baseline event/state prefix,
before that service's authored effect. Other actor bytes are checked unchanged,
except the exact fields written by the native caller or explicitly authored
service mutation. Stops do not synthesize exception unwinding or rollback.

CharacterView.AnimateIn/AnimateOut/Init and CardHighlight.ShowHighlight/
DisableHighlight are explicit **game-owned supplied boundaries** with pinned
metadata identities. Their bodies remain outside this caller audit. The view
and highlight state recorded in snapshots is authored service bookkeeping.
Unity liveness, transform lookup/rotation, metadata/class initialization and
callback implementation are also supplied. Real rendering/input, animations,
object lifetime and the remaining Oracle/art/description callers are unclaimed.
Private native bytes and bodies remain off-repository.
The report's 142 execution addresses include one authored callback gateway;
the 130 caller instructions exclude all supplied service entries.

Independent private reports are byte-identical: 2,467,374 bytes, SHA-256
`2710e0461d0bed26bc0091a4a04ffd78f8a8a92be4c4f9783940da8ab036290a`.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_presentation_helpers.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_character_presentation_helpers.json`.
