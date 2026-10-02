# PinDeckViewButton caller audit

This audit executes four exact `PinDeckViewButton` method bodies from the pinned
GameAssembly offline. Settings access, Unity UI, metadata, allocation, delegate
construction, Combine/Remove, write barriers and Action callback effects remain
explicit supplied services. No live game/process, scene lifecycle, preference
storage, renderer, or managed multicast implementation is executed.

## Exact identity and bounds

The pinned declaration is `PinDeckViewButton : MonoBehaviour`, TypeDefIndex 5728,
with the single custom field `public Toggle toggle; // 0x20`. These four exact
metadata declarations have `vii` signatures and `void` returns:

| Method ID | Method | RVA | End exclusive | Next managed entry |
| --- | --- | --- | --- | --- |
| tdi5728.m0000 | OnEnable | 0x3A5660 | 0x3A57C3 | 0x3A57D0 |
| tdi5728.m0001 | OnDisable | 0x3A5530 | 0x3A565D | 0x3A5660 |
| tdi5728.m0002 | UpdateView | 0x3A57D0 | 0x3A5816 | 0x3A5820 |
| tdi5728.m0003 | OnClick | 0x3A54C0 | 0x3A5524 | 0x3A5530 |

`OnEnable` spans three verified chained unwind chunks: `3A5660..3A56C4`,
`3A56C4..3A5738`, and `3A5738..3A57C3`. The first chunk is not the complete
method. The other three methods each have one verified unwind entry. The audit
checks file backing and complete reads, decodes from verified entries through
the final instruction, and checks all alignment bytes to the next managed
entry. Its coverage does not promote the folded constructor at `0x33E820`.

The exact static Settings declaration is TypeDefIndex 5795. `PinnedDeck` is an
`int`, with getter `0x3C70A0` and setter `0x3C7310`. The getter supplies a full
low-DWORD bit pattern, with poisoned upper RAX bits. The consumed truth gate is
`EAX != 0`, including negative and noncanonical integer values. No preference
key, persisted value, getter/setter body or automatic setter notification is
inferred from this caller audit.

The exact UIEvents declaration is TypeDefIndex 5523, containing 21 static
fields. `OnSettingsChange` is `Action` at static offset `0x88`. Its complete
`0xA8` fixture storage and every unrelated field are retained. The UIEvents
runtime class static-block pointer at `+0xB8` is an explicit input; the class
word at `+0xE0` is retained unchanged. These callers perform no class-init gate.

## Toggle and registration behavior

`UpdateView` and the inlined view prefix of `OnEnable` first call the supplied
getter, then load the current `toggle` reference. They pass the getter's
nonzero/zero result to `Toggle.SetIsOnWithoutNotify` at `0x1EDB120`. The method
name and its exact `bool` ABI signature are independently pinned. A missing
toggle reaches the caller's null guard after the getter. Getter-time replacement
of the component is consumed by the subsequent load. A null owner faults at
that owner field load; it is not repaired or declared valid Unity behavior.

For a true result, both callers write only `DL = 1`; poisoned upper RDX bits are
preserved. `OnEnable` writes only `DL = 0` for a false result, whereas the false
`UpdateView` branch clears all of EDX. The fixtures assert exact physical RDX,
its Boolean low byte, and the zero R8 MethodInfo argument. `UpdateView` restores
the caller's stack and tail-transfers to the supplied toggle implementation.

After updating the toggle, `OnEnable` captures the current settings-change
Action, allocates an Action, and supplies the exact retained owner and
`Method$PinDeckViewButton.UpdateView()` registration token to its constructor.
It calls supplied `Delegate.Combine` with that previously captured source.
`OnDisable` captures the event first, then allocates and constructs the same
registration value and calls supplied `Delegate.Remove`. Allocation- or
constructor-time event replacement does not alter the captured input.

Combine and Remove have authored bounded invocation bookkeeping: append for
Combine and remove the final matching segment for Remove. Null, source-alias
and foreign-class result profiles are independently supplied. This bookkeeping
does not execute or reconstruct the managed implementations. Native inline
class-pointer checks accept only the exact supplied Action class; a foreign
result enters a decoded cast-failure gateway before storing the new event.
For accepted results, the native caller reloads the UI static-block pointer,
stores the event, reloads the block again, and tail-calls the supplied write
barrier. A service-time static-block swap therefore changes the destination
while retaining the earlier captured source.

Three independent native metadata flags are raw bytes: `OnEnable` at
`0x288C3BD`, `OnDisable` at `0x288C3BE`, and `OnClick` at `0x288C3BF`. Cold
profiles perform the exact metadata services before writing byte 1; nonzero
warm profiles, including `0xFE`, skip those services. Full service-entry and
final snapshots retain those bytes and the three resolved metadata slot values.

## Click and physical Action dispatch

`OnClick` does not consume its owner. It calls the supplied getter, computes a
canonical integer `getter == 0` using `xor ecx, ecx; sete cl`, and passes that
0 or 1 to the supplied setter with zero RDX MethodInfo. After the setter returns,
it reloads the UIEvents class/static block and the Action at `+0x88`. A null
event returns directly; otherwise it tail-dispatches through that physical
Action's `invoke_impl` at `+0x18`, with its `method_code` at `+0x40` in RCX and
its `method` at `+0x28` in RDX. Setter-time event clearing, replacement and
static-block swapping are observed by these subsequent loads.

The offset names are pinned to the exact `System.Delegate` declaration,
TypeDefIndex 419, with `m_target` at `+0x20`. Its `method_code` field is an
IntPtr; it is not renamed as the managed target. `MulticastDelegate` is
TypeDefIndex 440, with its `delegates` field at `+0x78`; `Action` is the exact
sealed TypeDefIndex 153 declaration. Supplied registration construction and
callback records include these physical fields. Authored callback first-argument
tokens may equal the button or either Toggle, or be null; the callback is a
supplied acceptance boundary, with no claim about actual managed delegate
admission or dispatch of every logical invocation. R8/R9 at callback entry
retain supplied volatile poison; the caller does not write hidden MethodInfo
arguments for this indirect tail transfer.

## Evidence limits and verification

Each profile records complete initial, service-entry and final fixture memory,
metadata state, raw setting bits, UI values, delegate records, allocation order,
explicit supplied setting writes and callback observations. Diagnostic windows
and their sentinel bytes are not managed object size assertions or a claim that
unconsumed sentinel references form valid typed objects. Every unrelated byte
is checked, permitting only exact caller stores and authored supplied changes.
The bounded fixture graph never imports private native method bytes into the
report.

Normal returns check all eight Win64 integer nonvolatiles, XMM6–XMM15 and the
return stack. Supplied services poison integer volatile registers and XMM0–5,
so later native loads cannot silently rely on accidental register preservation.
Failure profiles stop at each actual service-entry boundary and require the
entire event/snapshot prefix and final snapshot to match the normal baseline.
They do not emulate engine exception unwinding.

All 198 instructions of the four complete caller bodies are decoded and 186
are executed. The explicit remaining 12 are six `int3` terminal traps and six
instructions in the second post-store cast-failure stubs (`3A5651/54/57` and
`3A57AB/AE/B1`). A typed matching result passes the first inline check, with no
intervening service before the second check; these supplied inert profiles
cannot trigger that second failure. The stubs remain decoded and recorded,
without an artificial concurrent metadata race or an execution-coverage claim.

The corpus contains 213 standalone normal/edge/mutation profiles, 12 retained
lifecycle sequences, and 22 exact service-entry stops. Lifecycle sequences
check repeated enable, final-match disable, click and explicit view refresh
with retained native metadata/event storage and supplied invocation bookkeeping.

## Reproduction

```powershell
python -m py_compile reverse_engineering/scripts/audit_pin_deck_view_button.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH = 'B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_pin_deck_view_button.py `
  --game-root 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' `
  --dumper-root 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/il2cppdumper-v6.7.46' `
  --output 'B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/pin_deck_view_button_peer.json'
```

The producer pins the installed GameAssembly and Dumper artifacts through the
checked-in build/extraction manifests for `f530404b0f3f_807de4a83df4`. The public
report is `reports/f530404b0f3f_807de4a83df4_pin_deck_view_button.json`.

Two successful independent final producers, each preceded by Python syntax
compilation, produced byte-identical reports: 13,460,664 bytes, SHA-256
`1587eb73aeb9abe8b5f2e32f6a73a28fcd9cb015b965193c273ccf4ed1668da2`.
The independent private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/pin_deck_view_button_peer.json`.
All 32 reverse-engineering infrastructure tests passed; no Rust build,
simulation suite, live game or Python bridge regression was run by this audit.
