# DeckView visibility and identity callers

Build `f530404b0f3f_807de4a83df4`. `scripts/audit_deck_view_visibility.py`
executes four complete native DeckView bodies against pinned GameAssembly and
Dumper metadata. The matching report is
`reports/f530404b0f3f_807de4a83df4_deck_view_visibility.json`.

| Declaration | Stable method ID | Complete native range | Instructions |
| --- | --- | --- | --- |
| Start | tdi5744.m0002 | 39C7E0..39C86B | 34 |
| OpenDeckView | tdi5744.m0010 | 39B7B0..39B8A1 | 59 |
| CloseDeckView | tdi5744.m0011 | 39B1A0..39B28C | 59 |
| Update | tdi5744.m0009 | 39D310..39D33C | 16 |

Ranges are end-exclusive and include each final instruction and native null
gateway call, excluding trailing alignment padding. Exact method names,
signatures, addresses and DeckView's TypeDefIndex 5744 field declaration are
verified. No Ghidra project or live game is opened. The shared constructor and
the other DeckView methods are outside this audit.

## Identity and visibility

Start requests the component's GameObject, rejects a null returned object,
obtains its instance ID, boxes the **low 32 bits** and requests String.Format with
the exact literal `deckView_{0}`. It stores the returned string in animId (+0x70)
before the reference barrier. Fixtures include signed extrema, poisoned upper
return bits, a supplied null formatted result, and callbacks before/after the
native store. A format callback's replacement animation ID is overwritten by
the native store; a barrier callback can replace the stored ID afterward. A
barrier stop retains the newly stored ID. String formatting, boxing and engine
identity lookup are explicitly supplied, not independently decompiled here.

Both visibility methods capture canvasGroup (+0x68), initialize Unity.Object if
its runtime class word is zero, and request comparison of that captured canvas
with null. **Only AL controls the comparison branch.** A nonzero low byte
returns without any tween or CanvasGroup setters, regardless of upper RAX bits.
The capture survives a supplied class-initialization callback that changes the
owner's canvas reference.

On the other branch they capture animId after the equality service and before
DOTween class initialization, then request DOTween.Kill with complete byte 1.
Both Open and Close pass complete=true; the Kill return value is ignored.
The full Kill RDX register retains the supplied caller-saved poison's high bits
while DL is 1. Open's two setter calls likewise preserve those high bits; Close's
`xor edx, edx` clears the entire register. Both the byte and full register are
asserted and recorded. Update's key argument is the exact zero-extended 27.
They reload canvasGroup for DOFade, with duration bits 0 in both methods. Open
passes alpha bits `0x3F800000`; Close passes alpha bits 0. SetId receives the
returned tween, the **reloaded** animId and the exact
`SetId<TweenerCore<float, float, FloatOptions>>` MethodInfo, even though the
native shared function's metadata name is `SetId<object>`.

Open then requests interactable=true and blocksRaycasts=true; Close requests
both false. Each setter loads and null-checks canvasGroup again. A callback can
therefore route fade, interactable and raycast requests to different physical
CanvasGroup records. Nulling the reference after fade/SetId stops before the
first setter; nulling it after interactable stops before blocksRaycasts,
retaining the first supplied setter's effect. The report records precise
service entry snapshots and request identity; a fade is a requested tween and
does not synthesize rendered alpha progression.

## Native Update composition

Update requests GetKeyDownInt(27). Only AL controls its branch. Zero returns
without touching visibility; any nonzero byte tail-calls the **actual native
CloseDeckView body**. The key service can replace canvasGroup/animId before
that body begins. This is an explicitly supplied input result, with no live
keyboard use, engine input polling reconstruction or UI focus claim.

Two retained six-call sequences execute Start, Open, Update(false),
Update(noncanonical true), Close, Open in the same physical state. Metadata
flags, class-initialization words, identity strings, UI bookkeeping and prior
requests persist between these calls. The second sequence uses a different
physical canvas. Additional fixtures preserve aliased, unconsumed Transform
parent fields and distinguish two physical animation strings with equal text.

## Evidence and boundaries

The corpus contains **142 cases**: 127 normal returns and 15 native null guards;
two retained six-call sequences; and **81 controlled service stops** across
eight cold complete/capture/reload baselines. Every stopped event list equals
its baseline's complete attempted prefix, and its full final snapshot equals
the last service-entry snapshot. All **168/168 body instructions** execute
across 183 total native/service addresses. There are 22 direct instruction
assertions and a verified file-backed float literal.

Normal returns verify stack balance, all eight Windows nonvolatile integer
registers and XMM6..XMM15. Services poison caller-saved integer registers and
XMM0..XMM5; return-bit probes distinguish AL from the rest of RAX. Every
unconsumed owner byte is retained outside documented callback writes. Snapshots
include all eleven declared owner references, a diagnostic 0x90 owner window,
runtime class words/bytes, metadata slots/flags, supplied strings/boxed integer,
CanvasGroup records and UI bookkeeping, opaque GameObject/tween records and
accumulated requests. Diagnostic byte windows and sentinel headers are not
claims about actual object sizes or valid managed layouts for unused fields.

Unity equality/liveness, component and instance-ID lookup, input collection,
CanvasGroup setters, DOTween Kill/DOFade/SetId, formatting/boxing,
metadata resolution, class initialization and barriers remain explicit supplied
implementations. Diagnostic null animation/tween responses and a false equality
result for a null canvas use permissive services solely to expose the caller's
native guards; this does not assert those inputs are accepted by the real
engine or tween library. Controlled stops do not execute managed exception
construction or unwinding. No renderer, scheduler, ObscuredCharacters lifecycle,
scene wiring, reveal coroutine or remaining DeckView body is reconstructed.

Reproduce with:

```
python scripts/audit_deck_view_visibility.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Use private Unicorn 2.1.4 and its accompanying dependencies through PYTHONPATH.
Python syntax compilation, the 32 reverse-engineering infrastructure tests and
two byte-identical final report producers pass.
