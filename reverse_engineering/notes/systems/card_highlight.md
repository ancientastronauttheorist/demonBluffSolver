# CardHighlight callers and iterator boundary

Build `f530404b0f3f_807de4a83df4`. `scripts/audit_card_highlight.py` and `reports/f530404b0f3f_807de4a83df4_card_highlight.json` add **76 native fixtures** for seven exact declarations: Awake396EB0, DisableHighlight396F40, HighlightCoroutine factory397020, ShowHighlight397090, constructor397150, generated MoveNext3A9F50 and Reset3AA090. Complete entry boundaries exclude trailing padding, and all seven signatures plus nine immediate gateway bindings are verified against pinned Dumper metadata. Shared iterator constructor/Current/Dispose declarations already have separate coverage.

## Animation identity and entry calls

The constructor writes the empty string to highlightAnimation (+28), performs the reference barrier, then tail-calls the MonoBehaviour constructor. Awake obtains this component's GameObject and its signed instance ID, boxes the integer and passes it to the exact `{0}_highlight` format string. It writes the returned string before the reference barrier. A barrier failure therefore preserves the stored ID. Formatting/boxing and engine identity lookup remain explicit services; a null GameObject reaches the native null failure.

Both ShowHighlight and DisableHighlight first request StopAllCoroutines on this component. They then load highlightAnimation and initialize DOTween if necessary before calling DOTween.Kill. **Show passes complete=true; Disable passes complete=false.** The ID used for Kill is cached after StopAllCoroutines but before class initialization: controlled callbacks verify those read positions. Kill's result is ignored.

Show then allocates the exact generated iterator, writes state0 and its captured receiver, performs the capture barrier and tail-calls StartCoroutine with that iterator. It contains the factory's allocation/capture sequence inline. The separately callable HighlightCoroutine factory performs that sequence and returns the iterator without starting it. It can capture a null receiver. Neither path itself executes MoveNext in these fixtures; StartCoroutine is a gateway, so no claim is made about synchronous first advancement or subsequent engine scheduling.

Disable captures the highlights array (+20), traverses every occurrence, requests DOFade(target, alpha0, duration bits `0x3DCCCCCD`), then calls the bound SetId generic helper with the returned tween and the current highlightAnimation. The array reference stays captured; the ID is reread after each fade. Controlled fade/SetId callbacks confirm that swapping the owner's array does not replace the current traversal, while ID mutations affect the immediately following SetId. Repeated references cause repeated requests.

## Iterator state and timing descriptor

On state0, MoveNext first writes state-1, allocates WaitForSeconds and calls its constructor with raw bits `0x3ED70A3D` (binary32 0.42). It stores that object in Current before the reference barrier, then writes state1 and returns true. Allocation/constructor failure leaves state-1 and the prior Current; barrier failure leaves the newly stored Current but state-1.

An explicit second invocation with state1 writes state-1 before checking its captured receiver and highlights array. It traverses the captured array, requests DOFade(target, alpha1, duration bits `0x3F000000`), applies the current animation ID to each returned tween, and returns false. Current is retained. Other states return false without changing state or Current. Null capture is harmless for the first yield but fails on state1; a null array fails after the state becomes negative. An empty array completes without fades.

These are constructor arguments and requested tween durations, not recovered scheduler deadlines or demonstrated elapsed time. No wait queue, time-scale behavior, DOTween update loop, rendered alpha progression or tween completion callback is reconstructed.

## Failures and explicit services

All reached baseline service failures retain exact attempted event and managed-field prefixes. Tests also cover cold metadata/type initialization, duplicate/null array elements, null tween responses, null IDs, captured-null factories and integer state extremes. Null CanvasGroup and null tween behavior is deliberately supplied by permissive DOFade/SetId gateways to expose caller behavior; it is not a claim that their real implementations accept those values.

Reset always allocates and constructs NotSupportedException, resolves its exact MethodInfo and invokes the throw gateway, preserving state, Current and capture. Stop/start coroutine internals, allocation, constructors, tween methods, formatting and managed exception unwinding remain service boundaries.

Normal returns verify stack balance, all eight Windows nonvolatile integer registers and XMM6/XMM7. Owner bytes outside the explicitly modeled array/animation fields and the traversed array remain unchanged. Reference barriers execute after the native stores; this audit retains those partial writes on injected barrier failure. Python compilation and the final 76-case rerun pass. No new targets, live game interaction or Rust replay are added.

Reproduce with `python scripts/audit_card_highlight.py GAME_ROOT DUMPER_ROOT --output REPORT`, using the private Unicorn2.1.4 runtime through PYTHONPATH.
