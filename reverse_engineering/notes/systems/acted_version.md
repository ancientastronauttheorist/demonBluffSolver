# ActedVersion immediate helpers

Build `f530404b0f3f_807de4a83df4`. `scripts/audit_acted_version.py` and `reports/f530404b0f3f_807de4a83df4_acted_version.json` add **30 native fixtures** for Awake35D860, GetActed35D8F0, Start35DBF0, UpdateActed35DC40 and constructor33E820. Exact declarations, full native boundaries, metadata references and consumed field offsets are verified against pinned GameAssembly/Dumper inputs. The constructor explicitly maps this declaration to the already audited shared MonoBehaviour constructor tail body.

GetActed directly checks blankText (+28) and tail-calls its virtual text getter through function/MethodInfo slots548/550. It does not check GameObject activeSelf; that gate belongs to the outer Acted.GetActed wrapper. UpdateActed likewise forwards the supplied description to blankText using558/560. Both preserve the separate text field (+30). Null blankText fails at the native guard; nullable getter results and setter arguments are passed unchanged through the explicit text service.

Awake obtains the component's GameObject and signed instance ID, boxes its integer bits and formats the exact `{0}_actedAnim` string. It writes the returned animationId (+38) before the reference barrier. Gateway failures before that store preserve the prior ID; barrier failure retains the newly stored ID. Null GameObject fails before ID lookup, while a nullable format response is stored under the supplied formatting service. Signed minimum, zero and maximum IDs verify the exact integer payload without assuming decimal formatting internals.

Start resolves UnityEngine.Vector3 metadata and copies **zeroVector at static offset0** into savedScale (+40), using an eight-byte move followed by a four-byte move. It does not obtain a transform or read current scale. Raw signed-zero, NaN-payload, subnormal and ordinary vector fixtures verify byte-preserving copying from the supplied static value. Modified zeroVector fixtures are adversarial provider data, not a claim about mutation of the actual readonly engine field.

All reached baseline service failures and cold metadata failures preserve exact event/field prefixes. Normal returns verify stack balance and all eight Windows nonvolatile integer registers. Owner bytes outside animationId and savedScale remain unchanged. Engine object/ID lookup, boxing/formatting, virtual text methods, metadata setup and the MonoBehaviour base constructor remain explicit services. No animation, layout, text rendering or managed unwind implementation is reconstructed.

ActedVersion.Show35D920 and Animate35D690 are covered separately in [the animation caller audit](acted_version_animation.md); this report itself covers the five helpers. No Rust module, shared target, Ghidra project change or live interaction is added. Final 30-case rerun and Python compilation pass.

Reproduce with `python scripts/audit_acted_version.py GAME_ROOT DUMPER_ROOT --output REPORT`, using the private Unicorn2.1.4 runtime through PYTHONPATH.
