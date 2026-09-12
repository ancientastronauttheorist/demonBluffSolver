# ActedVersion text and scale animation callers

Pinned build `f530404b0f3f_807de4a83df4`. The authored audit executes the complete native `ActedVersion.Animate` (0x35D690..0x35D858) and `Show(string description)` (0x35D920..0x35DBEC) bodies. The corpus passes 72 native cases, executing 228 instruction addresses. Both complete entry boundaries and exact metadata signatures are verified against pinned GameAssembly/script/dump inputs. Constructor and the other four helpers are covered by the separate ActedVersion helper audit.

## Show ordering

Show reads the animation ID, initializes DOTween if required, and calls `DOTween.Kill(id, complete: true)`. It then obtains its GameObject, requires it to be physically nonnull, and activates it. Next it invokes the blank TextMeshPro object's virtual text setter with the description, then the main text object's setter with the empty string. A missing main text fails after activation and blank-text update. Null descriptions are forwarded.

It requests TMP `DOText` with the description, duration bits `0x3E4CCCCD` (float approximately 0.2), rich text enabled, scramble mode zero, null scramble characters and null MethodInfo. It passes that tween through SetId, then executes the scale-animation logic inline. Show does not call the separate Animate entry; the native instructions implement the same logic inside its body.

Kill's complete flag can matter to the actual tween engine. Its completion callbacks remain outside this preserving-service corpus; the audit does not claim they cannot mutate the receiver or scene.

## Saved-scale selection and tween

Both methods compare `savedScale` (+0x40..+0x4B) to the supplied Unity Vector3 static zero vector. They perform float subtraction, square each component, and sum `(y² + x²) + z²`, rounding each operation as native float32 under MXCSR `0x1F80`. They capture the current transform's local scale only when the squared distance is strictly below bits `0x2EDBE6FE` (approximately 1e-10). This is a tolerance comparison, not a component-wise bitwise zero test. NaN/unordered comparisons skip capture. Signed zero and subnormal values can take the capture branch.

If capture is needed, transform lookup and physical-null checking precede get_localScale; the returned twelve raw bytes are then written to savedScale. A second lookup is checked for physical null before setting localScale to the static zero vector. A third transform lookup supplies DOScale with savedScale and the same duration bits `0x3E4CCCCD`. This third result has no caller null guard: null is forwarded to the declared DOScale service. The resulting tween goes through SetEase with `OutBack = 27`, then SetId using the current animation ID. Null tween returns are likewise forwarded through the declared extension services.

## Fixtures and boundaries

The corpus includes preserving success and every cold baseline gateway failure prefix; missing GameObject/TMP references; all three transform-null positions; null description/ID/tween returns; zero, signed zero, subnormal, finite, infinity and NaN saved scales; representable neighbors around the 1e-5 component threshold; and a supplied nonzero static zero-vector with raw nonfinite captured scale. All saved-scale and transform values use raw u32 bits. An independent authored event/state model checks the complete prefix and resulting saved scale, transform scale and text values.

All three generic SetEase/SetId MethodInfo slots are bound by their exact pinned metadata names and checked at the service call, including null tween returns. A separate returned-buffer fixture verifies get_localScale copies the returned RAX buffer, not an assumed out-buffer identity. Normal returns require exact RIP/RSP restoration and eight distinctly seeded nonvolatile integer registers. All owner bytes outside savedScale are preserved. Metadata flags are warmed. TMP virtual setters, Unity object/transform accessors, DOTween initialization and tween creation/settings remain explicit stable gateways. Failure fixtures stop at gateway entry and do not execute managed exception unwinding. No actual tween advancement, interpolation/ease behavior, completion callback or engine timing is inferred.

Artifacts: `scripts/audit_acted_version_animation.py` and `reports/f530404b0f3f_807de4a83df4_acted_version_animation.json`.
