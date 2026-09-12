# Characters layout and highlight loops

Pinned build `f530404b0f3f_807de4a83df4`. `scripts/audit_characters_layout_highlight.py` verifies GameAssembly/Dumper hashes, the exact three caller signatures and complete entry boundaries, five immediate callback/engine method identities, field offsets and the native 360.0 constant. Its report `reports/f530404b0f3f_807de4a83df4_characters_layout_highlight.json` contains 96 passing native cases and 200 executed instruction addresses. These three methods had no matching evidence row in the coverage ledger at the initial read; no shared ledger or target manifest is edited here.

## Layout

Characters.UpdateCharacterPositions (36E4E0) enumerates this.characters (+20). Despite its name, this body calls rotation setters, not a position setter. It computes a binary32 step as 360.0 divided by the signed list size converted to float32. For each occurrence in list order, it obtains the Character transform and calls set_localEulerAngles with (0,0,index*step). After that setter returns, it increments the index, resolves Character.icon (+20), obtains the icon transform, and calls set_eulerAngles with the UnityEngine.Vector3.zeroVector static value. The icon rotation is in world space, while the Character rotation is local.

No Unity destroyed-object comparison protects this loop. Null list, Character, Character transform, icon, or icon transform reaches a native null failure. A failure on the icon path leaves the current Character's local rotation already supplied, and prior characters' local/world setter calls remain completed. A failing setter gateway is recorded before its modeled effect; earlier setter effects persist. Repeated Character identities receive repeated rotations, with the last successful occurrence determining the recorded final rotation for that target.

The empty list performs the floating division under masked exceptions and then completes enumeration without any rotation call. Count-zero/nonempty-enumerator and signed-negative/extreme count fixtures are explicit adversarial provider configurations, not valid managed-list claims. They verify exact division/multiplication behavior, including the index-zero times infinity NaN result. The environment is MXCSR 0x1F80. Raw u32 bits preserve all vector values. A controlled altered zeroVector fixture confirms that the world setter receives the static value read at that point rather than an independently authored constant; it does not imply the real readonly vector is normally altered.

## Highlighting

Characters.HighlightCharacters (36CAF0) enumerates its supplied chList parameter. It does not first visit or clear the full board. Characters.DisableHighlightAll (369D60) instead enumerates this.characters. For each occurrence, both ensure UnityEngine.Object runtime initialization if needed, call Object.op_Inequality(character,null), and skip an entry when that result is false. This covers managed nulls and supplied destroyed-object equivalence. A nonzero high portion of the gateway return with AL=false still skips correctly.

For an accepted Character, the caller checks the physical pointer and its CardHighlight field (+90), then directly calls ShowHighlight (397090) or DisableHighlight (396F40). Duplicate input occurrences cause duplicate calls. No deduplication, membership restriction, or implicit clearing of unselected entries appears in these three bodies. A deliberately inconsistent Unity comparison that returns true for a physical null confirms the subsequent explicit null check.

## Fixture scope

The complete native caller instructions execute; enumeration, Unity comparison/class initialization, transform lookups/setters and CardHighlight callbacks are explicit gateways. All managed fixture object bytes remain unchanged by the callers. Separate effect records retain successful rotation/highlight callback prefixes, distinct from attempted calls. Cases cover empty/null lists, selected-vs-board distinction, duplicates, null entries/components, destroyed-object skipping, runtime initialization, exact rotation bits and every reached baseline gateway failure occurrence. Successful enumeration reaches its disposal gateway; managed exception unwinding/finally is not modeled.

The deeper CardHighlight coroutine/tween bodies and Unity transform implementation are not executed. This proves caller traversal, arguments and partial callback order, not visual interpolation, coroutine scheduling, parent transform composition or a live layout screenshot. Reproduce with game-root/Dumper-root positional arguments and `--output`, using the private Unicorn 2.1.4 environment. Python compilation passes; no native code or live game state is included.

## Offline caller replay

`crates/solver-core/src/character_visuals.rs` implements the versioned
`character_visuals_native_v1` boundary. Five test groups compare all 96 native
fixtures, duplicate/failure effects, shared stable transform references, strict
provenance and input limits, and actual host floating-point control rejection.
The reference registry rejects contradictory stable get_transform responses
while admitting shared icons and Transform self-identity. Local and world
setter effects remain separate channels; engine transform composition is not
inferred. The caller context is unchanged.

Rotation arguments execute the audited SSE conversion/division/multiplication
order with matching MXCSR control bits. The model checks host controls without
changing them and excludes sticky exception-status timing, including the native
empty-list division. It preserves vector bit patterns and attempted/successful
callback prefixes under the explicit stable services. Unsupported graphs or
bounds reject before publishing a replay result. There is no live UI integration.
