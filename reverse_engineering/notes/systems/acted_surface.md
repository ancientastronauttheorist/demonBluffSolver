# Immediate Acted surface

Build `f530404b0f3f_807de4a83df4`. `scripts/audit_acted_surface.py` and `reports/f530404b0f3f_807de4a83df4_acted_surface.json` add **68 native fixtures / 200 executed instructions** for GetActed35DE70, UpdateActed35E010, Act(string)35DD10, Hide35DF00, Highlight35DF40, UnHighlight35DFC0 and the aliased constructor33E820. Exact signatures, complete boundaries, consumed field metadata and ten immediate gateway identities are checked against pinned GameAssembly/Dumper inputs. The separate delayed Act overload and generated iterator are owned by the delayed-surface audit.

## Text and layout ordering

GetActed checks acted (+20), obtains that ActedVersion's GameObject and queries activeSelf. False returns the exact empty string without touching text. True rereads acted, checks its **blankText (+28)** and tail-calls its text getter using vtable function/MethodInfo slots548/550. This is distinct from ActedVersion.text (+30). Controlled activeSelf callbacks prove the acted reference is reread; when activeSelf returns false, even a callback clearing acted still returns empty. A nullable getter response is forwarded unchanged.

UpdateActed first obtains this component's GameObject, reads its name, concatenates the exact `Character: ` prefix, obtains this GameObject again, initializes Debug if required and logs using the second object as context. It then rereads acted and blankText, and tail-calls the text setter via558/560 with the original description. A nullable second GameObject is passed to the log gateway without another caller-side null check. Failures in the final acted/text chain occur after the completed log. Text operations and engine/string/log implementations are explicit services.

Act(string) requires acted and calls ActedVersion.Show before loading layoutsToRebuild (+28). It captures that array and calls ForceRebuildLayoutImmediate for each occurrence, initializing LayoutRebuilder when needed. A Show callback replacing the field changes the chosen array; replacements during later class initialization or rebuild calls do not replace the captured traversal. Empty arrays complete, duplicate entries produce duplicate calls, and null array failure follows the completed Show. Null element handling is deliberately supplied by the layout gateway, not inferred for the engine implementation.

Hide calls StopAllCoroutines, then rereads acted, obtains its GameObject and requests SetActive(false). It does not itself clear blankText. The constructor is an aliased native tail call to the MonoBehaviour constructor, with no additional local field writes.

## Highlight color ABI and saved state

Highlight checks the highlight GameObject (+30), requests SetActive(true), then reads arrowImage (+38). Its virtual color getter uses **RCX return buffer, RDX Image receiver, R8 MethodInfo**, with slots298/2A0. The returned RAX points to the color bytes. Both the ordinary supplied buffer and a separate returned buffer are exercised.

The caller rereads arrowImage after the getter. At35DF88 it loads the returned color, and at35DF8B it stores those bits into savedArrowColor (+40). Only then does it test the reread arrow pointer and overwrite its stack buffer at35DF9C with the constant `(0,1,1,1)`. It passes that buffer to the virtual color setter via2A8/2B0. Consequently a getter callback clearing arrowImage still leaves the returned color saved before the null failure; replacing the arrow sends the constant to the replacement. The saved color is never accidentally taken from the overwritten local constant buffer.

UnHighlight requests SetActive(false), rereads arrowImage, and sends savedArrowColor through the same setter ABI. It does not query a new color. Fixtures preserve raw signed-zero, NaN-payload, subnormal and ordinary component bits in the saved value. No parent-space or engine color processing is modeled.

## Failure boundaries

Every reached baseline gateway failure is checked against its exact event/managed-state prefix. Additional fixtures cover null acted/blankText/GameObject/highlight/arrow/layouts, nullable strings, cold class initialization, alternate color return buffers, array capture and reference rereads. Setter effects are recorded only after successful gateways; native saved-color writes persist independently of later setter success. Normal returns verify stack balance and all eight Windows nonvolatile integer registers. Owner bytes outside the declared mutable fields remain unchanged.

Metadata is warmed; class initialization, virtual text/color methods, ActedVersion.Show, layout rebuild, string/log calls and engine object operations remain explicit services. No ActedVersion animation, delayed scheduler, live speech history or managed exception unwind behavior is claimed. No typed targets, shared Rust edits or live game actions are added.

Reproduce with `python scripts/audit_acted_surface.py GAME_ROOT DUMPER_ROOT --output REPORT` and the private Unicorn2.1.4 runtime through PYTHONPATH. Final 68-case rerun and Python compilation pass.
