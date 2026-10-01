# Character hover and description-hide callers

Pinned build `f530404b0f3f_807de4a83df4`. This offline values-only audit executes
two complete Character bodies: OnHover (`tdi5487.m0054`, `0x3674F0..0x3674F8`)
and HideDescription (`tdi5487.m0062`, `0x365340..0x365458`). Exact signatures,
TypeSignatures, declarations, complete instruction ranges, unwind containment
and following managed entries/padding are asserted. The eight-byte hover leaf
has no unwind record. Ten operand assertions bind field reads, byte width,
loaded Action tokens, static reloads and returns. All 66 caller instructions
execute, including the native null-helper call and every normal return path.

OnHover writes hover to the canonical byte one and returns. It performs no
presentation or callback work. HideDescription tests the raw leftAct byte;
any nonzero value enters the left-Acted path. It captures leftActed before the
supplied StopAllCoroutines service, reads that captured object's ActedVersion,
obtains its GameObject and requests SetActive(false). It then reloads the main
Character.acteds and savedAct and calls the supplied Acted.Act with that exact
string reference, including null. Clearing Character.leftActed during the
stopping service does not replace the captured receiver. Clearing main acteds
after hiding stops at its native guard and retains the already-hidden object.

Both branches next load Characters.Instance and request the supplied
DisableHighlightAll caller. HideDescription then loads UIEvents.OnHideCustomHint
at static offset `0x98`, invokes its physical target/method pointer when present,
reloads UIEvents static storage, and loads OnHideHint at `0x40`. The second
Action can be cleared or replaced by the first callback. Both loaded delegate
identities are verified directly from native R8: shared target/MethodInfo values
do not identify which physical delegate was selected. Null Action targets and
targets aliased with the left Acted are explicit supplied callback probes.

No native store changes leftAct, savedAct or the main Acted pointer. Service
effects are separately authored and phase-labeled. Class-init DWORDs for
Characters/UIEvents are retained, even when supplied valid static storage has
a cold word; neither body initializes those classes. Every UIEvents static
slot retains its full pointer-width value. Normal calls check stack balance,
all eight integer nonvolatiles and XMM6–15 through the inherited native runner.

The corpus has 52 cases, including six native null stops, plus two explicitly
ordered hover/hide sequences and nine exact service-entry stop prefixes.
Null and mutation cases assert the expected return, missing later callbacks,
partial active state and restored speech. Every stop compares the complete
event/state prefix. Snapshots also retain an exact 0x200-byte authored Character
memory window. Its unused reference bytes contain diagnostic sentinels; this
does not establish valid runtime objects at those pointers or enlarge the
managed object's declared extent. Outside the exact hover store and authored
callback mutations, all bytes in that memory window are checked unchanged.

Acted.Act and Characters.DisableHighlightAll are explicitly game-owned supplied
boundaries with pinned metadata identities. Unity coroutine stopping, object
lookup/active changes, metadata resolution and both Action implementations are
also supplied. Their recorded effects are fixture bookkeeping. Actual renderer,
scheduler, scene/event readiness, service implementations and native exception
unwinding are outside this caller audit. No live game, registry or process
operation occurs; private native bytes remain off-repository.

Executable: [audit_character_description_hide.py](../../scripts/audit_character_description_hide.py).
Report: [f530404b0f3f_807de4a83df4_character_description_hide.json](../../reports/f530404b0f3f_807de4a83df4_character_description_hide.json).
Two successful independent native producers emitted identical 1,456,143-byte
reports, SHA-256 `787dabd8850d3f253c8bc481b31e4adf70472976405da3da532a63c9508fa7d4`.
The final source includes peer-reviewed loaded-delegate handling and explicit
null/mutation outcome checks.
