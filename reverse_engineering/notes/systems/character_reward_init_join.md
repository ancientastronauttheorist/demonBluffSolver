# Character reward selection, initialization and real presentation

Build `f530404b0f3f_807de4a83df4`. This audit executes actual Character.SetupObject,
InitReward and RevealReal in one retained physical graph. Frozen standalone
producers supply exact declaration, type, field, enum and instruction pins;
their source and reports are unchanged. InitReward's final native jump now
enters actual RevealReal rather than the earlier supplied whole-method boundary.

| Method | Exact identity | Verified native range | Instructions |
| --- | --- | --- | --- |
| SetupObject | `tdi5487.m0019` | `3689D0..368A44` | 26 |
| InitReward | `tdi5487.m0027` | `365640..365712` | 50 including terminal trap |
| RevealReal | `tdi5487.m0044` | `3682A0..36840E` | 87 nontraps |

Ranges are exclusive at the right edge. SetupObject is a verified leaf without
an unwind record; its next managed entry is `368A50`. InitReward has the exact
unwind range through `365712`, next managed entry `365720`, and terminal int3
at `365711`. RevealReal's unwind range ends at `36840F`, including terminal
int3 at `36840E`, before the next managed entry `368410`. Both terminal traps
are explicitly excluded from execution; trailing alignment is excluded.
All 162 nontrap instructions execute, across 178 native and supplied gateway
addresses. There are 20 InitReward and 23 presentation operand assertions.

SetupObject selects the exact Up 10, Left 20, Down 30 or Right 40 Acted
reference, stores it at actor+`A8`, and tails the supplied reference barrier.
Right additionally writes byte 1 at `B0`; other sides retain the prior byte.
Unknown low DWORD sides leave the actor untouched, including a null actor.
Recognized null-actor side loads retain their exact native read fault sites.

InitReward captures incoming CharacterData in RDI and Acted once from actor+`A8`.
The supplied Component getter returns an explicit GameObject, which is guarded
and deactivated before any reward pointer stores. Getter callbacks may replace
or clear actor.Acted or change its later mapping; the current deactivation still
uses the already returned GameObject. Fixtures include distinct and shared
GameObjects and an alternate or null supplied getter result.

Actual native stores then clear bluff+`58`, store captured input at dataRef+`50`,
clear registerAs+`60`, and set pickableUses+`DC` to 1. Each pointer store precedes
its own supplied GC barrier. Null input stops only after these effects; alignment,
previous/current state and Action callback remain untouched. Nonnull input's
current startingAlignment DWORD at `134` is read after the third barrier, from
the original captured input even if a callback has replaced actor.dataRef.
Current state+`E4` is copied to prevState+`E0`, current onStateChange+`180` is
loaded, and state becomes Hidden 5. Late third-barrier changes to input alignment,
state or Action therefore take effect at their respective native loads.

The actual indirect Action call passes captured delegate.method_code+`40` in
RCX and delegate.method+`28` in RDX via invoke_impl+`18`. Its unused managed
target+`20` remains a separate diagnostic record. R8/R9 preserve this machine's
full volatile poison bits. Nullable Action and nullable method_code fixtures
retain their distinct meanings. Callback-entry state already contains Hidden,
copied alignment/previous state and uses 1.

Actual RevealReal follows the InitReward tail jump with the same actor. A state
callback replacing dataRef or chName therefore changes the native presentation
loads. Alignment remains the earlier captured input's value. Compound fixtures
change input alignment before and after its copy, replace name before Action and
data during Action, or repair an initially absent Action at the third barrier.
The complete actual tail-entry register history is retained.

RevealReal preserves its standalone chronology: name receiver captured before
String.ToUpper, null uppercase replaced by the pinned empty String, TMP class
loaded after uppercase, and data/name reloaded after text for the exact 16 color
bytes. Art Sprite and art-type queries have separate data reloads; only EAX's
low DWORD reaches SetupArt. Background Sprite is captured before class setup
and Unity inequality. A false AL suppresses later data/image reads; a true AL
uses later current data/background and image receiver. The final actual jump
enters supplied Character.UpdateViewReal `3694D0`.

The graph retains full diagnostic bytes for 45 named records, including the
512-byte actor/Object class, 384-byte data records, 1536-byte authored TMP classes
with exact slot 66 text and slot 23 color functions/MethodInfos, Acted, Strings,
Sprites, Images, GameObjects, Action, code/method tokens and unused managed target.
New records occupy a distinct allocation range; physical identities and raw
addresses are serialized. This diagnostic graph does not establish engine or
managed-object admission. Aliases include shared Acted sides, shared GameObjects,
and an authored shared TMP/background component.

An independent ordered semantic model begins solely from authored initial storage
and options. It predicts every full service-entry snapshot, RCX/RDX/R8/R9,
native call/jump site and exact return target, every reached native store,
completed supplied effect, targeted callback, metadata/class gate, partial fault
and final state. Guard register residues are derived independently. Each supplied
return poisons caller-saved integer/XMM registers; normal returns preserve the
stack, eight nonvolatile integer registers and XMM6 through XMM15.

Byte retention grants only exact reached native pointer/DWORD/byte store ranges,
exact reached callback writes, and completed class E0 DWORD initialization.
Metadata slots retain their exact physical values; the metadata flag changes
only after both required metadata services complete. Shared GC entries qualify
callbacks by decoded call site and invocation phase. Failed services have no
effect, callback or completed request; earlier native stores and completed
effects remain in the stopped full snapshot.

The final corpus contains 144 cases: 128 normal returns and 16 native stops.
It independently varies metadata/class warmth, captured input identity, nullable
Action, raw state/alignment DWORDs, pointer guards, aliases, getter outputs and
mutations at getter, SetActive, all three barriers, Action and presentation
services. Ten retained sequences preserve one graph, including eight four-call
side-to-reward chains, recovery after a null-input partial initialization, and
aliased sides with Acted replacement during the getter. Each next initial state
equals the preceding final state. Nine normal baselines produce 141 exact full
stopped prefixes; every stopped event list and final snapshot equal the selected
baseline prefix and service-entry state.

Supplied boundaries remain GC/reference barriers, metadata and class setup,
Component.get_gameObject, GameObject.SetActive, Action callback, String.ToUpper,
TMP text/color setters, CharacterData.GetArt/GetArtType, Character.SetupArt,
Unity inequality, Image.set_sprite and Character.UpdateViewReal. No real
renderer, scheduler, runtime admission, native unwind or acquisition interleaving
is inferred from this bounded composition.

The report losslessly pools full raw memory with `sha256-authored-memory-hex-v1`
before full snapshots with `sha256-full-authored-snapshot-v1`. Expand full
snapshots first with `audit_report_snapshots.expand_snapshots`, then memory with
`audit_character_oracle_reveal_join.expand_memory`. Both codecs assert exact
round trips; no fields or diagnostic bytes are dropped. The final report has
173 memory blobs and 1730 full snapshot blobs.

Two independently launched, individually syntax-preceded final native producers
emitted identical 13,458,950-byte reports, SHA-256
`aee96fd63042a35751269f380a01f9be13750ac554de98f37b0aad4b6d9dda4f`.
Private peer:
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/character_reward_init_join.peer.private.json`.
All 36 reverse-engineering infrastructure tests and diff checks passed. Script,
note and report are frozen for parent integration.

Run `reverse_engineering/scripts/audit_character_reward_init_join.py` with the
pinned game and Dumper directories as positional arguments and `--output`.
PYTHONPATH must include the private python-emulation directory.
