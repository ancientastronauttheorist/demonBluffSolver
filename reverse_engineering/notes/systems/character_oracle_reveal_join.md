# Character Oracle callers joined to RevealOrder

Pinned build `f530404b0f3f_807de4a83df4`. This separate offline audit executes
Character.OracleEyeActive (`tdi5487.m0030`, `0x3675A0..0x36778D`) and
Character.HideOracleInfo (`tdi5487.m0031`, `0x3654F0..0x36563D`) with their actual
RevealOrder.Init (`tdi5735.m0000`, `0x3A71C0..0x3A721D`) and RevealOrder.Hide
(`tdi5735.m0001`, `0x3A7190..0x3A71B7`) callees. It replaces only those two
whole-game-owned supplied services from the earlier Oracle audit. The earlier
source and report remain unchanged.

The two frozen audit verifiers independently pin extraction hashes, exact
managed declarations/signatures, following managed entries, unwind families,
complete decode and padding. Their thirty exact instruction assertions bind
both caller families. All 192 Oracle instructions and all 38 nontrap
RevealOrder instructions execute. The two terminal RevealOrder `int3`
instructions remain unexecuted: the supplied native null gateway stops first.
The union has 248 observed execution addresses, including supplied gateways;
that address count does not establish additional method implementations.

The final corpus has 182 cases, eight retained Oracle/Hide/Oracle sequences,
eight service-stop baselines and 88 complete exact stopped-prefix fixtures.
Stops compare the entire final snapshot to the original service-entry snapshot
and every earlier event to its complete baseline prefix. Baselines include cold
metadata/class state, physical GameObject aliases, replacement TMP dispatch,
a supplied null formatter result and native-null guard paths. Each normally
returned invocation verifies all eight integer nonvolatile registers, XMM6-15
and the exact restored stack pointer. Supplied services poison caller-saved
integer registers and XMM0-5; callers consume only the documented widths.

## One physical state across the join

The same Character's revealOrder field names the actual RevealOrder receiver.
The physical record's text field at `+0x20` is serialized separately from its
text value. Two TextMeshProUGUI component records (pinned TypeDefIndex 8974)
have distinct authored class records, virtual setter functions and MethodInfo
records. TMP_Text (9110) slot 66 binds class function `+0x558` and MethodInfo
`+0x560`. Component.get_gameObject is shared with the later CharacterView path;
receiver identity distinguishes its RevealOrder request. Decoded return
locations distinguish shared SetActive requests even when RevealOrder's
GameObject aliases Oracle's description, pick or view GameObject. This is one
physical active-state map; later requests can overwrite an earlier effect.

Snapshots retain Character, Acted, history List/backing array, ActedInfo,
CharacterData, Images and CharacterView diagnostics from the Oracle audit.
Additional snapshots retain RevealOrder, both TMP components, both class
records, both MethodInfo records, both opaque formatter-result records,
GameObject diagnostics, Image class/MethodInfo records and metadata/class
records. They preserve all bytes outside precise native saved-color writes,
supplied class-initializer DWORD updates and explicitly authored mutations.
Unconsumed sentinel memory and the Character's larger `0x200` window are
authored diagnostics, not complete valid runtime object layouts. Formatter
string values, TMP text values, speech, color and active-state bookkeeping are
independently supplied effects, not implementation claims for those services.

## Capture, reload and chronology

Only exact Hidden state DWORD 5 suppresses RevealOrder.Init. Hidden profiles
with a null reveal field, null RevealOrder GameObject or null text still return
without entering that callee. Every other tested state enters Init with the
zero-extended Character.order DWORD. Init saves only the low DWORD into its
caller stack slot; the authored upper `0x11223344` DWORD survives. The report
records the callee entry's relative stack offset and the full order slot at
service entries and after return. Signed minimum/maximum, negative, zero and
positive order bits remain exact.

Init obtains the RevealOrder GameObject, guards the returned pointer and enables
it with a DL-only write. It then captures the physical text field, calls supplied
Int32.ToString using the saved order slot and reloads that captured component's
class before virtual dispatch. Replacing text during the GameObject getter or
SetActive changes the captured receiver. Replacing/clearing text during
formatting leaves the earlier capture intact. Replacing its class during
formatting changes the actual setter function and MethodInfo. Clearing the
field before capture preserves activation and formatting, then stops at the
native text guard. A supplied null formatter result is legal and reaches the
setter as a null string. Changing Character.order during any of these services
leaves the saved order consumed by formatting unchanged; changing the saved
stack DWORD before formatting changes its exact input. Mutating the List count
inside the joined callee affects the Oracle caller's later signed-count branch.

Hide obtains/guards the same RevealOrder GameObject and tail-calls SetActive
with the complete zero RDX value. It never reads text or formats an order.
Text replacement, clearing and count/order mutations during its supplied
services preserve that distinction. The parent then continues through its
actual Acted/history/color/pick/view paths. Parent mutations retain captured
Acted receiver identity through GetItem, exercise later Acted/arrow/view
reloads, and retain every earlier request/native write when a later guard stops.
Signed count and uses, noncanonical picking/killed bytes, exact color DWORDs
and noncanonical supplied AL responses remain represented.

## Remaining explicit boundaries

Acted.Act, Character.GetCharacterBluffIfAble, CharacterView.AnimateIn,
AnimateOut and Init remain whole-game-owned supplied services. The folded
List.GetItem, runtime metadata/class operations, Unity Object/Component/
GameObject methods, Image color virtual calls, Int32 formatting and TMP virtual
setter implementations remain supplied. Their exact physical receiver,
arguments, results and callback effects are observed at the actual native call
sites. No constructor, rendering, gameplay event admission, scheduler,
DOTween implementation, native exception unwinding or live game access is
claimed. Native bytes and instruction bodies remain private on B:.

Executable: [audit_character_oracle_reveal_join.py](../../scripts/audit_character_oracle_reveal_join.py).
Report: [f530404b0f3f_807de4a83df4_character_oracle_reveal_join.json](../../reports/f530404b0f3f_807de4a83df4_character_oracle_reveal_join.json).

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_character_oracle_reveal_join.py GAME_ROOT DUMPER_ROOT --output REPORT
```

The saved report losslessly pools diagnostic byte windows only. Each `memory`
map preserves its physical field/storage name and points through `memory_sha256`
to the report's `memory_blobs` table. `expand_memory(report)` verifies every blob
hash and restores the original full snapshots. Each producer asserts exact
round-trip equality to its fully expanded native report after all stop/prefix
checks. No state, event, field or byte window is omitted.

Two successful independent final producers emitted identical 17,669,542-byte reports,
SHA-256 `e639f1ec5e6c08e95436329ae820af2bee3c889f2f5f130e686ce60ae4290726`. Python syntax compilation,
all 32 reverse-engineering tests and focused diff checks passed.
