# RevealOrder Init and Hide native caller audit

The pinned build's complete two `RevealOrder` caller bodies execute offline in
`scripts/audit_reveal_order_presentation.py`. The report is
`reports/f530404b0f3f_807de4a83df4_reveal_order_presentation.json`.
Native PE bytes and the inspection decode remain private on B:. The report
contains authored fixture storage, service observations and scalar projections.
No game process, renderer, actual Unity services or scheduler is accessed.

## Exact targets and storage

The extraction-pinned Dumper declarations and direct native bindings establish:

| Inventory ID | Method | RVA | Complete decode end, exclusive | Next managed entry |
|---|---|---|---|---|
| `tdi5735.m0000` | `RevealOrder.Init(int order)` | `0x3A71C0` | `0x3A721D` | `0x3A7220` |
| `tdi5735.m0001` | `RevealOrder.Hide()` | `0x3A7190` | `0x3A71B7` | `0x3A71C0` |

Each range includes the final exception-path `int3` and excludes subsequent
alignment padding. The producer verifies the associated unwind chunks,
complete instruction decode, next metadata entry, padding, exact signatures
and 14 selected instruction/operand assertions. There are 40 instructions in
total; 38 execute across the corpus. Only the two terminal `int3` instructions
remain unexecuted because the supplied native exception gateway stops first.

`RevealOrder : MonoBehaviour`, TypeDefIndex 5735, declares exactly one custom
field: `TextMeshProUGUI text` at `+0x20`. The producer also pins
`TextMeshProUGUI : TMP_Text` (8974), `TMP_Text` (9110), and its virtual
`set_text(string)` slot 66 declaration/RVA `0x1BE7620`. The caller uses class
function `+0x558` and MethodInfo `+0x560`. Two authored class/function/MethodInfo
records allow the audit to distinguish replacement virtual dispatch.

Every snapshot preserves the complete authored RevealOrder, both text records,
both class records, GameObject, both opaque formatter-result records and both
MethodInfo records. Their unconsumed sentinel bytes are diagnostic authored
memory, not evidence of valid runtime headers or complete Unity object layouts.
The supplied formatting and TMP services own their separately recorded values;
the callers do not inspect string contents. An eight-byte caller stack sentinel
proves Init writes only the low order DWORD and retains its upper DWORD. Hide
retains that whole slot.

## Caller behavior and ABI

`Hide` calls `Component.get_gameObject(this, MethodInfo=0)`, guards the returned
GameObject against null, then tail-calls `GameObject.SetActive(game, false,
MethodInfo=0)`. It never reads the `text` field. The Boolean argument is produced
by clearing EDX, so the complete RDX register is zero.

`Init` first saves the low 32 bits of its order argument in its caller stack
slot. It calls the same GameObject getter, guards the returned object, and calls
`SetActive(game, true, MethodInfo=0)`. Its `mov dl,1` preserves the upper bits of
RDX returned by the supplied getter; fixtures poison those bits and assert the
complete outgoing register as well as its low Boolean byte.

After activation, Init loads and retains the physical `text` pointer. It then
calls `System.Int32.ToString` RVA `0x1117320` with RCX pointing to the saved
four-byte order and RDX MethodInfo zero. The exact Dumper signature is retained
in the report; the decoded call site establishes this address-of-value ABI.
Only after formatting returns does Init guard its retained text pointer and
dispatch the setter through that text object's current class. The formatter's
returned physical pointer, including an authored null result, is passed through
unchanged. No managed string implementation or culture policy is inferred.

Normal services return through a helper which poisons all six volatile integer
argument/scratch registers and XMM0–5. Each successful call verifies Win64 stack
restoration, RBX/RBP/RSI/RDI/R12–15 and XMM6–15 retention. The two caller bodies
contain no literal slots, metadata/class initialization gates or DOTween calls.

## Fixtures and partial effects

The corpus contains 50 normal/edge/mutation cases, two retained four-call
Init/Hide sequences, two normal baselines and six independently stopped service
prefixes. Signed order boundaries cover 0, 1, `0x7FFFFFFF`, `0x80000000`,
`0x80000003` and `0xFFFFFFFF` while the input register carries nonzero upper
bits. Supplied decimal results are authored from those signed values, and both
formatter-result identities and a null result are exercised.

The 50 cases include nine native null-guard stops and one native unmapped null
owner dereference, with the latter pinned to `0x3A71E5`. Null GameObjects stop
before activation. A null text field still permits activation and formatting
before the text guard stops. Hide accepts a null text field because it does not
read it. A null owner can pass through the explicitly permissive supplied getter
when that service returns a GameObject: Hide completes, whereas Init activates
the object then faults when loading the owner's text field. This is a caller
observation under supplied getter behavior, not a claim about Unity's actual
null-receiver policy.

Phase-qualified supplied mutations replace or clear the owner's text field,
replace the captured text object's class, and change the saved order DWORD.
Getter/activation changes to `text` affect the subsequent native load. Formatter
changes to the owner's field leave the already captured receiver intact, while
formatter changes to its class affect the subsequent native virtual lookup.
Mutating the saved order before the formatter changes its consumed value;
mutating it inside or after the formatter does not retroactively change the
captured formatting observation. Aliased text classes preserve shared physical
dispatch. The retained sequences preserve service bookkeeping across repeated
calls and leave the same physical caller/component storage in place.

Every stopped prefix compares its complete event list, including exact arguments
and pre-effect snapshots, against the corresponding normal baseline prefix.
Its final snapshot must equal the selected stopped service-entry snapshot,
the last event in that baseline prefix. All
native caller storage remains byte-identical except explicitly authored
mutation offsets; supplied activation/text effects remain separate projections.

## Reproduction and limits

The final repository and private producers both exited successfully and emitted
identical 2,944,924-byte reports. SHA-256:
`9fcd48180ddd485db7a6f28442cb8967d0aaf4e7945957881aea67f66eb06ace`.
Python syntax compilation and all 32 reverse-engineering infrastructure tests
passed. No Cargo build, Ghidra export or shared coverage update was performed by
this worker.

```powershell
$env:PYTHONPATH='B:\CodexTools\DemonBluffReverseEngineering\python-emulation'
python reverse_engineering/scripts/audit_reveal_order_presentation.py `
  --game-root 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output 'reverse_engineering/reports/f530404b0f3f_807de4a83df4_reveal_order_presentation.json'
```

Component lookup, SetActive, Int32 formatting, virtual TMP text effects and the
exception gateway are supplied named boundaries. The constructor is omitted;
component references are declared serialized/runtime fixture inputs. This audit
closes the two standalone RevealOrder caller bodies without claiming an actual
Oracle-to-RevealOrder joined execution, renderer effects or service internals.
