# Shared constructors and reference getters

This audit closes the direct native behavior of 503 concrete managed definitions
across four folded bodies. Every method retains its metadata identity. The
selection excludes generic-shared bindings and constructors that enter Unity's
MonoBehaviour initialization; neither can be inferred from these bodies.

| Native entry | Direct definitions | Observed body effect |
| --- | ---: | --- |
| `0x357920` | 152 | Empty base-constructor tail chain; no object writes |
| `0x3CFFF0` | 37 | Another empty constructor tail chain through `0x357920` |
| `0x357700` | 104 | Empty base call, then one four-byte argument store at receiver `+0x10` |
| `0x353580` | 210 | Return the reference at receiver `+0x18` |

The first two bodies clear the native metadata-argument register before the
tail transfer. Their receiver remains unchanged, including when the direct body
is invoked with null. This does not establish null behavior at managed call
sites, allocation semantics or runtime class initialization.

The four-byte constructor body serves 102 iterator state constructors plus
`AlchemistRuntimeData.cures` and `EnlightenedRuntimeData.direction`. Each
definition's declared parameter matches its field type. The direction enum's
underlying `int` is independently checked in the pinned dump. A shared RVA alone
would not justify calling every field an iterator state.

The reference getter serves 204 generic/non-generic iterator Current definitions
and six other properties: CustomOptionValue Label, IdsList Format, collider,
transform, current animation and TailSegment ChildBone. Every field is matched
by offset and type in its declaring metadata. The getter returns the stored
reference without inspecting its target, changing the object or advancing an
iterator. In particular, DelayReveal Current does not call MoveNext or Reveal.
Current may remain null or retain an earlier yielded object according to writes
elsewhere; this body alone does not establish those writes.

## Verification

[`audit_shared_scaffolding.py`](../../scripts/audit_shared_scaffolding.py)
first reruns the shared-return prerequisite, which regenerates and byte-compares
the complete method denominator and pins the native/metadata inputs. It then
verifies the exact instruction relationships of these four entries and their
shared return, and executes 96 isolated cases per entry (384 total).

The execution cases vary all non-stack general registers, all vector registers,
object bytes, stack bytes and incoming arguments, including signed boundaries
and nonzero high argument bits. The four-byte store truncates to the low 32 bits.
They verify exact object and stack effects, return control, nonvolatile-register
restoration, vector preservation and getter read addresses. Constructor stack
saves and the base-call return address are explicitly accounted for. Arithmetic
flags after constructor execution are unspecified; the getter preserves flags.
The field-accessing cases use mapped receivers; null exception translation is
outside this audit. No host DLL executes.

The [report](../../reports/f530404b0f3f_807de4a83df4_shared_scaffolding_audit.json)
lists every definition, its behavior and relevant metadata field. Existing 49
classifications are retained; 454 explicit classifications are added. Coverage
becomes 1,160 methods backed by 280 evidence records, with the original 4,207
method denominator unchanged. This does not classify complete iterators,
allocation routines, classes or game features.

```powershell
python reverse_engineering/scripts/audit_shared_scaffolding.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_shared_scaffolding_audit.json
```
