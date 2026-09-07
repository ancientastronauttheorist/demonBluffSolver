# Shared void return body

The regenerated 4,207-method Assembly-CSharp denominator contains 175 concrete
void declarations with one direct native binding to body `ga:rva:0033ED50`.
The pinned body consists of one immediate return with no extra argument-stack
cleanup. It makes no calls, reads no object fields and performs no memory writes.
This is a complete native-body observation, not a naming-based assumption that
every lifecycle hook or iterator Dispose method is empty.

Six definitions already had role-specific classifications. This checkpoint
explicitly classifies the remaining 169 as `understood`, preserving all existing
classifications. Every managed identity remains in the denominator and in the
report; sharing one native body does not collapse 175 methods into one method.
The complete public definition list is in the
[audit report](../../reports/f530404b0f3f_807de4a83df4_folded_return_audit.json).

## Relevant groups

The observed definitions include:

- Iterator `Dispose` entries for delayed Reveal, delayed Demon kill, clue and
  interference delays, game initialization, shuffling, UI and bundled examples.
- Empty base Role hooks and concrete Villager, Minion, Demon, Mutant,
  SaintVillager and Skinwalker Act entries.
- Selected mode, debug, menu, animation and pointer-event hooks.
- One compiled static constructor and several empty delegate bodies.

These are the exact selected methods, not all methods on their declaring types.
For example, `Character.<DelayReveal>d__84.Dispose` has no native rollback
code: invoking this entry alone cannot undo a prior role clone, Reveal effect
or latch write. This does not establish whether a particular engine stop path
invokes Dispose, how it releases references, or how the iterator's other methods
behave. Those remain separate call-chain and lifetime questions.

Similarly, an empty base Role Act does not make derived overrides empty, and an
empty static-constructor body does not remove the runtime's class-initialization
bookkeeping. Native-body classification does not classify the whole feature.

## Independent verification

[`audit_folded_return.py`](../../scripts/audit_folded_return.py) first rebuilds
the complete metadata denominator from the pinned Dumper outputs and native
image, verifies both generated files byte-for-byte, and checks the metadata
fingerprint. It selects only concrete definitions with one direct binding to
this exact body and requires a void return type for every selected declaration.
The executable-section check and native instruction check fail closed.

The script then executes the pinned instruction in isolated Unicorn memory
under 64 deterministic register/stack/object states. Each case verifies:

- all 15 non-stack general registers and all 16 XMM registers are unchanged;
- arithmetic/control flags and authored stack/heap bytes are unchanged; and
- return control reaches the supplied address and the stack pointer advances
  exactly eight bytes for the return address.

Mapped and null receivers both reach the direct return body without an object
access. This says nothing about null checks or exceptions in callers. No host
DLL executes, and the report emits no native bytes or decompiler bodies.

```powershell
python reverse_engineering/scripts/audit_folded_return.py `
  'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' `
  --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' `
  --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_folded_return_audit.json
```

Coverage becomes 706 classified methods backed by 279 evidence records. The
denominator remains 4,207 methods, 3,066 native bodies and 107 shared groups.
No live automation or solver behavior changes in this audit-only checkpoint.
