# Gameplay relic insertion and generic rule lookup

Build `f530404b0f3f_807de4a83df4`. `scripts/audit_gameplay_relic_rules.py` and `reports/f530404b0f3f_807de4a83df4_gameplay_relic_rules.json` add **55 passing native caller fixtures**. GameAssembly, Dumper script and dump hashes are pinned; three exact declaration signatures and consumed field metadata are verified. The bodies are inspected directly from the PE. No new typed targets or live state are introduced.

## AddRelic mutation order

Gameplay.AddRelic (37B2A0) ignores its receiver and uses static Gameplay.CurrentRelics. After any required Gameplay class initialization, it checks the list reference, then increments the list's uint32 version. It obtains the backing array only after that increment. A missing backing array therefore fails with the version changed but count unchanged; a missing list or failed class initialization precedes the increment.

With spare capacity it increases count, stores the supplied relic reference at the previous count, then emits the write barrier. It accepts literal null and repeated references: there is no local uniqueness/type validation, score update, relic-trigger invocation or UI notification. At capacity it delegates to the exact generic growth helper with the version already incremented. The harness checks that MethodInfo's generic context supplies the growth helper argument. Growth is an explicit service; its injected failure preserves the earlier version increment and old count. Tests also cover wrapping version 0xFFFFFFFF to zero.

## Exact generic ABI

Dumper binds these instantiated bodies:

- `Gameplay.GetSpecialRuleIfAble<object>` at 638DE0 returns an Il2CppObject reference.
- `Gameplay.GetRuleOfType<__Il2CppFullySharedGenericType>` at 638C80 returns a SpecialRule reference.

Both receive the Gameplay receiver in RCX and their MethodInfo in RDX, with a reference result in RAX. The fully shared generic type in the second name does not introduce a hidden value-return buffer: its declared return type is SpecialRule, and the native register flow is asserted. The fixture supplies the MethodInfo runtime generic context at +38, whose first entry is the target runtime type, rather than assuming the baseline function's alias is a closed generic type.

## Ordered lookup

Both bodies read the receiver's specialRules list (+88), create its enumerator and process occurrences in order. A null list fails; null elements are passed through the type-classification gateway and do not match in the supplied classifier contract. Before classification, the target type is initialized if its metadata readiness bit requires it. An absent method generic context follows its own initialization service first.

GetRuleOfType returns the original list element as soon as type classification succeeds. GetSpecialRuleIfAble repeats type resolution/classification for that matching element as a cast; the normal fixture returns the same reference. An adversarial second classification failure takes the cast-failure path. Neither body invokes a rule's ability, enabled flag, condition, Execute method or score multiplier. `IfAble` in the method name is not an additional native eligibility test here.

Repeated compatible entries still return the first. The native immediate-return Dispose stub executes on normal completion and early match. The authored enumerator enforces a stable captured version on each MoveNext. Version change during an unmatched classification fails at the next MoveNext; a first-match early return performs no later MoveNext/version check. These explicit callback/version fixtures do not reconstruct actual engine mutation history.

## Boundary

Relic growth, generic-context/type/class initialization, instance classification and enumerator behavior remain explicit gateways. The fast append instructions and generic getter bodies execute natively; managed list internals, type assignability, arbitrary generic instantiations, exception unwinding and live relic/rule state remain separate. Successful returns verify the stack and all eight Windows nonvolatile integer registers. Complete range endpoints and seven exact instruction relationships bind the audit.

Reproduce with `python scripts/audit_gameplay_relic_rules.py GAME_ROOT DUMPER_ROOT --output REPORT`, using private Unicorn 2.1.4 through PYTHONPATH.
