# Generic real-role Character filter

Build `f530404b0f3f_807de4a83df4`. `scripts/audit_character_role_filter.py` and `reports/f530404b0f3f_807de4a83df4_character_role_filter.json` add **40 native fixtures** for the exact `Characters.FilterRealCharacterRole<object>` declaration at RVA6028C0. GameAssembly, Dumper script and declarations are hash-pinned. PE decode covers6028C0..602A92, including failure branches and the final throw trap, excluding alignment padding before the next verified managed entry602AA0. No new typed targets or project exports are needed.

## Exact generic boundary

The managed generic definition has no standalone RVA; Dumper reports its object instantiation. Native ABI is RCX Characters receiver, RDX input List<Character>, R8 MethodInfo; RAX returns List<Character>. The caller does not dereference its Characters receiver. A null receiver is used throughout the fixtures.

When MethodInfo+38 is empty, seven exact metadata requests precede generic-context initialization. Each referenced named token is verified before fixtures execute. The generic target is the first pointer in MethodInfo+38's context. Its readiness flag at type+135 bit0 gates runtime type preparation. This is generic type readiness, not Gameplay or Unity engine class initialization.

The caller allocates a new List<Character>, runs its constructor, checks the input for null, and obtains a versioned enumerator. For each occurrence it checks Character nonnull, reads Character.dataRef+50 and checks that object nonnull, then reads CharacterData.role+140. It calls the runtime type classifier against the supplied generic target. A match appends the **original Character**, preserving occurrence order and repeated Character references. It does not return Role or CharacterData objects, classify the Character itself, or read a bluff-role field.

The runtime classifier is an explicit fixture gateway: selected role identities match a supplied target type and null does not match. These fixtures do not reconstruct inheritance/interface classification or assert that role asset names determine assignability. The output Add and input enumerator are explicit managed collection gateways. The enumerator checks its captured source version on each MoveNext. A service mutation after either matching or nonmatching classification is detected at the next MoveNext; a matching occurrence already appended before that check remains in the output prefix. Source contents are unchanged in all fixtures.

## Failures and limits

The fixtures distinguish null input, null Character, null CharacterData, and null role. The first three stop the caller; a null role simply proceeds through the classifier and contributes no output. Cold metadata, generic-context/type preparation, allocation, constructor, enumerator, iteration, classification and Add failures preserve the exact preceding event/output prefix. Failure injection stops at the gateway; managed unwind internals are outside this audit.

A deliberately permissive null-allocation constructor fixture isolates the caller's own delayed output-null check: empty input and all-nonmatching input can return null, whereas the first match reaches the null throw before Add. This is **not** a claim that managed allocation or List construction actually succeeds with a null receiver.

Normal returns verify stack balance and all eight Windows nonvolatile integer registers. The shared enumerator Dispose executes its actual single `ret 0` instruction, preserving RAX. The audit neither models a complete collection runtime nor supplies scheduling, live game data or a Rust replay for generic type relations.

Reproduce with `python scripts/audit_character_role_filter.py GAME_ROOT DUMPER_ROOT --output REPORT`, using the private Unicorn2.1.4 runtime through PYTHONPATH.
