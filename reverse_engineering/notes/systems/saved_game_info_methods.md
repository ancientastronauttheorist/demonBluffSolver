# SavedGameInfo native mutation and construction

All five pinned `SavedGameInfo` methods now execute offline: `AddTutorial`,
`AddCharacter`, both clear methods and the constructor. The audit checks exact
Dumper signatures, six metadata/literal bindings, complete unwind ranges and
25 instruction assertions. It passes 132 cases, 20 controlled service stops and
ten value-level joins to the independently audited native JSON pipeline.

Both add methods check membership first. A duplicate leaves the List unchanged.
For a new value, native code increments the version before either a supplied
resize/append or its own inline append. The inline path increments size and
stores the element before the write barrier. The resize service supplies growth;
the audit does not infer the runtime growth algorithm from its fixture.

Both clear methods increment the version, including when already empty, and
set size to zero before requesting array clearing. Version arithmetic wraps at
32 bits. The constructor stores the exact key `Tutorials`, then allocates,
constructs and publishes two distinct empty Lists in order. Existing fields are
overwritten. Null List receivers reach the native null-reference gateway.

Warm/cold fixtures cover duplicates, null strings, empty/nonempty Lists, spare
capacity and version wrapping. Every service occurrence in four representative
calls is stopped independently; the report retains the exact event prefix and
snapshot, including stored references and unpublished allocations. These stops
do not model native managed exception unwinding. Normal returns check the stack
and all eight nonvolatile registers.

Contains, resize/append, Array.Clear, allocation, List construction, metadata
initialization and write barriers remain explicit services. The constructor's
folded base tail is verified directly rather than named from decompiler aliases.
No preference access or persistence occurs in these five methods.

The JSON join transfers the exact three public values once into the audited
engine pipeline and reloads them. It does not transfer object identities or
private List versions between the two emulators. Actual persistence callers and
runtime metadata/allocation remain separate boundaries.

```powershell
python reverse_engineering/scripts/audit_saved_game_info_methods.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_saved_game_info_methods.json`. Private Unicorn
2.1.4 dependencies are required. No native bytes or decompiled bodies are kept.
