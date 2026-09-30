# Native generic FromJson inside SavedGameData.Load

The shared reference-returning `FromJson<object>` body at `0x645DA0` now executes
inside the pinned native SavedGameData.Load caller, using the exact
`FromJson<SavedGameInfo>` MethodInfo slot. This extends the caller audit without
claiming runtime generic metadata discovery. Complete decode, exact method/type
bindings and 22 instruction assertions precede 48 cases, 64 controlled service
stops and four value-level native engine JSON joins.

The wrapper tests the MethodInfo's context pointer at `0x38`. When absent, it
requests System.Type metadata, rechecks the context and invokes its initializer
if still absent. It consumes context entries zero and eight separately: the
first supplies the type handle and the second supplies the return class. Native
code checks the System.Type class word at `0xE0` before requesting its initializer.
GetTypeFromHandle runs through an explicit service, followed by the non-generic
FromJson(string, Type) gateway.

After the JSON service returns, the wrapper rereads its context and return class.
If bit one at class offset `0x135` is clear, it requests class resolution even
when the JSON result is null. A null result then returns null without casting.
A nonnull result goes through the cast helper. A failed cast reaches the native
failure gateway before SavedGameData.Load can replace its existing save.

Fixtures vary caller metadata warmth, context publication, the System.Type
initialization word, return-class flag and successful/null/failed-cast outcomes.
Controlled stops retain complete baseline event prefixes and exact snapshots.
Normal returns check the stack and all eight nonvolatile registers. Four joins
replace the non-generic JSON service's field-processing outcome with the audited
native engine reader over an authored zero-initialized destination. Only public
values cross the emulator boundary; identities and private List versions do not.

Runtime context publication, type initialization/conversion, class resolution,
cast helpers, preference access and the non-generic gateway remain services.
Actual runtime object construction and native exception unwinding are open.
The independent engine gateway and field processor audits remain separate from
this value-level join. No live preferences are read or written, and this
framework wrapper adds no Assembly-CSharp classification.

```powershell
python reverse_engineering/scripts/audit_saved_game_generic_json.py GAME_ROOT DUMPER_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_saved_game_generic_json.json`. Private Unicorn
2.1.4 dependencies are required; native bytes and decompiled bodies remain private.
