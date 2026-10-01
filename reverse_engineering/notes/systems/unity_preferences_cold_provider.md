# Cold token discovery joined to native preference storage

One offline emulator now executes cold security-token discovery, path selection,
provider acquisition/retry and the native preference entries/getter/setter. The
audit passes 25 provider cases, four entry joins and 99 controlled stops,
executing 1,730 distinct instruction/service addresses.

Successful discovery of SID subauthority zero equal to `0x1000` selects the
`Software\AppDataLow\Software\` prefix. Other successful values and authored
discovery failures select `Software\`. This is exact equality and preserves
the independent predicate's cache/error policy. No real machine's security token
or registry state is read.

The provider's handle recovery does not clear the token cache. After a successful
discovery, repeated `0x3FA` handle probes rebuild the path and both handles while
skipping further token API calls. A failed discovery with nonzero captured error
leaves the sentinel, so a later handle retry discovers again; zero captured error
leaves zero and skips discovery. Matching configuration can skip path construction
entirely, leaving an undiscovered token cache untouched.

Both read and write entries run through cold acquisition with distinct authored
registry handles. Their actual converted values, registry interfaces and caller
cleanup execute. Every supplied-service occurrence in the two composed baselines
has exact stopped-prefix/snapshot checks; normal returns preserve native
configuration headers/payloads, managed inputs, the stack and all eight
nonvolatile registers. Token cleanup history records API attempts, including
authored failures, as in the standalone token audit.

Windows security, registry and UTF-8 API results are supplied services. Runtime
configuration fields, metadata/runtime exports, memory primitives, assignment,
allocation and ownership remain explicit inputs/boundaries. Native exception
unwinding, real configuration initialization and actual OS access remain open.
No Assembly-CSharp classification is added.

```powershell
python reverse_engineering/scripts/audit_unity_preferences_cold_provider.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_preferences_cold_provider.json`. The reusable
`TokenServices` adapter binds only authored API callbacks; it does not replace
the native predicate. Independent private and repository report runs match.
