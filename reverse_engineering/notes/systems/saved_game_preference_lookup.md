# Preference request-to-registration join

The exact preference request literals now join their shipped bare-name
registrations through ten native GameAssembly lookup fixtures. The existing
registration-loop/export-sink audit runs first, and all 3,447 pointer/name pairs
are decoded with file-backed name and executable-target validation.

| Exact request | Registration index | UnityPlayer target RVA |
| --- | ---: | --- |
| `UnityEngine.PlayerPrefs::GetString(System.String,System.String)` | 2172 | `0xF3150` |
| `UnityEngine.PlayerPrefs::TrySetSetString(System.String,System.String)` | 2169 | `0xF22B0` |

Both registration names omit parameter signatures. The repeated `Set` appears
in both the request and registration. Native lookup tries the full request first,
then strips at the opening parenthesis and traverses the map again. A supplied
exact-signature registration takes precedence over the supplied bare entry.
Bare requests resolve directly; empty maps and wrong prefixes return zero.
Each fixture asserts the exact resolved pointer, complete constructed-string
sequence and normal-return ABI.

Map entries and target pointer values are authored fixtures derived from verified
registrations; the engine targets are not invoked. Runtime map population,
resolver failure/exception policy, preference marshalling and platform storage
remain open. Next entries for offline engine-body work are `0xF3150` and
`0xF22B0`; verify their complete unwind/chained ranges before export or emulation.

The shared JSON gateway audit accepts additional lookup fixtures. Its default
report reruns identically, and the preference report reruns independently.
No preference access, native bytes or decompiled bodies are retained.

```powershell
python reverse_engineering/scripts/audit_saved_game_preference_lookup.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_saved_game_preference_lookup.json`. Private
Unicorn 2.1.4 dependencies are required. No Assembly-CSharp classification is added.
