# Engine preference entries and string conversion

Both shipped engine entries now execute offline: GetString at `0xF3150` and
TrySetSetString at `0xF22B0`. Chained unwind metadata resolves their complete
audited bodies through `0xF388A` and `0xF298B`, respectively. Native managed-string
conversion at `0x4A86B0`, UTF-16 conversion, stack probing and caller cleanup run.
The audit passes 380 cases and 42 controlled service stops.

The setter converts its key and value separately, requests the backend with a
byte-valued argument of one, and checks the backend byte at offset eight. A set
byte skips the setter and returns false. Otherwise it passes an explicit key
slice, type value three, the value bytes and a count including one final NUL.
Its returned Boolean is normalized from the backend's byte-sized result.

The getter converts both key and default, passes their explicit slices to the
supplied backend getter, then creates a managed string using the returned native
string's explicit byte count. Authored backend results include embedded NUL and
Unicode. They are fixture outcomes, not claims about missing-key storage policy.

The string helper uses a fast ASCII path for at most 24 UTF-16 units. Longer
strings or a non-ASCII unit enter the native conversion path; at 500 units the
2000-byte temporary buffer changes from stack to supplied heap allocation.
Unicode and supplementary characters survive. Malformed surrogate fixtures use
the native replacement behavior; an unmatched high surrogate consumes its next
code unit. Null managed inputs convert to empty slices.

Explicit lengths preserve embedded NULs through these entry interfaces and
through the supplied length-aware managed-string constructor. This differs from
the separately audited JSON field conversion's C-string truncation. Registry
key/value processing is still a distinct boundary and may have further rules.

Warm input combinations cover null, empty, short ASCII, non-ASCII, embedded NUL,
inline/heap boundaries, malformed UTF-16 and long keys. Native cleanup traverses
the allocator-manager flag-zero path with supplied allocation ownership and
release. Other allocator modes and TLS allocator ownership remain outside this
contract. Normal returns preserve source UTF-16 buffers, the stack and all eight
nonvolatile registers. Every service occurrence in two long-value baselines is
independently stopped with exact event-prefix and snapshot equality.

Backend acquisition/get/set, runtime string and GC exports, string copy/assign,
allocation/ownership/free and native exception unwinding remain explicit
boundaries. Windows TIB/TLS conversion state is supplied, as in the string-field
audit. No actual registry reads/writes occur. No Assembly-CSharp classification
is added, and native bodies/bytes remain private.

```powershell
python reverse_engineering/scripts/audit_unity_preferences_entries.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_preferences_entries.json`. Private
Unicorn 2.1.4 dependencies are required. The existing string audit's exact export
and conversion checks are reused alongside 24 new assertions for these entries.
