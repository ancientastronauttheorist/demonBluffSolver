# Native JSON string fields

The pinned engine audit executes the string field readers and writers selected
from core source-table slot `+0x180`, native metadata construction, UTF-16 to UTF-8
conversion, JSON tree construction/rendering and subsequent native reload. The
inventory contains 38 compact/pretty save/load cases, 13 reader cases, two mixed
inherited-object cases and 67 controlled service stops. Fifty-two exact
instruction/export assertions cover nine complete chained-unwind families and
the two nine-byte registry wrappers. Required conversion and field-handler
entries must actually execute.

The reader wrapper at `0xA91B50` dispatches to `0xA9BD70`. Native member lookup
and `0xAA6450` convert a found value to UTF-8 text; the exact runtime export
`il2cpp_string_new_wrapper` creates the supplied managed string. The writer
wrapper at `0xA93240` dispatches to `0xA9E7A0`, which calls `0x755EB0`. That
method obtains UTF-16 length and characters through independently bound exports
`il2cpp_string_length` and `il2cpp_string_chars`; actual native `0x4A8590` and
`0x32EA90` perform conversion. The UTF-8 then enters actual JSON construction
and rendering. Python supplies string objects and valid UTF-8 runtime creation,
but does not substitute for the native writer or UTF-16 conversion.

Observed behavior:

- Missing fields and differently cased names preserve the old string pointer.
  Duplicate names use the first value.
- A found JSON null, object or array produces an empty managed string. Boolean
  values become `true` or `false`, integers become decimal text, and the tested
  floating value `1.5` becomes `1.500000`.
- A null managed string saves as an empty JSON string and reloads as an allocated
  empty string. Nullness does not survive this round trip.
- Embedded NUL truncates both reading and writing at the first NUL. This happens
  in native C-string length walks, even though the parser and UTF-16 conversion
  can hold the complete lengthful value.
- Valid Unicode, supplementary characters and escaped control/quote/backslash
  text survive. Length cases cover the JSON node's 15-byte inline threshold,
  native string storage thresholds and the converter's 2,000-byte temporary
  allocation boundary (499 versus 500 UTF-16 code units).
- With the replacement-character singleton explicitly initialized to U+FFFD,
  native conversion replaces isolated surrogate code units. An unmatched high
  surrogate followed by an ordinary character consumes that next code unit as
  well, yielding one replacement character for the tested pair.
- Mixed parent/child string and numeric fields reload with native parent-first
  descriptor order in compact and pretty modes. The writer preserves the entire
  authored input object, and successful returns preserve the stack and all eight
  nonvolatile integer registers.

Writer-created short JSON strings have inline bytes and tag `0x700005`, with
length encoded as `15 - byte[15]`; longer strings use tag `0x300005` and pointer
storage. The audit snapshot decoder accounts for both representations. This
decoder observes the native tree; it does not construct it.

Windows TIB/TLS slots and already-initialized conversion state are explicit
fixtures. Runtime metadata, type classifiers, cache lookup, allocation, GC stores
and vector cleanup also remain supplied services. Each observed read/write
service boundary is independently stopped and checked against the baseline
prefix, including whether a managed field write preceded cleanup failure. These
stops do not claim native exception unwinding. Actual IL2CPP string allocation,
invalid UTF-8 runtime creation policy, real managed metadata discovery, compound
reference graphs and callbacks remain open.

```powershell
python reverse_engineering/scripts/audit_unity_json_strings.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_strings.json`. Private Unicorn
2.1.4 dependencies are required. No native bytes or decompiled bodies are retained;
this engine audit adds no Assembly-CSharp classification.
