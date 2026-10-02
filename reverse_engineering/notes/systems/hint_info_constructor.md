# HintInfo constructor argument capture and late stack loads

Build `f530404b0f3f_807de4a83df4`. The audit executes the complete HintInfo
constructor, `tdi5800.m0000` at `0x3bc200`, through its return at `0x3bc296`.
The verified next managed entry is `0x3bc2a0`; alignment padding is excluded.
All 40 decoded constructor instructions are reached. Fourteen operand
assertions pin the exact captured registers, stores, late stack loads, color
load, and base call. GameAssembly and the exact Dumper inputs are hash-pinned
through the frozen DeckCharacter surface verifier.

The actual base call at `0x33ed50` executes its folded `ret 0` instruction. It
is not replaced with a supplied constructor and is not attributed to another
folded managed alias. The five reference GC barriers at `0x2b6ff0` are named
supplied inert services. Their callbacks are explicit authored diagnostics,
not a claim that a game engine or GC admits these mutations.

Before the base call, the constructor captures text in RBX, hints in RSI, and
Image/Sprite reference in RDI. It stores and barriers the fields in exactly
this order: text at `0x18`, title at `0x10`, image at `0x30`, hints at `0x20`,
and flavor at `0x28`. Title is loaded from the caller stack only after the
first barrier returns; flavor is loaded only after the fourth barrier returns.
After the fifth barrier returns, it loads the caller's color pointer and
copies exactly 16 bytes into `0x38` using unaligned SSE loads/stores.

The caller stack supplies flavor at entry RSP+`0x28`, title at `0x30`, the
color pointer at `0x38`, and MethodInfo bits at `0x40`. The constructor never
reads the MethodInfo slot. Nullable register and stack reference arguments are
copied as raw QWORD values without dereferencing their String/Sprite objects.
The corpus exhausts all 32 nullable masks, four authored raw color bit vectors,
and distinct/aliased String pointers. Colors include signed zero, NaN payload,
infinities, subnormal and arbitrary bit patterns; no floating-point conversion
or normalization occurs.

Twenty-five callback cases place each title/flavor/color-pointer/color-data
replacement and early-text overwrite after each of the five barrier stages.
Only an early title-slot mutation affects the stored title; flavor changes
before its late load affect the flavor. Every barrier callback can affect the
later color pointer/data load. Overwriting text after its initial native store
persists. Five further callbacks clear the color pointer at each stage, causing
the exact color read fault after all five references have been stored. A null
owner faults at the initial text store with WRITE_UNMAPPED at `0x3bc22e`;
a null color pointer faults with READ_UNMAPPED at `0x3bc285`. Fault types are
checked separately, and the independent model validates every partial field.

Full RCX/RDX/R8/R9 values and exact native caller return addresses are recorded
at every barrier, including stopped services. The first barrier retains the
entry R8/R9 image/hints values; later barriers carry the explicit volatile
register poison from prior supplied returns. Each barrier must match its exact
decoded call site. Normal returns preserve stack position, all eight
nonvolatile integer registers, and XMM6–XMM15; supplied barrier returns poison
caller-saved integer and XMM registers.

The independent ordered model starts from the authored initial memory and
argument slots. It simulates native stores, reached callbacks, the late reads,
completed-barrier history, retained native-entry history, and color byte copy.
It checks each complete event snapshot and final byte against its expected
state, rather than using the observed final fields or current native color
buffer to generate expectations. A stopped service has its complete native
store prefix but no callback or completed-barrier effect. The separate native
retention check permits only reached store widths and exact callback ranges;
unused diagnostic objects, headers, and argument slots remain intact.

There are 288 cases (281 normal returns and seven exact faults), two retained
three-call sequences, five baselines, and 25 full stopped prefixes. Service
ordinals reset per invocation while physical object bytes, completed service
history, and native-entry history persist. Caller argument-slot rewrites for
retained invocations are explicit and occur before the next initial snapshot.
No allocator, real GC, UI renderer, runtime object admission, or native
exception unwinding is reconstructed.

All complete initial/final/service snapshots are pooled losslessly using
`sha256-full-authored-snapshot-v1`. Every hash is verified and expansion must
round-trip to the exact original report before output. Two independent final
producer processes emitted identical 6,537,770-byte reports, SHA-256
`6ee95fb63c703853f102ff05fce329890001c8a8751c5989dbc17ca62efa1683`.
The private peer is
`B:/CodexTools/DemonBluffReverseEngineering/artifacts/f530404b0f3f_807de4a83df4/hint_info_constructor.peer.private.json`.
Python syntax compilation, all 36 RE infrastructure tests, and diff checks
passed. The script, note and report are frozen for parent integration.

Reproduce with `audit_hint_info_constructor.py <game-root> <dumper-root>
--output <report>`, using the pinned private directories and setting PYTHONPATH
to the private python-emulation runtime. No native binary bytes are published.
