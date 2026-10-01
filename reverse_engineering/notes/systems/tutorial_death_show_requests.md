# Death tutorial MoveNext request boundary

`audit_tutorial_death_show_requests.py` extends the frozen death-generator native
machine only with observations. It executes the actual two MoveNext bodies and
the actual Show caller at `0x38E1A0`; it supplies no replacement Show function.
The inherited audit pins the generated declarations, signatures, pointer entries,
complete unwind ranges, constructors and call operands. The input matrix covers
48 normal profiles: warmed/cold metadata, initialized/uninitialized Gameplay
class, Gameplay states 10/50/60 and four ordered status lists including duplicates.
Each routine receives two explicit resumes. No engine scheduler executes.

The new report observes every Show entry's actual RCX controller, low-DWORD
tutorial type, R8 pivot and R9 MethodInfo. Across the matrix there are 48 Show
entries, requesting KilledCharacter 100 or Poison 45 with MethodInfo zero.
The caller's complete routine state/current/capture projection is recorded at
each entry. Return observations are phase-gated: Poison's shared return address
also executes after a false membership result, so a return observation requires
an earlier pending Show entry.

The report records physical initial and final actor (0x1B8), status container
(0x80), active/resistance Lists (0x28 each), backing arrays (header plus full
capacity), Gameplay class (0x180), static block (0x100), icon/pivot Transform
records (0x80 each), controller (0x40), routines (0x80 each) and allocated waits
(0x80 each). Unconsumed Character bytes remain opaque; their sentinel values
are not decoded as unused typed references. The controller's represented
0..0x40 bytes are equal at entry and return for every empty-note call. This
does not establish general Show inertness or cover larger UI object graphs.

Two independent final producers completed successfully with byte-identical
reports: 4,223,282 bytes, SHA-256
`f543784be22cd236136af62ab5eb230b378f97e600cbbad6c143085ca6fc0477`.
The report is values-only; private native instructions remain outside the repo.

`tutorial_death_generators.rs` is a guarded offline caller replay, not a save/UI
implementation. It preserves full represented physical records and exact
32-bit states/class words/status values, 8-bit metadata flags, 64-bit identities
and 0x3E4CCCCD wait bits. The two generators retain reversed physical capture
fields. Summary 50 gates before actor/status/icon use; Poison uses native
Corrupted 10 membership before requesting a transform or Show. Wait current
references remain after completion. Caller requests are checked against
independently supplied class-initialization, membership, Unity-transform and
normal Show acceptance outcomes. These outcomes promise represented caller
storage retention, and do not establish tutorial display or persistence.

Unsupported null physical inputs, failures, class-state mutation, mutation by
downstream Show, missing/unused outcomes, implicit resumes, incorrect nominal
identities and malformed/capacity-invalid Lists are rejected. Complete physical
shape/aggregate budgets run without allocation before identity maps or clones:
128 initially retained objects, 16 routines, 32 resumes, 4096 aggregate backing slots and
1 MiB aggregate retained snapshot work including future waits. Tests compare all
48 native profiles, full final physical bytes, waits, class state, metadata,
manual resume state/current/captures and observed Show ABI. Additional tests
cover completion retention, nominal aliases, signed negative List count,
unsupported outcomes and aggregate work limits. The replay is declared as an offline API; live automation integration remains separate.
