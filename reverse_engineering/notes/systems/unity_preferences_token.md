# Unity preference provider: cold security-token predicate

Pinned build `f530404b0f3f_807de4a83df4`; UnityPlayer fingerprint is inherited
from the engine preference entry audit. This framework audit adds no
Assembly-CSharp method classification.

## Executed scope

`audit_unity_preferences_token.py` executes the complete native predicate at
UnityPlayer RVA `0x9E1D00`, including its five chained unwind chunks through
exclusive end `0x9E1E52`. The two normal returns are `0x9E1E3F` and `0x9E1E51`.
All eight Windows x64 nonvolatile registers and final stack restoration are
checked on every normal return. The native result is `AL`; upper `RAX` is not a
Boolean contract.

The synthetic report contains 87 individual cases, five cache/retry sequences,
17 exact controlled-stop prefixes, and 93 executed instruction/service
addresses. Windows services receive authored inputs; no real security token,
process handle, registry, or Windows API is accessed. Controlled stops are not
native exception unwinding.

## Native cache and discovery

The DWORD at RVA `0x1BD00A0` uses `0xFFFFFFFF` as the undiscovered sentinel.
Any other value skips discovery and returns whether that exact DWORD equals
`0x1000`. This is equality, not a lower/upper integrity-level comparison.

On a cold call the body first writes zero to the cache. It requests the current
process pseudo-handle, opens its token with access mask `8`, and requests token
information class `25`. A size query uses a null buffer and zero capacity. A
successful size query proceeds directly; a failed one proceeds only when its
first `GetLastError` result is `122`. In the other failure branch, a second
`GetLastError` call supplies the error retained for the final decision.

It allocates the authored size with `LocalAlloc(0x40, size)`, then performs a
second information query with that buffer and size. On success it reads the
first pointer in the returned buffer as the SID, calls
`GetSidSubAuthority(sid, 0)`, dereferences that returned DWORD pointer, and stores
the value in the cache. The subauthority index is exactly zero; this native body
does not query the SID subauthority count.

Microsoft's enumeration identifies class 25 as `TokenIntegrityLevel`, whose
output describes a mandatory label. This identifies the Windows API contract;
the executable audit supplies only the pointer/DWORD fields actually consumed
by this native body. See
[TOKEN_INFORMATION_CLASS](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ne-winnt-token_information_class),
[GetTokenInformation](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-gettokeninformation),
[GetSidSubAuthority](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-getsidsubauthority),
and [LocalAlloc](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-localalloc).

## Error and cleanup behavior

Open-token, size-query, allocation, and data-query failures capture a last-error
DWORD. After cleanup, a nonzero captured error restores the sentinel and returns
false. An authored zero last error leaves the cache at zero and returns false;
the next call then skips discovery. This distinction is executed in repeated
calls rather than inferred from the return value.

A nonnull token handle is closed even when the authored open operation failed
but wrote a handle. An allocated buffer is freed after both successful and
failed data queries. `CloseHandle` and `LocalFree` return values are ignored:
fixtures exercise their failures without changing the cached result. A
successfully obtained subauthority of `0xFFFFFFFF` itself leaves the sentinel,
so the next call discovers again.

BOOL and last-error results are consumed at 32-bit widths. Authored services
poison upper `RAX`; native `test eax,eax` and DWORD stores still behave exactly.
Negative/nonstandard nonzero BOOL values succeed. A value nonzero only above
bit 31 is a native failure. Size zero and 4096 are authored stress fixtures,
not claims that Windows normally returns those sizes for this information.

## Boundaries and reproduction

IAT bindings are checked against the pinned PE import directory:

| RVA | Library | Service |
| --- | --- | --- |
| `0x1825708` | KERNEL32.dll | GetCurrentProcess |
| `0x1825048` | ADVAPI32.dll | OpenProcessToken |
| `0x1825050` | ADVAPI32.dll | GetTokenInformation |
| `0x1825058` | ADVAPI32.dll | GetSidSubAuthority |
| `0x18253C8` | KERNEL32.dll | LocalAlloc |
| `0x1825298` | KERNEL32.dll | LocalFree |
| `0x1825798` | KERNEL32.dll | CloseHandle |
| `0x1825818` | KERNEL32.dll | GetLastError |

The audit checks all 13 native indirect call sites and 17 exact instruction
assertions in addition to the inherited preference-entry evidence. Invalid
SID pointers, service exceptions, actual Windows allocation semantics, and a
joined cold provider call remain outside this report. The provider audit can
consume the independently verified cache predicate without assuming any real
machine's token value.

`TokenServices` exposes `bind_token_services()`,
`initialize_token_services(options)`, `hook_token_service(address)`, and
`token_snapshot()` for a provider-hosted audit. The host supplies its existing
register, event, return, and snapshot helpers; the adapter resets only its own
service bookkeeping. Binding occupies synthetic callback offsets
`services + 0x300` through `services + 0x370`. The standalone report remained
identical after extraction of this reusable adapter.

```powershell
$env:PYTHONPATH='B:/CodexTools/DemonBluffReverseEngineering/python-emulation'
python reverse_engineering/scripts/audit_unity_preferences_token.py 'B:/SteamLibrary/steamapps/common/Demon Bluff Playtest' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_preferences_token.json
```

The committed report contains synthetic service inputs and summarized evidence,
not copied native bytes or decompiled method bodies. A separate process produced
the private peer report; the JSON values matched the committed report exactly.
