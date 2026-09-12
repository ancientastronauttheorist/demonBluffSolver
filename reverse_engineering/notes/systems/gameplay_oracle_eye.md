# Oracle-eye event forwarding

Pinned build `f530404b0f3f_807de4a83df4`. The two exact native Gameplay callers
have 144 passing fixtures and completed private baseline exports.

`HoverOverOracleEye` reads GameplayEvents.OnShowEyeOracleInfo (+0xB8 in static
fields); `HideOracleEyeInfo` reads OnHideEyeOracleInfo (+0xC0). With no installed
delegate each returns. With a delegate, the native caller forwards its +0x40
field in RCX and +0x28 field in RDX, then tail-jumps through its +0x18 invoke
pointer. The audit verifies these exact operands without equating an arbitrary
delegate field with a recovered subscriber identity.

Neither body reads its Gameplay receiver, card data, score, mode, eye state or
configuration. A null receiver therefore follows the same caller path. These
methods emit requests; they do not themselves establish that a tooltip appears
or disappears. Subscriber bodies and actual scene wiring remain separate.

Cold metadata resolution runs before reading the static event field. Its
failure prevents callback dispatch. A supplied callback failure occurs after
dispatch is attempted. Nonzero metadata bytes, including 255, skip resolution.
Fixtures exercise absent/present delegates, null/nonzero forwarded arguments,
null/nonzero receivers and both failures. All modeled heap bytes remain
unchanged under the explicit preserving-service contract. Normal returns check
the stack and eight nonvolatile integer registers.

The script pins GameAssembly and Dumper inputs, verifies both exact signatures,
GameplayEvents field names and its type slot, and checks eight native operand
relationships. Metadata and callback services are explicit; exception unwinding,
multicast execution and subscriber effects are not modeled. No live UI is used.

```powershell
python reverse_engineering/scripts/audit_gameplay_oracle_eye.py GAME_ROOT DUMPER_ROOT --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_gameplay_oracle_eye.json
```
