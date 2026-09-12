# Shipped Unity clock settings

The shipped `TimeManager` lives in `Demon Bluff_Data/globalgamemanagers`, class
ID 5, path ID 8. Its complete serialized payload is 16 bytes: four binary32
scalars in this order. The companion `globalgamemanagers.assets` contains no
TimeManager object.

| Serialized name | Exact stored value | Binary32 bits | Native clock offset |
| --- | --- | --- | --- |
| Fixed Timestep | 0.019999999552965164 | `3ca3d70a` | `+0x48` |
| Maximum Allowed Timestep | 0.3333333432674408 | `3eaaaaab` | `+0x100` |
| m_TimeScale | 1.0 | `3f800000` | `+0xfc` |
| Maximum Particle Timestep | 0.029999999329447746 | `3cf5c28f` | `+0x104` |

These correspond to fixedDeltaTime, maximumDeltaTime, timeScale and
maximumParticleDeltaTime. They are serialized configuration values, not an
observation of the running game's current clock.

## Evidence and reproducibility

`reverse_engineering/scripts/audit_unityplayer_clock_assets.py` pins the complete
source files and verifies the sole TimeManager identity:

- `globalgamemanagers`: 202,560 bytes, SHA-256
  `6bd99988279019eb3190f40940b1ff2242ed28a728be22b9c48a26f118198c32`.
- `globalgamemanagers.assets`: 705,652 bytes, SHA-256
  `38e56ef97cda4c5dfcfcb397f8077335aebbf199460fa5b536d91698c21ed2d9`.
- Native mappings use pinned UnityPlayer SHA-256
  `b5d48235e7cc02ff9496fb33a07d5921adfc4b40ded1bc64c96a7a7c10b4dfb2`.

UnityPy 1.25.0 reads the Unity 2022.3.10f1 object using its versioned fallback
schema; the asset does not embed a TimeManager type tree. The audit explicitly
checks this provenance, requests a complete checked read, verifies the reader
ends exactly after the 16-byte object, and independently unpacks all four raw
scalars with exact bit assertions. There is no omitted MonoBehaviour tail or
assumed object header.

Two independent native virtual transfer implementations pair the same names
with the same member addresses. The descriptor transfer at
`0x552210..0x552358` passes each member pointer, field name and `float` type
name to its helper and marks each scalar as four bytes. The named-reader
transfer at `0x552360..0x552443` passes each corresponding member address and
field name to its scalar helper. The audit verifies the LEA targets, pointed-to
strings, field offsets, calls, vtable pointers and complete instruction
boundaries. The descriptor ends in a tail jump, while the named reader ends in
a return; trailing alignment bytes are excluded. Native bodies remain private.

The authored JSON report is
`reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_clock_assets_audit.json`.
Run with the local pinned reverse-engineering dependencies on `PYTHONPATH`:

```powershell
python reverse_engineering/scripts/audit_unityplayer_clock_assets.py 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\Demon Bluff_Data' --unityplayer 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_clock_assets_audit.json
python -m py_compile reverse_engineering/scripts/audit_unityplayer_clock_assets.py
```

## Remaining boundary

The native transfer helpers are not executed by this audit. This establishes
serialized field provenance and static member bindings, not the complete
configuration loading path or runtime initialization. Constructor, load,
normalization, reciprocal refresh and later public setters remain distinct
operations. In particular, these settings do not prove when the engine fills
constructor-preserved fields, calls the two normalization/refresh virtuals, or
whether later game/runtime writers replace the shipped values. The asset
fixture must not be silently substituted for a captured live clock state.
