# Bounded inventory for unresolved wait phase eight

Phase bit `8` remains unresolved. A reproducible search of the pinned
UnityPlayer found no additional phase-eight dispatcher in the explicit forms
below. This is a bounded negative inventory, not a claim that phase eight is
unused or that indirect dispatch has been exhausted.

`audit_unityplayer_wait_phase8_inventory.py` searches 25,310,208 file-backed
executable bytes in UnityPlayer SHA-256
`b5d48235e7cc02ff9496fb33a07d5921adfc4b40ded1bc64c96a7a7c10b4dfb2`:

| Search surface | Result |
| --- | --- |
| REX.W RIP-relative MOV loads from wait global `0x1c6e720` | 23, all verified on instruction boundaries |
| Raw relative `E8`/`E9` branches targeting concrete consumer `0x43bd90` | 0 |
| Raw memory-indirect call/jump encodings with displacement `+0xb8`, including SIB forms | 162 |
| Such branches verified by decoding from an enclosing unwind entry | 155 |
| Additional branches verified from previously bound PlayerLoop leaf entries | 3 |
| Remaining raw candidates without a verified entry boundary | 4 |
| Verified branches with a preceding literal `mov edx, 8` in the same decoded range | 0 |

The four unverified raw candidates are `0x304a63`, `0x6d0bcb`, `0xefaa40` and
`0x1467e7f`. These refer to the raw `FF` opcode locations; any instruction prefix
would precede that address. They are not treated as proven instructions or
wait-manager calls. For the three known leaf branches, the full instruction
addresses including their prefix are used in the verified inventory.

The script independently rechecks the five known global-load, mask, vtable and
slot-call relationships on decoded boundaries:

| Wait dispatch instruction | Mask |
| --- | --- |
| `0x59f692` | 16 |
| `0x5b7b0c` | 32 |
| `0x5b7cef` | 4 |
| `0x5b7d38` | 1 |
| `0x5b7d5f` | 2 |

The mask-two callback binds two default PlayerLoop nodes, as established by the
separate phase audit; this table counts distinct native dispatch instructions.
The fixed callback is decoded through both its tail jump and conditional return.

A focused follow-up checks every memory operand with displacement `+0xb8`
after each of the 23 global loads, through that load's verified chunk or leaf
range. The only five occurrences are the known immediate call/jump sites above;
there is no intervening register-loaded `+0xb8` target in those ranges. This
narrows a plausible alias pattern without claiming inter-chunk or whole-function
dataflow. The report also records each global load's destination register and
first subsequent call/jump within the range for further bounded follow-up.

One unrelated unwind chunk, `0x107aa50..0x10874cc`, does not fully decode with
the installed Capstone: its contiguous decoded prefix ends at `0x108711a`.
The report records this explicitly. Verified candidate instructions in that
chunk occur within the successfully decoded prefix; no skipped bytes are used
to restart decoding or invent instruction boundaries. Unwind chunks are not
asserted to be complete method boundaries.

## Scope and reproduction

The syntactic absence of `mov edx, 8` does not exclude computed masks, masks
supplied by callers or other register operations. A mask can also include bit
8 together with other bits. The arbitrary-object slot scan does not identify
all its receivers as wait managers. Alternate global addressing, copied global
aliases, register-loaded targets outside the inspected ranges, indirect calls
and inter-chunk control flow remain open. No live PlayerLoop or startup order
is inferred, and this audit makes no scheduler admission change.

With the local pinned reverse-engineering dependencies on `PYTHONPATH`:

```powershell
python reverse_engineering/scripts/audit_unityplayer_wait_phase8_inventory.py 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest\UnityPlayer.dll' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_unity_wait_phase8_inventory.json
python -m py_compile reverse_engineering/scripts/audit_unityplayer_wait_phase8_inventory.py
```

The authored report contains counts, addresses, verified known relationships and
explicit exclusions. It contains no native method bodies. Ghidra was not opened.
