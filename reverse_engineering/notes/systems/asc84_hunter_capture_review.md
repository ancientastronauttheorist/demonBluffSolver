# Archived Hunter UI capture review: partial development evidence

The [reproducer](../../scripts/audit_hunter_capture_review.py) and
[normalized review](../../reports/asc84_hunter_capture_review_v1.json) bind two
private JPEG captures by SHA-256 and original 2560x1440 dimensions. Both images
were visually inspected before authoring the semantic observations. The script
checks that identity and reproduces the authored review; it does not perform
OCR, inspect different images, or certify player-history admission. Image pixels
remain in the private screenshot workspace.

## Public facts and unavailable facts

Both captures display ten numbered board positions, an objective to execute
three Evil characters, subtitle `2 Minions and 1 Demon`, zero of three Evils
killed, village one of seven, ascension 84, score zero and HP 10. The visible HUD
counts read `[6,1,2,1]` from left to right. Numbered positions run clockwise
`[10,1,2,3,4,5,6,7,8,9]` from the top. This is observed screen geometry, not the
native current-list orientation or a proof of underlying actor identity.

The first catalogued capture shows all board cards face down. The other shows
the cards face up and apparent Hunter #3 with the visible words
`I am 1 card away from closest Evil`. The visible line layout is three lines:
`I am 1 card`, `away from`, `closest Evil`. Visual wrapping does not prove the
managed string's newline encoding. No numbered targets appear in that speech.
Its geometric adjacent positions are #2 and #4; their sorted public labels do
not establish the native ActedInfo reference order.

Only five deck-strip labels are readable: Doppelganger, Chancellor, Minion,
Pooka and Plague Doctor. Neither capture establishes the full public role
occurrence multiset or how often an absent label occurs. Header counts do not
substitute for those identities. Wrong-execution cost and phase are unavailable
from the inspected pixels and stay unknown.

Catalog order and descriptive filenames do not establish actual action order,
single-card reveal chronology, absence of intervening callbacks/Nights, or
clock/frame samples. No capture-time binary/asset fingerprint is recorded in
these images. The current pinned build is a candidate scope, not a verified
binding of these archived captures. JPEG hashes prove the reviewed files are
unchanged; they do not prove that missing build binding.

## Admission outcome and next gate

The record is public-pixel development evidence and remains **unadmitted** as
`PlayerHistory`. It reads no archived true-role dictionary or native memory to
fill missing facts. The ten-card mixed-role board is outside the four/five-seat
[conditional Hunter/Baa domain](hunter_capture_admission.md), independently of
the missing build, deck and chronology provenance.

The [strict boundary](player_history_boundary.md) requires a trusted review of
each exact event and prefix. This specimen does not justify registration of a
complete synthetic history, acceptance of memory-only target order, or a
claim that native text publication guarantees visibility. Use a build-bound
capture family exposing the complete role multiset and verified reveal/action
prefixes before integrating actual captures with the independent world-set
comparison. Native-produced text and references stay separately labelled until
their public availability is established.

Reproduce with the two exact private images present:

```text
python reverse_engineering/scripts/audit_hunter_capture_review.py screenshots --output reverse_engineering/reports/asc84_hunter_capture_review_v1.json
```

This records two reviewed captures, unresolved build binding and no admitted
history. There was no live control, policy evaluation, generation certificate
or independently held-out outcome.
