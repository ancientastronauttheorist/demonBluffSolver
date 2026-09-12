# ManageCharacters prefix before per-card initialization

Pinned build `f530404b0f3f_807de4a83df4`. `audit_manage_pool_prefix.py` executes
Characters.ManageCharacters36CE30 from its verified entry, stopping at
Character.Init365A20 **before any callee instruction**, or at36D01E before
publication setup when the supplied enumeration is empty. This is a prefix audit,
not a completed method invocation. There are44 passing fixtures,103 executed
instruction addresses and9 exact call/load relationships. GameAssembly/script
hashes, exact caller metadata, four immediate managed callees and the exact
GetEnumerator/MoveNext/get_Item/Dispose method-context bindings are checked.

Native order is UpdateCharacterPositions36E4E0, PickRoundBluffs36D3A0,
PickRoundDuplicates36D720, then enumeration of this.characters+20. A failure at
any supplied builder/layout gateway prevents the following calls. Cold metadata
failures precede all three. No Character.Init or later publication, action pass,
onSetup callback or shuffle is executed by these fixtures.

After a successful MoveNext, the current Character is read from the captured
enumerator. The caller rereads this.characters+20 and captures its signed Count
before the Mathf class-initialization gateway. Its first displayed-ID argument
is wrapping abs(current board Count minus index0), not the supplied data-roster
Count. The signed-minimum adversarial count remains0x80000000. It next checks the
supplied roster reference and calls get_Item(0), and only then checks the physical
Character reference. A combined empty-roster/null-Character case therefore fails
at indexed access first. Null CharacterData is forwarded to the Init boundary.
These fixtures establish the argument; they do not claim an ID field write.

Controlled board replacement during any of the three initial gateways changes
the list subsequently enumerated. Replacement during MoveNext leaves the first
Character from the old captured list but changes the reread Count used for its
ID. Nulling the field there fails before data access. Empty enumeration disposes
its enumerator and reaches the36D01E sentinel without touching a null supplied
roster; publication and its class initialization remain beyond this boundary.
Zero/negative count with nonempty enumeration are explicit adversarial service
configurations, not valid managed-list states.

Positioning and both builders are supplied before-effect gateways; successful
service markers retain their order, and every reached warm/cold baseline failure
matches its exact attempted-event/state prefix. These markers do not reconstruct
builder-internal partial writes: those remain covered by their own audits and
must be composed explicitly. Enumeration, indexed bounds, metadata and class
initialization are also supplied services. Owner bytes outside controlled board
reference replacement remain unchanged. Because execution stops inside the caller,
no full-return/nonvolatile-restoration claim is made. Native exception unwinding,
per-card Init, later callbacks and engine effects are outside scope.

Run `audit_manage_pool_prefix.py GAME_ROOT DUMPER_ROOT --output REPORT` with the
private Unicorn2.1.4 runtime. Native fixtures, Python compilation and scoped diff
checks pass. No Rust, proprietary bytes, decompiler bodies or shared inventory
changes are included.
