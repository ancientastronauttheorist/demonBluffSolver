# Original first-village onSetup subscriber binding

Pinned build `f530404b0f3f_807de4a83df4`. This is native-static and authored
asset-configuration evidence. The producer executes **zero native bodies**.
It does not establish actual Unity lifecycle invocation or admit any player
observation.

[Producer](../../scripts/audit_first_village_on_setup_binding.py) and
[report](../../reports/f530404b0f3f_807de4a83df4_first_village_on_setup_binding.json)
retain hashes, selected operand assertions, exact metadata joins and complete
selected serialized component fields. Complete native bytes/disassembly remain
in the private artifact workspace.

Every native fingerprint interval is checked against its PE section's raw
extent and a complete file read. Selected instructions come from a contiguous
linear decode starting at the verified managed entry; the report retains that
decoded prefix's end. This checks instruction boundaries without claiming
complete control-flow coverage. The first transform is derived from the
component's GameObject component list; local references and all ancestor
transform membership are checked, with cycle rejection.

The original `level0` `CharacterShuffleAnimation` component is path `137027`,
MonoScript `920`, an 84-byte object consumed in full. Its Characters reference
is `(file0,path137026)`, the original manager used by the retained N5 audits.
The component is authored enabled and belongs to GameObject `415` (`Characters`)
alongside manager `137026`. Its complete transform-parent chain reaches
Canvas `1529`, Content `98`, Gameplay `1654`, then Game `7`; all five authored
GameObjects have active-self set. The report binds component, script, GameObject
and transform identities and hashes. These flags establish scene configuration;
actual hydration, lifecycle order and surviving enabled state remain a named
Unity provider contract.

Original `OnEnable` (`0x363500`) reads its Characters field at `+0x20`, constructs
`System.Action` bound to this animation component and `Animates`, calls
`Delegate.Combine` (`0x116BCC0`) with the existing receiver delegate, and writes
Characters `onSetup` at `+0x58`. `OnDisable` (`0x363160`) constructs the equivalent
bound action, calls `Delegate.Remove` (`0x116E070`), and writes the result back.
The exact MethodInfo slot `0x2710138` joins to `Animates()` at `0x363060`.
Removal uses target/method identity, not equality of newly allocated delegate
object pointers. Combine/Remove implementation and preexisting invocation-list
contents are not executed here. Repeated enables, matching disables and unrelated
subscribers therefore cannot be flattened into an assumed singleton/null value.

The same lifecycle subscribes/removes `ShuffleCards` on
`GameplayEvents.OnNextChallenge`, `OnRestartCurrentLevel`, and `OnRestartGame`.
These later event invocations are excluded from the first-village setup trace.
`Characters.ManageCharacters` reads `onSetup` at `0x36D2DB` and invokes its
delegate before starting the manager's ShuffleDeck coroutine: the native call
loads target `delegate+0x40` into RCX, method context `delegate+0x28` into RDX,
then calls the code pointer at `delegate+0x18`.

The selected native first-state trace is non-inert:

1. `Animates` allocates an Animate iterator, captures the component, writes
   state zero and tail-calls StartCoroutine on the animation component.
2. Animate (`0x375130`) enumerates current card occurrences and resolves each
   `SingleCharacterDrawAnimation`. It sets that component's pivot local position
   to Vector3.zero, disposes the first enumeration, and starts a separate
   PlayAudioDelay iterator on the same animation component.
3. PlayAudioDelay (`0x375DB0`) invokes the AudioEvents action with integer `200`
   when present, then yields `.4f` (bits `3ECCCCCD`, exactly
   `0.4000000059604645` when promoted). The action's subscribers remain unknown.
4. Animate begins a fresh enumeration, selects its first card, requests
   DOLocalMoveY to `390.0f` with duration `.3f` (bits `3E99999A`), then yields
   `.05f` (bits `3D4CCCCD`, exactly `0.05000000074505806` when promoted), stores
   state one and returns true.

With a stable nonempty N5 board and the independently audited synchronous
first-yield coroutine service, the audio wait would precede the outer animation
wait. Both have the animation component as physical owner, separate from the
five card owners and manager. This predicts two additional admissions before
the manager ShuffleDeck admission; **no retained subscriber/engine join or
eight-record count is certified by this static audit**. The existing
[conditional-null six-admission witness](first_village_shuffle_admission.md)
remains a distinct supplied-binding case.

The selected first-state trace has no direct role, status or Gameplay-phase
write. Transform/tween/audio providers can still have effects, and this is not a
proof that arbitrary subscribers or later resumes preserve the solver state.
The animation is neither public clue publication nor a screenshot or legal
reveal-readiness certificate.

The smallest next join should invoke the original installer under explicit
Unity lifecycle/identity providers, retain its exact target/method delegate,
then execute Manage's actual subscriber dispatch through both first waits and
the subsequent manager ShuffleDeck admission. Preserve all five existing card
waits, actor/status/pool storage, shared native queue order and distinct owner
keys, with two records sharing the animation owner. Supply and label physical
GetComponent/pivot mappings, transform/tween services and AudioEvents state;
do not silently treat their callbacks as inert. Later drain/acquisition,
Gameplay startup/Day availability and reviewed UI captures remain separate
dependencies. Minion copying Confessor can subsequently be compared only after
that acquisition's original dispatch and status effects are retained; the
animation trace itself establishes neither lying/truth status nor visibility.

If the subscriber admissions use the same producer time/frame as the frozen
N5 witness, its `.05f` animation wait is eligible before the five `.3f` card
waits. A subsequent claim of original first-wait drain must therefore retain
the actual consumer selection and animation continuation, rather than skip
directly to Minion acquisition. This is conditional timing inference from the
exact native literals, not an executed drain or an assumed scheduling law.

Rerun using the existing private Python runtime on PYTHONPATH:

```powershell
$env:PYTHONPATH='B:\CodexTools\DemonBluffReverseEngineering\python-emulation;B:\Codex\DBclone\reverse_engineering\scripts'
python reverse_engineering/scripts/audit_first_village_on_setup_binding.py --game-root 'B:\SteamLibrary\steamapps\common\Demon Bluff Playtest' --dumper-root 'B:\CodexTools\DemonBluffReverseEngineering\artifacts\f530404b0f3f_807de4a83df4\il2cppdumper-v6.7.46' --output reverse_engineering/reports/f530404b0f3f_807de4a83df4_first_village_on_setup_binding.json
```

Validation: syntax passed. Final private producer `72027` and independent private
reproduction `73585` both exited zero; byte equality, producer-source hash and
the assigned report copy passed. Exact counters: one original component, five
ancestor GameObjects, seven native interval fingerprints, 41 operand assertions,
nine metadata joins, four exact float literals, zero executed native bodies.
