# Demon Bluff: solver-first plan

Date: 2026-10-02. Planning baseline: `0a43323a163a00eabdac083d9be862b39fb4d453`.

## Outcome and authority

The project owner clarified the goal: **"the real goal for both is to make as perfect a solver as possible."** The game-playing and solver-learning project remains the product; reverse engineering supplies missing rules and validation evidence.

Build an exact, explicitly scoped rules model, a sound deduction engine, and a strong policy for information gathering and actions. The solver should distinguish certainty, ambiguity, model gaps and compute limits. Complete Unity recreation or a byte-identical GameAssembly DLL is not an acceptance gate for that outcome.

This additive plan preserves the existing native audits, Rust reconstructions and [historical reconstruction roadmap](ROADMAP.md). It does not mark unresolved methods complete, change AGENTS.md, authorize live automation, or restart any task. The user authorized publication on `codex/full-decompile`.

Future GitHub publications must omit personal names, personal email addresses and identifying local paths, regardless of repository visibility. Check document content and commit metadata before publication.

## Existing foundation and immediate frontier

Use the existing Rust solver, Python observation/automation tools, scenario regressions and native-backed reconstruction work. The [README](README.md) and [roadmap](ROADMAP.md) document the pinned IL2CPP baseline, explicit service contracts and bounded comparisons. A method classified as unresolved is accounted for, not recovered. Keep whole-program accounting distinct from solver-ready behavior.

Preserve the concrete [pool-to-acquisition frontier](notes/systems/round_pool_acquisition_frontier.md): join actual `ManageCharacters` (`tdi5505.m0006`, `36CE30`) through actual Init occurrences (`tdi5487.m0025`, `365A20..365D73`) over retained actor/pool/continuation state. First stop before publication at `36D01E`; then connect Hidden refresh, first `DelayReveal` yield/resume, Act Init/Start and queue/Reveal admission. Separate supplied scheduling from established native scheduling behavior.

Start with the existing alias-sensitive example: board `[A,A]`, roster `[D0,D1]`, displayed IDs `[2,1]`, final data `D1`, and two distinct state-zero iterators retaining owner `A`. Preserve the first iterator across the second Init; add board replacement during the first Init and a stopped second callback. This closes a setup-to-observation blocker rather than merely accumulating unrelated leaf audits. Existing unfinished ActivatePick drafts remain drafts until independently validated; they are not prerequisites unless a named solver input/action requires them.

## Four independent claims

| Claim | Required evidence | Insufficient evidence |
| --- | --- | --- |
| Model fidelity | Exact admitted generation, role, clue, status and phase transitions agree with original evidence | Synthetic simulations agree with themselves |
| Information legality | Deduction and policy receive only the selected player's observation/history | Hidden identity or corruption state is readable in memory |
| Deduction/search correctness | Complete possible-world equality on tractable references; sound proofs or bounds on larger domains | A plausible unique answer, sampled worlds, or timed-out search |
| Practical policy strength | Held-out decision quality and win/loss outcomes under declared uncertainty and compute budgets | Native-method counts, fixture totals or individual wins |

Correct rules do not guarantee enough information to identify the actual world. Multiple worlds may be observationally equivalent. Exact deduction need not select a unique answer; optimal action policy depends on the objective, risk preference and justified probability model. A timeout or heuristic cannot establish optimality or impossibility.

## Build, assets and provenance

Initial scope is Steam app `3749680`, **Demon Bluff Playtest**, build `23084916`, Windows x86-64, Unity `2022.3.10f1`, IL2CPP metadata 29, build identity `f530404b0f3f_807de4a83df4`. Use the exact binary/metadata fingerprints in [build manifests](manifests/builds/) and the [toolchain lock](toolchain/toolchain.lock.json). The app ID corrects the original plan's `3749960` transcription against the build manifest and installed app manifest; the binary scope is unchanged.

Pin GameAssembly, global metadata, relevant Unity assets/serialized role and script data, mode/ascension configuration, parser version and solver commit for each corpus. Asset serialization semantics are evidence, not interchangeable with runtime field layouts. A new build needs an explicit manifest diff and rule triage before prior certificates carry over.

Each rule/fixture records its original method or asset identity, extraction/observation method, source hashes, chronological inputs/results, admitted domain and supplied services. Separate original-game observations, execution of original native bodies, independently authored references and synthetic fixtures. Keep proprietary binaries/assets/raw exports private while retaining reproducible public manifests and normalized evidence. Byte-identical report producers establish serialization reproducibility, not reconstruction of a byte-identical DLL.

## Fair information and operating modes

Define a player-observation history with timestamps/phase order: publicly exposed deck/role possibilities, revealed cards and clues, permitted ability results, HP, public execution/status feedback and prior actions. Validate that each extracted observation is actually visible or otherwise legitimately available at that point. Memory can transcribe an already visible clue only if the observation gate prevents premature or extra disclosure.

Privileged true roles, evil positions, hidden corruption/truth flags, unrevealed choices and RNG state/seed belong to an isolated validation oracle. They may explain a post-game mismatch or grade world-set inclusion, but may not narrow solver beliefs or choose an action. Test paired hidden worlds with identical permitted histories: planner inputs and policy distributions must remain identical under the same planner seed. Deduction must not patch an input until it fits privileged truth.

Open user choices: human-assist versus autonomous play; supported Standard/Ascension/deck scope; maximize village win, full-ascension win, expected HP or another declared objective; risk preference; and latency/memory limits. These choices are not implied authorization for live game control. Perfect-information diagnostics, if later requested, require separate labels and results. No numeric win-rate or performance target is agreed by this plan.

## Acceptance gates

These are proposed gates, not claims of achieved coverage. Declare the supported domain and frozen evaluation protocol first. An exact-domain promotion requires zero unexplained disagreements; unsupported cases remain visible in coverage accounting.

1. **Generation and transition fidelity.** Compare original-derived deck/pool generation, multiplicity, must-include consumption, ordering, RNG draws, character construction and Reveal/queue sequence. Cover truth/lie and corruption/disguise effects, active abilities, target legality, use limits, execution/protection/damage, death triggers, night/phase order, scoring and ascension modifiers. Include repeated identities, aliasing, retained callbacks, contradictory-looking but valid clues, and interactions across roles. Distinguish actually composed native behavior from supplied Unity, scheduler, RNG or subscriber contracts.
2. **Independent held-out differential corpus.** Freeze development and held-out partitions by role interaction, deck/mode and chronological scenario family before tuning. Compare original-game before/action/after observations and, in a separate oracle lane, hidden-state transitions. Use existing native-body fixtures for their admitted scope; they do not establish live scheduling beyond supplied boundaries. Label model-generated scenarios separately. Record attempted/admitted/excluded/mismatched cases, original provenance and ordered effects. After promoting a held-out failure into regression, retain a fresh held-out family.
3. **Complete possible worlds on tractable cases.** Implement an independent small-case enumerator from reviewed rules. Compare satisfiability and the complete canonical world set, not just one predicted evil position. Check soundness (no admitted impossible world) and completeness (no valid world omitted) at every observation prefix. Canonicalization must preserve behaviorally relevant identities, order and correlations. Under a correct admitted model the actual oracle world remains included, without exposing it to the solver.
4. **Ambiguity, contradictions and unsupported input.** Include deliberately ambiguous valid histories, genuinely impossible histories, missing/noisy observations and unsupported rule combinations. Exact emptiness may be reported only after complete search in a supported model; return an evidence-linked contradictory subset where feasible. A timeout is `incomplete`, an unsupported rule is `unsupported`, and multiple valid worlds are `ambiguous`, not parser failure or permission to guess a unique world. Corrections require observation evidence, not privileged answer-fitting.
5. **Probability and policy.** Uniform worlds are not an assumed prior. Derive generation/path weights, correlated choices and lie/clue likelihoods from reviewed rules; retain exact rational mass on tractable cases and account for failure/rejection mass. If likelihoods or priors are unknown, expose bounds or an explicitly selected robust policy instead of invented confidence. Compare information-gathering and execution policies with exhaustive belief-state search on small cases for the declared horizon/objective, including action costs and all legal abilities. Larger search reports `best_found`, budget, explored worlds/nodes and valid bounds if available. Certify optimality only with complete enumeration or sound bound closure.
6. **Deterministic chronology.** Fixed build, initial admitted state, action/resume schedule and recorded RNG stream reproduce the same transition and observation sequence. Keep planner RNG distinct from original RNG. Test draw ordering and retained state through repeated reveals, callbacks and phase transitions; do not reorder independent-looking events without proof. Fair-play decisions cannot consume a future recorded draw. Stochastic distribution checks need predeclared tolerances and sampling plans.
7. **Regression and mutation sensitivity.** Minimize original mismatches into regression cases. Deliberately change a truth/corruption rule, reveal ordering, multiplicity weight, ability limit or phase trigger and verify the corresponding test fails. Test extraction, chronology, deduction and policy independently. A model-only suite is necessary engineering coverage, not sufficient original-game validation.
8. **Practical performance and outcomes.** Compare a frozen baseline and candidate on the same held-out decks/modes/history families under the same fair-information policy and compute budget. Report satisfiability/world-set failures, unsupported decisions, calibration where probabilities are justified, objective returns, wins/losses and causes, p50/p95/worst latency and peak memory. Use uncertainty intervals for stochastic results and distinguish unavoidable ambiguity from rule, parser or policy defects. Select proposed numeric budgets and promotion thresholds before evaluation; do not infer them from whatever run passes.

## Milestones with usable end-to-end results

| Milestone | Deliverable and acceptance |
| --- | --- |
| S0: contract and blocker map | Freeze observation schema, build/assets, provisional objective and a reachable-subsystem matrix. Separate accounted/reconstructed/validated/solver-integrated states. Prioritize the existing Manage-to-Init frontier. |
| S1: generation through observation | Close the retained-state setup/Init join, then the queue/Reveal path needed by one declared deck/role domain. Compare complete chronological original traces and feed the resulting legal observation history to the existing solver. |
| S2: trustworthy deduction | Complete world-set agreement with an independent enumerator on tractable cases, including ambiguity, contradictions, correlation and unsupported-domain handling. Deliver useful human-readable deductions with evidence and uncertainty. |
| S3: information-aware decisions | Recommend legal abilities/reveals/executions from belief state, with exact small-case policy comparisons and explicit best-found status elsewhere. Evaluate actual objective improvement, not just world-count reduction. |
| S4: broader ascension play | Expand prioritized role interactions, modifiers, nights and scoring into complete village/ascension continuations. Each increment passes held-out differential gates and a fixed-budget baseline comparison before its supported scope grows. |

Coverage rows are reachable gameplay subsystems: observation/extraction, deck/pool generation, initialization/Reveal scheduling, role clues/truth/corruption, legal abilities/actions, execution/death/protection, night/phase transitions, and scoring/ascension progression. Each row carries a supported domain, evidence source, unresolved boundaries, solver integration state and held-out result. Method totals and test counts are supplemental, not the definition of solver completion.

## Work selection, stop rules and reporting

Before a reverse-engineering tranche, name the wrong or uncertain solver decision, reachable scenario, missing rule and exit test. Reuse existing proofs. Preserve low-level results as dependency evidence; pursue more allocator/string/constructor/UI work only when it closes a named input, transition or action-delivery blocker. Rendering fidelity alone is outside the solver priority queue.

Proposed stop rule: after one bounded tranche without a new supported scenario, resolved contradiction, reduced uncertainty or necessary dependency closure, pause and reprioritize. If supplied service boundaries prevent an end-to-end claim, report that obstruction and target its smallest solver-relevant contract. Do not rerun unchanged large suites or extend micro-audits merely to increase passing counts.

Use this progress report:

```text
Build/assets / solver commit / corpus version / information mode / objective:
Named decision blocker and reachable deck/phase/role scenario:
Before -> after supported behavior:
Original evidence; supplied runtime/scheduler boundaries:
Differential attempted / admitted / passed / failed / excluded by subsystem:
World-set soundness/completeness; ambiguity/contradiction/unsupported results:
Policy certificate or best-found budget; prior/likelihood assumptions:
Held-out outcomes and latency/memory versus frozen baseline:
New uncertainty, regressions and remaining exclusions:
Next end-to-end milestone; continue / stop / reprioritize rationale:
```

The measure of progress is increasingly reliable deduction and better decisions under legitimate information, with explicit guarantees and limits.
