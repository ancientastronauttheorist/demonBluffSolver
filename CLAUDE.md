# Demon Bluff Solver

## Current Goal
Harden the solver. The automation loop works — vision, data entry, and game interaction are reliable. The focus now is on eliminating rule gaps, fixing bad heuristics, and handling edge cases so the solver wins consistently at high ascensions.

Priority order:
1. Use live runs to find rule gaps, wrong assumptions, and missing constraints in the solver.
2. Research each issue on the wiki, fix the code, and add regression tests.
3. Improve strategy decisions — ability timing, execution ordering, corruption-aware reasoning.
4. Increase win rate and push to higher ascension levels.

## Autonomous Workflow
**Always do what the solver recommends. No second-guessing, no manual overrides.** If the solver returns 0 scenarios or an error, stop and fix the problem immediately — do not guess, do not pick a random target, do not continue hoping for the best. Diagnose the root cause (bad data entry, missing constraint, wrong rule), fix the solver code, re-run, and only proceed when the solver gives a valid recommendation. (One-time exception: the empirical dead-card targeting test — see Empirical Tests.)

Claude should operate in this cycle:
1. Capture the current puzzle state from screenshots, hover text, compendium pages, and other in-game ground truth.
2. Enter or infer that state into the solver, then choose the best next action.
3. Act in game, observe the result, and update the state.
4. After each puzzle, log what happened and save regression coverage for anything learned.
5. **After every loss, stop and deeply analyze whether the solver could have done better.** Do not move on to the next game until this analysis is complete. Spawn an agent to examine the surviving scenarios at each critical decision point, check for missing constraints or deductions the solver should have made, and determine if the loss was genuinely unwinnable or if there's a solver improvement to make. Only proceed once you've either fixed the issue or confirmed the loss was unavoidable.
6. **Fix solver issues before the next game.** If the game exposed a bug, wrong assumption, missing rule, or bad heuristic — fix the code NOW, run regression, and verify the fix. Do not just log the issue and move on. Common triggers:
   - 0 scenarios (constraint bug or missing rule)
   - Wrong execution the solver was confident about (bad validation logic)
   - Solver couldn't narrow candidates (missing constraint or strategy gap)
   - New game mechanic or role interaction discovered

   **Research first:** Search the Demon Bluff wiki (https://demon-bluff.fandom.com) for the specific cards involved in the issue. Look for edge cases, interaction rules, and ability details that may not be in our knowledge base. Update `cards/` memory files and solver code with any new findings.
7. At the end of each completed game, commit and push the resulting code, test, and regression updates immediately. Do not batch multiple games into one commit.

### Lilis Night Handling
When the deck contains Lilis, night falls every 4 card reveals (kills 1 random unrevealed card + 2 HP damage). To correctly track which cards were flipped vs killed:
- Flip cards in batches of 4 (e.g., #1-#4, then #5-#8).
- **After each batch of 4 flips, wait 5 seconds** for the Lilis kill animation to complete, then take a screenshot.
- Read the screenshot to identify which card was killed (red skull overlay) vs which cards you successfully flipped.
- Enter the flipped card info and `night_kill` before continuing to the next batch.
- This ensures reveal_order and night_kill tracking stay accurate.

## Interaction Rules
- Use the mouse only for in-game interaction. Do not use keyboard shortcuts during live runs.
- To open the current deck, click the purple card icon in the top-right corner instead of using `Tab`.
- Prefer hover-and-screenshot verification before committing to a click when a UI target is ambiguous.
- For revealed-board clicks, prefer detected card-box centers from the current screenshot over the rough circle formula. The formula is only a fallback.
- Do not issue state-mutating `game_loop.py` commands in parallel. The session lock prevents file corruption, but it does not preserve command ordering. For `new`, `deck`, `card`, `execute`, `ability_used`, `slayer_result`, and similar updates, use one serialized command stream or a single in-process script.

## Mouse Parking
- Before taking screenshots, park the mouse in a neutral area so hover tooltips do not block card text or UI.
- Preferred parking spots on a 2560x1440 screen:
  - Board-center void: around `(1280, 690)` when no modal is open. This keeps the cursor off cards and side UI.
  - Lower-right dark margin: around `(2400, 1300)` when no settings tooltip is open.
- Avoid parking on:
  - the left status panel
  - the purple deck icon / top-right markers
  - the eye button on the right
  - the execute/settings cluster in the lower-right
  - any revealed or facedown card
- If a screenshot includes a hover panel, move to a parking spot and immediately retake it instead of trying to read through the obstruction.

## Step-by-Step Game Loop (a real game example)

### Pre-Interaction
- **Focus the game window first** before any mouse interaction. Hover highlights won't show and clicks may not register if the game isn't focused.
- `safe_click` handles this automatically — it detects if the game is unfocused and focuses it before proceeding. **Prefer `safe_click` over manual move+click.**
- For manual hover/click sequences, click a neutral area first to focus.

### 1. Start a New Game
- `safe_click menu_play_demo` → `safe_click mode_standard` → intro dialog appears.
- Intro dialog shows evil count. Dismiss with `safe_click btn_close_dialog`.
- Evil count and kill progress always visible in top-left panel ("Evils killed: 0/2").
- After closing dialog, the deck auto-opens.

### 2. Read the Deck
- Use `python capture_deck.py <name>` for an enhanced crop of the deck view (`_crop.png`).
- The header under "CURRENT DECK" shows board counts: e.g. "Villagers 5, Outcasts 2, Minions 1, Demons 1".
- Same numbers always visible as icons in the top-right next to the purple deck icon (V, O, M, D order).
- These are the **actual board counts**, NOT the pool size. The pool is larger at Ascension 10+.
- Close the deck by clicking anywhere (e.g. the mouse parking spot at 400, 780).

### 3. Enter Deck into Solver
- `python game_loop.py new <n_cards> <n_evil>`
- `python game_loop.py deck V=... O=... M=... D=... nv=<villager_count> no=<outcast_count>`
- Include ALL roles from the pool, not just the board. Prefixes (V=, O=, M=, D=) are REQUIRED.

### 4. Flip All Cards
- Use `detect_card_positions` or `all_game_card_coords(n)` to get coords.
- Click each card #1 through #N in order. No popup dismissal needed between flips.
- **Lilis night** triggers every 4 reveals — kills a random unrevealed card (red skull, 2 HP damage). This interrupts the flip sequence with a death animation.
- After clicking all cards, **verify all are flipped**:
  - `python template_match.py find_all card_facedown` — finds remaining unflipped cards.
  - **Visually check for skull icons** — dead cards (Lilis kills) have a red skull overlay, role unknown. `detect_dead` in card_vision.py is unreliable.
  - If any facedown cards remain, click them to flip.
- Check "Evils killed:" in top-left after Lilis night to see if she killed any evils.

### 5. Enter Card Info
- Cards with **active abilities** (lightning bolt icon) like Jester, Fortune Teller, Druid, Slayer — enter as `card no_info <pos> <RoleName>` until ability is used.
- Cards with **passive info** (speech bubbles) like Oracle, Bard, Confessor, Alchemist — enter info immediately with appropriate `card` command.
- `python game_loop.py night_kill <positions> <n_evil>` — for Lilis kills.
- `python game_loop.py set_hp <hp>` — update HP after Lilis nights or wrong executions.

### 6. Run Solver and Execute Recommendations
- `python game_loop.py next` — runs solver, gives strategy recommendation.
- **Always do what the solver recommends.**

#### Using Active Abilities
- Click the card with the ability icon to activate it.
- "Pick N characters" prompt appears → click the target cards → speech bubble shows result.
- **After using an active ability, immediately run `ability_used <pos>`** to tell the solver it's been consumed.
- Enter the result (e.g. `card jester <pos> <targets> <evil_count>`, `card druid <pos> <targets> none`).
- **WARNING**: When clicking targets for an ability, clicking a card with an unused active ability will activate THAT card's ability instead of selecting it as a target.

#### Executing a Card
- **First click the red execute sword button** (bottom-right) using `safe_click btn_execute_sword`.
- **Then click the target card** to execute it. Do NOT click the card first.
- Screenshot after to see the result (evil role revealed, HP change, evils killed count).
- Feed result into solver: `python game_loop.py execute <pos> <evil_role>` or `execute <pos> good`.

#### Loop
- After each ability use or execution, run `python game_loop.py next` again for the next recommendation.
- Repeat until all evils are executed or game over.

### 7. End of Game
- "Village is safe!" = WIN. All true roles are revealed on the board.
- Record the final execution: `python game_loop.py execute <pos> <role>`
- Set final HP: `python game_loop.py set_hp <hp>`
- Screenshot the end screen. Read true evil positions/roles and **check for "<Corrupted>" tags** on any cards.
- Log game over: `python game_loop.py game_over win/loss <name> "<pos=Role,...>" "[notes]"`
- This auto-saves a regression test and validates it.
- Click "Next" to proceed to next village.

### 8. Post-Game
- If loss or 0 scenarios: diagnose and fix solver immediately.
- Run regression: `python -m tests.test_regression`
- Commit and push.

## Empirical Tests (run when opportunity arises)
- **Can active abilities target dead/Lilis-killed cards?** E.g. can Fortune Teller pick a dead seat as one of its 2 targets? Wiki and forums have no answer. Next time we have an active ability and a dead card on the board, try targeting it and note whether the game allows it.

## Setup
- Screen: 2560x1440
- Python 3.13
- Dependencies: `mss`, `pyautogui`, `Pillow`
- `screenshot.py`: capture screenshots
- `mouse.py`: mouse control
- `card_vision.py`: card detection and template-based recognition
- `game_loop.py`: CLI/session bridge for solver state
- `solver.py`: constraint solver
- `strategy.py`: action selection
- `knowledge_base.py`: card role database

## Game Overview
- Demon Bluff is a puzzle and deduction game with a circle of face-down character cards.
- Reveal cards to get role info, deduce which characters are Evil, and execute them.
- Evil characters disguise themselves as Villagers and can lie.
- Good characters can also become corrupted, which makes their info unreliable without changing their apparent role.
- Win by executing all Evil characters before running out of health.
