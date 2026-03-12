# Loop Automation Notes

Step-by-step notes for perfecting the autonomous game loop.

## Pre-Interaction
- **Focus the game window first** before any mouse interaction. Hover highlights won't show and clicks may not register if the game isn't focused.
- `safe_click` handles this automatically — it detects if the game is unfocused and focuses it before proceeding. **Prefer `safe_click` over manual move+click.**
- For manual hover/click sequences, click a neutral area first to focus.

## Main Menu → Play Demo
- Template `menu_play_demo` finds "Play Demo" at ~(1280, 702) — accurate.
- After clicking Play Demo, lands on "Pick Game Mode" screen (Standard / Endless / Deckbuilding).

## Pick Game Mode → Standard
- Template `mode_standard` finds Standard card at ~(600, 445) — accurate.
- After clicking, game loads Village 1 with intro dialog: "There are N Evils among all characters. Find and Kill them."
- Dialog has a "Close" button to dismiss. Template `btn_close_dialog` at ~(1282, 822).
- Evil count and kill progress always visible in top-left panel ("Evils killed: 0/2").
- After closing dialog, the deck auto-opens showing all possible roles. Click anywhere to close deck view.
- **Use `python capture_deck.py <name>`** to get an enhanced crop of the deck view. Outputs both a full screenshot and a `_crop.png` with just the deck area — much easier to read role names from.
- Now uses template matching to find and click the purple deck icon (`icon_deck_purple`) instead of Tab key. Falls back to Tab if icon not found.
- Mouse auto-parks at (400, 780) before capturing so cursor doesn't obstruct cards.

## Reading the Deck
- The header under "CURRENT DECK" shows board counts: e.g. "Villagers 5, Outcasts 2, Minions 1, Demons 1".
- Same numbers always visible as icons in the top-right next to the purple deck icon (V, O, M, D order).
- These are the **actual board counts** (how many of each type are on the board), NOT the pool size.
- The deck view shows all possible roles (pool), which is larger than the board at Ascension 10+.
- Enter into solver: `python game_loop.py deck V=... O=... M=... D=... nv=<villager_count> no=<outcast_count>`
- Prefixes (V=, O=, M=, D=) are REQUIRED — positional args without them are silently ignored.

## Flipping Cards
- Click each card #1 through #N in order. No popup dismissal needed between flips.
- **Lilis night** triggers every 4 reveals — kills a random unrevealed card (red skull) and costs 2 HP. This interrupts the flip sequence with a death animation.
- After clicking all cards, **verify all are flipped**:
  - `python template_match.py find_all card_facedown` — finds any remaining unflipped cards (returns coords)
  - Dead cards (#5 here) show a red skull and won't match as facedown
  - If any facedown cards remain, click them to flip
- `detect_dead` in card_vision.py is unreliable — don't depend on it.
- **Visually check the screenshot for skull icons** to identify Lilis-killed cards. Dead cards have a red skull overlay and their role is unknown.
- Check "Evils killed:" in top-left after Lilis night to see if she killed any evils.

## Entering Card Info
- `python game_loop.py new <n_cards> <n_evil>` — start session
- `python game_loop.py deck V=... O=... M=... D=... nv=X no=Y` — enter full deck (include ALL roles from pool, not just board)
- Cards with **active abilities** (lightning bolt icon) like Jester, Fortune Teller, Druid, Slayer — enter as `card no_info <pos> <RoleName>` until ability is used.
- Cards with **passive info** (speech bubbles) like Oracle, Bard, Confessor, Alchemist — enter info immediately.
- `python game_loop.py night_kill <positions> <n_evil>` — for Lilis kills.
- `python game_loop.py set_hp <hp>` — update HP after Lilis nights or wrong executions.

## Executing a Card
- **First click the red execute sword button** (bottom-right) using template `btn_execute_sword`.
- **Then click the target card** to execute it.
- Do NOT click the card first — that just selects/views it. Execute button must come first.
- Screenshot after to see the result (evil role revealed, HP change, evils killed count).
- Feed result into solver: `python game_loop.py execute <pos> <evil_role>` or `execute <pos> good`.
- After execution, check: evils killed count (top-left), HP, and the executed card's revealed true role (shown on the card with red background if evil).

## Using Active Abilities
- Click the card with the ability icon to activate it.
- For Jester: "Pick 3 characters" prompt appears → click the 3 target cards → speech bubble shows result.
- Enter result: `python game_loop.py card jester <pos> <t1,t2,t3> <evil_count>`
- Solver can recommend abilities multiple times on the same card if it has uses left.
- **After using an active ability, immediately run `ability_used <pos>`** to tell the solver it's been consumed.
- **WARNING**: When clicking targets for an ability (e.g. Jester "Pick 3"), if you click a card that ALSO has an unused active ability, it will activate THAT card's ability instead of selecting it as a target. Be aware of which cards still have active abilities when picking targets.

## End of Game
- "Village is safe!" = WIN. All true roles are revealed on the board.
- Record the execution: `python game_loop.py execute <pos> <role>`
- Set final HP: `python game_loop.py set_hp <hp>`
- Log game over: `python game_loop.py game_over win/loss <name> "<pos=Role,...>" "[notes]"`
- This auto-saves a regression test and validates it.
- Screenshot the end screen to read true evil positions/roles.
- **Check for "<Corrupted>" tags** on any cards — record which positions were corrupted. Important for regression test accuracy.
- Click "Next" to proceed to next village or summary.
