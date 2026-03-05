# Demon Bluff Solver — Session Journal

## Session 30 — Village 1/7, Ascension 6 (LOSS)
- **Result**: Evils Win (ran out of health)
- **Score**: 410 (lost on restart)
- **HP**: 0/10 (started 10, wrong exec -5, Lilis damage -5)
- **Deck**: 8 cards, 3 evil (Puppeteer, Lilis) + Puppet
- **Layout**: #1=Puppeteer(disguised PD), #2=Puppet, #3=Lilis, #4=Wretch(killed by Lilis), #5=Knitter, #6=Jester, #7=Medium, #8=Enlightened
- **Key mistake**: Trusted fake PD ability at #1 (Puppeteer disguised as PD). PD said "#8 is Evil" — was a lie. Executed #8 (Enlightened, Good) = wrong exec, -5 HP. This was fatal.
- **Deduction after mistake**: Enlightened #8's info "Closest Evil is CW" pointed back at #1 (fake PD). Correctly identified #1=Puppeteer, #2=Puppet, then used Jester ability + Lilis night kill logic to find #3=Lilis. But no HP left.
- **Solver fixes**: Fixed Knitter validator to exclude executed (dead) positions from evil adjacency count.
- **Lessons**:
  1. NEVER trust a single revealed card's ability at face value — it could be Evil disguised as that role
  2. When deck shows an Outcast (PD) but Evil disguises as that Outcast, there's NO real copy — no corruption occurs
  3. Lilis damage (2/night) compounds with wrong exec cost (5) — one wrong move at Asc 6 is fatal
  4. Lilis night kill reveals who Lilis is NOT (can't self-kill) — powerful deduction tool

## Session 31 — Village 1/7, Ascension 6 (WIN with 1 wrong exec)
- **Result**: Village saved
- **Score**: 550 (running total 850)
- **HP**: 5/10 (wrong exec of #6 Knitter cost 5 HP)
- **Deck**: 8 cards, 2 evil (Witch, Baa). Villagers: Architect, Dreamer, Enlightened, Druid, Knitter, Medium, Alchemist. Outcasts: PD, Drunk.
- **Layout**: #1=Alchemist, #2=Enlightened, #3=Drunk(as Dreamer), #4=?, #5=Baa, #6=Knitter, #7=Medium, #8=Witch(as Druid)
- **Key deduction**: Medium #7 said "#3 is Drunk" but Druid #8 said "Drunk among #2,#4,#6". These contradicted — one was lying. Since #7-Evil scenarios all failed (Enlightened CCW constraint), #8 must be Evil. Then #5 + #8 satisfied all constraints.
- **Wrong exec**: Solver found only 1 scenario (#6+#7 evil) which was wrong because of Druid validator bug with unrevealed targets. Executed #6 (Good Knitter) = -5 HP. After fixing, manual deduction found the real evils.
- **Solver fixes**: Fixed Druid validator to handle unrevealed targets (unrevealed Good positions can plausibly be any Outcast).
- **Lessons**:
  1. Druid "found Outcast" on unrevealed targets needs special handling — the Outcast COULD be at any unrevealed Good position
  2. When Medium and Druid disagree, one is Evil — check which one's Evil theory produces valid scenarios
  3. Drunk mechanic: Drunk appears as a Villager (Dreamer), doesn't know it's Drunk. Medium reveals the true role.
  4. When solver has 0 scenarios, fall back on manual constraint-by-constraint analysis
  5. Manual deduction saved the game after solver bug led to wrong exec
