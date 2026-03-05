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
