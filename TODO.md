# Solver TODOs

## Wiki Audit: All 40 Cards vs Solver

Systematically verify every card in the knowledge base against the wiki (demonbluff.wiki.gg). For each of the 40 cards:

1. Read the wiki page for the role
2. Compare ability description against `knowledge_base.py`
3. Check the validator function in `solver.py` handles all documented edge cases
4. Search regression tests for games featuring that card and verify correct logic
5. Flag any wiki-vs-game discrepancies (like Baker "original")

**Approach**: Hybrid — research all 40 cards first (no code changes), catalog findings, then fix in priority order with individual commits.

### Villagers (24)
- [ ] Alchemist
- [ ] Architect
- [ ] Baker
- [ ] Bard
- [ ] Bishop
- [ ] Confessor
- [ ] Dreamer
- [ ] Druid
- [ ] Empress
- [ ] Enlightened
- [ ] Fortune Teller
- [ ] Gemcrafter
- [ ] Hunter
- [ ] Jester
- [ ] Judge
- [ ] Knight
- [ ] Knitter
- [ ] Lover
- [ ] Medium
- [ ] Oracle
- [ ] Poet
- [ ] Scout
- [ ] Slayer
- [ ] Witness

### Outcasts (5)
- [ ] Bombardier
- [ ] Doppelganger
- [ ] Drunk
- [ ] Plague Doctor
- [ ] Wretch

### Minions (8)
- [ ] Chancellor
- [ ] Minion
- [ ] Poisoner
- [ ] Puppet
- [ ] Puppeteer
- [ ] Shaman
- [ ] Twin Minion
- [ ] Witch

### Demons (3)
- [ ] Baa
- [ ] Lilis
- [ ] Pooka
