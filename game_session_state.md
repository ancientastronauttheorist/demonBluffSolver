# Demon Bluff â€” Game Session Log (v2, card_vision pipeline)

---

# New Game — 2026-03-10 12:59:24
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Hunter, Empress, Oracle, Medium, Knitter, Poet
- Outcasts: Wretch
- Minions: 
- Demons: Pooka

### [13:08:51] Revealed #1 Knitter
Info: {'evil_pairs': 0}

### [13:08:57] Revealed #2 Medium
Info: {'good_position': 1, 'good_role': 'Knitter'}

### [13:09:02] Revealed #3 Hunter
Info: {'distance': 3}

### [13:09:07] Revealed #4 Empress
Info: {'targets': [1, 3, 6]}

### [13:09:13] Revealed #5 Poet
Info: {'copied_role': '1,6'}

### [13:09:31] Revealed #5 Poet
Info: {'targets': [1, 6], 'minion_role': 'Poisoner', 'copied_role': 'Oracle'}

### [13:09:36] Revealed #6 Wretch
Info: {}

### [13:09:41] Revealed #7 Oracle
Info: {'targets': [2, 5], 'minion_role': 'Witch'}

#### [13:10:12] Solver Output
Scenarios: 0/7
  Generated 7 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #5 Poet: rejected 7/7 (100%)
    #7 Oracle: rejected 4/7 (57%)
    #1 Knitter: rejected 3/7 (43%)
    #3 Hunter: rejected 3/7 (43%)
    #4 Empress: rejected 3/7 (43%)
    #2 Medium: rejected 2/7 (29%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Knitter: still 0
    WITHOUT #2 Medium: still 0
    WITHOUT #3 Hunter: still 0
    WITHOUT #4 Empress: still 0
    WITHOUT #5 Poet: 2 scenarios survive  <-- SUSPECT
    WITHOUT #7 Oracle: still 0

#### [13:10:12] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [13:13:43] Solver Output
Scenarios: 0/7
  Generated 7 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #5 Poet: rejected 4/7 (57%)
    #7 Oracle: rejected 4/7 (57%)
    #1 Knitter: rejected 3/7 (43%)
    #3 Hunter: rejected 3/7 (43%)
    #4 Empress: rejected 3/7 (43%)
    #2 Medium: rejected 2/7 (29%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Knitter: 1 scenarios survive  <-- SUSPECT
    WITHOUT #2 Medium: 1 scenarios survive  <-- SUSPECT
    WITHOUT #3 Hunter: 1 scenarios survive  <-- SUSPECT
    WITHOUT #4 Empress: 1 scenarios survive  <-- SUSPECT
    WITHOUT #5 Poet: 2 scenarios survive  <-- SUSPECT
    WITHOUT #7 Oracle: 1 scenarios survive  <-- SUSPECT

#### [13:13:43] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [13:19:18] Solver Output
Scenarios: 1/7
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [13:19:18] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [13:20:38] Executed #7 -> Pooka (EVIL)

## [13:20:44] GAME OVER — WIN
Final HP: 10
Notes: Poet-Oracle+Wretch bug fix game. 0 scenarios initially, fixed Wretch-Oracle minion pool check


---

# New Game — 2026-03-10 13:26:41
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Slayer, Judge, Druid, Poet
- Outcasts: Drunk, Wretch, Bombardier, Plague Doctor
- Minions: Witch
- Demons: Baa

## Deck
- Villagers: Architect, Slayer, Judge, Druid, Poet
- Outcasts: Drunk, Wretch, Bombardier, Plague Doctor
- Minions: Witch
- Demons: Baa

### [13:29:24] Revealed #2 Judge
Info: {}

### [13:29:29] Revealed #3 Wretch
Info: {}

### [13:29:47] Revealed #4 Poet
Info: {'corruption_distance': 3, 'copied_role': 'Bard'}

### [13:29:53] Revealed #5 Druid
Info: {}

### [13:29:53] Revealed #6 Bombardier
Info: {}

### [13:29:53] Revealed #7 Plague Doctor
Info: {}

### [13:29:54] Revealed #8 Slayer
Info: {}

#### [13:30:10] Solver Output
Scenarios: 36/774
Definite good: ['#1', '#2', '#4', '#5', '#8']
Evil probabilities: #3=89%, #6=89%, #7=22%
  Generated 774 candidate scenarios
  36 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 6, 7]

#### [13:30:10] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#1']
Reason: Entropy 1.864 (adjusted 1.864) | timing x1.00

### [13:31:37] Ability used at #7

#### [13:31:42] Solver Output
Scenarios: 12/774
Definite evil: ['#3', '#6']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#8']
  Generated 774 candidate scenarios
  12 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch', 'Baa'})
    #6 is DEFINITELY EVIL (possible roles: {'Witch', 'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [13:31:42] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 12 scenarios (roles: {'Witch', 'Baa'})

## Deck
- Villagers: Architect, Slayer, Judge, Druid, Poet
- Outcasts: Drunk, Wretch, Bombardier, Plague Doctor
- Minions: Witch
- Demons: Baa

#### [13:34:45] Solver Output
Scenarios: 84/762
Definite good: ['#1', '#7']
Evil probabilities: #3=52%, #6=52%, #2=29%, #5=29%, #4=19%, #8=19%
  Generated 762 candidate scenarios
  84 scenarios survived validation
    #1 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6, 8]

#### [13:34:45] Recommendation
Action: **USE_ABILITY** #5 (Druid) -> targets ['#1', '#2', '#6']
Reason: Entropy 1.956 (adjusted 1.863) | timing x1.00
WARNING: Corruption risk: 10%

### [13:37:27] Revealed #5 Druid
Info: {'targets': [1, 2, 6], 'found_outcast': None}

### [13:37:33] Ability used at #5

#### [13:37:33] Solver Output
Scenarios: 32/762
Definite good: ['#1', '#2', '#7', '#8']
Evil probabilities: #5=75%, #3=50%, #6=50%, #4=25%
  Generated 762 candidate scenarios
  32 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 5, 6]

#### [13:37:33] Recommendation
Action: **USE_ABILITY** #2 (Judge) -> targets ['#3']
Reason: Expected posterior 16.0 scenarios (adjusted 16.0, info gain 1.000 bits) | timing x1.00

### [13:38:35] Revealed #2 Judge
Info: {'target': 3, 'is_lying': True}

### [13:38:40] Ability used at #2

#### [13:38:40] Solver Output
Scenarios: 16/762
Definite evil: ['#3']
Definite good: ['#1', '#2', '#6', '#7', '#8']
Evil probabilities: #5=75%, #4=25%
  Generated 762 candidate scenarios
  16 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch', 'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [4, 5]

#### [13:38:40] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 16 scenarios (roles: {'Witch', 'Baa'})

#### [13:42:27] Solver Output
Scenarios: 22/762
Definite good: ['#1', '#7', '#8']
Evil probabilities: #3=73%, #5=55%, #2=27%, #6=27%, #4=18%
  Generated 762 candidate scenarios
  22 scenarios survived validation
    #1 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6]

#### [13:42:27] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#3']
Reason: Target #3 is 73% evil (adjusted 0.43)
WARNING: Corruption risk: 27% -- Slayer ability disabled if corrupted
WARNING: Wretch kill risk: 27% -- costs 5 HP

### [13:45:55] Ability used at #8

#### [13:46:02] Solver Output
Scenarios: 6/130
Definite evil: ['#3']
Definite good: ['#1', '#2', '#6', '#7', '#8']
Evil probabilities: #5=67%, #4=33%
  Generated 130 candidate scenarios
  6 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [4, 5]

#### [13:46:02] Recommendation
Action: **ERROR** #5
Reason: #5 is 67% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 67% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 67% < 80% threshold. Consider manual override if you have extra information.

#### [13:46:45] Solver Output
Scenarios: 4/502
Definite evil: ['#2', '#6']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']
  Generated 502 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa', 'Witch'})
    #6 is DEFINITELY EVIL (possible roles: {'Baa', 'Witch'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [13:46:45] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Baa', 'Witch'})

### [13:47:27] Executed #2 -> Baa (EVIL)

### [13:49:02] Executed #6 -> Witch (EVIL)

## [13:49:12] GAME OVER — WIN
Final HP: 5
Notes: Wretch Slayer kill cost 5HP. Fixed Druid unrevealed-target bug + Wretch-Oracle minion pool. Baa warning wrong - no=2 was correct.

