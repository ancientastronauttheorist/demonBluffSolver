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

