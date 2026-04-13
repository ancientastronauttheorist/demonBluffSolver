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


---

# New Game — 2026-03-10 13:55:57
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Medium, Enlightened, Confessor, Baker, Dreamer
- Outcasts: Bombardier, Drunk, Doppelganger
- Minions: Witch
- Demons: Baa

### [13:58:05] Revealed #2 Confessor
Info: {'dizzy': False}

### [13:58:09] Revealed #3 Alchemist
Info: {'cured_count': 0}

### [13:58:14] Revealed #4 Baker
Info: {'original_role': 'original'}

### [13:58:18] Revealed #5 Baker
Info: {'original_role': 'Medium'}

### [13:58:22] Revealed #6 Dreamer
Info: {}

### [13:58:27] Revealed #7 Enlightened
Info: {'direction': 'cw'}

### [13:58:31] Revealed #8 Baker
Info: {'original_role': 'Medium'}

#### [13:58:37] Solver Output
Scenarios: 236/2156
Definite good: ['#2', '#3']
Evil probabilities: #5=44%, #8=44%, #4=39%, #7=35%, #1=21%, #6=17%
  Generated 2156 candidate scenarios
  236 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7, 8]

#### [13:58:37] Recommendation
Action: **USE_ABILITY** #6 (Dreamer) -> targets ['#8']
Reason: Entropy 2.349 (adjusted 2.150) | timing x1.00
WARNING: Corruption risk: 17%

### [13:59:24] Revealed #6 Dreamer
Info: {'target': 8, 'evil_role': 'Witch'}

### [13:59:29] Ability used at #6

#### [13:59:35] Solver Output
Scenarios: 184/2156
Definite good: ['#2', '#3']
Evil probabilities: #5=48%, #7=45%, #4=39%, #8=28%, #1=22%, #6=19%
  Generated 2156 candidate scenarios
  184 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7, 8]

#### [13:59:35] Recommendation
Action: **EXECUTE** #5
Reason: No reveals available. #5 is 48% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 48% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Low confidence (48%) -- consider gathering more info

### [14:00:18] Executed #5 -> Witch (EVIL)

### [14:00:57] Revealed #1 Medium
Info: {'good_position': 8, 'good_role': 'real'}

#### [14:01:02] Solver Output
Scenarios: 2/301
Definite evil: ['#5']
Definite good: ['#2', '#3', '#4', '#6', '#8']
Evil probabilities: #1=50%, #7=50%
  Generated 301 candidate scenarios
  2 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [14:01:02] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 50% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [14:01:47] Executed #1 -> GOOD (WRONG!)

#### [14:01:57] Solver Output
Scenarios: 0/258
  Generated 258 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #1 Medium: rejected 247/258 (96%)
    #7 Enlightened: rejected 154/258 (60%)
    #2 Confessor: rejected 73/258 (28%)
    #3 Alchemist: rejected 73/258 (28%)
    #6 Dreamer: rejected 37/258 (14%)
    #4 Baker: rejected 30/258 (12%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Medium: 22 scenarios survive  <-- SUSPECT
    WITHOUT #2 Confessor: 1 scenarios survive  <-- SUSPECT
    WITHOUT #3 Alchemist: 1 scenarios survive  <-- SUSPECT
    WITHOUT #4 Baker: 1 scenarios survive  <-- SUSPECT
    WITHOUT #5 Baker: 1 scenarios survive  <-- SUSPECT
    WITHOUT #6 Dreamer: 7 scenarios survive  <-- SUSPECT
    WITHOUT #7 Enlightened: 3 scenarios survive  <-- SUSPECT
    WITHOUT #8 Baker: 1 scenarios survive  <-- SUSPECT

#### [14:01:57] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [14:03:47] Claude Reasoning


#### [14:08:42] Solver Output
Scenarios: 3/258
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#8']
Evil probabilities: #4=33%, #6=33%, #7=33%
  Generated 258 candidate scenarios
  3 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [4, 6, 7]

#### [14:08:42] Recommendation
Action: **ERROR** #4
Reason: #4 is 33% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 33% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 33% < 80% threshold. Consider manual override if you have extra information.

#### [14:26:33] Solver Output
Scenarios: 2/258
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4', '#8']
Evil probabilities: #6=50%, #7=50%
  Generated 258 candidate scenarios
  2 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [6, 7]

#### [14:26:33] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 50% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [14:27:59] Executed #6 -> Baa (EVIL)

## [14:28:08] GAME OVER — WIN
Final HP: 5
Notes: Baker conversion chain constraint added: evil Bakers don't trigger conversion, so converted Baker needs a good Baker revealed before it. Fixed 0-scenario bug (Medium good_role=real not parsed). Went from 3 scenarios (33%) to 2 (50%) with chain fix. 5HP, 1 wrong exec (#1 Doppelganger).


---

# New Game — 2026-03-10 14:35:05
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Fortune_Teller, Medium, Druid, Alchemist, Jester, Hunter
- Outcasts: Doppelganger, Plague_Doctor
- Minions: Puppeteer, Witch
- Demons: Lilis

### [14:37:34] Revealed #1 Judge
Info: {}

### [14:37:40] Revealed #2 Fortune_Teller
Info: {}

### [14:37:45] Revealed #3 Jester
Info: {}

### [14:37:51] Revealed #4 Hunter
Info: {'distance': 4}

### [14:39:32] Revealed #5 Judge
Info: {}

### [14:39:32] Revealed #7 Plague_Doctor
Info: {}

### [14:39:32] Revealed #8 Medium
Info: {'good_position': 6, 'good_role': 'Alchemist'}

#### [14:39:40] Solver Output
Scenarios: 1494/10080
Definite evil: ['#4']
Definite good: ['#6']
Evil probabilities: #1=38%, #5=38%, #9=38%, #2=34%, #3=34%, #7=10%, #8=8%
  Generated 10080 candidate scenarios
  1494 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis', 'Puppeteer', 'Witch'})
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 7, 8, 9]

#### [14:39:40] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1494 scenarios (roles: {'Lilis', 'Puppeteer', 'Witch'})

### [14:40:26] Executed #4 -> Lilis (EVIL)

#### [14:40:32] Solver Output
Scenarios: 498/1196
Definite evil: ['#4']
Definite good: ['#6']
Evil probabilities: #1=38%, #5=38%, #9=38%, #2=34%, #3=34%, #7=10%, #8=8%
  Generated 1196 candidate scenarios
  498 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 7, 8, 9]

#### [14:40:32] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#2']
Reason: Entropy 1.213 (adjusted 1.213) | timing x1.00

### [14:41:30] Ability used at #7

#### [14:41:35] Solver Output
Scenarios: 66/1196
Definite evil: ['#4']
Definite good: ['#6']
Evil probabilities: #7=58%, #1=42%, #9=30%, #5=27%, #3=24%, #2=12%, #8=6%
  Generated 1196 candidate scenarios
  66 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 7, 8, 9]

#### [14:41:35] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#8', '#9']
Reason: Expected posterior 25.5 scenarios (adjusted 25.5, info gain 1.371 bits) | timing x1.00

### [14:42:37] Revealed #1 Judge
Info: {'target': 8, 'is_lying': False}

### [14:42:43] Ability used at #1

#### [14:42:48] Solver Output
Scenarios: 38/1196
Definite evil: ['#4']
Definite good: ['#6']
Evil probabilities: #7=95%, #5=26%, #9=26%, #2=21%, #3=21%, #1=5%, #8=5%
  Generated 1196 candidate scenarios
  38 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 7, 8, 9]

#### [14:42:48] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#5', '#8']
Reason: Expected posterior 15.0 scenarios (adjusted 15.0, info gain 1.337 bits) | timing x1.00

### [14:45:26] Revealed #5 Judge
Info: {'target': 8, 'is_lying': False}

### [14:45:29] Ability used at #5

#### [14:45:33] Solver Output
Scenarios: 26/1196
Definite evil: ['#4', '#7']
Definite good: ['#1', '#5', '#6', '#8']
Evil probabilities: #9=38%, #2=31%, #3=31%
  Generated 1196 candidate scenarios
  26 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Witch', 'Puppeteer'})
    #1 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3, 9]

#### [14:45:33] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 26 scenarios (roles: {'Witch', 'Puppeteer'})

### [14:46:08] Executed #7 -> Puppeteer (EVIL)

#### [14:46:21] Solver Output
Scenarios: 13/43
Definite evil: ['#4', '#7']
Definite good: ['#1', '#5', '#6', '#8']
Evil probabilities: #9=38%, #2=31%, #3=31%
  Generated 43 candidate scenarios
  13 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3, 9]

#### [14:46:21] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#2', '#5']
Reason: Expected posterior 5.8 scenarios (adjusted 5.8, info gain 1.174 bits) | timing x1.00

### [14:47:31] Revealed #3 Jester
Info: {'targets': [1, 2, 5], 'evil_count': 0}

### [14:47:36] Ability used at #3

#### [14:47:40] Solver Output
Scenarios: 5/43
Definite evil: ['#4', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#8']
  Generated 43 candidate scenarios
  5 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #9 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [14:47:40] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 5 scenarios (roles: {'Witch'})

### [14:48:33] Executed #9 -> Witch (EVIL)

#### [14:48:37] Solver Output
Scenarios: 5/7
Definite evil: ['#4', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#8']
  Generated 7 candidate scenarios
  5 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #9 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [14:48:37] Recommendation
Action: **WIN**
Reason: All evil characters have been executed!

### [14:51:16] Executed #8 -> Puppet (EVIL)

## [14:51:24] GAME OVER — WIN
Final HP: 8
Notes: Solver WIN bug: recommended WIN at 3/4 evils, didn't track Puppet needing execution. Manual deduction found #8=Puppet via Jester elimination.


---

# New Game — 2026-03-10 15:01:45
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Medium, Fortune_Teller, Knight, Baker, Confessor, Lover, Alchemist
- Outcasts: Plague_Doctor, Bombardier
- Minions: Minion, Twin_Minion
- Demons: Pooka

### [15:07:03] Revealed #1 Alchemist
Info: {'cured_count': 2}

### [15:07:09] Revealed #2 Plague_Doctor
Info: {}

### [15:07:14] Revealed #3 Confessor
Info: {'dizzy': False}

### [15:07:20] Revealed #4 Bard
Info: {'corruption_distance': 1}

### [15:07:25] Revealed #5 Bombardier
Info: {}

### [15:07:30] Revealed #6 Lover
Info: {'evil_adjacent': 0}

### [15:07:35] Revealed #7 Baker
Info: {'original_role': 'original'}

### [15:07:41] Revealed #8 Fortune_Teller
Info: {}

### [15:07:47] Revealed #9 Baker
Info: {'original_role': 'Confessor'}

### [15:07:52] Revealed #10 Knight
Info: {}

#### [15:07:59] Solver Output
Scenarios: 34/2904
Definite good: ['#3']
Evil probabilities: #5=94%, #1=53%, #9=53%, #10=29%, #4=24%, #6=18%, #7=12%, #8=12%, #2=6%
  Generated 2904 candidate scenarios
  34 scenarios survived validation
    #3 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7, 8, 9, 10]

#### [15:07:59] Recommendation
Action: **EXECUTE** #10
Reason: Knight check: #10 is 29% evil, 6% corruption risk. Expected HP cost: 0.4 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 6% -- corrupted Knight loses immunity + 4 extra damage

### [15:08:44] Executed #10 -> Pooka (EVIL)

#### [15:08:49] Solver Output
Scenarios: 6/310
Definite evil: ['#10']
Definite good: ['#1', '#3', '#7', '#8', '#9']
Evil probabilities: #4=67%, #5=67%, #2=33%, #6=33%
  Generated 310 candidate scenarios
  6 scenarios survived validation
    #10 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 6]

#### [15:08:49] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#1']
Reason: Entropy 1.585 (adjusted 1.585) | timing x1.00

### [15:09:46] Ability used at #2

#### [15:09:52] Solver Output
Scenarios: 4/310
Definite evil: ['#5', '#10']
Definite good: ['#1', '#2', '#3', '#7', '#8', '#9']
Evil probabilities: #4=50%, #6=50%
  Generated 310 candidate scenarios
  4 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Minion'})
    #10 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [15:09:52] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 4 scenarios (roles: {'Twin_Minion', 'Minion'})

### [15:10:29] Executed #5 -> Minion (EVIL)

#### [15:10:35] Solver Output
Scenarios: 2/43
Definite evil: ['#5', '#10']
Definite good: ['#1', '#2', '#3', '#7', '#8', '#9']
Evil probabilities: #4=50%, #6=50%
  Generated 43 candidate scenarios
  2 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #10 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [15:10:35] Recommendation
Action: **USE_ABILITY** #8 (Fortune Teller) -> targets ['#1', '#4']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [15:11:32] Revealed #8 Fortune Teller
Info: {'targets': [1, 4], 'has_evil': False}

### [15:11:38] Ability used at #8

#### [15:11:45] Solver Output
Scenarios: 1/43
Definite evil: ['#5', '#6', '#10']
Definite good: ['#1', '#2', '#3', '#4', '#7', '#8', '#9']
  Generated 43 candidate scenarios
  1 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #6 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #10 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [15:11:45] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [15:12:26] Executed #6 -> Twin_Minion (EVIL)

## [15:12:38] GAME OVER — WIN
Final HP: 10
Notes: Perfect village 10HP. Knight check on #10 found Pooka demon immediately. PD confirmed #1 corrupted by Pooka.


---

# New Game — 2026-03-10 15:13:41
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Dreamer, Architect, Druid, Hunter, Confessor, Gemcrafter
- Outcasts: Drunk, Bombardier
- Minions: Chancellor
- Demons: Pooka

### [15:16:08] Revealed #1 Bombardier
Info: {}

### [15:16:08] Revealed #2 Architect
Info: {'side': 'equal'}

### [15:16:08] Revealed #3 Druid
Info: {}

### [15:16:09] Revealed #4 Dreamer
Info: {}

### [15:16:09] Revealed #5 Poet
Info: {'copied_role': '2'}

### [15:16:09] Revealed #6 Confessor
Info: {'dizzy': True}

### [15:16:09] Revealed #7 Hunter
Info: {'distance': 2}

### [15:16:09] Revealed #8 Bombardier
Info: {}

### [15:16:40] Revealed #5 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 2}

#### [15:16:47] Solver Output
Scenarios: 11/340
Definite good: ['#3', '#4', '#7']
Evil probabilities: #1=82%, #6=45%, #2=27%, #5=27%, #8=18%
  Generated 340 candidate scenarios
  11 scenarios survived validation
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 8]

#### [15:16:47] Recommendation
Action: **USE_ABILITY** #4 (Dreamer) -> targets ['#6']
Reason: Entropy 1.868 (adjusted 1.528) | timing x1.00
WARNING: Corruption risk: 36%

### [15:17:46] Revealed #4 Dreamer
Info: {'target': 6, 'evil_role': 'Pooka'}

### [15:17:52] Ability used at #4

#### [15:17:59] Solver Output
Scenarios: 10/340
Definite good: ['#3', '#4', '#7']
Evil probabilities: #1=80%, #6=40%, #2=30%, #5=30%, #8=20%
  Generated 340 candidate scenarios
  10 scenarios survived validation
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 8]

#### [15:17:59] Recommendation
Action: **USE_ABILITY** #3 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 1.485 (adjusted 1.263) | timing x1.00
WARNING: Corruption risk: 30%

### [15:19:04] Revealed #3 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': None}

### [15:19:11] Ability used at #3

#### [15:19:18] Solver Output
Scenarios: 4/340
Definite evil: ['#1']
Definite good: ['#3', '#4', '#7', '#8']
Evil probabilities: #6=50%, #2=25%, #5=25%
  Generated 340 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka', 'Chancellor'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 5, 6]

#### [15:19:18] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Pooka', 'Chancellor'})

### [15:20:02] Executed #1 -> Chancellor (EVIL)

#### [15:20:09] Solver Output
Scenarios: 3/36
Definite evil: ['#1']
Definite good: ['#2', '#3', '#4', '#7', '#8']
Evil probabilities: #6=67%, #5=33%
  Generated 36 candidate scenarios
  3 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 6]

#### [15:20:09] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (67% evil Pooka, 33% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [15:20:54] Executed #6 -> Pooka (EVIL)

## [15:20:54] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Two Bombardiers on board, Druid found no outcasts among 1,2,4 confirming #1 evil. Dreamer tagged #6 as Pooka.


---

# New Game — 2026-03-10 15:25:03
Cards: 8, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Hunter, Dreamer, Knight, Lover, Fortune_Teller
- Outcasts: Wretch
- Minions: Puppeteer
- Demons: Lilis

### [15:29:06] Revealed #1 Hunter
Info: {'distance': 1}

### [15:29:11] Revealed #2 Wretch
Info: {}

### [15:29:16] Revealed #3 Dreamer
Info: {}

### [15:29:20] Revealed #4 Fortune_Teller
Info: {}

### [15:32:30] Revealed #5 Lover
Info: {'evil_adjacent': 1}

### [15:32:34] Revealed #6 Fortune_Teller
Info: {}

### [15:32:39] Revealed #7 Knight
Info: {}

### [15:32:51] Revealed #8 Empress
Info: {'targets': [1, 6, 7]}

#### [15:32:57] Solver Output
Scenarios: 17/84
Evil probabilities: #4=59%, #6=53%, #8=47%, #7=41%, #3=35%, #5=35%, #2=18%, #1=12%
  Generated 84 candidate scenarios
  17 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [15:32:57] Recommendation
Action: **EXECUTE** #7
Reason: Knight free check: #7 is 41% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

#### [15:34:05] Solver Output
Scenarios: 10/50
Definite good: ['#7']
Evil probabilities: #4=60%, #6=60%, #5=50%, #3=40%, #8=40%, #2=30%, #1=20%
  Generated 50 candidate scenarios
  10 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8]

#### [15:34:05] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#4']
Reason: Entropy 2.446 (adjusted 2.446) | timing x1.00

### [15:34:58] Revealed #3 Dreamer
Info: {'target': 4, 'evil_role': 'Puppeteer'}

### [15:35:03] Ability used at #3

#### [15:35:08] Solver Output
Scenarios: 7/50
Definite good: ['#7']
Evil probabilities: #3=57%, #6=57%, #8=57%, #4=43%, #5=43%, #2=29%, #1=14%
  Generated 50 candidate scenarios
  7 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8]

#### [15:35:08] Recommendation
Action: **USE_ABILITY** #4 (Fortune Teller) -> targets ['#1', '#5']
Reason: Entropy 0.985 (adjusted 0.985) | timing x1.00

### [15:36:03] Revealed #4 Fortune Teller
Info: {'targets': [1, 5], 'has_evil': False}

### [15:36:08] Ability used at #4

#### [15:36:13] Solver Output
Scenarios: 3/50
Definite good: ['#1', '#7']
Evil probabilities: #3=67%, #4=67%, #8=67%, #2=33%, #5=33%, #6=33%
  Generated 50 candidate scenarios
  3 scenarios survived validation
    #1 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6, 8]

#### [15:36:13] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.918 (adjusted 0.918) | timing x1.00

### [15:36:55] Revealed #6 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': True}

### [15:37:00] Ability used at #6

#### [15:37:05] Solver Output
Scenarios: 2/50
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#6', '#7']
Evil probabilities: #3=50%, #5=50%
  Generated 50 candidate scenarios
  2 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Puppet'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [3, 5]

#### [15:37:05] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Puppeteer', 'Puppet'})

### [15:37:51] Executed #4 -> Puppeteer (EVIL)

#### [15:37:55] Solver Output
Scenarios: 1/10
Definite evil: ['#4', '#5', '#8']
Definite good: ['#1', '#2', '#3', '#6', '#7']
  Generated 10 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [15:37:55] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [15:38:30] Executed #5 -> Puppet (EVIL)

#### [15:38:35] Solver Output
Scenarios: 1/5
Definite evil: ['#4', '#5', '#8']
Definite good: ['#1', '#2', '#3', '#6', '#7']
  Generated 5 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [15:38:35] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [15:39:08] Executed #8 -> Lilis (EVIL)

## [15:39:49] GAME OVER — WIN
Final HP: 6
Notes: Clean solve with Knight free check and FT abilities. Puppeteer+Puppet+Lilis game. HP mystery: showed 6/10 but no night kills visible.


---

# New Game — 2026-03-10 15:40:24
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Gemcrafter, Scout, Bard, Architect, Baker, Poet, Oracle
- Outcasts: Plague_Doctor
- Minions: Minion, Twin_Minion
- Demons: Pooka

### [15:44:01] Revealed #1 Plague_Doctor
Info: {}

### [15:44:07] Revealed #2 Plague_Doctor
Info: {}

### [15:44:11] Revealed #3 Baker
Info: {'original_role': 'original'}

### [15:44:16] Revealed #4 Baker
Info: {'original_role': 'Scout'}

### [15:44:21] Revealed #5 Poet
Info: {'evil_pairs': 0, 'copied_role': 'Knitter'}

### [15:44:25] Revealed #6 Oracle
Info: {'targets': [3, 4], 'minion_role': 'Twin_Minion'}

### [15:44:30] Revealed #7 Baker
Info: {'original_role': 'Scout'}

### [15:44:35] Revealed #8 Bard
Info: {'corruption_distance': 3}

### [15:44:40] Revealed #9 Scout
Info: {'evil_role': 'Minion', 'distance': 1}

### [15:44:45] Revealed #10 Architect
Info: {'side': 'Left'}

#### [15:44:51] Solver Output
Scenarios: 27/3744
Definite good: ['#3', '#5', '#10']
Evil probabilities: #9=70%, #1=52%, #7=52%, #2=48%, #4=33%, #6=22%, #8=22%
  Generated 3744 candidate scenarios
  27 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 7, 8, 9]

#### [15:44:51] Recommendation
Action: **USE_ABILITY** #1 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.989 (adjusted 1.989) | timing x1.00

### [15:45:49] Ability used at #1

#### [15:45:54] Solver Output
Scenarios: 13/3744
Definite good: ['#3', '#5', '#10']
Evil probabilities: #9=69%, #2=54%, #7=54%, #1=46%, #4=31%, #6=23%, #8=23%
  Generated 3744 candidate scenarios
  13 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 7, 8, 9]

#### [15:45:54] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#4']
Reason: Entropy 1.950 (adjusted 1.950) | timing x1.00

### [15:46:37] Ability used at #2

#### [15:46:42] Solver Output
Scenarios: 7/3744
Definite good: ['#3', '#4', '#5', '#10']
Evil probabilities: #7=71%, #1=57%, #9=57%, #2=43%, #6=43%, #8=29%
  Generated 3744 candidate scenarios
  7 scenarios survived validation
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 6, 7, 8, 9]

#### [15:46:42] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (57% evil Pooka, 29% good Baker (corrupted), 14% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 71%, but all reveal branches still lead to a forced win.

### [15:47:23] Executed #7 -> Minion (EVIL)

#### [15:47:29] Solver Output
Scenarios: 1/380
Definite evil: ['#2', '#6', '#7']
Definite good: ['#1', '#3', '#4', '#5', '#8', '#9', '#10']
  Generated 380 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #7 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [15:47:29] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [15:48:09] Executed #2 -> Twin_Minion (EVIL)

### [15:48:53] Executed #6 -> Pooka (EVIL)

## [15:49:11] GAME OVER — WIN
Final HP: 10
Notes: Perfect solve at 10HP. PD checks narrowed from 27->13->7->1 scenarios. Pooka corrupted #5 Poet, PD corrupted #4 Baker.


---

# New Game — 2026-03-10 15:50:11
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Dreamer, Gemcrafter, Knitter, Alchemist, Oracle, Jester
- Outcasts: Wretch, Bombardier
- Minions: Shaman, Minion
- Demons: Lilis

### [15:51:34] Revealed #2 Alchemist
Info: {'cured_count': 0}

### [15:51:40] Revealed #3 Gemcrafter
Info: {'good_position': 2}

### [15:51:46] Revealed #4 Dreamer
Info: {}

### [15:52:39] Revealed #1 Bombardier
Info: {}

### [15:55:05] Revealed #6 Wretch
Info: {}

### [15:55:05] Revealed #7 Bombardier
Info: {}

### [15:55:05] Revealed #8 Jester
Info: {}

### [15:55:05] Revealed #9 Gemcrafter
Info: {'good_position': 8}

#### [15:55:17] Solver Output
Scenarios: 24/504
Definite good: ['#2', '#3', '#5', '#8', '#9']
Evil probabilities: #1=75%, #4=75%, #6=75%, #7=75%
  Generated 504 candidate scenarios
  24 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4, 6, 7]

#### [15:55:17] Recommendation
Action: **USE_ABILITY** #4 (Dreamer) -> targets ['#1']
Reason: Entropy 2.689 (adjusted 2.689) | timing x1.00

### [15:56:06] Revealed #4 Dreamer
Info: {'target': 1, 'evil_role': 'Lilis'}

### [15:56:12] Ability used at #4

#### [15:56:18] Solver Output
Scenarios: 16/504
Definite good: ['#2', '#3', '#5', '#8', '#9']
Evil probabilities: #4=88%, #6=75%, #7=75%, #1=62%
  Generated 504 candidate scenarios
  16 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4, 6, 7]

#### [15:56:18] Recommendation
Action: **USE_ABILITY** #8 (Jester) -> targets ['#1', '#2', '#4']
Reason: Expected posterior 8.0 scenarios (adjusted 8.0, info gain 1.000 bits) | timing x1.00

### [15:57:09] Revealed #8 Jester
Info: {'targets': [1, 2, 4], 'evil_count': 2}

### [15:57:14] Ability used at #8

#### [15:57:21] Solver Output
Scenarios: 8/504
Definite evil: ['#1', '#4']
Definite good: ['#2', '#3', '#5', '#8', '#9']
Evil probabilities: #6=50%, #7=50%
  Generated 504 candidate scenarios
  8 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion', 'Shaman'})
    #4 is DEFINITELY EVIL (possible roles: {'Lilis', 'Minion', 'Shaman'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [6, 7]

#### [15:57:21] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 8 scenarios (roles: {'Minion', 'Shaman'})

### [15:58:01] Executed #1 -> Shaman (EVIL)

#### [15:58:08] Solver Output
Scenarios: 4/56
Definite evil: ['#1', '#4']
Definite good: ['#2', '#3', '#5', '#8', '#9']
Evil probabilities: #6=50%, #7=50%
  Generated 56 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #4 is DEFINITELY EVIL (possible roles: {'Lilis', 'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [6, 7]

#### [15:58:08] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 4 scenarios (roles: {'Lilis', 'Minion'})

### [15:58:48] Executed #4 -> Lilis (EVIL)

#### [15:58:54] Solver Output
Scenarios: 2/7
Definite evil: ['#1', '#4']
Definite good: ['#2', '#3', '#5', '#8', '#9']
Evil probabilities: #6=50%, #7=50%
  Generated 7 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [6, 7]

#### [15:58:54] Recommendation
Action: **ERROR** #2
Reason: #2 is 0% likely evil but budget=1 requires >=80% confidence (HP=6, cost=5).
WARNING: Probabilistic execution -- 0% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 0% < 80% threshold. Consider manual override if you have extra information.

### [16:00:38] Executed #6 -> GOOD (WRONG!)

### [16:01:22] Executed #7 -> Minion (EVIL)

## [16:01:43] GAME OVER — WIN
Final HP: 1
Notes: Close win at 1HP. Lilis killed #5 Oracle. Solver stuck 50/50 on #6/#7 (both outcasts, only 1 on board). Manual deduction: execute Wretch first (safe -5HP) then Bombardier. Wretch wrong exec cost 5HP, survived.


---

# New Game — 2026-03-10 16:22:27
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Judge, Jester, Dreamer, Fortune_Teller, Bard
- Outcasts: Drunk, Wretch, Plague_Doctor
- Minions: Chancellor
- Demons: Pooka

### [16:25:38] Revealed #1 Judge
Info: {}

### [16:25:43] Revealed #2 Fortune_Teller
Info: {}

### [16:25:46] Revealed #3 Jester
Info: {}

### [16:25:50] Revealed #4 Wretch
Info: {}

### [16:25:55] Revealed #5 Bard
Info: {'corruption_distance': -1}

### [16:25:59] Revealed #6 Plague_Doctor
Info: {}

### [16:26:03] Revealed #7 Dreamer
Info: {}

### [16:26:07] Revealed #8 Bishop
Info: {'targets': [3, 7, 8], 'types': ['Outcast', 'Villager', 'Minion']}

### [16:26:11] Revealed #9 Judge
Info: {}

#### [16:26:15] Solver Output
Scenarios: 686/2444
Evil probabilities: #4=33%, #5=32%, #8=31%, #7=27%, #9=24%, #2=20%, #1=18%, #3=13%, #6=3%
  Generated 2444 candidate scenarios
  686 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [16:26:15] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#8']
Reason: Entropy 2.640 (adjusted 2.640) | timing x1.00

### [16:27:19] Ability used at #6

#### [16:27:24] Solver Output
Scenarios: 404/2444
Definite good: ['#6', '#8']
Evil probabilities: #5=39%, #9=37%, #4=36%, #7=28%, #2=25%, #1=23%, #3=12%
  Generated 2444 candidate scenarios
  404 scenarios survived validation
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 9]

#### [16:27:24] Recommendation
Action: **USE_ABILITY** #7 (Dreamer) -> targets ['#5']
Reason: Entropy 2.302 (adjusted 2.149) | timing x1.00
WARNING: Corruption risk: 13%

### [16:28:16] Revealed #7 Dreamer
Info: {'target': 5, 'evil_role': 'Chancellor'}

### [16:28:22] Ability used at #7

#### [16:28:27] Solver Output
Scenarios: 308/2444
Definite good: ['#6', '#8']
Evil probabilities: #4=44%, #9=41%, #7=29%, #2=27%, #1=24%, #5=20%, #3=14%
  Generated 2444 candidate scenarios
  308 scenarios survived validation
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 9]

#### [16:28:27] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#4', '#8']
Reason: Entropy 1.000 (adjusted 0.881) | follow-up bonus 0.883 | timing x1.00
WARNING: Corruption risk: 24%

### [16:29:17] Revealed #2 Fortune Teller
Info: {'targets': [4, 8], 'has_evil': True}

### [16:29:21] Ability used at #2

#### [16:29:26] Solver Output
Scenarios: 151/2444
Definite good: ['#2', '#6', '#8']
Evil probabilities: #4=56%, #9=50%, #7=37%, #1=28%, #5=18%, #3=10%
  Generated 2444 candidate scenarios
  151 scenarios survived validation
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 7, 9]

#### [16:29:26] Recommendation
Action: **USE_ABILITY** #9 (Judge) -> targets ['#1']
Reason: Expected posterior 84.0 scenarios (adjusted 88.8, info gain 0.766 bits) | timing x1.00
WARNING: Corruption risk: 11% -- corrupted Judge results are unreliable

### [16:30:02] Revealed #9 Judge
Info: {'target': 1, 'is_lying': False}

### [16:30:07] Ability used at #9

#### [16:30:12] Solver Output
Scenarios: 86/2444
Definite good: ['#2', '#6', '#8']
Evil probabilities: #4=58%, #9=56%, #7=35%, #5=20%, #1=17%, #3=14%
  Generated 2444 candidate scenarios
  86 scenarios survived validation
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 7, 9]

#### [16:30:12] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#2', '#4', '#6']
Reason: Expected posterior 43.0 scenarios (adjusted 50.7, info gain 0.761 bits) | timing x1.00
WARNING: Corruption risk: 36%

### [16:31:02] Revealed #3 Jester
Info: {'targets': [2, 4, 6], 'evil_count': 0}

### [16:31:07] Ability used at #3

#### [16:31:11] Solver Output
Scenarios: 43/2444
Definite good: ['#2', '#6', '#8']
Evil probabilities: #4=56%, #9=49%, #3=28%, #7=28%, #5=26%, #1=14%
  Generated 2444 candidate scenarios
  43 scenarios survived validation
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 7, 9]

#### [16:31:11] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#4']
Reason: Expected posterior 31.1 scenarios (adjusted 38.0, info gain 0.178 bits) | timing x1.00
WARNING: Corruption risk: 44% -- corrupted Judge results are unreliable

### [16:31:50] Revealed #1 Judge
Info: {'target': 4, 'is_lying': True}

### [16:31:55] Ability used at #1

#### [16:32:00] Solver Output
Scenarios: 33/2444
Definite good: ['#2', '#6', '#8']
Evil probabilities: #9=64%, #4=61%, #5=30%, #3=21%, #7=18%, #1=6%
  Generated 2444 candidate scenarios
  33 scenarios survived validation
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 7, 9]

#### [16:32:00] Recommendation
Action: **EXECUTE** #9
Reason: No reveals available. #9 is 64% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 64% confident (budget: 2 wrong execs)

### [16:32:41] Executed #9 -> Pooka (EVIL)

#### [16:32:52] Solver Output
Scenarios: 17/259
Definite evil: ['#9']
Definite good: ['#2', '#6', '#8']
Evil probabilities: #3=29%, #5=29%, #4=24%, #1=12%, #7=6%
  Generated 259 candidate scenarios
  17 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 7]

#### [16:32:52] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 29% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 29% confident (budget: 2 wrong execs)
WARNING: Low confidence (29%) -- consider gathering more info

### [16:33:39] Executed #3 -> GOOD (WRONG!)

#### [16:33:50] Solver Output
Scenarios: 0/234
  Generated 234 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #3 Jester: rejected 156/234 (67%)
    #5 Bard: rejected 140/234 (60%)
    #2 Fortune Teller: rejected 110/234 (47%)
    #8 Bishop: rejected 40/234 (17%)
    #7 Dreamer: rejected 9/234 (4%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Judge: 13 scenarios survive  <-- SUSPECT
    WITHOUT #2 Fortune Teller: 19 scenarios survive  <-- SUSPECT
    WITHOUT #3 Jester: 38 scenarios survive  <-- SUSPECT
    WITHOUT #5 Bard: 40 scenarios survive  <-- SUSPECT
    WITHOUT #7 Dreamer: 15 scenarios survive  <-- SUSPECT
    WITHOUT #8 Bishop: 15 scenarios survive  <-- SUSPECT
    WITHOUT #9 Judge: 13 scenarios survive  <-- SUSPECT

#### [16:33:50] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [16:35:59] Solver Output
Scenarios: 12/234
Definite evil: ['#9']
Definite good: ['#2', '#3', '#6', '#8']
Evil probabilities: #5=42%, #4=33%, #1=17%, #7=8%
  Generated 234 candidate scenarios
  12 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 7]

#### [16:35:59] Recommendation
Action: **ERROR** #5
Reason: #5 is 42% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 42% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 42% < 80% threshold. Consider manual override if you have extra information.

## [16:38:40] GAME OVER — WIN
Final HP: 5
Notes: PD+Pooka dual corruption. 0-scenario bug from executed_good_corrupted=false when #3 was corrupted. Fixed mid-game.


---

# New Game — 2026-03-10 16:40:32
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Alchemist, Judge, Knight, Enlightened, Lover, Jester
- Outcasts: Plague_Doctor
- Minions: Puppeteer, Minion
- Demons: Pooka

### [16:42:07] Revealed #1 Knight
Info: {}

### [16:42:14] Revealed #2 Jester
Info: {}

### [16:42:14] Revealed #3 Hunter
Info: {'distance': 2}

### [16:42:14] Revealed #4 Lover
Info: {'evil_adjacent': 2}

### [16:42:15] Revealed #5 Judge
Info: {}

### [16:42:15] Revealed #6 Enlightened
Info: {'direction': 'ccw'}

### [16:42:15] Revealed #7 Alchemist
Info: {'cured_count': 1}

### [16:42:15] Revealed #8 Lover
Info: {'evil_adjacent': 0}

### [16:42:15] Revealed #9 Plague_Doctor
Info: {}

### [16:42:15] Revealed #10 Enlightened
Info: {'direction': 'ccw'}

#### [16:42:21] Solver Output
Scenarios: 56/3696
Evil probabilities: #4=93%, #10=91%, #5=48%, #1=46%, #3=46%, #2=41%, #6=12%, #7=12%, #8=7%, #9=2%
  Generated 3696 candidate scenarios
  56 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#### [16:42:21] Recommendation
Action: **EXECUTE** #1
Reason: Knight check: #1 is 46% evil, 27% corruption risk. Expected HP cost: 1.3 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 27% -- corrupted Knight loses immunity + 4 extra damage

### [16:43:22] Executed #1 -> Puppet (EVIL)

#### [16:43:27] Solver Output
Scenarios: 23/448
Definite evil: ['#1', '#10']
Definite good: ['#6', '#9']
Evil probabilities: #4=96%, #2=43%, #3=30%, #5=17%, #7=9%, #8=4%
  Generated 448 candidate scenarios
  23 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #10 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Minion'})
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 7, 8]

#### [16:43:27] Recommendation
Action: **EXECUTE** #10
Reason: #10 is evil in ALL 23 scenarios (roles: {'Puppeteer', 'Minion'})

### [16:44:08] Executed #10 -> Puppeteer (EVIL)

#### [16:44:08] Solver Output
Scenarios: 20/224
Definite evil: ['#1', '#10']
Definite good: ['#6', '#9']
Evil probabilities: #4=95%, #2=35%, #3=35%, #5=20%, #7=10%, #8=5%
  Generated 224 candidate scenarios
  20 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #10 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 7, 8]

#### [16:44:08] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.559 (adjusted 1.559) | timing x1.00

### [16:44:48] Ability used at #9

#### [16:44:49] Solver Output
Scenarios: 1/224
Definite evil: ['#1', '#4', '#7', '#10']
Definite good: ['#2', '#3', '#5', '#6', '#8', '#9']
  Generated 224 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #10 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [16:44:49] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Minion'})

### [16:45:24] Executed #4 -> Minion (EVIL)

### [16:46:06] Executed #7 -> Pooka (EVIL)

## [16:46:13] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. PD check solved it to 1 scenario.


---

# New Game — 2026-03-10 16:47:58
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Fortune_Teller, Empress, Jester, Oracle, Knight
- Outcasts: Plague_Doctor
- Minions: 
- Demons: Pooka

### [16:50:16] Revealed #1 Knight
Info: {}

### [16:50:16] Revealed #2 Empress
Info: {'targets': [1, 5, 6]}

### [16:50:16] Revealed #3 Fortune_Teller
Info: {}

### [16:50:16] Revealed #4 Plague_Doctor
Info: {}

### [16:50:16] Revealed #5 Oracle
Info: {}

### [16:50:17] Revealed #6 Jester
Info: {}

#### [16:50:22] Solver Output
Scenarios: 14/21
Definite good: ['#1', '#4']
Evil probabilities: #2=29%, #3=29%, #5=21%, #6=21%
  Generated 21 candidate scenarios
  14 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [2, 3, 5, 6]

#### [16:50:22] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#1']
Reason: Entropy 2.067 (adjusted 2.067) | timing x1.00

### [16:51:06] Ability used at #4

#### [16:51:06] Solver Output
Scenarios: 5/21
Definite good: ['#1', '#2', '#4', '#6']
Evil probabilities: #3=60%, #5=40%
  Generated 21 candidate scenarios
  5 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [3, 5]

#### [16:51:06] Recommendation
Action: **USE_ABILITY** #3 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.722 (adjusted 0.650) | timing x1.00
WARNING: Corruption risk: 20%

### [16:51:53] Revealed #3 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': True}

### [16:51:53] Ability used at #3

#### [16:51:53] Solver Output
Scenarios: 4/21
Definite good: ['#1', '#2', '#4', '#6']
Evil probabilities: #3=75%, #5=25%
  Generated 21 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [3, 5]

#### [16:51:53] Recommendation
Action: **USE_ABILITY** #6 (Jester) -> targets ['#1', '#2', '#4']
Reason: Expected posterior 2.0 scenarios (adjusted 2.5, info gain 0.678 bits) | timing x1.00
WARNING: Corruption risk: 50%

### [16:52:56] Revealed #6 Jester
Info: {'targets': [1, 2, 4], 'evil_count': 2}

### [16:52:56] Ability used at #6

#### [16:52:56] Solver Output
Scenarios: 2/21
Definite good: ['#1', '#2', '#4', '#6']
Evil probabilities: #3=50%, #5=50%
  Generated 21 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [3, 5]

#### [16:52:56] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% good Fortune Teller (corrupted), 50% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [16:53:44] Executed #3 -> Pooka (EVIL)

## [16:53:44] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. FT on confirmed-Good caught the lie. Jester corruption confirmed Pooka position.


---

# New Game — 2026-03-10 16:55:00
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Hunter, Architect, Bard, Jester, Knitter, Enlightened
- Outcasts: Plague_Doctor
- Minions: Shaman, Minion
- Demons: Pooka

### [16:56:29] Revealed #1 Enlightened
Info: {'direction': 'cw'}

### [16:56:29] Revealed #2 Enlightened
Info: {'direction': 'cw'}

### [16:56:29] Revealed #3 Jester
Info: {}

### [16:56:30] Revealed #4 Druid
Info: {}

### [16:56:30] Revealed #5 Jester
Info: {}

### [16:56:30] Revealed #6 Plague_Doctor
Info: {}

### [16:56:30] Revealed #7 Hunter
Info: {'distance': 1}

### [16:56:30] Revealed #8 Druid
Info: {}

### [16:56:30] Revealed #9 Bard
Info: {'corruption_distance': 1}

### [16:56:31] Revealed #10 Architect
Info: {'side': 'left'}

#### [16:56:38] Solver Output
Scenarios: 66/3240
Definite good: ['#6', '#9']
Evil probabilities: #8=58%, #10=45%, #1=39%, #4=39%, #5=33%, #7=33%, #3=27%, #2=24%
  Generated 3240 candidate scenarios
  66 scenarios survived validation
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 10]

#### [16:56:38] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#9']
Reason: Entropy 1.783 (adjusted 1.783) | timing x1.00

### [16:57:19] Ability used at #6

#### [16:57:20] Solver Output
Scenarios: 10/3240
Definite evil: ['#3', '#10']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#9']
Evil probabilities: #8=80%, #4=20%
  Generated 3240 candidate scenarios
  10 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Shaman', 'Minion'})
    #10 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 8]

#### [16:57:20] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 10 scenarios (roles: {'Shaman', 'Minion'})

### [16:59:09] Executed #3 -> Shaman (EVIL)

#### [16:59:17] Solver Output
Scenarios: 5/352
Definite evil: ['#3', '#10']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#9']
Evil probabilities: #8=80%, #4=20%
  Generated 352 candidate scenarios
  5 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #10 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 8]

#### [16:59:17] Recommendation
Action: **EXECUTE** #10
Reason: #10 is evil in ALL 5 scenarios (roles: {'Pooka'})

### [17:03:19] Executed #10 -> Pooka (EVIL)

#### [17:03:24] Solver Output
Scenarios: 5/43
Definite evil: ['#3', '#10']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#9']
Evil probabilities: #8=80%, #4=20%
  Generated 43 candidate scenarios
  5 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #10 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 8]

#### [17:03:24] Recommendation
Action: **USE_ABILITY** #5 (Jester) -> targets ['#1', '#2', '#4']
Reason: Expected posterior 2.1 scenarios (adjusted 2.4, info gain 1.085 bits) | timing x1.00
WARNING: Corruption risk: 20%

### [17:04:41] Revealed #5 Jester
Info: {'targets': [1, 2, 4], 'evil_count': 1}

### [17:04:46] Ability used at #5

#### [17:04:52] Solver Output
Scenarios: 2/43
Definite evil: ['#3', '#10']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#9']
Evil probabilities: #4=50%, #8=50%
  Generated 43 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #10 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 8]

#### [17:04:52] Recommendation
Action: **USE_ABILITY** #4 (Druid) -> targets ['#1', '#2', '#5']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [17:06:15] Revealed #4 Druid
Info: {'targets': [1, 2, 5], 'found_outcast': 'Doppelganger'}

### [17:06:20] Ability used at #4

#### [17:06:25] Solver Output
Scenarios: 1/43
Definite evil: ['#3', '#4', '#10']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8', '#9']
  Generated 43 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #10 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [17:06:25] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Minion'})

### [17:07:35] Executed #4 -> Minion (EVIL)

## [17:07:43] GAME OVER — WIN
Final HP: 10
Notes: PD active confirmed #3 evil + #9 corrupted. Jester on #1,#2,#4 found 1 evil. Druid #4 lied (Doppelganger). Perfect 10HP.


---

# New Game — 2026-03-10 17:09:39
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Dreamer, Judge, Gemcrafter, Hunter, Bishop
- Outcasts: Bombardier
- Minions: Minion
- Demons: Pooka

### [17:10:55] Revealed #1 Scout
Info: {'evil_role': 'Pooka', 'distance': 4}

### [17:11:00] Revealed #2 Hunter
Info: {'distance': 4}

### [17:11:06] Revealed #3 Gemcrafter
Info: {'good_position': 7}

### [17:11:11] Revealed #4 Dreamer
Info: {}

### [17:11:17] Revealed #5 Bishop
Info: {'targets': [4, 6, 7], 'types': ['Outcast', 'Villager', 'Minion']}

### [17:11:22] Revealed #6 Bombardier
Info: {}

### [17:11:28] Revealed #7 Scout
Info: {'evil_role': 'Minion', 'distance': 1}

### [17:11:36] Revealed #8 Judge
Info: {}

#### [17:11:41] Solver Output
Scenarios: 2/56
Definite evil: ['#7']
Definite good: ['#1', '#4', '#5', '#6', '#8']
Evil probabilities: #2=50%, #3=50%
  Generated 56 candidate scenarios
  2 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3]

#### [17:11:41] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Minion'})

### [17:12:22] Executed #7 -> Minion (EVIL)

#### [17:12:28] Solver Output
Scenarios: 2/7
Definite evil: ['#7']
Definite good: ['#1', '#4', '#5', '#6', '#8']
Evil probabilities: #2=50%, #3=50%
  Generated 7 candidate scenarios
  2 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3]

#### [17:12:28] Recommendation
Action: **USE_ABILITY** #8 (Judge) -> targets ['#1']
Reason: Expected posterior 1.0 scenarios (adjusted 1.0, info gain 1.000 bits) | timing x1.00

### [17:13:30] Revealed #8 Judge
Info: {'target': 1, 'is_lying': False}

### [17:13:36] Ability used at #8

#### [17:13:41] Solver Output
Scenarios: 1/7
Definite evil: ['#3', '#7']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#8']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #7 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [17:13:41] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [17:14:23] Executed #3 -> Pooka (EVIL)

## [17:14:31] GAME OVER — WIN
Final HP: 10
Notes: Judge #8 on #1 = truthful, confirmed #3=Pooka. Corrupted: #2 Hunter, #4 Dreamer. Perfect 10HP.


---

# New Game — 2026-03-10 17:15:25
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Oracle, Medium, Baker, Dreamer, Gemcrafter, Empress
- Outcasts: Bombardier, Plague_Doctor
- Minions: Witch, Minion
- Demons: Pooka

### [17:17:02] Revealed #1 Medium
Info: {'good_position': 2, 'good_role': 'Baker'}

### [17:17:09] Revealed #2 Baker
Info: {'original_role': 'original'}

### [17:18:30] Revealed #3 Empress
Info: {'targets': [2, 6, 7]}

### [17:18:37] Revealed #4 Bombardier
Info: {}

### [17:18:37] Revealed #5 Knitter
Info: {'evil_pairs': 2}

### [17:18:37] Revealed #6 Plague_Doctor
Info: {}

### [17:18:37] Revealed #7 Gemcrafter
Info: {'good_position': 8}

### [17:18:37] Revealed #8 Dreamer
Info: {}

### [17:18:37] Revealed #9 Baker
Info: {'original_role': 'original'}

#### [17:19:03] Solver Output
Scenarios: 20/2904
Definite good: ['#7', '#8']
Evil probabilities: #4=90%, #3=40%, #5=40%, #9=40%, #10=40%, #2=30%, #1=10%, #6=10%
  Generated 2904 candidate scenarios
  20 scenarios survived validation
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 9, 10]

#### [17:19:03] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.046 (adjusted 2.046) | timing x1.00

### [17:20:05] Ability used at #6

#### [17:20:11] Solver Output
Scenarios: 6/2904
Definite evil: ['#4', '#5']
Definite good: ['#1', '#3', '#6', '#7', '#8']
Evil probabilities: #2=33%, #9=33%, #10=33%
  Generated 2904 candidate scenarios
  6 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Minion', 'Pooka', 'Witch'})
    #5 is DEFINITELY EVIL (possible roles: {'Minion', 'Pooka', 'Witch'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 9, 10]

#### [17:20:11] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 6 scenarios (roles: {'Minion', 'Pooka', 'Witch'})

### [17:20:48] Executed #4 -> Minion (EVIL)

#### [17:20:48] Solver Output
Scenarios: 1/352
Definite evil: ['#2', '#4', '#5']
Definite good: ['#1', '#3', '#6', '#7', '#8', '#9', '#10']
  Generated 352 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Witch'})
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [17:20:48] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Witch'})

### [17:21:22] Executed #2 -> Witch (EVIL)

### [17:22:02] Executed #5 -> Pooka (EVIL)

## [17:22:02] GAME OVER — WIN
Final HP: 10
Notes: Witch blocked #10. PD clean on #5. Solver found #4+#5 definite evil, then 1 scenario after Minion exec. Medium #1 corrupted. Perfect 10HP.


---

# New Game — 2026-03-10 17:23:11
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Hunter, Baker, Alchemist, Bishop, Empress, Oracle
- Outcasts: Doppelganger, Drunk
- Minions: Twin_Minion
- Demons: Baa

### [17:24:18] Revealed #1 Hunter
Info: {'distance': 2}

### [17:24:18] Revealed #2 Architect
Info: {'side': 'left'}

### [17:24:18] Revealed #3 Alchemist
Info: {'cured_count': 0}

### [17:24:18] Revealed #4 Oracle
Info: {'targets': [2, 6], 'minion_role': 'Twin_Minion'}

### [17:24:18] Revealed #5 Empress
Info: {'targets': [3, 6, 8]}

### [17:24:18] Revealed #6 Bishop
Info: {'targets': [1, 4, 7], 'types': ['Villager', 'Outcast', 'Minion']}

### [17:24:19] Revealed #7 Empress
Info: {'targets': [1, 2, 8]}

### [17:24:19] Revealed #8 Baker
Info: {'original_role': 'original'}

#### [17:24:25] Solver Output
Scenarios: 29/2408
Definite good: ['#3', '#4', '#8']
Evil probabilities: #6=83%, #2=41%, #7=38%, #1=21%, #5=17%
  Generated 2408 candidate scenarios
  29 scenarios survived validation
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 7]

#### [17:24:25] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (62% evil Twin_Minion, 21% evil Baa, 14% good Bishop).
WARNING: Execution lookahead override -- immediate hit chance is 83%, but all reveal branches still lead to a forced win.

### [17:25:11] Executed #6 -> GOOD (WRONG!)

#### [17:25:17] Solver Output
Scenarios: 5/1806
Definite evil: ['#5', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#8']
  Generated 1806 candidate scenarios
  5 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #7 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [17:25:17] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 5 scenarios (roles: {'Baa'})

### [17:25:55] Executed #5 -> Baa (EVIL)

### [17:26:37] Executed #7 -> Twin_Minion (EVIL)

## [17:26:37] GAME OVER — WIN
Final HP: 5
Notes: Wrong exec on #6 Bishop (5HP). Solver found #5+#7 definite evil after. #4 Drunk(Oracle) corrupted. Won at 5HP.


---

# New Game — 2026-03-10 17:29:12
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Witness, Enlightened, Bard, Lover, Poet
- Outcasts: Wretch, Plague_Doctor, Bombardier
- Minions: Poisoner
- Demons: Baa

### [17:32:17] Revealed #1 Lover
Info: {'evil_adjacent': 1}

### [17:32:27] Revealed #2 Jester
Info: {}

### [17:32:27] Revealed #3 Witness
Info: {'affected_position': 8}

### [17:32:27] Revealed #4 Bard
Info: {'corruption_distance': 2}

### [17:32:28] Revealed #5 Plague_Doctor
Info: {}

### [17:32:28] Revealed #6 Poet
Info: {'evil_role': 'Poisoner', 'distance': 3, 'copied_role': 'Scout'}

### [17:32:28] Revealed #7 Bombardier
Info: {}

### [17:32:28] Revealed #8 Enlightened
Info: {'direction': 'equidistant'}

#### [17:32:35] Solver Output
Scenarios: 2/274
Definite evil: ['#7']
Definite good: ['#1', '#3', '#5', '#6', '#8']
Evil probabilities: #2=50%, #4=50%
  Generated 274 candidate scenarios
  2 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 4]

#### [17:32:35] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Poisoner'})

### [17:33:16] Executed #7 -> Poisoner (EVIL)

#### [17:33:16] Solver Output
Scenarios: 2/52
Definite evil: ['#7']
Definite good: ['#1', '#3', '#5', '#6', '#8']
Evil probabilities: #2=50%, #4=50%
  Generated 52 candidate scenarios
  2 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 4]

#### [17:33:16] Recommendation
Action: **USE_ABILITY** #2 (Jester) -> targets ['#1', '#3', '#5']
Reason: Expected posterior 1.0 scenarios (adjusted 1.0, info gain 1.000 bits) | timing x1.00

### [17:34:30] Revealed #2 Jester
Info: {'targets': [1, 3, 5], 'evil_count': 2}

### [17:34:30] Ability used at #2

#### [17:34:30] Solver Output
Scenarios: 1/52
Definite evil: ['#2', '#7']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#8']
  Generated 52 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [17:34:30] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Baa'})

### [17:35:13] Executed #2 -> Baa (EVIL)

## [17:35:13] GAME OVER — WIN
Final HP: 10
Notes: Jester #2 lied (2 evils among confirmed good). #8 Enlightened + #4 Bard corrupted by Poisoner. Perfect 10HP.


---

# New Game — 2026-03-10 17:36:16
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Poet, Druid, Medium, Fortune_Teller, Scout, Judge
- Outcasts: Bombardier, Plague_Doctor
- Minions: Minion, Chancellor
- Demons: Pooka

### [17:37:32] Revealed #1 Medium
Info: {'good_position': 8, 'good_role': 'Bombardier'}

### [17:37:32] Revealed #2 Poet
Info: {'targets': [5, 6, 10], 'copied_role': 'Empress'}

### [17:37:33] Revealed #3 Bombardier
Info: {}

### [17:37:33] Revealed #4 Fortune_Teller
Info: {}

### [17:37:33] Revealed #5 Scout
Info: {'evil_role': 'Pooka', 'distance': 2}

### [17:37:33] Revealed #6 Judge
Info: {}

### [17:37:33] Revealed #7 Druid
Info: {}

### [17:37:33] Revealed #8 Bombardier
Info: {}

### [17:37:33] Revealed #9 Scout
Info: {'evil_role': 'Chancellor', 'distance': 3}

### [17:37:34] Revealed #10 Plague_Doctor
Info: {}

#### [17:37:42] Solver Output
Scenarios: 66/3186
Definite good: ['#6', '#10']
Evil probabilities: #5=62%, #9=62%, #8=55%, #3=52%, #1=26%, #4=20%, #7=17%, #2=8%
  Generated 3186 candidate scenarios
  66 scenarios survived validation
    #6 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [17:37:42] Recommendation
Action: **USE_ABILITY** #7 (Druid) -> targets ['#1', '#2', '#8']
Reason: Entropy 1.970 (adjusted 1.731) | timing x1.00
WARNING: Corruption risk: 24%

### [17:41:33] Revealed #7 Druid
Info: {'targets': [1, 2, 8], 'found_outcast': None}

### [17:41:38] Ability used at #7

#### [17:41:43] Solver Output
Scenarios: 33/3186
Definite good: ['#6', '#10']
Evil probabilities: #9=70%, #8=64%, #5=52%, #1=36%, #3=36%, #7=24%, #4=12%, #2=6%
  Generated 3186 candidate scenarios
  33 scenarios survived validation
    #6 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [17:41:43] Recommendation
Action: **USE_ABILITY** #10 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.777 (adjusted 1.777) | timing x1.00

### [17:42:40] Ability used at #10

#### [17:42:45] Solver Output
Scenarios: 7/3186
Definite evil: ['#9']
Definite good: ['#2', '#4', '#6', '#10']
Evil probabilities: #8=57%, #3=43%, #7=43%, #1=29%, #5=29%
  Generated 3186 candidate scenarios
  7 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Minion', 'Chancellor'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 7, 8]

#### [17:42:45] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 7 scenarios (roles: {'Minion', 'Chancellor'})

### [17:43:19] Executed #9 -> Chancellor (EVIL)

#### [17:43:24] Solver Output
Scenarios: 3/268
Definite evil: ['#9']
Definite good: ['#2', '#4', '#6', '#10']
Evil probabilities: #8=67%, #1=33%, #3=33%, #5=33%, #7=33%
  Generated 268 candidate scenarios
  3 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 7, 8]

#### [17:43:24] Recommendation
Action: **USE_ABILITY** #4 (Fortune Teller) -> targets ['#6', '#8']
Reason: Entropy 0.918 (adjusted 0.765) | follow-up bonus 0.667 | timing x1.00
WARNING: Corruption risk: 33%

### [17:44:20] Revealed #4 Ft
Info: {}

### [17:44:26] Ability used at #4

#### [17:44:31] Solver Output
Scenarios: 3/226
Definite evil: ['#9']
Definite good: ['#2', '#4', '#6', '#10']
Evil probabilities: #8=67%, #1=33%, #3=33%, #5=33%, #7=33%
  Generated 226 candidate scenarios
  3 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 7, 8]

#### [17:44:31] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#1']
Reason: Expected posterior 3.0 scenarios (adjusted 4.5, info gain 0.000 bits) | timing x1.00
WARNING: Corruption risk: 100% -- corrupted Judge results are unreliable

### [17:45:26] Revealed #6 Judge
Info: {'target': 1, 'is_lying': False}

### [17:45:31] Ability used at #6

#### [17:45:37] Solver Output
Scenarios: 3/226
Definite evil: ['#9']
Definite good: ['#2', '#4', '#6', '#10']
Evil probabilities: #8=67%, #1=33%, #3=33%, #5=33%, #7=33%
  Generated 226 candidate scenarios
  3 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 7, 8]

#### [17:45:37] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (33% good Medium, 33% good Medium (corrupted), 33% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 33%, but all reveal branches still lead to a forced win.

### [17:46:18] Executed #1 -> GOOD (WRONG!)

#### [17:46:28] Solver Output
Scenarios: 1/176
Definite evil: ['#3', '#7', '#9']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#8', '#10']
  Generated 176 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [17:46:28] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Minion'})

## [17:48:26] GAME OVER — LOSS
Final HP: 5
Notes: Solver had 1 wrong scenario: predicted #3=Minion #7=Pooka but truth was #5=Pooka #8=Minion. Executed #3 (real Bombardier) = instant loss. Corrupted: #1(PD), #4/#6(Pooka). Need to investigate why correct scenario was eliminated.

### [17:52:58] Revealed #4 Fortune Teller
Info: {'targets': [6, 8], 'has_evil': False}


---

# New Game — 2026-03-10 17:56:21
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Fortune_Teller, Jester, Bishop, Enlightened, Bard, Baker, Architect
- Outcasts: Drunk, Wretch
- Minions: Minion, Twin_Minion
- Demons: Baa

### [17:58:14] Revealed #1 Hunter
Info: {'distance': 1}

### [17:58:14] Revealed #2 Hunter
Info: {'distance': 3}

### [17:58:15] Revealed #3 Jester
Info: {}

### [17:58:15] Revealed #4 Enlightened
Info: {'direction': 'Equidistant'}

### [17:58:15] Revealed #5 Bard
Info: {'corruption_distance': 3}

### [17:58:15] Revealed #6 Architect
Info: {'side': 'Equal'}

### [17:58:15] Revealed #7 Bishop
Info: {'targets': [2, 3, 4], 'types': ['Outcast', 'Minion', 'Villager']}

### [17:58:15] Revealed #8 Baker
Info: {'original_role': 'original'}

### [17:58:15] Revealed #9 Fortune_Teller
Info: {}

#### [17:58:21] Solver Output
Scenarios: 16/3528
Definite good: ['#3', '#8']
Evil probabilities: #2=62%, #6=62%, #7=50%, #4=38%, #9=38%, #1=25%, #5=25%
  Generated 3528 candidate scenarios
  16 scenarios survived validation
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7, 9]

#### [17:58:21] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#4', '#5']
Reason: Expected posterior 5.5 scenarios (adjusted 5.5, info gain 1.541 bits) | timing x1.00

### [17:59:21] Revealed #3 Jester
Info: {'targets': [1, 4, 5], 'evil_count': 1}

### [17:59:22] Ability used at #3

#### [17:59:22] Solver Output
Scenarios: 6/3528
Definite evil: ['#9']
Definite good: ['#1', '#3', '#6', '#8']
Evil probabilities: #2=67%, #5=67%, #4=33%, #7=33%
  Generated 3528 candidate scenarios
  6 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Baa', 'Twin_Minion', 'Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 7]

#### [17:59:22] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 6 scenarios (roles: {'Baa', 'Twin_Minion', 'Minion'})

### [17:59:55] Executed #9 -> Minion (EVIL)

#### [17:59:55] Solver Output
Scenarios: 2/392
Definite evil: ['#9']
Definite good: ['#1', '#3', '#6', '#8']
Evil probabilities: #2=50%, #4=50%, #5=50%, #7=50%
  Generated 392 candidate scenarios
  2 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 7]

#### [17:59:55] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (50% good Drunk (corrupted), 50% evil Twin_Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [18:00:31] Executed #2 -> Twin_Minion (EVIL)

#### [18:00:31] Solver Output
Scenarios: 1/49
Definite evil: ['#2', '#5', '#9']
Definite good: ['#1', '#3', '#4', '#6', '#7', '#8']
  Generated 49 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #9 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [18:00:31] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Baa'})

### [18:01:08] Executed #5 -> Baa (EVIL)

## [18:01:08] GAME OVER — WIN
Final HP: 10
Notes: Perfect game 10HP. Jester narrowed to 6 scenarios, exec #9 Minion, exec #2 Twin_Minion, exec #5 Baa.


---

# New Game — 2026-03-10 18:03:00
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Druid, Knitter, Slayer, Architect, Gemcrafter, Enlightened
- Outcasts: Doppelganger, Plague_Doctor
- Minions: Chancellor
- Demons: Lilis

### [18:04:03] Revealed #1 Enlightened
Info: {'direction': 'CCW'}

### [18:04:04] Revealed #2 Knitter
Info: {'evil_pairs': 0}

### [18:04:04] Revealed #3 Gemcrafter
Info: {'good_position': 2}

### [18:04:04] Revealed #4 Plague_Doctor
Info: {}

### [18:05:05] Revealed #5 Knitter
Info: {'evil_pairs': 1}

### [18:05:05] Revealed #6 Druid
Info: {}

### [18:05:05] Revealed #7 Architect
Info: {'side': 'Equal'}

#### [18:05:05] Solver Output
Scenarios: 24/1140
Definite good: ['#2', '#3', '#4', '#8']
Evil probabilities: #5=67%, #7=67%, #1=33%, #6=33%
  Generated 1140 candidate scenarios
  24 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 5, 6, 7]

#### [18:05:05] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.585 (adjusted 1.585) | timing x1.00

### [18:05:58] Ability used at #4

#### [18:05:58] Solver Output
Scenarios: 8/1140
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']
  Generated 1140 candidate scenarios
  8 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [18:05:58] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 8 scenarios (roles: {'Chancellor', 'Lilis'})

### [18:06:36] Executed #6 -> Chancellor (EVIL)

#### [18:06:36] Solver Output
Scenarios: 4/188
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']
  Generated 188 candidate scenarios
  4 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #7 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [18:06:36] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 4 scenarios (roles: {'Lilis'})

### [18:07:16] Executed #7 -> Lilis (EVIL)

## [18:07:16] GAME OVER — WIN
Final HP: 8
Notes: Lilis game, 8HP. Night killed #8. PD clean check on #6 confirmed evil. Both evils adjacent (#6,#7). Corrupted: #2 Knitter.


---

# New Game — 2026-03-10 18:08:15
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Enlightened, Gemcrafter, Bishop, Lover, Alchemist
- Outcasts: Doppelganger
- Minions: Minion
- Demons: Pooka

### [18:09:32] Revealed #1 Bishop
Info: {'targets': [4, 5, 6], 'types': ['Villager', 'Outcast', 'Minion']}

### [18:09:32] Revealed #2 Lover
Info: {'evil_adjacent': 1}

### [18:09:32] Revealed #3 Gemcrafter
Info: {'good_position': 5}

### [18:09:32] Revealed #4 Alchemist
Info: {'cured_count': 1}

### [18:09:33] Revealed #5 Lover
Info: {'evil_adjacent': 1}

### [18:09:33] Revealed #6 Alchemist
Info: {'cured_count': 0}

### [18:09:33] Revealed #7 Druid
Info: {}

### [18:09:33] Revealed #8 Enlightened
Info: {'direction': 'Equidistant'}

#### [18:09:40] Solver Output
Scenarios: 1/336
Definite evil: ['#3', '#5']
Definite good: ['#1', '#2', '#4', '#6', '#7', '#8']
  Generated 336 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [18:09:40] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [18:10:18] Executed #3 -> Pooka (EVIL)

### [18:10:51] Executed #5 -> Minion (EVIL)

## [18:10:51] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP, solver found single scenario immediately. Bishop+Lover+Alchemist constraints very powerful.


---

# New Game — 2026-03-10 18:11:48
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Jester, Knitter, Slayer, Baker, Lover, Architect
- Outcasts: Plague_Doctor
- Minions: Shaman
- Demons: Lilis

### [18:12:49] Revealed #1 Judge
Info: {}

### [18:12:49] Revealed #2 Plague_Doctor
Info: {}

### [18:12:49] Revealed #3 Baker
Info: {'original_role': 'original'}

### [18:12:50] Revealed #4 Architect
Info: {'side': 'Right'}

### [18:13:28] Revealed #6 Knitter
Info: {'evil_pairs': 1}

### [18:13:28] Revealed #7 Jester
Info: {}

### [18:13:28] Revealed #8 Lover
Info: {'evil_adjacent': 2}

#### [18:13:29] Solver Output
Scenarios: 14/224
Definite good: ['#2', '#5']
Evil probabilities: #8=86%, #1=29%, #4=29%, #6=29%, #3=14%, #7=14%
  Generated 224 candidate scenarios
  14 scenarios survived validation
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 6, 7, 8]

#### [18:13:29] Recommendation
Action: **USE_ABILITY** #7 (Jester) -> targets ['#1', '#3', '#8']
Reason: Expected posterior 5.6 scenarios (adjusted 6.0, info gain 1.213 bits) | timing x1.00
WARNING: Corruption risk: 14%

### [18:14:29] Revealed #7 Jester
Info: {'targets': [1, 3, 8], 'evil_count': 0}

### [18:14:29] Ability used at #7

#### [18:14:29] Solver Output
Scenarios: 6/224
Definite good: ['#2', '#3', '#5']
Evil probabilities: #8=67%, #1=33%, #4=33%, #6=33%, #7=33%
  Generated 224 candidate scenarios
  6 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 4, 6, 7, 8]

#### [18:14:29] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#4']
Reason: Entropy 0.918 (adjusted 0.918) | timing x1.00

### [18:16:40] Ability used at #2

#### [18:16:43] Solver Output
Scenarios: 4/224
Definite good: ['#2', '#3', '#5', '#7']
Evil probabilities: #1=50%, #4=50%, #6=50%, #8=50%
  Generated 224 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 4, 6, 8]

#### [18:16:43] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#2']
Reason: Expected posterior 2.0 scenarios (adjusted 2.0, info gain 1.000 bits) | timing x1.00

### [18:19:10] Revealed #1 Judge
Info: {'target': 1, 'is_lying': False}

### [18:19:14] Ability used at #1

#### [18:19:18] Solver Output
Scenarios: 4/224
Definite good: ['#2', '#3', '#5', '#7']
Evil probabilities: #1=50%, #4=50%, #6=50%, #8=50%
  Generated 224 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 4, 6, 8]

#### [18:19:18] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% good Judge, 25% evil Lilis, 25% evil Shaman).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [18:20:10] Executed #1 -> GOOD (WRONG!)

#### [18:20:19] Solver Output
Scenarios: 2/162
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8']
  Generated 162 candidate scenarios
  2 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis', 'Shaman'})
    #6 is DEFINITELY EVIL (possible roles: {'Lilis', 'Shaman'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [18:20:19] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Lilis', 'Shaman'})

### [18:20:50] Executed #4 -> Lilis (EVIL)

#### [18:20:54] Solver Output
Scenarios: 1/26
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8']
  Generated 26 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [18:20:54] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Shaman'})

## [18:21:52] GAME OVER — WIN
Final HP: 3
Notes: Win 3HP. Lilis game, night killed #5. Judge self-targeted (wasted ability). PD found #4 clean. Corrupted: #8(Lover). Wrong exec #1 cost 5HP.


---

# New Game — 2026-03-10 18:25:23
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Bishop, Lover, Judge, Enlightened, Baker, Gemcrafter, Slayer
- Outcasts: Bombardier, Drunk
- Minions: Poisoner, Puppeteer
- Demons: Lilis

### [18:27:16] Revealed #1 Gemcrafter
Info: {'good_position': 3}

### [18:27:21] Revealed #2 Bombardier
Info: {}

### [18:27:26] Revealed #3 Judge
Info: {}

### [18:27:31] Revealed #4 Bishop
Info: {'targets': [1, 6, 8], 'types': ['Minion', 'Outcast', 'Villager']}

### [18:28:48] Revealed #5 Slayer
Info: {}

### [18:28:53] Revealed #6 Knitter
Info: {'evil_pairs': 1}

### [18:28:58] Revealed #7 Lover
Info: {'evil_adjacent': 2}

### [18:29:03] Revealed #8 Baker
Info: {'original_role': 'Baker'}

#### [18:29:14] Solver Output
Scenarios: 142/5268
Definite good: ['#9']
Evil probabilities: #8=76%, #7=75%, #4=69%, #6=48%, #5=45%, #2=42%, #1=26%, #3=19%
  Generated 5268 candidate scenarios
  142 scenarios survived validation
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [18:29:14] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#8']
Reason: Target #8 is 76% evil (adjusted 0.61)
WARNING: Corruption risk: 20% -- Slayer ability disabled if corrupted

### [18:30:34] Ability used at #5

#### [18:30:40] Solver Output
Scenarios: 20/418
Definite evil: ['#8']
Definite good: ['#9']
Evil probabilities: #4=70%, #6=70%, #7=50%, #5=40%, #2=35%, #1=20%, #3=15%
  Generated 418 candidate scenarios
  20 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [18:30:40] Recommendation
Action: **USE_ABILITY** #3 (Judge) -> targets ['#2']
Reason: Expected posterior 12.0 scenarios (adjusted 13.2, info gain 0.599 bits) | timing x1.00
WARNING: Corruption risk: 20% -- corrupted Judge results are unreliable

### [18:31:34] Revealed #3 Judge
Info: {'target': 2, 'is_lying': False}

### [18:31:39] Ability used at #3

#### [18:31:45] Solver Output
Scenarios: 12/418
Definite evil: ['#8']
Definite good: ['#9']
Evil probabilities: #4=92%, #6=83%, #5=58%, #7=42%, #1=8%, #2=8%, #3=8%
  Generated 418 candidate scenarios
  12 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [18:31:45] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (50% evil Lilis, 42% evil Puppeteer, 8% good Bishop).
WARNING: Execution lookahead override -- immediate hit chance is 92%, but all reveal branches still lead to a forced win.

### [18:32:34] Executed #4 -> GOOD (WRONG!)

#### [18:32:45] Solver Output
Scenarios: 1/276
Definite evil: ['#2', '#6', '#7', '#8']
Definite good: ['#1', '#3', '#4', '#5', '#9']
  Generated 276 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #8 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [18:32:45] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [18:33:55] Executed #2 -> Poisoner (EVIL)

#### [18:34:01] Solver Output
Scenarios: 7/70
Definite evil: ['#2', '#8']
Definite good: ['#1', '#4', '#5', '#9']
Evil probabilities: #6=43%, #7=43%, #3=14%
  Generated 70 candidate scenarios
  7 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #8 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 6, 7]

#### [18:34:01] Recommendation
Action: **ERROR** #6
Reason: #6 is 43% likely evil but HP too low to risk (HP=1, cost=5). Need more info.
WARNING: Probabilistic execution -- 43% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

#### [18:35:27] Solver Output
Scenarios: 1/96
Definite evil: ['#2', '#3', '#7', '#8']
Definite good: ['#1', '#4', '#5', '#6', '#9']
  Generated 96 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #8 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [18:35:27] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [18:36:06] Executed #3 -> Lilis (EVIL)

## [18:36:55] GAME OVER — WIN
Final HP: 1
Notes: Clutch win 1HP! Lilis game. Slayer killed Puppet#8. Wrong exec Bishop#4 cost 5HP. Needed to manually add Puppet role to executed_evil_roles for solver to deduce Puppeteer at #7. Corrupted: #1(Gemcrafter by Poisoner), #6(Drunk always corrupted).


---

# New Game — 2026-03-10 18:39:14
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Poet, Scout, Gemcrafter, Judge, Enlightened, Empress
- Outcasts: Plague_Doctor, Doppelganger
- Minions: Minion, Chancellor
- Demons: Pooka

### [18:41:46] Revealed #1 Poet
Info: {'copied_role': '2,9'}

### [18:42:13] Revealed #1 Poet
Info: {'evil_adjacent': 1, 'copied_role': 'Lover'}

### [18:42:19] Revealed #2 Scout
Info: {'evil_role': 'Chancellor', 'distance': 3}

### [18:42:25] Revealed #3 Gemcrafter
Info: {'good_position': 8}

### [18:42:33] Revealed #4 Plague_Doctor
Info: {}

### [18:42:33] Revealed #5 Empress
Info: {'targets': [1, 3, 7]}

### [18:42:33] Revealed #6 Druid
Info: {}

### [18:42:33] Revealed #7 Scout
Info: {'evil_role': 'Minion', 'distance': 2}

### [18:42:33] Revealed #8 Druid
Info: {}

### [18:42:33] Revealed #9 Enlightened
Info: {'direction': 'ccw'}

#### [18:42:40] Solver Output
Scenarios: 169/12612
Evil probabilities: #2=60%, #6=56%, #1=40%, #8=40%, #9=30%, #7=25%, #3=21%, #5=20%, #4=9%
  Generated 12612 candidate scenarios
  169 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [18:42:40] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#9']
Reason: Entropy 1.639 (adjusted 1.639) | timing x1.00

### [18:43:40] Ability used at #4

#### [18:43:46] Solver Output
Scenarios: 112/12612
Evil probabilities: #2=71%, #9=45%, #6=41%, #8=41%, #1=29%, #5=27%, #3=23%, #4=14%, #7=9%
  Generated 12612 candidate scenarios
  112 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [18:43:46] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 1.432 (adjusted 1.368) | timing x1.00
WARNING: Corruption risk: 9%

### [18:44:55] Revealed #8 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': None}

### [18:44:55] Ability used at #8

#### [18:45:03] Solver Output
Scenarios: 72/12612
Evil probabilities: #8=64%, #6=58%, #2=56%, #3=36%, #1=28%, #4=22%, #7=14%, #9=14%, #5=8%
  Generated 12612 candidate scenarios
  72 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [18:45:03] Recommendation
Action: **USE_ABILITY** #6 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 1.516 (adjusted 1.432) | timing x1.00
WARNING: Corruption risk: 11%

### [18:46:11] Revealed #6 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': None}

### [18:46:11] Ability used at #6

#### [18:46:20] Solver Output
Scenarios: 34/12612
Definite evil: ['#8']
Definite good: ['#1', '#4', '#5', '#9']
Evil probabilities: #6=76%, #2=53%, #3=53%, #7=18%
  Generated 12612 candidate scenarios
  34 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Pooka', 'Minion', 'Chancellor'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 6, 7]

#### [18:46:20] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 34 scenarios (roles: {'Pooka', 'Minion', 'Chancellor'})

### [18:48:35] Executed #8 -> Minion (EVIL)

#### [18:48:35] Solver Output
Scenarios: 11/1482
Definite evil: ['#8']
Definite good: ['#1', '#4', '#5', '#9']
Evil probabilities: #6=91%, #3=73%, #2=27%, #7=9%
  Generated 1482 candidate scenarios
  11 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 6, 7]

#### [18:48:35] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (91% evil Chancellor, 9% good Druid (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 91%, but all reveal branches still lead to a forced win.

### [18:49:13] Executed #6 -> Chancellor (EVIL)

#### [18:49:13] Solver Output
Scenarios: 10/262
Definite evil: ['#6', '#8']
Definite good: ['#1', '#4', '#5', '#7', '#9']
Evil probabilities: #3=80%, #2=20%
  Generated 262 candidate scenarios
  10 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3]

#### [18:49:13] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (80% evil Pooka, 20% good Gemcrafter (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 80%, but all reveal branches still lead to a forced win.

## [18:50:07] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Druid abilities narrowed down, then execution lookahead guaranteed wins. Corrupted: #1(Poet), #2(Scout) from Pooka#3. #7 was Doppelganger (as Scout). Final village of Asc29!


---

# New Game — 2026-03-10 18:55:25
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Alchemist, Fortune_Teller, Medium, Slayer, Scout
- Outcasts: Doppelganger, Plague_Doctor
- Minions: Witch
- Demons: Baa

#### [19:00:30] Solver Output
Scenarios: 56/56
Evil probabilities: #1=25%, #2=25%, #3=25%, #4=25%, #5=25%, #6=25%, #7=25%, #8=25%
  Generated 56 candidate scenarios
  56 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [19:00:30] Recommendation
Action: **REVEAL** #2
Reason: #2: 25% evil, entropy 0.911

### [19:00:52] Revealed #2 Alchemist
Info: {'cured_count': 0}

### [19:00:54] Revealed #3 Alchemist
Info: {'cured_count': 0}

### [19:00:56] Revealed #4 Medium
Info: {'good_position': 6, 'good_role': 'Medium'}

### [19:01:03] Revealed #5 Confessor
Info: {'dizzy': False}

### [19:01:07] Revealed #6 Medium
Info: {'good_position': 2, 'good_role': 'Alchemist'}

### [19:01:10] Revealed #7 Scout
Info: {'evil_role': 'Witch', 'distance': 2}

### [19:01:16] Revealed #8 Fortune_Teller
Info: {}

#### [19:01:20] Solver Output
Scenarios: 4/350
Definite evil: ['#4']
Definite good: ['#1', '#2', '#3', '#5', '#8']
Evil probabilities: #6=50%, #7=50%
  Generated 350 candidate scenarios
  4 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Witch', 'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [6, 7]

#### [19:01:20] Recommendation
Action: **USE_ABILITY** #8 (Fortune Teller) -> targets ['#1', '#6']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [19:02:11] Revealed #8 Fortune Teller
Info: {'targets': [1], 'has_evil': False}

### [19:03:33] Revealed #8 Fortune Teller
Info: {'targets': [1, 6], 'has_evil': True}

### [19:03:38] Ability used at #8

#### [19:03:42] Solver Output
Scenarios: 2/350
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8']
  Generated 350 candidate scenarios
  2 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Witch', 'Baa'})
    #6 is DEFINITELY EVIL (possible roles: {'Witch', 'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [19:03:42] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 50% evil Witch).

### [19:04:27] Executed #4 -> Baa (EVIL)

#### [19:04:32] Solver Output
Scenarios: 1/43
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8']
  Generated 43 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Baa'})
    #6 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [19:04:32] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (100% evil Witch).

### [19:06:50] Executed #6 -> Witch (EVIL)

## [19:06:57] GAME OVER — WIN
Final HP: 10
Notes: Perfect game 10HP. FT#8 on 1,6=True confirmed #6 evil. Witch blocked #1(Slayer). #2=Doppelganger appeared as Alchemist.


---

# New Game — 2026-03-10 19:07:04
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Druid, Bard, Architect, Gemcrafter, Knitter
- Outcasts: Wretch
- Minions: 
- Demons: Pooka

### [19:09:18] Revealed #1 Druid
Info: {}

### [19:09:24] Revealed #2 Architect
Info: {'side': 'left'}

### [19:09:29] Revealed #3 Bard
Info: {'corruption_distance': 3}

### [19:09:34] Revealed #4 Gemcrafter
Info: {'good_position': 7}

### [19:09:40] Revealed #5 Wretch
Info: {}

### [19:09:46] Revealed #6 Lover
Info: {'evil_adjacent': 0}

### [19:09:58] Revealed #7 Knitter
Info: {'evil_pairs': 0}

#### [19:10:09] Solver Output
Scenarios: 1/7
Definite evil: ['#6']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#7']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [19:10:09] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [19:10:47] Executed #6 -> Pooka (EVIL)

## [19:10:57] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. 1 scenario from all info. #7 Knitter corrupted by Pooka.


---

# New Game — 2026-03-10 19:11:34
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Dreamer, Druid, Slayer, Alchemist, Enlightened, Scout
- Outcasts: Drunk
- Minions: Puppeteer, Twin_Minion
- Demons: Lilis

### [19:13:46] Revealed #1 Scout
Info: {'evil_role': 'Twin_Minion', 'distance': 2}

### [19:13:52] Revealed #2 Alchemist
Info: {'cured_count': 0}

### [19:13:57] Revealed #3 Dreamer
Info: {}

### [19:14:03] Revealed #4 Scout
Info: {'evil_role': 'Twin_Minion', 'distance': 3}

#### [19:14:33] Solver Output
Scenarios: 253/3780
Definite good: ['#5']
Evil probabilities: #4=63%, #9=60%, #3=57%, #1=56%, #6=47%, #7=43%, #8=42%, #2=32%
  Generated 3780 candidate scenarios
  253 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:14:33] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#9']
Reason: Entropy 3.100 (adjusted 2.978) | timing x0.75
WARNING: Corruption risk: 8%

### [19:15:16] Revealed #3 Dreamer
Info: {'target': 9, 'evil_role': 'Lilis'}

### [19:15:25] Ability used at #3

#### [19:15:30] Solver Output
Scenarios: 176/3780
Definite good: ['#5']
Evil probabilities: #3=72%, #4=62%, #1=53%, #6=49%, #7=43%, #9=43%, #2=39%, #8=39%
  Generated 3780 candidate scenarios
  176 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:15:30] Recommendation
Action: **REVEAL** #6
Reason: #6: 49% evil, entropy 1.100

### [19:16:37] Revealed #6 Druid
Info: {}

#### [19:16:42] Solver Output
Scenarios: 176/3780
Definite good: ['#5']
Evil probabilities: #3=72%, #4=62%, #1=53%, #6=49%, #7=43%, #9=43%, #2=39%, #8=39%
  Generated 3780 candidate scenarios
  176 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:16:42] Recommendation
Action: **REVEAL** #7
Reason: #7: 43% evil, entropy 1.088

### [19:17:31] Revealed #7 Enlightened
Info: {'direction': 'Equidistant'}

#### [19:17:36] Solver Output
Scenarios: 90/3780
Definite good: ['#5']
Evil probabilities: #3=72%, #4=60%, #7=56%, #1=49%, #6=48%, #2=43%, #8=39%, #9=33%
  Generated 3780 candidate scenarios
  90 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:17:36] Recommendation
Action: **REVEAL** #8
Reason: #8: 39% evil, entropy 1.064

### [19:18:18] Revealed #8 Poet
Info: {'evil_role': 'Twin_Minion', 'distance': 3, 'copied_role': 'Scout'}

#### [19:18:23] Solver Output
Scenarios: 41/3780
Definite good: ['#5']
Evil probabilities: #3=61%, #6=61%, #8=61%, #7=56%, #4=54%, #1=44%, #2=41%, #9=22%
  Generated 3780 candidate scenarios
  41 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:18:23] Recommendation
Action: **USE_ABILITY** #6 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.947 (adjusted 0.924) | timing x1.00
WARNING: Corruption risk: 5%

### [19:20:01] Revealed #6 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [19:20:07] Ability used at #6

#### [19:20:14] Solver Output
Scenarios: 19/3780
Definite good: ['#5']
Evil probabilities: #4=63%, #7=63%, #1=53%, #2=53%, #3=53%, #8=47%, #6=42%, #9=26%
  Generated 3780 candidate scenarios
  19 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:20:14] Recommendation
Action: **REVEAL** #9
Reason: #9: 26% evil, entropy 0.931

### [19:20:59] Revealed #9 Slayer
Info: {}

#### [19:21:13] Solver Output
Scenarios: 19/3780
Definite good: ['#5']
Evil probabilities: #4=63%, #7=63%, #1=53%, #2=53%, #3=53%, #8=47%, #6=42%, #9=26%
  Generated 3780 candidate scenarios
  19 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:21:13] Recommendation
Action: **USE_ABILITY** #9 (Slayer) -> targets ['#4']
Reason: Target #4 is 63% evil (adjusted 0.57)
WARNING: Corruption risk: 11% -- Slayer ability disabled if corrupted

### [19:22:29] Ability used at #9

#### [19:22:35] Solver Output
Scenarios: 3/336
Definite evil: ['#4']
Definite good: ['#2', '#5', '#7', '#9']
Evil probabilities: #6=67%, #8=67%, #1=33%, #3=33%
  Generated 336 candidate scenarios
  3 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 6, 8]

#### [19:22:35] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (33% good Druid, 33% evil Lilis, 33% evil Twin_Minion).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [19:23:21] Executed #6 -> Lilis (EVIL)

#### [19:23:25] Solver Output
Scenarios: 1/42
Definite evil: ['#4', '#6', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#9']
  Generated 42 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [19:23:25] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [19:24:29] Executed #8 -> Puppeteer (EVIL)

#### [19:24:37] Solver Output
Scenarios: 3/10
Definite evil: ['#4', '#6', '#8']
Definite good: ['#1', '#2', '#3', '#5']
Evil probabilities: #7=67%, #9=33%
  Generated 10 candidate scenarios
  3 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [7, 9]

#### [19:24:37] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (67% evil Puppet, 33% good Enlightened).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [19:25:21] Executed #7 -> GOOD (WRONG!)

#### [19:25:28] Solver Output
Scenarios: 1/5
Definite evil: ['#4', '#6', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#7']
  Generated 5 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #9 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [19:25:28] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [19:26:09] Executed #9 -> Puppet (EVIL)

## [19:26:18] GAME OVER — WIN
Final HP: 1
Notes: Lilis game, 1HP clutch. Puppet#9 disguised as Slayer - its ability WORKED and killed Twin_Minion#4! Drunk#3 disguised as Dreamer was corrupted. Wrong exec on #7 Enlightened cost 5HP. Night killed #5 Poet.


---

# New Game — 2026-03-10 19:40:01
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Poet, Knitter, Knight, Oracle, Dreamer, Druid
- Outcasts: Wretch, Drunk
- Minions: Witch
- Demons: Baa

### [19:42:19] Revealed #1 Druid
Info: {}

### [19:42:19] Revealed #2 Confessor
Info: {'dizzy': False}

### [19:42:19] Revealed #3 Oracle
Info: {'targets': [5, 8], 'minion_role': 'Witch'}

### [19:42:19] Revealed #4 Dreamer
Info: {}

### [19:42:50] Revealed #5 Poet
Info: {}

### [19:43:11] Revealed #6 Knitter
Info: {'evil_pairs': 1}

### [19:43:11] Revealed #7 Knight
Info: {}

#### [19:43:19] Solver Output
Scenarios: 49/392
Definite good: ['#2']
Evil probabilities: #3=53%, #6=41%, #4=35%, #8=27%, #1=16%, #5=16%, #7=12%
  Generated 392 candidate scenarios
  49 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8]

#### [19:43:19] Recommendation
Action: **EXECUTE** #7
Reason: Knight check: #7 is 12% evil, 14% corruption risk. Expected HP cost: 1.1 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 14% -- corrupted Knight loses immunity + 4 extra damage

### [19:44:11] Executed #7 -> GOOD (WRONG!)

#### [19:44:17] Solver Output
Scenarios: 36/294
Definite good: ['#2', '#7']
Evil probabilities: #3=56%, #6=47%, #4=39%, #8=22%, #1=19%, #5=17%
  Generated 294 candidate scenarios
  36 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8]

#### [19:44:17] Recommendation
Action: **USE_ABILITY** #4 (Dreamer) -> targets ['#3']
Reason: Entropy 2.547 (adjusted 2.405) | timing x1.00
WARNING: Corruption risk: 11%

### [19:45:02] Revealed #4 Dreamer
Info: {'target': 3, 'evil_role': 'Baa'}

### [19:45:09] Ability used at #4

#### [19:45:16] Solver Output
Scenarios: 26/294
Definite good: ['#2', '#7']
Evil probabilities: #6=46%, #3=38%, #4=38%, #8=31%, #1=23%, #5=23%
  Generated 294 candidate scenarios
  26 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8]

#### [19:45:16] Recommendation
Action: **USE_ABILITY** #1 (Druid) -> targets ['#2', '#3', '#4']
Reason: Entropy 0.961 (adjusted 0.887) | timing x1.00
WARNING: Corruption risk: 15%

### [19:47:26] Revealed #1 Druid
Info: {'targets': [2, 3, 4], 'found_outcast': None}

### [19:47:30] Ability used at #1

#### [19:47:35] Solver Output
Scenarios: 15/294
Definite good: ['#2', '#7']
Evil probabilities: #3=40%, #4=40%, #6=40%, #8=33%, #5=27%, #1=20%
  Generated 294 candidate scenarios
  15 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8]

#### [19:47:35] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 40% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 40% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Low confidence (40%) -- consider gathering more info

### [19:48:19] Executed #3 -> Witch (EVIL)

### [19:51:45] Revealed #8 Dreamer
Info: {}

### [19:52:39] Revealed #8 Dreamer
Info: {'target': 6, 'evil_role': 'Baa'}

### [19:52:44] Ability used at #8

#### [19:52:49] Solver Output
Scenarios: 2/42
Definite evil: ['#3', '#4']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8']
  Generated 42 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #4 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [19:52:49] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Baa'})

### [19:53:21] Executed #4 -> Baa (EVIL)

## [19:53:28] GAME OVER — WIN
Final HP: 10
Notes: Clean win 10HP. Dreamer#8 was Drunk(Corrupted), gave false info '#6 could be Baa'. Druid#1 found no outcasts among 2,3,4. Knight#7 free check confirmed good. Solver nailed both evils.


---

# New Game — 2026-03-10 19:55:54
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Medium, Confessor, Fortune_Teller, Slayer, Hunter
- Outcasts: Plague_Doctor, Bombardier, Doppelganger
- Minions: Chancellor
- Demons: Lilis

### [19:58:03] Revealed #1 Medium
Info: {'good_position': 6, 'good_role': 'Doppelganger'}

### [19:58:09] Revealed #2 Plague_Doctor
Info: {}

### [19:58:14] Revealed #3 Enlightened
Info: {'direction': 'CCW'}

### [19:58:19] Revealed #4 Enlightened
Info: {'direction': 'CW'}

### [19:59:33] Revealed #5 Bombardier
Info: {}

### [19:59:39] Revealed #6 Medium
Info: {'good_position': 7, 'good_role': 'Slayer'}

### [19:59:44] Revealed #7 Slayer
Info: {}

### [19:59:49] Revealed #8 Ft
Info: {}

#### [19:59:59] Solver Output
Scenarios: 7/1010
Definite good: ['#2', '#7', '#8', '#9']
Evil probabilities: #3=71%, #5=43%, #1=29%, #4=29%, #6=29%
  Generated 1010 candidate scenarios
  7 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6]

#### [19:59:59] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#7']
Reason: Entropy 1.557 (adjusted 1.557) | timing x1.00

### [20:01:04] Ability used at #2

#### [20:01:09] Solver Output
Scenarios: 3/1010
Definite evil: ['#3', '#5']
Definite good: ['#1', '#2', '#4', '#6', '#7', '#8', '#9']
  Generated 1010 candidate scenarios
  3 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis', 'Chancellor'})
    #5 is DEFINITELY EVIL (possible roles: {'Lilis', 'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [20:01:09] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 3 scenarios (roles: {'Lilis', 'Chancellor'})

### [20:02:02] Executed #3 -> Lilis (EVIL)

#### [20:02:10] Solver Output
Scenarios: 2/104
Definite evil: ['#3', '#5']
Definite good: ['#1', '#2', '#4', '#6', '#7', '#8', '#9']
  Generated 104 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #5 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [20:02:10] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 2 scenarios (roles: {'Chancellor'})

### [20:04:00] Executed #5 -> GOOD (WRONG!)

## [20:04:10] GAME OVER — LOSS
Final HP: 6
Notes: BOMBARDIER LOSS. Solver said #5 DEFINITELY EVIL but was Good Bombardier. True evils #3=Lilis #4=Chancellor. Bug: Chancellor at #4 has neighbors #3(evil) and #5(Outcast) - no adjacent Villager to convert. Solver likely rejected this Chancellor placement. PD#2 corrupted #8(FT). Need to fix Chancellor constraint to allow no-conversion when no adjacent Villager exists.


---

# New Game — 2026-03-10 20:17:17
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Alchemist, Fortune_Teller, Slayer, Medium, Architect, Empress
- Outcasts: Wretch
- Minions: Minion
- Demons: Lilis

### [20:18:42] Revealed #1 Slayer
Info: {}

### [20:18:42] Revealed #2 Alchemist
Info: {'cured_count': 0}

### [20:18:42] Revealed #3 Empress
Info: {'targets': [5, 6, 8]}

### [20:18:42] Revealed #4 Oracle
Info: {'targets': [7, 8], 'minion_role': 'Minion'}

### [20:19:41] Revealed #6 Fortune_Teller
Info: {}

### [20:19:41] Revealed #7 Wretch
Info: {}

### [20:19:41] Revealed #8 Medium
Info: {'good_position': 4, 'good_role': 'Oracle'}

#### [20:19:48] Solver Output
Scenarios: 8/56
Definite good: ['#2', '#4', '#5', '#8']
Evil probabilities: #1=50%, #3=50%, #6=50%, #7=50%
  Generated 56 candidate scenarios
  8 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 6, 7]

#### [20:19:48] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#1', '#7']
Reason: Entropy 1.000 (adjusted 1.000) | follow-up bonus 0.250 | timing x1.00

### [20:21:03] Revealed #6 Fortune Teller
Info: {'targets': [1, 7], 'has_evil': True}

### [20:21:10] Ability used at #6

#### [20:21:18] Solver Output
Scenarios: 4/56
Definite evil: ['#3']
Definite good: ['#2', '#4', '#5', '#6', '#8']
Evil probabilities: #1=50%, #7=50%
  Generated 56 candidate scenarios
  4 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [20:21:18] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 4 scenarios (roles: {'Minion', 'Lilis'})

### [20:22:08] Executed #3 -> Minion (EVIL)

#### [20:22:15] Solver Output
Scenarios: 2/7
Definite evil: ['#3']
Definite good: ['#2', '#4', '#5', '#6', '#8']
Evil probabilities: #1=50%, #7=50%
  Generated 7 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [20:22:15] Recommendation
Action: **USE_ABILITY** #1 (Slayer) -> targets ['#7']
Reason: Target #7 is 50% evil (adjusted 0.25)
WARNING: Wretch kill risk: 50% -- costs 5 HP

### [20:23:48] Ability used at #1

#### [20:23:55] Solver Output
Scenarios: 1/7
Definite evil: ['#1', '#3']
Definite good: ['#2', '#4', '#5', '#6', '#7', '#8']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #3 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [20:23:55] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [20:24:38] Executed #1 -> Lilis (EVIL)

## [20:24:47] GAME OVER — WIN
Final HP: 8
Notes: Lilis game, 8HP. Fake Slayer #1 couldn't kill #7 Wretch - proved #1 evil. Night killed #5 Architect. FT#6 confirmed evil in {1,7}. Empress#3 correctly identified as evil. Clean solver deduction.


---

# New Game — 2026-03-10 20:25:58
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Bard, Poet, Enlightened, Judge, Knitter, Hunter
- Outcasts: Bombardier
- Minions: Puppeteer, Minion
- Demons: Lilis

### [20:27:45] Revealed #1 Enlightened
Info: {'direction': 'Equidistant'}

### [20:27:54] Revealed #2 Knitter
Info: {'evil_pairs': 0}

### [20:28:10] Revealed #3 Hunter
Info: {'distance': 1}

### [20:28:17] Revealed #4 Lover
Info: {'evil_adjacent': 0}

### [20:29:30] Revealed #5 Judge
Info: {}

### [20:29:30] Revealed #7 Bard
Info: {'corruption_distance': -1}

### [20:29:30] Revealed #8 Poet
Info: {'targets': [3, 4], 'minion_role': 'Puppeteer', 'copied_role': 'Oracle'}

### [20:29:31] Revealed #9 Judge
Info: {}

#### [20:29:38] Solver Output
Scenarios: 12/756
Definite evil: ['#2']
Definite good: ['#6']
Evil probabilities: #9=83%, #1=50%, #4=50%, #8=50%, #5=33%, #3=17%, #7=17%
  Generated 756 candidate scenarios
  12 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Lilis', 'Minion', 'Puppeteer'})
    #6 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 7, 8, 9]

#### [20:29:38] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 12 scenarios (roles: {'Lilis', 'Minion', 'Puppeteer'})

### [20:30:15] Executed #2 -> Lilis (EVIL)

#### [20:30:16] Solver Output
Scenarios: 5/84
Definite evil: ['#2']
Definite good: ['#6']
Evil probabilities: #9=80%, #4=60%, #1=40%, #5=40%, #8=40%, #3=20%, #7=20%
  Generated 84 candidate scenarios
  5 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 7, 8, 9]

#### [20:30:16] Recommendation
Action: **USE_ABILITY** #5 (Judge) -> targets ['#4']
Reason: Expected posterior 2.6 scenarios (adjusted 2.6, info gain 0.943 bits) | timing x1.00

### [20:31:09] Revealed #5 Judge
Info: {'target': 4, 'is_lying': False}

### [20:31:09] Ability used at #5

#### [20:31:09] Solver Output
Scenarios: 2/84
Definite evil: ['#2', '#8', '#9']
Definite good: ['#3', '#4', '#5', '#6']
Evil probabilities: #1=50%, #7=50%
  Generated 84 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion', 'Puppeteer'})
    #9 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Minion'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [20:31:09] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 2 scenarios (roles: {'Minion', 'Puppeteer'})

### [20:31:54] Executed #8 -> Minion (EVIL)

#### [20:31:54] Solver Output
Scenarios: 1/10
Definite evil: ['#1', '#2', '#8', '#9']
Definite good: ['#3', '#4', '#5', '#6', '#7']
  Generated 10 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #2 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #9 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [20:31:54] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [20:32:34] Executed #1 -> Puppet (EVIL)

### [20:33:17] Executed #9 -> Puppeteer (EVIL)

## [20:33:17] GAME OVER — WIN
Final HP: 6
Notes: Lilis+Puppeteer game 4 evils. 6HP. Judge#5 confirmed Lover#4 truthful, narrowed to 1 scenario. Bard#7 confirmed no corruption. Night killed #6 Bombardier. Solver nailed all 4 evils.


---

# New Game — 2026-03-10 20:36:10
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Oracle, Dreamer, Bishop, Gemcrafter, Knight, Enlightened, Architect
- Outcasts: Bombardier, Plague_Doctor
- Minions: Minion, Shaman
- Demons: Pooka

### [20:38:57] Revealed #1 Bombardier
Info: {}

### [20:39:02] Revealed #2 Architect
Info: {'side': 'CW'}

### [20:39:06] Revealed #3 Knight
Info: {}

### [20:39:09] Revealed #4 Plague_Doctor
Info: {}

### [20:39:13] Revealed #5 Gemcrafter
Info: {'good_position': 10}

### [20:39:17] Revealed #6 Enlightened
Info: {'direction': 'Equidistant'}

### [20:39:22] Revealed #7 Knight
Info: {}

### [20:39:26] Revealed #8 Dreamer
Info: {}

### [20:39:30] Revealed #9 Oracle
Info: {'targets': [4, 10], 'minion_role': 'Minion'}

### [20:39:34] Revealed #10 Bishop
Info: {'targets': [2, 6, 7], 'types': ['Minion', 'Outcast', 'Villager']}

#### [20:39:38] Solver Output
Scenarios: 26/2904
Definite evil: ['#1']
Definite good: ['#3', '#7']
Evil probabilities: #6=58%, #9=54%, #2=31%, #10=23%, #8=19%, #4=8%, #5=8%
  Generated 2904 candidate scenarios
  26 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Shaman', 'Minion', 'Pooka'})
    #3 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 6, 8, 9, 10]

#### [20:39:38] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 26 scenarios (roles: {'Shaman', 'Minion', 'Pooka'})

### [20:40:10] Executed #1 -> Shaman (EVIL)

#### [20:40:15] Solver Output
Scenarios: 10/352
Definite evil: ['#1']
Definite good: ['#3', '#4', '#7']
Evil probabilities: #6=60%, #9=60%, #10=30%, #2=20%, #8=20%, #5=10%
  Generated 352 candidate scenarios
  10 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 5, 6, 8, 9, 10]

#### [20:40:15] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#10']
Reason: Entropy 1.685 (adjusted 1.685) | timing x1.00

### [20:40:53] Ability used at #4

#### [20:40:58] Solver Output
Scenarios: 3/352
Definite evil: ['#1', '#10']
Definite good: ['#3', '#4', '#7', '#8', '#9']
Evil probabilities: #2=33%, #5=33%, #6=33%
  Generated 352 candidate scenarios
  3 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #10 is DEFINITELY EVIL (possible roles: {'Minion'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 5, 6]

#### [20:40:58] Recommendation
Action: **EXECUTE** #10
Reason: #10 is evil in ALL 3 scenarios (roles: {'Minion'})

### [20:41:30] Executed #10 -> Minion (EVIL)

#### [20:41:35] Solver Output
Scenarios: 3/43
Definite evil: ['#1', '#10']
Definite good: ['#3', '#4', '#7', '#8', '#9']
Evil probabilities: #2=33%, #5=33%, #6=33%
  Generated 43 candidate scenarios
  3 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #10 is DEFINITELY EVIL (possible roles: {'Minion'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 5, 6]

#### [20:41:35] Recommendation
Action: **USE_ABILITY** #8 (Dreamer) -> targets ['#2']
Reason: Entropy 0.918 (adjusted 0.918) | timing x1.00

### [20:42:23] Revealed #8 Dreamer
Info: {'target': 2, 'evil_role': 'Shaman'}

### [20:42:26] Ability used at #8

#### [20:42:31] Solver Output
Scenarios: 2/43
Definite evil: ['#1', '#10']
Definite good: ['#2', '#3', '#4', '#7', '#8', '#9']
Evil probabilities: #5=50%, #6=50%
  Generated 43 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #10 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [5, 6]

#### [20:42:31] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (50% good Gemcrafter (corrupted), 50% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [20:43:42] Executed #5 -> Pooka (EVIL)

## [20:43:51] GAME OVER — WIN
Final HP: 10
Notes: Perfect win 10HP. PD#4 clean check on #10 narrowed to 3 scenarios. Dreamer#8 on #2 got Shaman (noise, confirmed #2 good). 50/50 on #5/#6, exec #5=Pooka. Corrupted: #2(PD), #6(Pooka). Ascension 30 complete!


---

# New Game — 2026-03-10 20:45:55
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Empress, Hunter, Bard, Gemcrafter, Confessor
- Outcasts: Drunk, Wretch, Plague_Doctor
- Minions: Witch
- Demons: Baa

### [20:48:03] Revealed #1 Alchemist
Info: {'cured_count': 0}

### [20:48:08] Revealed #2 Wretch
Info: {}

### [20:48:14] Revealed #3 Empress
Info: {'targets': [4, 7, 8]}

### [20:48:19] Revealed #4 Gemcrafter
Info: {'good_position': 8}

### [20:48:25] Revealed #5 Bard
Info: {'corruption_distance': -1}

### [20:48:29] Revealed #6 Empress
Info: {'targets': [4, 7, 8]}

### [20:48:34] Revealed #7 Confessor
Info: {'dizzy': False}

#### [20:48:46] Solver Output
Scenarios: 10/350
Definite good: ['#1', '#2', '#4', '#7']
Evil probabilities: #3=60%, #5=60%, #6=60%, #8=20%
  Generated 350 candidate scenarios
  10 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [3, 5, 6, 8]

#### [20:48:46] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (30% evil Baa, 30% evil Witch, 20% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [20:49:27] Executed #3 -> Witch (EVIL)

### [20:50:01] Revealed #8 Hunter
Info: {'distance': 2}

#### [20:50:06] Solver Output
Scenarios: 2/43
Definite evil: ['#3']
Definite good: ['#1', '#2', '#4', '#7', '#8']
Evil probabilities: #5=50%, #6=50%
  Generated 43 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 6]

#### [20:50:06] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 50% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [20:50:54] Executed #5 -> Baa (EVIL)

## [20:51:02] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Witch blocked #8. Exec #3=Witch (60% hit). Unblocked #8=Hunter. 50/50 #5/#6, exec #5=Baa. #6=Drunk(corrupted) disguised as Empress.


---

# New Game — 2026-03-10 20:51:46
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Gemcrafter, Dreamer, Empress, Confessor
- Outcasts: Doppelganger
- Minions: 
- Demons: Pooka

### [20:52:43] Revealed #1 Confessor
Info: {'dizzy': True}

### [20:52:49] Revealed #2 Gemcrafter
Info: {'good_position': 2}

### [20:52:56] Revealed #3 Dreamer
Info: {}

### [20:53:02] Revealed #4 Judge
Info: {}

### [20:53:08] Revealed #5 Dreamer
Info: {}

### [20:53:16] Revealed #6 Empress
Info: {'targets': [2, 4, 5]}

#### [20:53:22] Solver Output
Scenarios: 4/30
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4', '#5', '#6']
  Generated 30 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [20:53:22] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Pooka'})

### [20:54:12] Executed #2 -> Pooka (EVIL)

## [20:54:12] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP, 1 exec. Pooka at #2 disguised as Gemcrafter said self-is-Good (lie). Confessor corrupted by adjacent Pooka. #3=Doppelganger.


---

# New Game — 2026-03-10 20:55:15
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Hunter, Gemcrafter, Medium, Bard, Druid, Knight
- Outcasts: Bombardier, Doppelganger, Plague_Doctor
- Minions: Poisoner, Shaman
- Demons: Baa

### [20:56:27] Revealed #1 Medium
Info: {'good_position': 2, 'good_role': 'Druid'}

### [20:56:33] Revealed #2 Druid
Info: {}

### [20:56:38] Revealed #3 Gemcrafter
Info: {'good_position': 5}

### [20:56:45] Revealed #4 Poet
Info: {'targets': [1, 2, 6], 'copied_role': 'Empress'}

### [20:56:52] Revealed #5 Gemcrafter
Info: {'good_position': 9}

### [20:56:58] Revealed #6 Bombardier
Info: {}

### [20:56:59] Revealed #7 Hunter
Info: {'distance': 1}

### [20:56:59] Revealed #8 Plague_Doctor
Info: {}

### [20:56:59] Revealed #9 Bard
Info: {'corruption_distance': -1}

#### [20:57:06] Solver Output
Scenarios: 44/11858
Definite good: ['#3', '#5', '#7']
Evil probabilities: #6=82%, #4=50%, #2=45%, #9=45%, #8=41%, #1=36%
  Generated 11858 candidate scenarios
  44 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 8, 9]

#### [20:57:06] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#9']
Reason: Entropy 1.742 (adjusted 1.742) | timing x1.00

### [20:58:07] Ability used at #8

#### [20:58:13] Solver Output
Scenarios: 18/11858
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#7']
Evil probabilities: #6=56%, #4=44%
  Generated 11858 candidate scenarios
  18 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Shaman', 'Baa'})
    #9 is DEFINITELY EVIL (possible roles: {'Shaman', 'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [20:58:13] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 18 scenarios (roles: {'Shaman', 'Baa'})

### [20:59:00] Executed #8 -> Shaman (EVIL)

#### [20:59:06] Solver Output
Scenarios: 9/509
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#7']
Evil probabilities: #6=56%, #4=44%
  Generated 509 candidate scenarios
  9 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #9 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [20:59:06] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 9 scenarios (roles: {'Baa'})

### [20:59:46] Executed #9 -> Baa (EVIL)

#### [20:59:53] Solver Output
Scenarios: 9/68
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#7']
Evil probabilities: #6=56%, #4=44%
  Generated 68 candidate scenarios
  9 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #9 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [20:59:53] Recommendation
Action: **USE_ABILITY** #2 (Druid) -> targets ['#1', '#3', '#6']
Reason: Entropy 0.991 (adjusted 0.991) | timing x1.00

### [21:00:47] Revealed #2 Druid
Info: {'targets': [1, 3, 6], 'found_outcast': 'Bombardier'}

### [21:00:47] Ability used at #2

#### [21:00:54] Solver Output
Scenarios: 4/68
Definite evil: ['#4', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7']
  Generated 68 candidate scenarios
  4 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #8 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #9 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [21:00:54] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 4 scenarios (roles: {'Poisoner'})

### [21:01:46] Executed #4 -> Poisoner (EVIL)

## [21:01:46] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Fake PD#8 lied about corruption, solver caught it. Druid confirmed #6=Bombardier. Poisoner#4 corrupted adjacent #5. 3 confident execs, 0 wrong.


---

# New Game — 2026-03-10 21:02:37
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Fortune_Teller, Poet, Scout, Hunter, Bishop
- Outcasts: Drunk
- Minions: 
- Demons: Pooka

### [21:03:38] Revealed #1 Poet
Info: {'good_position': 2, 'copied_role': 'Gemcrafter'}

### [21:03:38] Revealed #2 Scout
Info: {'evil_role': 'Pooka', 'distance': 2}

### [21:03:38] Revealed #3 Bishop
Info: {'targets': [3, 4, 5], 'types': ['Villager', 'Outcast', 'Demon']}

### [21:03:38] Revealed #4 Hunter
Info: {'distance': 2}

### [21:03:38] Revealed #5 Knitter
Info: {'evil_pairs': 0}

### [21:03:39] Revealed #6 Fortune_Teller
Info: {}

#### [21:03:45] Solver Output
Scenarios: 5/30
Definite good: ['#1', '#4', '#5', '#6']
Evil probabilities: #2=60%, #3=40%
  Generated 30 candidate scenarios
  5 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [2, 3]

#### [21:03:45] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.971 (adjusted 0.777) | timing x1.00
WARNING: Corruption risk: 40%

### [21:04:38] Revealed #6 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [21:04:38] Ability used at #6

#### [21:04:38] Solver Output
Scenarios: 2/30
Definite good: ['#1', '#4', '#5', '#6']
Evil probabilities: #2=50%, #3=50%
  Generated 30 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [2, 3]

#### [21:04:38] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (50% good Drunk (corrupted), 50% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [21:05:34] Executed #2 -> Pooka (EVIL)

## [21:05:35] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. FT#6 was Drunk(corrupted), lied about #1/#2. 50/50 exec #2=Pooka. Pooka corrupted #1(Poet), #3(Bishop). #6=Drunk corrupted.


---

# New Game — 2026-03-10 21:06:18
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Judge, Gemcrafter, Jester, Empress, Alchemist, Hunter
- Outcasts: Plague_Doctor, Bombardier
- Minions: Witch
- Demons: Baa

### [21:07:16] Revealed #1 Oracle
Info: {'targets': [1, 2], 'minion_role': 'Witch'}

### [21:07:16] Revealed #2 Hunter
Info: {'distance': 2}

### [21:07:17] Revealed #3 Judge
Info: {}

### [21:07:17] Revealed #4 Gemcrafter
Info: {'good_position': 3}

### [21:07:17] Revealed #5 Alchemist
Info: {'cured_count': 1}

### [21:07:17] Revealed #6 Plague_Doctor
Info: {}

### [21:07:17] Revealed #7 Jester
Info: {}

#### [21:07:24] Solver Output
Scenarios: 8/224
Definite good: ['#1', '#3', '#4', '#6']
Evil probabilities: #2=75%, #5=62%, #7=38%, #8=25%
  Generated 224 candidate scenarios
  8 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [2, 5, 7, 8]

#### [21:07:24] Recommendation
Action: **USE_ABILITY** #7 (Jester) -> targets ['#1', '#2', '#8']
Reason: Expected posterior 4.0 scenarios (adjusted 4.2, info gain 0.913 bits) | timing x1.00
WARNING: Corruption risk: 12%

### [21:08:21] Revealed #7 Jester
Info: {'targets': [1, 2, 8], 'evil_count': 3}

### [21:08:21] Ability used at #7

#### [21:08:21] Solver Output
Scenarios: 4/224
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4', '#6', '#8']
Evil probabilities: #7=75%, #5=25%
  Generated 224 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 7]

#### [21:08:21] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Witch'})

### [21:08:57] Executed #2 -> Witch (EVIL)

### [21:09:33] Revealed #8 Empress
Info: {'targets': [1, 2, 4]}

#### [21:09:34] Solver Output
Scenarios: 4/31
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4', '#6', '#8']
Evil probabilities: #7=75%, #5=25%
  Generated 31 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 7]

#### [21:09:34] Recommendation
Action: **USE_ABILITY** #3 (Judge) -> targets ['#5']
Reason: Expected posterior 2.0 scenarios (adjusted 2.0, info gain 1.000 bits) | timing x1.00

### [21:10:21] Revealed #3 Judge
Info: {'target': 5, 'is_lying': False}

### [21:10:21] Ability used at #3

#### [21:10:22] Solver Output
Scenarios: 2/31
Definite evil: ['#2', '#7']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#8']
  Generated 31 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Witch'})
    #7 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [21:10:22] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Baa'})

### [21:11:15] Executed #7 -> Baa (EVIL)

## [21:11:15] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Jester#7 claimed 3 evils among 3 positions (impossible with 2 evils) = lying = evil. Witch#2 blocked #8. Judge confirmed #5 truthful. All confident execs.


---

# New Game — 2026-03-10 21:12:01
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Architect, Knitter, Baker, Alchemist, Empress
- Outcasts: Bombardier
- Minions: 
- Demons: Pooka

### [21:14:15] Revealed #1 Bombardier
Info: {}

### [21:14:18] Revealed #2 Baker
Info: {'original_role': 'Baker'}

### [21:14:20] Revealed #3 Alchemist
Info: {'cured_count': 1}

### [21:14:23] Revealed #4 Empress
Info: {'targets': [1, 6, 7]}

### [21:14:25] Revealed #5 Architect
Info: {'side': 'CCW'}

### [21:14:28] Revealed #6 Knitter
Info: {'evil_pairs': 0}

### [21:14:32] Revealed #7 Lover
Info: {'evil_adjacent': 0}

#### [21:14:35] Solver Output
Scenarios: 1/7
Definite evil: ['#3']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [21:14:35] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [21:15:33] Executed #3 -> Pooka (EVIL)

## [21:15:39] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. 1 scenario, solver nailed it instantly. Corrupted: #2(Baker), #4(Empress) adjacent to Pooka.


---

# New Game — 2026-03-10 21:17:45
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Oracle, Knitter, Druid, Confessor, Gemcrafter, Lover
- Outcasts: Plague_Doctor
- Minions: Shaman, Minion
- Demons: Pooka

### [21:20:03] Revealed #1 Gemcrafter
Info: {'good_position': 9}

### [21:20:07] Revealed #2 Confessor
Info: {'dizzy': False}

### [21:20:12] Revealed #3 Confessor
Info: {'dizzy': False}

### [21:20:16] Revealed #4 Poet
Info: {'direction': 'CW', 'copied_role': 'Enlightened'}

### [21:20:21] Revealed #5 Lover
Info: {'evil_adjacent': 2}

### [21:20:25] Revealed #6 Knitter
Info: {'evil_pairs': 2}

### [21:20:29] Revealed #7 Knitter
Info: {'evil_pairs': 2}

### [21:20:34] Revealed #8 Plague_Doctor
Info: {}

### [21:20:38] Revealed #9 Oracle
Info: {'targets': [2, 8], 'minion_role': 'Minion'}

#### [21:20:44] Solver Output
Scenarios: 18/1848
Definite good: ['#2', '#3', '#4']
Evil probabilities: #9=89%, #6=72%, #5=56%, #7=50%, #1=22%, #8=11%
  Generated 1848 candidate scenarios
  18 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 5, 6, 7, 8, 9]

#### [21:20:44] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#7']
Reason: Entropy 1.948 (adjusted 1.948) | timing x1.00

### [21:21:34] Ability used at #8

#### [21:21:39] Solver Output
Scenarios: 9/1848
Definite good: ['#1', '#2', '#3', '#4']
Evil probabilities: #7=89%, #9=89%, #5=56%, #6=56%, #8=11%
  Generated 1848 candidate scenarios
  9 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [5, 6, 7, 8, 9]

#### [21:21:39] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (33% evil Minion, 33% evil Shaman, 22% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 89%, but all reveal branches still lead to a forced win.

### [21:22:21] Executed #7 -> Shaman (EVIL)

#### [21:22:25] Solver Output
Scenarios: 3/224
Definite evil: ['#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#8']
Evil probabilities: #6=67%, #5=33%
  Generated 224 candidate scenarios
  3 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #9 is DEFINITELY EVIL (possible roles: {'Minion', 'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 6]

#### [21:22:25] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 3 scenarios (roles: {'Minion', 'Pooka'})

### [21:23:08] Executed #9 -> Minion (EVIL)

#### [21:23:14] Solver Output
Scenarios: 1/31
Definite evil: ['#6', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']
  Generated 31 candidate scenarios
  1 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #7 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #9 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [21:23:14] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [21:24:00] Executed #6 -> Pooka (EVIL)

## [21:24:08] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Shaman duplicated Confessor. PD clean check on #7, solver narrowed to 1 scenario after 2 execs. Corrupted: #1(Gemcrafter), #5(Lover) from Pooka#6.


---

# New Game — 2026-03-10 21:29:05
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Knight, Baker, Medium, Poet, Bard
- Outcasts: Plague_Doctor, Bombardier
- Minions: Minion
- Demons: Lilis

### [21:31:28] Revealed #1 Medium
Info: {'good_position': 7, 'good_role': 'Medium'}

### [21:31:34] Revealed #2 Baker
Info: {'original_role': 'Baker'}

### [21:31:39] Revealed #3 Plague_Doctor
Info: {}

### [21:31:44] Revealed #4 Bard
Info: {'corruption_distance': 1}

### [21:33:10] Revealed #5 Empress
Info: {'targets': [2, 8, 9]}

### [21:33:16] Revealed #7 Medium
Info: {'good_position': 1, 'good_role': 'Medium'}

### [21:33:22] Revealed #8 Baker
Info: {'original_role': 'Knight'}

### [21:33:29] Revealed #9 Bombardier
Info: {}

#### [21:33:41] Solver Output
Scenarios: 0/298
  Generated 298 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #4 Bard: rejected 166/298 (56%)
    #1 Medium: rejected 142/298 (48%)
    #7 Medium: rejected 142/298 (48%)
    #5 Empress: rejected 134/298 (45%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Medium: 2 scenarios survive  <-- SUSPECT
    WITHOUT #2 Baker: 2 scenarios survive  <-- SUSPECT
    WITHOUT #4 Bard: 2 scenarios survive  <-- SUSPECT
    WITHOUT #5 Empress: 4 scenarios survive  <-- SUSPECT
    WITHOUT #7 Medium: 2 scenarios survive  <-- SUSPECT
    WITHOUT #8 Baker: 2 scenarios survive  <-- SUSPECT

#### [21:33:41] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [21:42:26] Solver Output
Scenarios: 2/298
Definite evil: ['#1', '#7']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#8', '#9']
  Generated 298 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [21:42:26] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Minion', 'Lilis'})

### [21:42:59] Executed #1 -> Minion (EVIL)

### [21:43:46] Executed #7 -> Lilis (EVIL)

## [21:44:06] GAME OVER — WIN
Final HP: 6
Notes: 6HP. Both evils disguised as Medium, mutually lying. PD corrupted #5(Empress). Lilis killed #6(Poet). Baker chain bug found and fixed: 'Baker' as original_role wasn't recognized as original.


---

# New Game — 2026-03-10 21:46:08
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Slayer, Alchemist, Judge, Baker, Medium
- Outcasts: Doppelganger
- Minions: 
- Demons: Pooka

### [21:47:27] Revealed #1 Judge
Info: {}

### [21:47:33] Revealed #2 Medium
Info: {'good_position': 7, 'good_role': 'Slayer'}

### [21:47:40] Revealed #3 Judge
Info: {}

### [21:47:46] Revealed #4 Alchemist
Info: {'cured_count': 1}

### [21:47:54] Revealed #5 Bishop
Info: {'targets': [2, 3, 7], 'types': ['Outcast', 'Villager', 'Demon']}

### [21:48:00] Revealed #6 Baker
Info: {'original_role': 'original'}

### [21:48:07] Revealed #7 Baker
Info: {'original_role': 'Slayer'}

#### [21:48:14] Solver Output
Scenarios: 9/42
Definite good: ['#1', '#6', '#7']
Evil probabilities: #4=44%, #3=22%, #5=22%, #2=11%
  Generated 42 candidate scenarios
  9 scenarios survived validation
    #1 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5]

#### [21:48:14] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#3']
Reason: Expected posterior 5.2 scenarios (adjusted 5.5, info gain 0.713 bits) | timing x1.00
WARNING: Corruption risk: 11% -- corrupted Judge results are unreliable

### [21:49:11] Revealed #1 Judge
Info: {'target': 3, 'is_lying': True}

### [21:49:19] Ability used at #1

#### [21:49:26] Solver Output
Scenarios: 6/42
Definite good: ['#1', '#5', '#6', '#7']
Evil probabilities: #4=50%, #3=33%, #2=17%
  Generated 42 candidate scenarios
  6 scenarios survived validation
    #1 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 4]

#### [21:49:26] Recommendation
Action: **USE_ABILITY** #3 (Judge) -> targets ['#2']
Reason: Expected posterior 4.6 scenarios (adjusted 5.7, info gain 0.075 bits) | timing x1.00
WARNING: Corruption risk: 50% -- corrupted Judge results are unreliable

### [21:50:16] Revealed #3 Judge
Info: {'target': 2, 'is_lying': True}

### [21:50:16] Ability used at #3

#### [21:50:24] Solver Output
Scenarios: 5/42
Definite good: ['#1', '#5', '#6', '#7']
Evil probabilities: #4=60%, #2=20%, #3=20%
  Generated 42 candidate scenarios
  5 scenarios survived validation
    #1 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 4]

#### [21:50:24] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (60% evil Pooka, 20% good Alchemist, 20% good Alchemist (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [21:51:22] Executed #4 -> Pooka (EVIL)

## [21:51:32] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Doppelganger#1 disguised as Judge. Pooka#4 corrupted #3(Judge) and #5(Bishop). Both Judges caught lying targets. Solver exec lookahead guaranteed win at 60% hit.


---

# New Game — 2026-03-10 21:53:29
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Lover, Baker, Oracle, Fortune_Teller, Hunter, Bishop
- Outcasts: Drunk, Bombardier
- Minions: Witch
- Demons: Baa

### [21:55:22] Revealed #2 Hunter
Info: {'distance': 2}

### [21:55:30] Revealed #3 Fortune_Teller
Info: {}

### [21:55:30] Revealed #4 Lover
Info: {'evil_adjacent': 2}

### [21:55:30] Revealed #5 Slayer
Info: {}

### [21:55:30] Revealed #6 Bishop
Info: {'targets': [1, 4, 7], 'types': ['Minion', 'Outcast', 'Villager']}

### [21:55:30] Revealed #7 Oracle
Info: {'targets': [3, 4], 'minion_role': 'Witch'}

#### [21:55:38] Solver Output
Scenarios: 13/252
Definite good: ['#1', '#2']
Evil probabilities: #4=69%, #5=46%, #6=38%, #7=31%, #3=15%
  Generated 252 candidate scenarios
  13 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    Uncertain: [3, 4, 5, 6, 7]

#### [21:55:38] Recommendation
Action: **USE_ABILITY** #3 (Fortune Teller) -> targets ['#4', '#5']
Reason: Entropy 0.996 (adjusted 0.919) | follow-up bonus 0.284 | timing x1.00
WARNING: Corruption risk: 15%

### [21:56:39] Revealed #3 Fortune Teller
Info: {'targets': [4, 5], 'has_evil': True}

### [21:56:40] Ability used at #3

#### [21:56:47] Solver Output
Scenarios: 7/252
Definite good: ['#1', '#2', '#3']
Evil probabilities: #4=71%, #5=71%, #6=29%, #7=29%
  Generated 252 candidate scenarios
  7 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 7]

#### [21:56:47] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#4']
Reason: Target #4 is 71% evil (adjusted 0.61)
WARNING: Corruption risk: 14% -- Slayer ability disabled if corrupted

### [21:57:42] Ability used at #5

#### [21:57:50] Solver Output
Scenarios: 6/252
Definite good: ['#1', '#2', '#3']
Evil probabilities: #5=83%, #4=67%, #7=33%, #6=17%
  Generated 252 candidate scenarios
  6 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 7]

#### [21:57:50] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (67% evil Baa, 17% good Drunk (corrupted), 17% evil Witch).
WARNING: Execution lookahead override -- immediate hit chance is 83%, but all reveal branches still lead to a forced win.

### [22:00:06] Executed #5 -> baa (EVIL)

#### [22:00:11] Solver Output
Scenarios: 5/72
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#6']
Evil probabilities: #4=60%, #7=40%
  Generated 72 candidate scenarios
  5 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [4, 7]

#### [22:00:11] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (60% evil Witch, 40% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [22:01:00] Executed #4 -> witch (EVIL)

## [22:01:33] GAME OVER — WIN
Final HP: 10
Notes: Perfect game, 10HP. Solver 83% on #5=Baa, then 60% on #4=Witch with guaranteed win.


---

# New Game — 2026-03-10 22:09:35
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Fortune_Teller, Knight, Empress, Oracle, Baker, Knitter, Architect
- Outcasts: Drunk, Plague_Doctor
- Minions: Twin_Minion, Chancellor
- Demons: Lilis

### [22:10:48] Revealed #1 Baker
Info: {'original_role': 'Knight'}

### [22:10:52] Revealed #2 Fortune_Teller
Info: {}

### [22:10:55] Revealed #3 Oracle
Info: {'targets': [8, 9], 'minion_role': 'Chancellor'}

### [22:10:59] Revealed #4 Plague_Doctor
Info: {}

#### [22:11:10] Solver Output
Scenarios: 540/4282
Definite good: ['#7']
Evil probabilities: #1=54%, #6=49%, #5=49%, #3=42%, #2=40%, #8=23%, #9=23%, #4=19%
  Generated 4282 candidate scenarios
  540 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8, 9]

#### [22:11:10] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#1']
Reason: Entropy 2.442 (adjusted 2.442) | timing x0.75

### [22:11:54] Ability used at #4

#### [22:11:58] Solver Output
Scenarios: 246/4282
Definite good: ['#7']
Evil probabilities: #1=88%, #6=50%, #5=49%, #3=38%, #2=36%, #8=14%, #9=14%, #4=12%
  Generated 4282 candidate scenarios
  246 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8, 9]

#### [22:11:58] Recommendation
Action: **REVEAL** #6
Reason: #6: 50% evil, entropy 1.000

### [22:12:53] Revealed #5 Knitter
Info: {'evil_pairs': 1}

### [22:12:57] Revealed #6 Alchemist
Info: {'cured_count': 0}

### [22:13:02] Revealed #8 Oracle
Info: {'targets': [3, 6], 'minion_role': 'Chancellor'}

### [22:13:06] Revealed #9 Empress
Info: {'targets': [2, 3, 4]}

#### [22:13:14] Solver Output
Scenarios: 18/10292
Definite evil: ['#1']
Definite good: ['#4', '#6', '#7']
Evil probabilities: #5=67%, #8=67%, #3=33%, #9=22%, #2=11%
  Generated 10292 candidate scenarios
  18 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Lilis'})
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 5, 8, 9]

#### [22:13:14] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 18 scenarios (roles: {'Twin_Minion', 'Lilis'})

### [22:13:45] Executed #1 -> Lilis (EVIL)

#### [22:13:49] Solver Output
Scenarios: 11/1120
Definite evil: ['#1']
Definite good: ['#4', '#6', '#7']
Evil probabilities: #8=73%, #5=55%, #9=36%, #3=27%, #2=9%
  Generated 1120 candidate scenarios
  11 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 5, 8, 9]

#### [22:13:49] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#4', '#5']
Reason: Entropy 0.994 (adjusted 0.723) | timing x1.00
WARNING: Corruption risk: 55%

### [22:14:38] Revealed #2 Fortune Teller
Info: {'targets': [4, 5], 'has_evil': True}

### [22:14:42] Ability used at #2

#### [22:14:46] Solver Output
Scenarios: 5/1120
Definite evil: ['#1']
Definite good: ['#4', '#6', '#7']
Evil probabilities: #8=80%, #5=40%, #9=40%, #2=20%, #3=20%
  Generated 1120 candidate scenarios
  5 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 5, 8, 9]

#### [22:14:46] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (60% evil Chancellor, 20% good Oracle, 20% evil Twin_Minion).
WARNING: Execution lookahead override -- immediate hit chance is 80%, but all reveal branches still lead to a forced win.

### [22:15:28] Executed #8 -> Chancellor (EVIL)

#### [22:15:31] Solver Output
Scenarios: 3/131
Definite evil: ['#1', '#8']
Definite good: ['#2', '#3', '#4', '#6', '#7']
Evil probabilities: #5=67%, #9=33%
  Generated 131 candidate scenarios
  3 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [5, 9]

#### [22:15:31] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (67% evil Twin_Minion, 33% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [22:16:13] Executed #5 -> Twin_Minion (EVIL)

## [22:16:20] GAME OVER — WIN
Final HP: 6
Notes: Lilis game, 6HP. Corrupted: #9 Empress. Night killed #7. PD checked #1 clean. Solver nailed all 3 evils.


---

# New Game — 2026-03-10 22:17:28
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Bishop, Judge, Fortune_Teller, Empress, Medium
- Outcasts: Plague_Doctor
- Minions: Witch, Puppeteer
- Demons: Lilis

### [22:18:27] Revealed #1 Judge
Info: {}

### [22:18:27] Revealed #2 Slayer
Info: {}

### [22:18:27] Revealed #3 Fortune_Teller
Info: {}

### [22:18:27] Revealed #4 Fortune_Teller
Info: {}

#### [22:18:36] Solver Output
Scenarios: 540/1260
Definite good: ['#8']
Evil probabilities: #3=64%, #4=64%, #2=42%, #5=42%, #1=40%, #6=40%, #7=34%, #9=34%
  Generated 1260 candidate scenarios
  540 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [22:18:36] Recommendation
Action: **REVEAL** #5
Reason: #5: 42% evil, entropy 0.981

### [22:19:16] Revealed #5 Judge
Info: {}

#### [22:19:22] Solver Output
Scenarios: 376/1260
Definite good: ['#8']
Evil probabilities: #4=63%, #3=60%, #5=60%, #1=57%, #2=37%, #6=34%, #9=30%, #7=27%
  Generated 1260 candidate scenarios
  376 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [22:19:22] Recommendation
Action: **USE_ABILITY** #4 (Fortune Teller) -> targets ['#1', '#6']
Reason: Entropy 1.000 (adjusted 1.000) | timing x0.96

### [22:20:26] Revealed #4 Fortune Teller
Info: {'targets': [1, 6], 'has_evil': True}

### [22:20:26] Ability used at #4

#### [22:20:30] Solver Output
Scenarios: 358/1260
Definite good: ['#8']
Evil probabilities: #1=66%, #5=61%, #6=49%, #2=45%, #3=42%, #9=39%, #7=38%, #4=21%
  Generated 1260 candidate scenarios
  358 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [22:20:30] Recommendation
Action: **REVEAL** #6
Reason: #6: 49% evil, entropy 1.000

### [22:21:03] Revealed #6 Plague_Doctor
Info: {}

#### [22:21:07] Solver Output
Scenarios: 892/3822
Definite good: ['#8']
Evil probabilities: #1=65%, #5=63%, #2=47%, #3=46%, #9=41%, #7=38%, #4=32%, #6=17%
  Generated 3822 candidate scenarios
  892 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [22:21:07] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#7']
Reason: Entropy 1.466 (adjusted 1.466) | timing x1.00

### [22:21:55] Ability used at #6

#### [22:22:00] Solver Output
Scenarios: 624/3822
Definite good: ['#6', '#8']
Evil probabilities: #1=65%, #5=63%, #3=47%, #2=47%, #7=45%, #9=41%, #4=37%
  Generated 3822 candidate scenarios
  624 scenarios survived validation
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 9]

#### [22:22:00] Recommendation
Action: **REVEAL** #7
Reason: #7: 45% evil, entropy 0.992
WARNING: Witch may be alive -- be cautious about revealing

### [22:22:30] Revealed #7 Bishop
Info: {'targets': [1, 3, 6], 'types': ['Villager', 'Minion', 'Outcast']}

#### [22:22:35] Solver Output
Scenarios: 276/3822
Definite good: ['#6', '#8']
Evil probabilities: #5=65%, #7=58%, #1=56%, #3=48%, #2=43%, #4=41%, #9=34%
  Generated 3822 candidate scenarios
  276 scenarios survived validation
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 9]

#### [22:22:35] Recommendation
Action: **USE_ABILITY** #3 (Fortune Teller) -> targets ['#7', '#9']
Reason: Entropy 0.999 (adjusted 0.928) | follow-up bonus 0.261 | timing x1.00
WARNING: Corruption risk: 14%

### [22:23:20] Revealed #3 Fortune Teller
Info: {'targets': [7, 9], 'has_evil': True}

### [22:23:20] Ability used at #3

#### [22:23:26] Solver Output
Scenarios: 85/3822
Definite good: ['#6', '#8']
Evil probabilities: #5=93%, #4=86%, #7=53%, #3=51%, #2=33%, #9=19%, #1=14%
  Generated 3822 candidate scenarios
  85 scenarios survived validation
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 9]

#### [22:23:26] Recommendation
Action: **USE_ABILITY** #5 (Judge) -> targets ['#3']
Reason: Expected posterior 43.6 scenarios (adjusted 44.1, info gain 0.948 bits) | timing x1.00
WARNING: Corruption risk: 2% -- corrupted Judge results are unreliable

### [22:24:10] Revealed #5 Judge
Info: {'target': 3, 'is_lying': False}

### [22:24:10] Ability used at #5

#### [22:24:15] Solver Output
Scenarios: 45/3822
Definite good: ['#6', '#8']
Evil probabilities: #5=87%, #4=78%, #3=71%, #2=51%, #7=40%, #1=16%, #9=13%
  Generated 3822 candidate scenarios
  45 scenarios survived validation
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 9]

#### [22:24:15] Recommendation
Action: **USE_ABILITY** #2 (Slayer) -> targets ['#5']
Reason: Target #5 is 87% evil (adjusted 0.75)
WARNING: Corruption risk: 13% -- Slayer ability disabled if corrupted

### [22:24:52] Ability used at #2

#### [22:24:57] Solver Output
Scenarios: 40/1200
Definite evil: ['#4', '#5']
Definite good: ['#1', '#6', '#8']
Evil probabilities: #3=55%, #7=45%, #2=15%, #9=10%
  Generated 1200 candidate scenarios
  40 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Lilis', 'Witch'})
    #5 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #1 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3, 7, 9]

#### [22:24:57] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 40 scenarios (roles: {'Puppeteer', 'Lilis', 'Witch'})

### [22:25:40] Executed #4 -> Lilis (EVIL)

#### [22:25:45] Solver Output
Scenarios: 15/104
Definite evil: ['#4', '#5']
Definite good: ['#1', '#6', '#8', '#9']
Evil probabilities: #3=60%, #7=40%, #2=20%
  Generated 104 candidate scenarios
  15 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #5 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #1 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 7]

#### [22:25:45] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#9']
Reason: Expected posterior 10.0 scenarios (adjusted 11.7, info gain 0.363 bits) | timing x1.00
WARNING: Corruption risk: 33% -- corrupted Judge results are unreliable

### [22:26:56] Revealed #1 Judge
Info: {'target': 9, 'is_lying': True}

### [22:26:56] Ability used at #1

#### [22:27:02] Solver Output
Scenarios: 10/104
Definite evil: ['#4', '#5']
Definite good: ['#1', '#6', '#8', '#9']
Evil probabilities: #3=60%, #7=40%, #2=20%
  Generated 104 candidate scenarios
  10 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #5 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #1 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 7]

#### [22:27:02] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (40% good Fortune Teller, 40% evil Puppeteer, 20% evil Witch).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [22:27:45] Executed #3 -> Witch (EVIL)

## [22:27:54] GAME OVER — WIN
Final HP: 8
Notes: Lilis game, 8HP. Corrupted: #1 Judge. Slayer killed Puppeteer. Two FTs (both evil lies), two Judges. Solver found all 3.


---

# New Game — 2026-03-10 22:28:42
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Knight, Judge, Dreamer, Gemcrafter
- Outcasts: Doppelganger, Drunk
- Minions: Witch
- Demons: Baa

### [22:29:50] Revealed #1 Judge
Info: {}

### [22:29:51] Revealed #2 Dreamer
Info: {}

### [22:29:51] Revealed #3 Knight
Info: {}

### [22:29:51] Revealed #4 Hunter
Info: {'distance': 2}

### [22:29:51] Revealed #5 Knight
Info: {}

### [22:29:51] Revealed #6 Dreamer
Info: {}

#### [22:30:13] Solver Output
Scenarios: 482/1152
Evil probabilities: #4=36%, #7=35%, #2=31%, #6=31%, #1=29%, #3=19%, #5=19%
  Generated 1152 candidate scenarios
  482 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [22:30:13] Recommendation
Action: **EXECUTE** #3
Reason: Knight check: #3 is 19% evil, 14% corruption risk. Expected HP cost: 1.0 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 14% -- corrupted Knight loses immunity + 4 extra damage

### [22:31:06] Executed #3 -> GOOD (WRONG!)

#### [22:31:15] Solver Output
Scenarios: 390/830
Definite good: ['#3']
Evil probabilities: #7=41%, #2=36%, #6=36%, #1=34%, #4=31%, #5=22%
  Generated 830 candidate scenarios
  390 scenarios survived validation
    #3 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7]

#### [22:31:15] Recommendation
Action: **EXECUTE** #5
Reason: Knight check: #5 is 22% evil, 15% corruption risk. Expected HP cost: 1.1 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 15% -- corrupted Knight loses immunity + 4 extra damage

### [22:31:56] Executed #5 -> GOOD (WRONG!)

#### [22:32:01] Solver Output
Scenarios: 306/560
Definite good: ['#3', '#5']
Evil probabilities: #7=48%, #2=44%, #6=44%, #1=41%, #4=24%
  Generated 560 candidate scenarios
  306 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 7]

#### [22:32:01] Recommendation
Action: **USE_ABILITY** #2 (Dreamer) -> targets ['#7']
Reason: Entropy 2.459 (adjusted 2.330) | timing x1.00
WARNING: Corruption risk: 10%

### [22:32:49] Revealed #2 Dreamer
Info: {'target': 7, 'evil_role': 'Baa'}

### [22:32:49] Ability used at #2

#### [22:32:58] Solver Output
Scenarios: 232/560
Definite good: ['#3', '#5']
Evil probabilities: #1=52%, #2=47%, #6=47%, #7=32%, #4=22%
  Generated 560 candidate scenarios
  232 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 7]

#### [22:32:58] Recommendation
Action: **USE_ABILITY** #6 (Dreamer) -> targets ['#1']
Reason: Entropy 2.469 (adjusted 2.346) | timing x1.00
WARNING: Corruption risk: 10%

### [22:33:35] Revealed #6 Dreamer
Info: {'target': 1, 'evil_role': 'Baa'}

### [22:33:35] Ability used at #6

#### [22:33:41] Solver Output
Scenarios: 170/560
Definite good: ['#3', '#5']
Evil probabilities: #2=52%, #6=52%, #7=41%, #1=34%, #4=21%
  Generated 560 candidate scenarios
  170 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 7]

#### [22:33:41] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#4']
Reason: Expected posterior 98.7 scenarios (adjusted 105.1, info gain 0.694 bits) | timing x1.00
WARNING: Corruption risk: 13% -- corrupted Judge results are unreliable

### [22:34:17] Revealed #1 Judge
Info: {'target': 4, 'is_lying': False}

### [22:34:17] Ability used at #1

#### [22:34:25] Solver Output
Scenarios: 112/560
Definite good: ['#3', '#5']
Evil probabilities: #2=60%, #6=60%, #7=48%, #4=18%, #1=14%
  Generated 560 candidate scenarios
  112 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 7]

#### [22:34:25] Recommendation
Action: **EXECUTE** #6
Reason: No reveals available. #6 is 60% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 60% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card

### [22:35:08] Executed #6 -> Witch (EVIL)

#### [22:35:13] Solver Output
Scenarios: 41/109
Definite evil: ['#6']
Definite good: ['#1', '#3', '#4', '#5']
Evil probabilities: #2=51%, #7=49%
  Generated 109 candidate scenarios
  41 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [2, 7]

#### [22:35:13] Recommendation
Action: **REVEAL** #7
Reason: #7: 49% evil, entropy 1.000

### [22:35:50] Revealed #7 Gemcrafter
Info: {'good_position': 6}

#### [22:35:56] Solver Output
Scenarios: 25/124
Definite evil: ['#6']
Definite good: ['#1', '#3', '#4', '#5']
Evil probabilities: #7=80%, #2=20%
  Generated 124 candidate scenarios
  25 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [2, 7]

#### [22:35:56] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (80% evil Baa, 20% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 80%, but all reveal branches still lead to a forced win.

### [22:36:40] Executed #7 -> Baa (EVIL)

## [22:36:40] GAME OVER — WIN
Final HP: 10
Notes: Perfect game 10HP. Knight checks on #3 and #5 both passed (uncorrupted). Dreamer said #7=Baa, confirmed. #5=Doppelganger-as-Knight had immunity.


---

# New Game — 2026-03-10 22:39:31
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Gemcrafter, Slayer, Druid, Bishop, Lover
- Outcasts: Bombardier, Plague_Doctor
- Minions: Poisoner
- Demons: Baa

### [22:40:54] Revealed #1 Slayer
Info: {}

### [22:40:59] Revealed #2 Gemcrafter
Info: {'good_position': 6}

### [22:41:03] Revealed #2 Gemcrafter
Info: {'good_position': 3}

### [22:41:07] Revealed #3 Lover
Info: {'evil_adjacent': 2}

### [22:41:10] Revealed #4 Druid
Info: {}

### [22:41:16] Revealed #5 Plague_Doctor
Info: {}

### [22:41:16] Revealed #6 Druid
Info: {}

### [22:41:16] Revealed #7 Judge
Info: {}

### [22:41:20] Revealed #8 Bishop
Info: {'targets': [4, 5, 6], 'types': ['Minion', 'Outcast', 'Villager']}

#### [22:41:25] Solver Output
Scenarios: 21/334
Definite good: ['#2', '#5']
Evil probabilities: #4=62%, #6=43%, #1=29%, #7=29%, #3=19%, #8=19%
  Generated 334 candidate scenarios
  21 scenarios survived validation
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 6, 7, 8]

#### [22:41:25] Recommendation
Action: **USE_ABILITY** #5 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.956 (adjusted 1.956) | timing x1.00

### [22:42:30] Ability used at #5

#### [22:42:34] Solver Output
Scenarios: 11/334
Definite evil: ['#4']
Definite good: ['#2', '#3', '#5']
Evil probabilities: #1=36%, #7=36%, #8=18%, #6=9%
  Generated 334 candidate scenarios
  11 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner', 'Baa'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 6, 7, 8]

#### [22:42:34] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 11 scenarios (roles: {'Poisoner', 'Baa'})

### [22:43:12] Executed #4 -> Poisoner (EVIL)

#### [22:43:16] Solver Output
Scenarios: 7/31
Definite evil: ['#4']
Definite good: ['#2', '#3', '#5', '#8']
Evil probabilities: #1=43%, #7=43%, #6=14%
  Generated 31 candidate scenarios
  7 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 6, 7]

#### [22:43:16] Recommendation
Action: **USE_ABILITY** #6 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.985 (adjusted 0.844) | timing x1.00
WARNING: Corruption risk: 29%

### [22:44:10] Revealed #6 Druid
Info: {'targets': [1, 2, 6], 'found_outcast': 'Doppelganger'}

### [22:44:19] Ability used at #6

#### [22:44:23] Solver Output
Scenarios: 3/31
Definite evil: ['#4']
Definite good: ['#2', '#3', '#5', '#8']
Evil probabilities: #1=33%, #6=33%, #7=33%
  Generated 31 candidate scenarios
  3 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 6, 7]

#### [22:44:23] Recommendation
Action: **USE_ABILITY** #7 (Judge) -> targets ['#1']
Reason: Expected posterior 1.7 scenarios (adjusted 1.7, info gain 0.848 bits) | timing x1.00

### [22:45:12] Revealed #7 Judge
Info: {'target': 1, 'is_lying': True}

### [22:45:17] Ability used at #7

#### [22:45:21] Solver Output
Scenarios: 2/31
Definite evil: ['#4']
Definite good: ['#2', '#3', '#5', '#6', '#8']
Evil probabilities: #1=50%, #7=50%
  Generated 31 candidate scenarios
  2 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [22:45:21] Recommendation
Action: **USE_ABILITY** #1 (Slayer) -> targets ['#7']
Reason: Target #7 is 50% evil (adjusted 0.50)

### [22:49:15] Ability used at #1

#### [22:49:20] Solver Output
Scenarios: 1/31
Definite evil: ['#1', '#4']
Definite good: ['#2', '#3', '#5', '#6', '#7', '#8']
  Generated 31 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Baa'})
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [22:49:20] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Baa'})

### [22:50:02] Executed #1 -> Baa (EVIL)

## [22:50:08] GAME OVER — WIN
Final HP: 10
Notes: Perfect game, 10HP. PD check nailed #4 as evil. Slayer no-kill on #7 confirmed #1=Baa. Corrupted: #3 Lover, #6 Druid.


---

# New Game — 2026-03-10 22:52:15
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Alchemist, Scout, Empress, Confessor, Enlightened, Knitter
- Outcasts: Drunk, Plague_Doctor
- Minions: Shaman
- Demons: Pooka

### [22:53:44] Revealed #1 Scout
Info: {'evil_role': 'Pooka', 'distance': 2}

### [22:53:49] Revealed #2 Scout
Info: {'evil_role': 'Pooka', 'distance': 2}

### [22:53:57] Revealed #3 Enlightened
Info: {'direction': 'equidistant'}

### [22:54:03] Revealed #4 Alchemist
Info: {'cured_count': 2}

### [22:54:09] Revealed #5 Knitter
Info: {'evil_pairs': 1}

### [22:54:15] Revealed #6 Judge
Info: {}

### [22:54:15] Revealed #7 Plague_Doctor
Info: {}

### [22:54:20] Revealed #8 Confessor
Info: {'dizzy': False}

### [22:54:26] Revealed #9 Enlightened
Info: {'direction': 'ccw'}

#### [22:54:34] Solver Output
Scenarios: 38/2128
Definite good: ['#8']
Evil probabilities: #1=45%, #5=45%, #4=39%, #3=24%, #6=18%, #2=16%, #9=11%, #7=3%
  Generated 2128 candidate scenarios
  38 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [22:54:34] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#4']
Reason: Entropy 1.870 (adjusted 1.870) | timing x1.00

### [22:55:29] Ability used at #7

#### [22:55:35] Solver Output
Scenarios: 13/2128
Definite evil: ['#1']
Definite good: ['#3', '#4', '#6', '#7', '#8', '#9']
Evil probabilities: #5=69%, #2=31%
  Generated 2128 candidate scenarios
  13 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka', 'Shaman'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 5]

#### [22:55:35] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 13 scenarios (roles: {'Pooka', 'Shaman'})

### [22:56:17] Executed #1 -> Pooka (EVIL)

#### [22:56:22] Solver Output
Scenarios: 9/259
Definite evil: ['#1']
Definite good: ['#3', '#4', '#6', '#7', '#8', '#9']
Evil probabilities: #5=78%, #2=22%
  Generated 259 candidate scenarios
  9 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 5]

#### [22:56:22] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#3']
Reason: Expected posterior 5.9 scenarios (adjusted 6.6, info gain 0.455 bits) | timing x1.00
WARNING: Corruption risk: 22% -- corrupted Judge results are unreliable

### [22:57:20] Revealed #6 Judge
Info: {'target': 3, 'is_lying': False}

### [22:57:20] Ability used at #6

#### [22:57:26] Solver Output
Scenarios: 7/259
Definite evil: ['#1', '#5']
Definite good: ['#2', '#3', '#4', '#6', '#7', '#8', '#9']
  Generated 259 candidate scenarios
  7 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #5 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [22:57:26] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 7 scenarios (roles: {'Shaman'})

### [22:58:10] Executed #5 -> Shaman (EVIL)

## [22:58:18] GAME OVER — WIN
Final HP: 10
Notes: Perfect game, 10HP. PD nailed #1 evil, Judge confirmed #3 truth -> solver locked #5=Shaman. Corrupted: #2 Drunk, #4 Alchemist, #9 Enlightened.


---

# New Game — 2026-03-10 22:59:12
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Enlightened, Jester, Gemcrafter, Druid, Baker
- Outcasts: Bombardier, Wretch
- Minions: Poisoner, Chancellor
- Demons: Lilis

### [23:01:30] Revealed #1 Jester
Info: {}

### [23:01:30] Revealed #2 Druid
Info: {}

### [23:01:30] Revealed #3 Jester
Info: {}

### [23:01:30] Revealed #4 Wretch
Info: {}

### [23:02:37] Revealed #5 Enlightened
Info: {'direction': 'cw'}

### [23:02:44] Revealed #6 Druid
Info: {}

### [23:02:50] Revealed #7 Bombardier
Info: {}

### [23:02:57] Revealed #8 Poet
Info: {'evil_pairs': 0, 'copied_role': 'Knitter'}

#### [23:03:20] Solver Output
Scenarios: 7/792
Definite evil: ['#6']
Definite good: ['#2', '#5', '#7', '#9']
Evil probabilities: #1=71%, #4=71%, #3=29%, #8=29%
  Generated 792 candidate scenarios
  7 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner', 'Chancellor', 'Lilis'})
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 8]

#### [23:03:20] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 7 scenarios (roles: {'Poisoner', 'Chancellor', 'Lilis'})

### [23:04:05] Executed #6 -> Poisoner (EVIL)

#### [23:04:12] Solver Output
Scenarios: 5/74
Definite evil: ['#6']
Definite good: ['#2', '#5', '#7', '#9']
Evil probabilities: #1=60%, #4=60%, #3=40%, #8=40%
  Generated 74 candidate scenarios
  5 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 8]

#### [23:04:12] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#2', '#8']
Reason: Expected posterior 2.3 scenarios (adjusted 2.3, info gain 1.100 bits) | timing x1.00

### [23:05:29] Revealed #3 Jester
Info: {'targets': [1, 2, 8], 'evil_count': 2}

### [23:05:37] Ability used at #3

#### [23:05:44] Solver Output
Scenarios: 2/74
Definite evil: ['#3', '#6', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#9']
  Generated 74 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Lilis'})
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [23:05:44] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Chancellor', 'Lilis'})

### [23:06:29] Executed #3 -> Chancellor (EVIL)

#### [23:06:38] Solver Output
Scenarios: 1/7
Definite evil: ['#3', '#6', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#9']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [23:06:38] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [23:07:28] Executed #8 -> Lilis (EVIL)

## [23:07:38] GAME OVER — WIN
Final HP: 6
Notes: Lilis game, 6HP. Jester-Empress ability (lie about 2 evils in #1,#2,#8) actually exposed #3 and #8 as evil. Corrupted: #5 Enlightened. Night killed: #9.


---

# New Game — 2026-03-10 23:08:30
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Hunter, Bishop, Slayer, Alchemist, Poet, Medium
- Outcasts: Drunk, Plague_Doctor
- Minions: Twin_Minion
- Demons: Pooka

### [23:09:44] Revealed #1 Plague_Doctor
Info: {}

### [23:09:44] Revealed #2 Bard
Info: {'corruption_distance': 1}

### [23:09:44] Revealed #3 Bishop
Info: {'targets': [4, 5, 6], 'types': ['Villager', 'Minion', 'Outcast']}

### [23:09:44] Revealed #4 Slayer
Info: {}

### [23:09:53] Revealed #5 Poet
Info: {'evil_role': 'Twin_Minion', 'distance': 4, 'copied_role': 'Scout'}

### [23:10:00] Revealed #6 Alchemist
Info: {'cured_count': 2}

### [23:10:00] Revealed #7 Hunter
Info: {'distance': 3}

### [23:10:00] Revealed #8 Medium
Info: {'good_position': 1, 'good_role': 'Plague_Doctor'}

### [23:10:00] Revealed #9 Hunter
Info: {'distance': 1}

#### [23:10:07] Solver Output
Scenarios: 26/2128
Definite good: ['#1', '#2', '#8']
Evil probabilities: #6=73%, #9=38%, #7=31%, #3=23%, #4=19%, #5=15%
  Generated 2128 candidate scenarios
  26 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 5, 6, 7, 9]

#### [23:10:07] Recommendation
Action: **USE_ABILITY** #1 (Plague Doctor) -> targets ['#9']
Reason: Entropy 2.134 (adjusted 2.134) | timing x1.00

### [23:10:56] Ability used at #1

#### [23:11:03] Solver Output
Scenarios: 8/2128
Definite evil: ['#7']
Definite good: ['#1', '#2', '#4', '#8', '#9']
Evil probabilities: #3=50%, #5=25%, #6=25%
  Generated 2128 candidate scenarios
  8 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 5, 6]

#### [23:11:03] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 8 scenarios (roles: {'Twin_Minion'})

### [23:11:43] Executed #7 -> Twin_Minion (EVIL)

#### [23:11:43] Solver Output
Scenarios: 8/259
Definite evil: ['#7']
Definite good: ['#1', '#2', '#4', '#8', '#9']
Evil probabilities: #3=50%, #5=25%, #6=25%
  Generated 259 candidate scenarios
  8 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 5, 6]

#### [23:11:43] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#3']
Reason: Target #3 is 50% evil (adjusted 0.25)
WARNING: Corruption risk: 50% -- Slayer ability disabled if corrupted

### [23:12:33] Ability used at #4

### [23:12:33] Executed #3 -> Pooka (EVIL)

## [23:12:41] GAME OVER — WIN
Final HP: 10
Notes: Perfect game, 10HP. PD found #7 evil, Slayer killed #3=Pooka. Corrupted: #2 Bard, #9 Drunk.


---

# New Game — 2026-03-10 23:13:39
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Dreamer, Lover, Scout, Jester, Gemcrafter, Fortune_Teller
- Outcasts: Drunk, Plague_Doctor
- Minions: Poisoner
- Demons: Baa

### [23:14:56] Revealed #1 Dreamer
Info: {}

### [23:14:56] Revealed #2 Lover
Info: {'evil_adjacent': 2}

### [23:14:56] Revealed #3 Scout
Info: {'evil_role': 'Baa', 'distance': 3}

### [23:14:57] Revealed #4 Gemcrafter
Info: {'good_position': 5}

### [23:14:57] Revealed #5 Confessor
Info: {'dizzy': True}

### [23:14:57] Revealed #6 Jester
Info: {}

### [23:14:57] Revealed #7 Fortune_Teller
Info: {}

### [23:14:57] Revealed #8 Plague_Doctor
Info: {}

#### [23:15:04] Solver Output
Scenarios: 164/1988
Evil probabilities: #2=52%, #5=52%, #3=29%, #6=23%, #1=18%, #7=17%, #4=7%, #8=1%
  Generated 1988 candidate scenarios
  164 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [23:15:04] Recommendation
Action: **USE_ABILITY** #1 (Dreamer) -> targets ['#5']
Reason: Entropy 2.449 (adjusted 2.076) | timing x1.00
WARNING: Corruption risk: 30%

### [23:17:08] Revealed #1 Dreamer
Info: {'target': 5, 'evil_role': 'Baa'}

### [23:17:12] Ability used at #1

#### [23:17:15] Solver Output
Scenarios: 115/1988
Evil probabilities: #2=53%, #5=31%, #6=31%, #3=30%, #1=24%, #7=23%, #4=7%, #8=1%
  Generated 1988 candidate scenarios
  115 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [23:17:15] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.007 (adjusted 2.007) | timing x1.00

### [23:17:52] Ability used at #8

#### [23:17:55] Solver Output
Scenarios: 37/1988
Definite good: ['#6', '#7']
Evil probabilities: #5=97%, #3=43%, #2=30%, #4=22%, #1=5%, #8=3%
  Generated 1988 candidate scenarios
  37 scenarios survived validation
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 8]

#### [23:17:55] Recommendation
Action: **USE_ABILITY** #6 (Jester) -> targets ['#3', '#5', '#7']
Reason: Expected posterior 15.2 scenarios (adjusted 17.3, info gain 1.096 bits) | timing x1.00
WARNING: Corruption risk: 27%

### [23:21:04] Revealed #6 Jester
Info: {'targets': [3, 5, 7], 'evil_count': 1}

### [23:21:09] Ability used at #6

### [23:21:13] Revealed #7 Fortune Teller
Info: {'targets': [1, 3], 'has_evil': False}

### [23:21:16] Ability used at #7

#### [23:21:20] Solver Output
Scenarios: 8/1988
Definite evil: ['#5']
Definite good: ['#1', '#3', '#6', '#7', '#8']
Evil probabilities: #2=50%, #4=50%
  Generated 1988 candidate scenarios
  8 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Baa', 'Poisoner'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 4]

#### [23:21:20] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 8 scenarios (roles: {'Baa', 'Poisoner'})

### [23:21:54] Executed #5 -> Poisoner (EVIL)

#### [23:21:59] Solver Output
Scenarios: 4/306
Definite evil: ['#2', '#5']
Definite good: ['#1', '#3', '#4', '#6', '#7', '#8']
  Generated 306 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #5 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [23:21:59] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Baa'})

### [23:23:06] Executed #2 -> Baa (EVIL)

## [23:23:14] GAME OVER — WIN
Final HP: 10
Notes: Perfect game, 10HP. Dreamer #1 saw #5=Baa, PD #8 found #5 clean, Jester #6 (Empress) found 1 evil in #3,#5,#7, FT #7 found no evil in #1,#3. Solver locked #5=Poisoner then #2=Baa. Corrupted: #1,#4.


---

# New Game — 2026-03-10 23:25:13
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Gemcrafter, Architect, Knight, Knitter, Enlightened, Lover, Druid
- Outcasts: Plague_Doctor
- Minions: Shaman, Twin_Minion
- Demons: Lilis

### [23:27:25] Revealed #1 Architect
Info: {'side': 'left'}

### [23:27:29] Revealed #2 Scout
Info: {'evil_role': 'Lilis', 'distance': 2}

### [23:27:33] Revealed #3 Knight
Info: {}

### [23:27:37] Revealed #4 Knitter
Info: {'evil_pairs': 2}

### [23:28:39] Revealed #5 Plague_Doctor
Info: {}

### [23:28:44] Revealed #6 Knight
Info: {}

### [23:28:48] Revealed #7 Lover
Info: {'evil_adjacent': 2}

### [23:28:54] Revealed #8 Druid
Info: {}

#### [23:29:06] Solver Output
Scenarios: 0/930
  Generated 930 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #7 Lover: rejected 564/930 (61%)
    #4 Knitter: rejected 528/930 (57%)
    #1 Architect: rejected 432/930 (46%)
    #2 Scout: rejected 408/930 (44%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Architect: 52 scenarios survive  <-- SUSPECT
    WITHOUT #2 Scout: 40 scenarios survive  <-- SUSPECT
    WITHOUT #4 Knitter: 71 scenarios survive  <-- SUSPECT
    WITHOUT #7 Lover: 70 scenarios survive  <-- SUSPECT
    WITHOUT #8 Druid: 27 scenarios survive  <-- SUSPECT

#### [23:29:06] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [23:30:13] Solver Output
Scenarios: 27/930
Definite evil: ['#9']
Definite good: ['#3', '#5']
Evil probabilities: #7=67%, #1=41%, #8=41%, #4=19%, #2=15%, #6=4%
  Generated 930 candidate scenarios
  27 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Unknown', 'Lilis', 'Shaman'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 7, 8]

#### [23:30:13] Recommendation
Action: **USE_ABILITY** #5 (Plague Doctor) -> targets ['#4']
Reason: Entropy 1.293 (adjusted 1.293) | timing x1.00

### [23:30:57] Ability used at #5

#### [23:31:03] Solver Output
Scenarios: 19/930
Definite evil: ['#9']
Definite good: ['#3', '#5', '#6']
Evil probabilities: #7=58%, #8=53%, #1=42%, #4=26%, #2=21%
  Generated 930 candidate scenarios
  19 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 7, 8]

#### [23:31:03] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.998 (adjusted 0.998) | timing x1.00

### [23:32:18] Revealed #8 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Drunk'}

### [23:32:24] Ability used at #8

#### [23:32:29] Solver Output
Scenarios: 10/930
Definite evil: ['#8', '#9']
Definite good: ['#2', '#3', '#4', '#5', '#6']
Evil probabilities: #7=80%, #1=20%
  Generated 930 candidate scenarios
  10 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Lilis', 'Twin_Minion', 'Shaman'})
    #9 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [23:32:29] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 10 scenarios (roles: {'Lilis', 'Twin_Minion', 'Shaman'})

### [23:33:04] Executed #8 -> Twin_Minion (EVIL)

#### [23:33:09] Solver Output
Scenarios: 4/74
Definite evil: ['#8', '#9']
Definite good: ['#2', '#3', '#4', '#5', '#6']
Evil probabilities: #7=75%, #1=25%
  Generated 74 candidate scenarios
  4 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #9 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [23:33:09] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (75% evil Shaman, 25% good Lover (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [23:33:53] Executed #7 -> Lilis (EVIL)

## [23:34:02] GAME OVER — WIN
Final HP: 6
Notes: Lilis game. Night killed #9 (Shaman, evil). Druid #8 (Twin Minion) lied about Drunk among #1,#2,#3 -> solver locked #8 as evil. Exec lookahead guaranteed win on #7 (75% Lilis). HP 6/10. Corrupted: #2.


---

# New Game — 2026-03-10 23:35:53
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Knitter, Alchemist, Judge, Empress, Bishop, Lover
- Outcasts: Bombardier
- Minions: Puppeteer, Twin_Minion
- Demons: Lilis

### [23:37:12] Revealed #1 Bishop
Info: {'targets': [2, 4, 9], 'types': ['Villager', 'Minion', 'Outcast']}

### [23:37:17] Revealed #2 Knitter
Info: {'evil_pairs': 2}

### [23:37:23] Revealed #3 Jester
Info: {}

### [23:37:23] Revealed #4 Judge
Info: {}

### [23:38:15] Revealed #5 Bishop
Info: {'targets': [2, 3, 4], 'types': ['Minion', 'Outcast', 'Villager']}

### [23:38:22] Revealed #6 Alchemist
Info: {'cured_count': 0}

### [23:38:27] Revealed #7 Lover
Info: {'evil_adjacent': 0}

### [23:38:32] Revealed #8 Bombardier
Info: {}

#### [23:38:43] Solver Output
Scenarios: 26/672
Definite evil: ['#5']
Definite good: ['#9']
Evil probabilities: #3=69%, #4=69%, #1=62%, #2=54%, #7=23%, #6=15%, #8=8%
  Generated 672 candidate scenarios
  26 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Lilis', 'Puppeteer', 'Twin_Minion'})
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8]

#### [23:38:43] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 26 scenarios (roles: {'Lilis', 'Puppeteer', 'Twin_Minion'})

### [23:39:20] Executed #5 -> Lilis (EVIL)

#### [23:39:25] Solver Output
Scenarios: 9/72
Definite evil: ['#5']
Definite good: ['#8', '#9']
Evil probabilities: #3=78%, #1=67%, #2=67%, #4=67%, #6=11%, #7=11%
  Generated 72 candidate scenarios
  9 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7]

#### [23:39:25] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#2', '#6']
Reason: Expected posterior 4.3 scenarios (adjusted 4.3, info gain 1.068 bits) | timing x1.00

### [23:40:35] Revealed #3 Jester
Info: {'targets': [1, 2, 6], 'evil_count': 2}

### [23:40:40] Ability used at #3

#### [23:40:46] Solver Output
Scenarios: 4/72
Definite evil: ['#5']
Definite good: ['#8', '#9']
Evil probabilities: #1=75%, #4=75%, #2=50%, #3=50%, #6=25%, #7=25%
  Generated 72 candidate scenarios
  4 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7]

#### [23:40:46] Recommendation
Action: **USE_ABILITY** #4 (Judge) -> targets ['#1']
Reason: Expected posterior 2.0 scenarios (adjusted 2.0, info gain 1.000 bits) | timing x1.00

### [23:41:25] Revealed #4 Judge
Info: {'target': 1, 'is_lying': True}

### [23:41:30] Ability used at #4

#### [23:41:37] Solver Output
Scenarios: 2/72
Definite evil: ['#1', '#5']
Definite good: ['#2', '#8', '#9']
Evil probabilities: #3=50%, #4=50%, #6=50%, #7=50%
  Generated 72 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #2 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 4, 6, 7]

#### [23:41:37] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Twin_Minion'})

### [23:42:19] Executed #1 -> Twin_Minion (EVIL)

#### [23:42:25] Solver Output
Scenarios: 2/9
Definite evil: ['#1', '#5']
Definite good: ['#2', '#8', '#9']
Evil probabilities: #3=50%, #4=50%, #6=50%, #7=50%
  Generated 9 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #2 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 4, 6, 7]

#### [23:42:25] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% good Jester, 50% evil Puppeteer).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [23:43:23] Executed #3 -> GOOD (WRONG!)

#### [23:43:29] Solver Output
Scenarios: 1/5
Definite evil: ['#1', '#5', '#6', '#7']
Definite good: ['#2', '#3', '#4', '#8', '#9']
  Generated 5 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [23:43:29] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [23:44:12] Executed #6 -> Puppet (EVIL)

### [23:45:06] Executed #7 -> Puppeteer (EVIL)

## [23:45:19] GAME OVER — WIN
Final HP: 1
Notes: Lilis+Puppeteer game, 4 evil. Night killed #9 (good Empress). Bishop #1 (evil Twin Minion) lied. Jester #3 (Empress) found 2 evils in #1,#2,#6. Judge #4 found #1 lying. Exec #5=Lilis, #1=Twin Minion, then 50/50 on #3 (good Jester, wrong exec HP 6->1), solver locked #6=Puppet #7=Puppeteer. HP 1/10.


---

# New Game — 2026-03-10 23:46:52
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Architect, Slayer, Oracle, Judge, Dreamer
- Outcasts: Plague_Doctor, Wretch
- Minions: Minion, Chancellor
- Demons: Lilis

### [23:48:13] Revealed #1 Oracle
Info: {'targets': [3, 5], 'minion_role': 'Minion'}

### [23:48:19] Revealed #2 Judge
Info: {}

### [23:48:19] Revealed #3 Jester
Info: {}

### [23:48:20] Revealed #4 Slayer
Info: {}

### [23:49:19] Revealed #6 Oracle
Info: {'targets': [2, 4], 'minion_role': 'Chancellor'}

### [23:49:25] Revealed #7 Plague_Doctor
Info: {}

### [23:49:25] Revealed #8 Jester
Info: {}

### [23:49:26] Revealed #9 Wretch
Info: {}

#### [23:49:41] Solver Output
Scenarios: 69/1952
Definite good: ['#5']
Evil probabilities: #8=84%, #6=68%, #1=67%, #9=28%, #3=25%, #4=17%, #2=9%, #7=3%
  Generated 1952 candidate scenarios
  69 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [23:49:41] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#2']
Reason: Entropy 1.134 (adjusted 1.134) | timing x1.00

### [23:51:38] Ability used at #7

#### [23:51:45] Solver Output
Scenarios: 52/1952
Definite good: ['#5', '#7']
Evil probabilities: #8=88%, #1=69%, #6=65%, #9=31%, #3=19%, #4=15%, #2=12%
  Generated 1952 candidate scenarios
  52 scenarios survived validation
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 8, 9]

#### [23:51:45] Recommendation
Action: **USE_ABILITY** #2 (Judge) -> targets ['#3']
Reason: Expected posterior 26.2 scenarios (adjusted 26.2, info gain 0.991 bits) | timing x1.00

### [23:52:45] Revealed #2 Judge
Info: {'target': 3, 'is_lying': False}

### [23:52:45] Ability used at #2

#### [23:52:53] Solver Output
Scenarios: 28/1952
Definite evil: ['#8']
Definite good: ['#3', '#5', '#7']
Evil probabilities: #1=79%, #6=64%, #9=36%, #4=14%, #2=7%
  Generated 1952 candidate scenarios
  28 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Lilis', 'Chancellor', 'Minion'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 9]

#### [23:52:53] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 28 scenarios (roles: {'Lilis', 'Chancellor', 'Minion'})

### [23:53:51] Executed #8 -> Chancellor (EVIL)

#### [23:53:57] Solver Output
Scenarios: 8/184
Definite evil: ['#8']
Definite good: ['#2', '#3', '#4', '#5', '#7']
Evil probabilities: #1=75%, #6=75%, #9=50%
  Generated 184 candidate scenarios
  8 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 6, 9]

#### [23:53:57] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#2', '#6']
Reason: Expected posterior 4.0 scenarios (adjusted 4.0, info gain 1.000 bits) | timing x1.00

### [23:55:12] Revealed #3 Jester
Info: {'targets': [1, 2, 6], 'evil_count': 2}

### [23:55:12] Ability used at #3

#### [23:55:20] Solver Output
Scenarios: 4/184
Definite evil: ['#1', '#6', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#9']
  Generated 184 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #6 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [23:55:20] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Minion', 'Lilis'})

### [23:56:06] Executed #1 -> Minion (EVIL)

### [23:56:54] Executed #6 -> Lilis (EVIL)

## [23:57:05] GAME OVER — WIN
Final HP: 6
Notes: Lilis game, Wretch in deck. Night killed #5 (good Dreamer). Oracle #1 (evil Minion) lied '#3 or #5 is Minion'. Judge #2 found #3 truthful. Jester #3 (Empress) found 2 evils in #1,#2,#6 -> locked #1 and #6 as evil. HP 6/10. Ascension 33 complete (7/7).


---

# New Game — 2026-03-11 16:18:15
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Baker, Scout, Lover, Confessor, Druid, Poet
- Outcasts: Plague_Doctor
- Minions: Witch, Minion
- Demons: Pooka

#### [16:36:58] Claude Reasoning


### [16:37:11] Revealed #9 Oracle
Info: {'targets': [4, 5], 'minion_role': 'Minion'}

### [16:37:15] Revealed #1 Lover
Info: {'evil_adjacent': 2}

### [16:37:18] Revealed #2 Plague_Doctor
Info: {}

### [16:37:22] Revealed #3 Druid
Info: {}

### [16:37:25] Revealed #4 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 4}

### [16:37:30] Revealed #5 Baker
Info: {'original_role': 'original'}

### [16:37:34] Revealed #6 Scout
Info: {'evil_role': 'Witch', 'distance': 3}

### [16:37:37] Revealed #7 Baker
Info: {'original_role': 'Confessor'}

#### [16:37:46] Solver Output
Scenarios: 23/1848
Definite good: ['#2', '#4']
Evil probabilities: #1=74%, #6=65%, #9=52%, #7=39%, #3=35%, #8=30%, #5=4%
  Generated 1848 candidate scenarios
  23 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 6, 7, 8, 9]

#### [16:37:46] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#8']
Reason: Entropy 1.209 (adjusted 1.209) | timing x1.00

### [16:38:22] Ability used at #2

#### [16:38:26] Solver Output
Scenarios: 17/1848
Definite good: ['#2', '#4']
Evil probabilities: #1=82%, #6=71%, #3=41%, #8=41%, #9=35%, #7=24%, #5=6%
  Generated 1848 candidate scenarios
  17 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 6, 7, 8, 9]

#### [16:38:26] Recommendation
Action: **USE_ABILITY** #3 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 0.977 (adjusted 0.977) | timing x1.00

### [16:39:03] Revealed #3 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': 'Plague_Doctor'}

### [16:39:07] Ability used at #3

#### [16:39:14] Solver Output
Scenarios: 10/1848
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4']
Evil probabilities: #1=80%, #8=60%, #9=30%, #7=20%, #5=10%
  Generated 1848 candidate scenarios
  10 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka', 'Minion', 'Witch'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 5, 7, 8, 9]

#### [16:39:14] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 10 scenarios (roles: {'Pooka', 'Minion', 'Witch'})

### [16:39:51] Executed #6 -> Witch (EVIL)

### [16:40:33] Revealed #8 Scout
Info: {'evil_role': 'Witch', 'distance': 1}

#### [16:40:37] Solver Output
Scenarios: 4/224
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5']
Evil probabilities: #1=75%, #8=75%, #7=25%, #9=25%
  Generated 224 candidate scenarios
  4 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 7, 8, 9]

#### [16:40:37] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% evil Pooka, 25% good Lover (corrupted), 25% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [16:41:13] Executed #1 -> Pooka (EVIL)

#### [16:41:17] Solver Output
Scenarios: 2/31
Definite evil: ['#1', '#6']
Definite good: ['#2', '#3', '#4', '#5', '#9']
Evil probabilities: #7=50%, #8=50%
  Generated 31 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #6 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [7, 8]

#### [16:41:17] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% good Baker, 50% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

#### [17:01:42] Solver Output
Scenarios: 2/31
Definite evil: ['#1', '#6']
Definite good: ['#2', '#3', '#4', '#5', '#9']
Evil probabilities: #7=50%, #8=50%
  Generated 31 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #6 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [7, 8]

#### [17:01:42] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% good Baker, 50% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [17:02:19] Executed #7 -> GOOD (WRONG!)

#### [17:02:29] Solver Output
Scenarios: 1/26
Definite evil: ['#1', '#6', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#9']
  Generated 26 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #6 is DEFINITELY EVIL (possible roles: {'Witch'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [17:02:29] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Minion'})

### [17:03:09] Executed #8 -> Minion (EVIL)

## [17:03:15] GAME OVER — WIN
Final HP: 5
Notes: Witch blocked #8. PD corrupted #4 Poet and #9 Oracle. Memory reader perfect match.


---

# New Game — 2026-03-11 17:05:47
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Dreamer, Hunter, Architect, Medium, Fortune_Teller, Enlightened, Empress
- Outcasts: Drunk, Bombardier
- Minions: Twin_Minion
- Demons: Baa

#### [17:08:22] Claude Reasoning


#### [17:08:27] Claude Reasoning


### [17:08:43] Revealed #1 Enlightened
Info: {'direction': 'Equidistant'}

### [17:08:47] Revealed #2 Hunter
Info: {'distance': 3}

### [17:08:50] Revealed #3 Medium
Info: {'good_position': 2, 'good_role': 'Hunter'}

### [17:08:54] Revealed #4 Fortune_Teller
Info: {}

### [17:08:57] Revealed #5 Empress
Info: {'targets': [3, 6, 8]}

### [17:09:01] Revealed #6 Dreamer
Info: {}

### [17:09:06] Revealed #7 Bombardier
Info: {}

### [17:09:09] Revealed #8 Architect
Info: {'side': 'Left'}

#### [17:09:13] Solver Output
Scenarios: 6/350
Definite good: ['#1', '#3', '#4']
Evil probabilities: #7=67%, #2=33%, #5=33%, #6=33%, #8=33%
  Generated 350 candidate scenarios
  6 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [2, 5, 6, 7, 8]

#### [17:09:13] Recommendation
Action: **USE_ABILITY** #6 (Dreamer) -> targets ['#7']
Reason: Entropy 2.252 (adjusted 2.252) | timing x1.00

### [17:09:53] Revealed #6 Dreamer
Info: {'target': 7, 'evil_role': 'Twin_Minion'}

### [17:09:57] Ability used at #6

#### [17:10:00] Solver Output
Scenarios: 4/350
Definite good: ['#1', '#3', '#4']
Evil probabilities: #2=50%, #7=50%, #8=50%, #5=25%, #6=25%
  Generated 350 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [2, 5, 6, 7, 8]

#### [17:10:00] Recommendation
Action: **USE_ABILITY** #4 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [17:10:33] Revealed #4 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [17:10:37] Ability used at #4

#### [17:10:41] Solver Output
Scenarios: 2/350
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#4', '#8']
Evil probabilities: #5=50%, #6=50%
  Generated 350 candidate scenarios
  2 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Baa', 'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 6]

#### [17:10:41] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Baa', 'Twin_Minion'})

### [17:11:09] Executed #7 -> Twin_Minion (EVIL)

#### [17:11:12] Solver Output
Scenarios: 1/49
Definite evil: ['#5', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#8']
  Generated 49 candidate scenarios
  1 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #7 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [17:11:12] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Baa'})

### [17:12:01] Executed #5 -> Baa (EVIL)

## [17:12:10] GAME OVER — WIN
Final HP: 10
Notes: Perfect game 10HP. Dreamer on #7 got Twin_Minion, FT confirmed #1/#2 good. #1 Drunk Corrupted.


---

# New Game — 2026-03-11 17:13:26
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Hunter, Baker, Lover, Confessor
- Outcasts: Plague_Doctor, Drunk, Wretch
- Minions: Chancellor
- Demons: Baa

### [17:14:41] Revealed #1 Confessor
Info: {'dizzy': True}

### [17:14:45] Revealed #2 Bishop
Info: {'targets': [1, 2, 7], 'types': ['Minion', 'Outcast', 'Villager']}

### [17:14:49] Revealed #3 Confessor
Info: {'dizzy': True}

### [17:14:52] Revealed #4 Hunter
Info: {'distance': 1}

### [17:14:56] Revealed #5 Baker
Info: {'original_role': 'original'}

### [17:15:00] Revealed #6 Wretch
Info: {}

### [17:15:04] Revealed #7 Confessor
Info: {'dizzy': True}

#### [17:15:09] Solver Output
Scenarios: 2/330
Definite evil: ['#3']
Definite good: ['#2', '#4', '#5', '#6']
Evil probabilities: #1=50%, #7=50%
  Generated 330 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Baa'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [17:15:09] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Baa'})

### [17:15:40] Executed #3 -> Baa (EVIL)

#### [17:15:44] Solver Output
Scenarios: 2/42
Definite evil: ['#3']
Definite good: ['#2', '#4', '#5', '#6']
Evil probabilities: #1=50%, #7=50%
  Generated 42 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Baa'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [17:15:44] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% evil Chancellor, 50% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [17:16:17] Executed #1 -> Chancellor (EVIL)

## [17:16:24] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. 3 Confessors on board (2 evil disguises + Drunk). Bishop+Hunter nailed it. #7 Drunk Corrupted.


---

# New Game — 2026-03-11 17:17:54
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Fortune_Teller, Baker, Poet, Scout, Confessor, Medium
- Outcasts: Plague_Doctor, Doppelganger, Bombardier
- Minions: Chancellor
- Demons: Lilis

### [17:18:56] Revealed #1 Medium
Info: {'good_position': 5, 'good_role': 'Plague_Doctor'}

### [17:19:00] Revealed #2 Confessor
Info: {'dizzy': False}

### [17:19:04] Revealed #3 Bombardier
Info: {}

### [17:19:09] Revealed #4 Fortune_Teller
Info: {}

### [17:23:03] Revealed #5 Plague_Doctor
Info: {}

### [17:23:07] Revealed #7 Baker
Info: {'original_role': 'Poet'}

### [17:23:20] Revealed #8 Alchemist
Info: {'cured_count': 2}

### [17:23:27] Revealed #9 Scout
Info: {'evil_role': 'Lilis', 'distance': 2}

#### [17:23:40] Solver Output
Scenarios: 16/1687
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']
Evil probabilities: #7=50%, #9=50%
  Generated 1687 candidate scenarios
  16 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Lilis', 'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [7, 9]

#### [17:23:40] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 16 scenarios (roles: {'Lilis', 'Chancellor'})

### [17:24:38] Executed #8 -> Lilis (EVIL)

#### [17:24:45] Solver Output
Scenarios: 8/176
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']
Evil probabilities: #7=50%, #9=50%
  Generated 176 candidate scenarios
  8 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [7, 9]

#### [17:24:45] Recommendation
Action: **USE_ABILITY** #4 (Fortune Teller) -> targets ['#1', '#7']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [17:25:27] Revealed #4 Fortune Teller
Info: {'targets': [1, 7], 'has_evil': True}

### [17:25:33] Ability used at #4

#### [17:25:38] Solver Output
Scenarios: 4/176
Definite evil: ['#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#9']
  Generated 176 candidate scenarios
  4 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [17:25:38] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 4 scenarios (roles: {'Chancellor'})

### [17:26:15] Executed #7 -> Chancellor (EVIL)

## [17:26:23] GAME OVER — WIN
Final HP: 6
Notes: Lilis game 6HP. Night killed #6 Doppelganger. #9 Scout corrupted. FT confirmed evil in #1/#7 -> solver locked #7. flip --lilis bug: re-clicked already-flipped cards on 2nd batch, accidentally activated FT ability (cancelled).


---

# New Game — 2026-03-11 17:28:51
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Lover, Fortune_Teller, Gemcrafter, Scout, Dreamer
- Outcasts: Doppelganger, Bombardier
- Minions: Twin_Minion
- Demons: Pooka

### [17:29:59] Revealed #1 Baker
Info: {'original_role': 'original'}

### [17:30:04] Revealed #2 Baker
Info: {'original_role': 'Fortune_Teller'}

### [17:30:09] Revealed #3 Gemcrafter
Info: {'good_position': 4}

### [17:30:16] Revealed #4 Scout
Info: {'evil_role': 'Twin_Minion', 'distance': 2}

### [17:30:21] Revealed #5 Fortune_Teller
Info: {}

### [17:30:31] Revealed #6 Baker
Info: {'original_role': 'Dreamer'}

### [17:30:31] Revealed #7 Baker
Info: {'original_role': 'Lover'}

### [17:30:31] Revealed #8 Bombardier
Info: {}

### [17:30:32] Revealed #9 Bombardier
Info: {}

#### [17:30:37] Solver Output
Scenarios: 1/392
Definite evil: ['#2', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7', '#8']
  Generated 392 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #9 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [17:30:37] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [17:31:36] Executed #2 -> GOOD (WRONG!)

#### [17:31:47] Solver Output
Scenarios: 0/308
  Generated 308 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #2 Baker: rejected 248/308 (81%)
    #4 Scout: rejected 158/308 (51%)
    #3 Gemcrafter: rejected 102/308 (33%)
    #1 Baker: rejected 31/308 (10%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Baker: 5 scenarios survive  <-- SUSPECT
    WITHOUT #2 Baker: 51 scenarios survive  <-- SUSPECT
    WITHOUT #3 Gemcrafter: 13 scenarios survive  <-- SUSPECT
    WITHOUT #4 Scout: 10 scenarios survive  <-- SUSPECT
    WITHOUT #6 Baker: 5 scenarios survive  <-- SUSPECT
    WITHOUT #7 Baker: 5 scenarios survive  <-- SUSPECT

#### [17:31:47] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [17:37:14] Solver Output
Scenarios: 15/308
Definite good: ['#1', '#2', '#3']
Evil probabilities: #5=67%, #8=53%, #9=47%, #4=13%, #6=13%, #7=7%
  Generated 308 candidate scenarios
  15 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 7, 8, 9]

#### [17:37:14] Recommendation
Action: **USE_ABILITY** #5 (Fortune Teller) -> targets ['#1', '#8']
Reason: Entropy 0.997 (adjusted 0.997) | timing x1.00

### [17:37:56] Revealed #5 Fortune Teller
Info: {'targets': [1, 8], 'has_evil': False}

### [17:37:56] Ability used at #5

#### [17:38:01] Solver Output
Scenarios: 7/308
Definite good: ['#1', '#2', '#3', '#6']
Evil probabilities: #5=71%, #8=71%, #9=29%, #4=14%, #7=14%
  Generated 308 candidate scenarios
  7 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [4, 5, 7, 8, 9]

#### [17:38:01] Recommendation
Action: **EXECUTE** #5
Reason: No reveals available. #5 is 71% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 71% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #5 (71%) despite low confidence — Bombardier candidate(s) [8, 9] risk instant game loss if executed first.

### [17:39:42] Executed #5 -> GOOD (WRONG!)

## [17:39:57] GAME OVER — LOSS
Final HP: 0
Notes: LOSS from Baker normalization bug. Solver had 1 scenario (wrong: #2=TM,#9=Pooka) due to 'Fortune_Teller' vs 'Fortune Teller' mismatch in Baker validator. Wrong exec #2 cost 5HP. After hotfix, solver had 7 scenarios with true pair only 1/7. Then 71% on #5 (good Doppelganger) killed us. #3 Gemcrafter Corrupted by adjacent Pooka.


---

# New Game — 2026-03-11 17:43:57
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Oracle, Jester, Empress, Druid, Architect, Gemcrafter, Medium
- Outcasts: Drunk, Bombardier, Plague_Doctor
- Minions: Chancellor, Twin_Minion
- Demons: Pooka

### [17:45:18] Revealed #1 Oracle
Info: {'targets': [3, 5], 'minion_role': 'Twin_Minion'}

### [17:45:24] Revealed #2 Jester
Info: {}

### [17:45:25] Revealed #3 Empress
Info: {'targets': [4, 5, 7]}

### [17:45:25] Revealed #4 Druid
Info: {}

### [17:45:25] Revealed #5 Architect
Info: {'side': 'Equal'}

### [17:45:31] Revealed #6 Gemcrafter
Info: {'good_position': 8}

### [17:45:31] Revealed #7 Plague_Doctor
Info: {}

### [17:45:31] Revealed #8 Medium
Info: {'good_position': 4, 'good_role': 'Druid'}

### [17:45:31] Revealed #9 Bombardier
Info: {}

#### [17:45:37] Solver Output
Scenarios: 193/11740
Evil probabilities: #4=77%, #9=51%, #8=45%, #1=34%, #2=31%, #5=19%, #6=19%, #7=15%, #3=10%
  Generated 11740 candidate scenarios
  193 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [17:45:37] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.476 (adjusted 2.476) | timing x1.00

### [17:46:25] Ability used at #7

#### [17:46:26] Solver Output
Scenarios: 66/11740
Evil probabilities: #4=80%, #8=64%, #5=47%, #9=42%, #2=23%, #3=17%, #7=12%, #1=9%, #6=6%
  Generated 11740 candidate scenarios
  66 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [17:46:26] Recommendation
Action: **USE_ABILITY** #4 (Druid) -> targets ['#1', '#2', '#9']
Reason: Entropy 1.213 (adjusted 1.121) | timing x1.00
WARNING: Corruption risk: 15%

### [17:47:15] Revealed #4 Druid
Info: {'targets': [1, 2, 9], 'found_outcast': 'Drunk'}

### [17:47:15] Ability used at #4

#### [17:47:21] Solver Output
Scenarios: 19/11740
Definite evil: ['#8', '#9']
Definite good: ['#1', '#6', '#7']
Evil probabilities: #4=79%, #2=11%, #3=5%, #5=5%
  Generated 11740 candidate scenarios
  19 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Pooka', 'Twin_Minion'})
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Pooka', 'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5]

#### [17:47:21] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 19 scenarios (roles: {'Chancellor', 'Pooka', 'Twin_Minion'})

### [17:47:57] Executed #8 -> Chancellor (EVIL)

#### [17:47:57] Solver Output
Scenarios: 5/1046
Definite evil: ['#8', '#9']
Definite good: ['#1', '#3', '#5', '#6', '#7']
Evil probabilities: #4=80%, #2=20%
  Generated 1046 candidate scenarios
  5 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #9 is DEFINITELY EVIL (possible roles: {'Pooka', 'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 4]

#### [17:47:57] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 5 scenarios (roles: {'Pooka', 'Twin_Minion'})

### [17:48:34] Executed #9 -> Twin_Minion (EVIL)

#### [17:48:34] Solver Output
Scenarios: 1/187
Definite evil: ['#2', '#8', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7']
  Generated 187 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #9 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [17:48:34] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [17:49:18] Executed #2 -> Pooka (EVIL)

## [17:49:19] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP on 3-evil retry. Massive corruption from Pooka (#2): Oracle#1, Empress#3, Drunk#4, Gemcrafter#6 all corrupted. PD check + Druid ability narrowed to 19 scenarios, then 5->1.


---

# New Game — 2026-03-11 17:52:39
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Poet, Knight, Druid, Scout, Baker, Judge
- Outcasts: Plague_Doctor
- Minions: Twin_Minion, Minion
- Demons: Pooka

### [17:55:03] Revealed #1 Poet
Info: {'direction': 'CCW', 'copied_role': 'Enlightened'}

### [17:55:06] Revealed #2 Plague_Doctor
Info: {}

### [17:55:09] Revealed #3 Baker
Info: {'original_role': 'original'}

### [17:55:12] Revealed #4 Plague_Doctor
Info: {}

### [17:55:15] Revealed #5 Bishop
Info: {'targets': [3, 6, 7], 'types': ['Minion', 'Villager', 'Outcast']}

### [17:55:18] Revealed #6 Scout
Info: {'evil_role': 'Twin_Minion', 'distance': 1}

### [17:55:21] Revealed #7 Druid
Info: {}

### [17:55:24] Revealed #8 Judge
Info: {}

### [17:55:27] Revealed #9 Poet
Info: {'corruption_distance': 2, 'copied_role': 'Bard'}

### [17:55:31] Revealed #10 Baker
Info: {'original_role': 'Knight'}

#### [17:55:35] Solver Output
Scenarios: 37/3744
Definite good: ['#3']
Evil probabilities: #4=78%, #9=59%, #1=49%, #5=38%, #6=32%, #2=22%, #8=8%, #10=8%, #7=5%
  Generated 3744 candidate scenarios
  37 scenarios survived validation
    #3 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7, 8, 9, 10]

#### [17:55:35] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.061 (adjusted 2.061) | timing x1.00

### [17:56:44] Ability used at #2

#### [17:56:48] Solver Output
Scenarios: 20/3744
Definite good: ['#3', '#6', '#7']
Evil probabilities: #4=65%, #5=65%, #9=60%, #1=55%, #2=35%, #8=10%, #10=10%
  Generated 3744 candidate scenarios
  20 scenarios survived validation
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 8, 9, 10]

#### [17:56:48] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#10']
Reason: Entropy 2.202 (adjusted 2.202) | timing x1.00

### [17:57:35] Ability used at #4

#### [17:57:39] Solver Output
Scenarios: 4/3744
Definite evil: ['#4', '#5']
Definite good: ['#2', '#3', '#6', '#7', '#8', '#10']
Evil probabilities: #1=50%, #9=50%
  Generated 3744 candidate scenarios
  4 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 9]

#### [17:57:39] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 4 scenarios (roles: {'Minion'})

### [17:58:16] Executed #4 -> Minion (EVIL)

#### [17:58:19] Solver Output
Scenarios: 4/352
Definite evil: ['#4', '#5']
Definite good: ['#2', '#3', '#6', '#7', '#8', '#10']
Evil probabilities: #1=50%, #9=50%
  Generated 352 candidate scenarios
  4 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 9]

#### [17:58:19] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 4 scenarios (roles: {'Pooka'})

### [17:58:46] Executed #5 -> Pooka (EVIL)

#### [17:58:50] Solver Output
Scenarios: 4/43
Definite evil: ['#4', '#5']
Definite good: ['#2', '#3', '#6', '#7', '#8', '#10']
Evil probabilities: #1=50%, #9=50%
  Generated 43 candidate scenarios
  4 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 9]

#### [17:58:50] Recommendation
Action: **USE_ABILITY** #7 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.811 (adjusted 0.710) | timing x1.00
WARNING: Corruption risk: 25%

### [17:59:53] Revealed #7 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Plague_Doctor'}

### [17:59:58] Ability used at #7

#### [18:00:02] Solver Output
Scenarios: 3/43
Definite evil: ['#4', '#5']
Definite good: ['#2', '#3', '#6', '#7', '#8', '#10']
Evil probabilities: #9=67%, #1=33%
  Generated 43 candidate scenarios
  3 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 9]

#### [18:00:02] Recommendation
Action: **USE_ABILITY** #8 (Judge) -> targets ['#2']
Reason: Expected posterior 2.5 scenarios (adjusted 2.9, info gain 0.041 bits) | timing x1.00
WARNING: Corruption risk: 33% -- corrupted Judge results are unreliable

### [18:00:53] Revealed #8 Judge
Info: {'target': 2, 'is_lying': True}

### [18:00:57] Ability used at #8

#### [18:01:01] Solver Output
Scenarios: 1/43
Definite evil: ['#4', '#5', '#9']
Definite good: ['#1', '#2', '#3', '#6', '#7', '#8', '#10']
  Generated 43 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #9 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [18:01:01] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [18:01:54] Executed #9 -> Twin_Minion (EVIL)

## [18:02:02] GAME OVER — WIN
Final HP: 10
Notes: Perfect game 10HP. PD#2 clean on #5, PD#4 (evil) lied about #7/#10. Judge#8 corrupted said #2 lying -> solver narrowed to 1 scenario. 3 correct execs.


---

# New Game — 2026-03-11 18:04:38
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Bishop, Jester, Poet, Scout, Judge
- Outcasts: Plague_Doctor
- Minions: 
- Demons: Pooka

### [18:04:56] Revealed #7 Bishop
Info: {'targets': [1, 4, 7], 'types': ['Villager', 'Demon', 'Outcast']}

### [18:06:03] Revealed #1 Empress
Info: {'targets': [4, 6, 7]}

### [18:06:08] Revealed #2 Plague_Doctor
Info: {}

### [18:07:36] Revealed #3 Jester
Info: {}

### [18:07:43] Revealed #4 Judge
Info: {}

### [18:07:43] Revealed #5 Scout
Info: {'evil_role': 'Pooka', 'distance': 3}

### [18:07:43] Revealed #6 Poet
Info: {'targets': [1, 3, 5], 'types': ['Demon', 'Villager', 'Outcast'], 'copied_role': 'Bishop'}

#### [18:07:49] Solver Output
Scenarios: 5/31
Definite good: ['#2', '#3', '#4', '#5', '#7']
Evil probabilities: #6=80%, #1=20%
  Generated 31 candidate scenarios
  5 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 6]

#### [18:07:49] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#2', '#4']
Reason: Expected posterior 2.1 scenarios (adjusted 2.4, info gain 1.085 bits) | timing x1.00
WARNING: Corruption risk: 20%

### [18:08:49] Revealed #3 Jester
Info: {'targets': [1, 2, 4], 'evil_count': 2}

### [18:08:54] Ability used at #3

#### [18:09:00] Solver Output
Scenarios: 1/31
Definite evil: ['#6']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#7']
  Generated 31 candidate scenarios
  1 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [18:09:00] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [18:09:40] Executed #6 -> Pooka (EVIL)

## [18:09:47] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Corrupted Jester said 2 evils among 1,2,4 (impossible with 1 evil) -> solver narrowed to 1 scenario confirming #6 Pooka. Ascension 34 complete 7/7!


---

# New Game — 2026-03-11 18:30:06
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Knight, Medium, Druid, Jester, Fortune_Teller, Bard
- Outcasts: Bombardier, Wretch
- Minions: Chancellor, Shaman
- Demons: Lilis

### [18:31:46] Revealed #1 Medium
Info: {'good_position': 3, 'good_role': 'Bombardier'}

### [18:31:51] Revealed #2 Medium
Info: {'good_position': 4, 'good_role': 'Fortune_Teller'}

### [18:31:56] Revealed #3 Bombardier
Info: {}

### [18:32:02] Revealed #4 Fortune_Teller
Info: {}

#### [18:32:13] Solver Output
Scenarios: 126/742
Definite good: ['#6']
Evil probabilities: #5=40%, #7=38%, #10=38%, #8=37%, #9=37%, #1=32%, #3=32%, #2=24%, #4=24%
  Generated 742 candidate scenarios
  126 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9, 10]

#### [18:32:13] Recommendation
Action: **REVEAL** #5
Reason: #5: 40% evil, entropy 1.069

### [18:34:06] Revealed #5 Jester
Info: {}

### [18:34:11] Revealed #7 Knitter
Info: {'evil_pairs': 2}

### [18:34:16] Revealed #8 Wretch
Info: {}

### [18:34:22] Revealed #9 Bombardier
Info: {}

### [18:35:08] Revealed #10 Knight
Info: {}

#### [18:35:14] Solver Output
Scenarios: 28/862
Definite good: ['#2', '#4', '#6']
Evil probabilities: #9=64%, #10=64%, #7=57%, #1=36%, #3=36%, #5=29%, #8=14%
  Generated 862 candidate scenarios
  28 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 7, 8, 9, 10]

#### [18:35:14] Recommendation
Action: **EXECUTE** #10
Reason: Knight free check: #10 is 64% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [18:36:21] Executed #10 -> Lilis (EVIL)

#### [18:36:26] Solver Output
Scenarios: 7/90
Definite evil: ['#9', '#10']
Definite good: ['#1', '#2', '#3', '#4', '#6']
Evil probabilities: #5=43%, #7=29%, #8=29%
  Generated 90 candidate scenarios
  7 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Shaman'})
    #10 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [5, 7, 8]

#### [18:36:26] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 7 scenarios (roles: {'Chancellor', 'Shaman'})

### [18:37:13] Executed #9 -> Chancellor (EVIL)

#### [18:37:19] Solver Output
Scenarios: 3/8
Definite evil: ['#9', '#10']
Definite good: ['#1', '#2', '#3', '#4', '#6']
Evil probabilities: #5=33%, #7=33%, #8=33%
  Generated 8 candidate scenarios
  3 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #10 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [5, 7, 8]

#### [18:37:19] Recommendation
Action: **USE_ABILITY** #5 (Jester) -> targets ['#1', '#2', '#3']
Reason: Expected posterior 1.4 scenarios (adjusted 1.4, info gain 1.100 bits) | timing x1.00

### [18:38:43] Revealed #5 Jester
Info: {'targets': [1, 2, 3], 'evil_count': 0}

### [18:38:49] Ability used at #5

#### [18:38:54] Solver Output
Scenarios: 2/8
Definite evil: ['#9', '#10']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']
Evil probabilities: #7=50%, #8=50%
  Generated 8 candidate scenarios
  2 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #10 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [7, 8]

#### [18:38:54] Recommendation
Action: **USE_ABILITY** #4 (Fortune Teller) -> targets ['#1', '#7']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [18:40:05] Revealed #4 Fortune Teller
Info: {'targets': [1, 7], 'has_evil': True}

### [18:40:11] Ability used at #4

#### [18:40:20] Solver Output
Scenarios: 1/8
Definite evil: ['#7', '#9', '#10']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#8']
  Generated 8 candidate scenarios
  1 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #10 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [18:40:20] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [18:41:21] Executed #7 -> Shaman (EVIL)

## [18:41:29] GAME OVER — WIN
Final HP: 6
Notes: 6HP. Lilis game, 10 cards. Knight free check found Lilis. Shaman dup Medium. FT+Jester narrowed last evil.


---

# New Game — 2026-03-11 18:43:57
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Alchemist, Confessor, Enlightened, Judge, Knitter, Fortune_Teller
- Outcasts: Bombardier
- Minions: Shaman
- Demons: Lilis

### [18:45:14] Revealed #1 Alchemist
Info: {'cured_count': 0}

### [18:45:14] Revealed #2 Alchemist
Info: {'cured_count': 0}

### [18:45:14] Revealed #3 Knitter
Info: {'evil_pairs': 1}

### [18:45:14] Revealed #4 Bombardier
Info: {}

### [18:46:37] Revealed #5 Enlightened
Info: {'direction': 'CCW'}

### [18:46:37] Revealed #6 Fortune_Teller
Info: {}

### [18:46:37] Revealed #8 Confessor
Info: {}

#### [18:46:46] Solver Output
Scenarios: 4/56
Definite good: ['#1', '#2', '#4', '#7']
Evil probabilities: #3=50%, #5=50%, #6=50%, #8=50%
  Generated 56 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [3, 5, 6, 8]

#### [18:46:46] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [18:48:02] Revealed #6 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [18:48:02] Ability used at #6

#### [18:48:03] Solver Output
Scenarios: 2/56
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7']
  Generated 56 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Shaman', 'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Shaman', 'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [18:48:03] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Shaman', 'Lilis'})

### [18:48:54] Executed #3 -> Lilis (EVIL)

### [18:51:03] Executed #8 -> Shaman (EVIL)

## [18:51:06] GAME OVER — WIN
Final HP: 8
Notes: 8HP. Lilis game, 8 cards. Shaman dup Alchemist. Confessor dizzy+no corruption source=evil disguise. FT alignment test confirmed.


---

# New Game — 2026-03-11 18:52:52
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Confessor, Lover, Scout, Dreamer, Poet
- Outcasts: Drunk
- Minions: 
- Demons: Pooka

### [18:53:58] Revealed #1 Confessor
Info: {'dizzy': False}

### [18:54:33] Revealed #2 Scout
Info: {}

### [18:54:37] Revealed #3 Lover
Info: {'evil_adjacent': 1}

### [18:54:40] Revealed #4 Dreamer
Info: {}

### [18:54:44] Revealed #5 Lover
Info: {'evil_adjacent': 1}

### [18:54:54] Revealed #6 Poet
Info: {'evil_role': 'Pooka', 'distance': 2, 'copied_role': 'Scout'}

### [18:54:59] Revealed #7 Enlightened
Info: {'direction': 'ccw'}

#### [18:55:02] Solver Output
Scenarios: 1/42
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#7']
  Generated 42 candidate scenarios
  1 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [18:55:02] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [18:55:36] Executed #5 -> Pooka (EVIL)

## [18:55:42] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. 7 cards, 1 evil Pooka. Solver 1 scenario, instant solve. Pooka corrupted #4 Dreamer and #6 Poet. Drunk at #3 disguised as Lover.


---

# New Game — 2026-03-11 18:57:24
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Confessor, Medium, Witness, Poet, Scout
- Outcasts: Bombardier, Doppelganger
- Minions: Minion
- Demons: Baa

### [18:58:23] Revealed #1 Poet
Info: {'evil_role': 'Baa', 'distance': 1, 'copied_role': 'Scout'}

### [18:58:27] Revealed #2 Medium
Info: {'good_position': 3, 'good_role': 'Medium'}

### [18:58:30] Revealed #3 Medium
Info: {'good_position': 2, 'good_role': 'Medium'}

### [18:58:34] Revealed #4 Confessor
Info: {'dizzy': False}

### [18:58:39] Revealed #5 Bombardier
Info: {}

### [18:58:43] Revealed #6 Hunter
Info: {'distance': 3}

### [18:58:48] Revealed #7 Scout
Info: {'evil_role': 'Baa', 'distance': 1}

### [18:59:09] Revealed #8 Witness
Info: {'affected_position': 0}

#### [18:59:14] Solver Output
Scenarios: 20/350
Definite good: ['#1', '#4', '#7', '#8']
Evil probabilities: #2=60%, #3=60%, #5=40%, #6=40%
  Generated 350 candidate scenarios
  20 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3, 5, 6]

#### [18:59:14] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (40% good Medium, 30% evil Baa, 30% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [18:59:50] Executed #2 -> Minion (EVIL)

#### [18:59:54] Solver Output
Scenarios: 6/43
Definite evil: ['#2', '#3']
Definite good: ['#1', '#4', '#5', '#6', '#7', '#8']
  Generated 43 candidate scenarios
  6 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Minion'})
    #3 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [18:59:54] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 6 scenarios (roles: {'Baa'})

### [19:00:26] Executed #3 -> Baa (EVIL)

## [19:00:33] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. 8 cards, 2 evil. Both Mediums were evil (Minion+Baa disguised). Scout+Poet both reported Baa 1 from Minion. Execution lookahead guaranteed win on #2.


---

# New Game — 2026-03-11 19:02:19
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Lover, Slayer, Confessor, Empress, Knight
- Outcasts: Bombardier, Wretch
- Minions: Twin_Minion
- Demons: Lilis

### [19:03:18] Revealed #1 Confessor
Info: {'dizzy': False}

### [19:03:23] Revealed #2 Empress
Info: {'targets': [3, 4, 8]}

### [19:03:27] Revealed #3 Knight
Info: {}

### [19:03:31] Revealed #4 Hunter
Info: {'distance': 1}

#### [19:03:57] Solver Output
Scenarios: 14/72
Definite good: ['#1', '#7']
Evil probabilities: #3=43%, #5=43%, #4=29%, #6=29%, #9=29%, #2=14%, #8=14%
  Generated 72 candidate scenarios
  14 scenarios survived validation
    #1 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6, 8, 9]

#### [19:03:57] Recommendation
Action: **EXECUTE** #3
Reason: Knight free check: #3 is 43% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [19:04:34] Executed #3 -> Twin_Minion (EVIL)

#### [19:04:38] Solver Output
Scenarios: 3/8
Definite evil: ['#3']
Definite good: ['#1', '#2', '#4', '#7', '#8']
Evil probabilities: #5=33%, #6=33%, #9=33%
  Generated 8 candidate scenarios
  3 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 6, 9]

#### [19:04:38] Recommendation
Action: **REVEAL** #5
Reason: #5: 33% evil, entropy 1.018

### [19:05:41] Revealed #5 Wretch
Info: {}

### [19:05:45] Revealed #6 Slayer
Info: {}

### [19:05:49] Revealed #8 Lover
Info: {'evil_adjacent': 0}

### [19:05:53] Revealed #9 Knight
Info: {}

#### [19:05:59] Solver Output
Scenarios: 1/8
Definite evil: ['#3', '#6']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#8', '#9']
  Generated 8 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [19:05:59] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [19:06:39] Executed #6 -> Lilis (EVIL)

## [19:06:46] GAME OVER — WIN
Final HP: 6
Notes: 6HP. Lilis game, 9 cards. Knight free check caught Twin Minion at #3. Lilis at #6 disguised as Slayer. Lilis killed Bombardier #7 night 1. No corruption sources.


---

# New Game — 2026-03-11 19:08:43
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Gemcrafter, Bishop, Dreamer, Knitter, Medium
- Outcasts: Plague_Doctor
- Minions: Shaman, Minion
- Demons: Pooka

### [19:10:07] Revealed #1 Poet
Info: {'targets': [3, 5, 9], 'copied_role': 'Empress'}

### [19:10:12] Revealed #2 Knitter
Info: {'evil_pairs': 2}

### [19:10:17] Revealed #3 Dreamer
Info: {}

### [19:10:23] Revealed #4 Knitter
Info: {'evil_pairs': 1}

### [19:10:28] Revealed #5 Plague_Doctor
Info: {}

### [19:10:33] Revealed #6 Medium
Info: {'good_position': 2, 'good_role': 'Knitter'}

### [19:10:38] Revealed #7 Plague_Doctor
Info: {}

### [19:10:43] Revealed #8 Poet
Info: {'targets': [3, 4, 6], 'copied_role': 'Empress'}

### [19:10:48] Revealed #9 Gemcrafter
Info: {'good_position': 6}

#### [19:10:56] Solver Output
Scenarios: 12/2142
Definite good: ['#8', '#9']
Evil probabilities: #2=67%, #5=67%, #3=50%, #1=33%, #6=33%, #7=33%, #4=17%
  Generated 2142 candidate scenarios
  12 scenarios survived validation
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [19:10:56] Recommendation
Action: **USE_ABILITY** #5 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.792 (adjusted 1.792) | timing x1.00

### [19:12:08] Ability used at #5

#### [19:12:14] Solver Output
Scenarios: 8/2142
Definite good: ['#8', '#9']
Evil probabilities: #5=75%, #1=50%, #2=50%, #3=50%, #4=25%, #6=25%, #7=25%
  Generated 2142 candidate scenarios
  8 scenarios survived validation
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [19:12:14] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#5']
Reason: Entropy 2.250 (adjusted 1.969) | timing x1.00
WARNING: Corruption risk: 25%

### [19:13:09] Revealed #3 Dreamer
Info: {'target': 5, 'evil_role': 'Minion'}

### [19:13:15] Ability used at #3

#### [19:13:20] Solver Output
Scenarios: 5/2142
Definite good: ['#8', '#9']
Evil probabilities: #2=60%, #5=60%, #1=40%, #3=40%, #4=40%, #7=40%, #6=20%
  Generated 2142 candidate scenarios
  5 scenarios survived validation
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [19:13:20] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#2']
Reason: Entropy 1.522 (adjusted 1.522) | timing x1.00

### [19:14:20] Ability used at #7

#### [19:14:26] Solver Output
Scenarios: 4/2142
Definite good: ['#6', '#8', '#9']
Evil probabilities: #1=50%, #2=50%, #3=50%, #4=50%, #5=50%, #7=50%
  Generated 2142 candidate scenarios
  4 scenarios survived validation
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7]

#### [19:14:26] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% evil Minion, 50% good Poet (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [19:15:11] Executed #1 -> GOOD (WRONG!)

#### [19:15:18] Solver Output
Scenarios: 2/1416
Definite evil: ['#2', '#4', '#7']
Definite good: ['#1', '#3', '#5', '#6', '#8', '#9']
  Generated 1416 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #4 is DEFINITELY EVIL (possible roles: {'Shaman', 'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Shaman', 'Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [19:15:18] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [19:15:58] Executed #2 -> Pooka (EVIL)

#### [19:16:03] Solver Output
Scenarios: 2/182
Definite evil: ['#2', '#4', '#7']
Definite good: ['#1', '#3', '#5', '#6', '#8', '#9']
  Generated 182 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #4 is DEFINITELY EVIL (possible roles: {'Minion', 'Shaman'})
    #7 is DEFINITELY EVIL (possible roles: {'Minion', 'Shaman'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [19:16:03] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Minion', 'Shaman'})

### [19:16:44] Executed #4 -> Minion (EVIL)

#### [19:16:50] Solver Output
Scenarios: 1/26
Definite evil: ['#2', '#4', '#7']
Definite good: ['#1', '#3', '#5', '#6', '#8', '#9']
  Generated 26 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [19:16:50] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [19:17:34] Executed #7 -> Shaman (EVIL)

## [19:17:45] GAME OVER — WIN
Final HP: 5
Notes: 5HP. 9 cards, 3 evil. Shaman dup Poet. Pooka corrupted #1,#3. PD#5 corrupted #6. PD abilities narrowed to 2 scenarios. Execution lookahead on #1 (50% wrong exec, still wins). Medium #6 lying (corrupted) said #2 real Knitter (lie).


---

# New Game — 2026-03-11 19:19:48
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Confessor, Empress, Poet, Hunter
- Outcasts: Bombardier
- Minions: 
- Demons: Pooka

### [19:21:51] Revealed #1 Hunter
Info: {'distance': 2}

### [19:21:57] Revealed #2 Confessor
Info: {'dizzy': False}

### [19:22:04] Revealed #3 Bombardier
Info: {}

### [19:22:20] Revealed #4 Poet
Info: {'targets': [3, 4, 6], 'types': ['Outcast', 'Demon', 'Villager'], 'copied_role': 'Bishop'}

### [19:22:27] Revealed #5 Bishop
Info: {'targets': [1, 2, 5], 'types': ['Outcast', 'Villager', 'Demon']}

### [19:22:34] Revealed #6 Empress
Info: {'targets': [2, 4, 5]}

#### [19:22:41] Solver Output
Scenarios: 1/6
Definite evil: ['#6']
Definite good: ['#1', '#2', '#3', '#4', '#5']
  Generated 6 candidate scenarios
  1 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD

#### [19:22:41] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [19:23:28] Executed #6 -> Pooka (EVIL)

## [19:23:37] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. 6 cards, 1 evil Pooka. 1-scenario instant solve. Pooka corrupted #1 Hunter and #5 Bishop. Ascension 35 complete (7/7).


---

# New Game — 2026-03-11 20:19:14
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Hunter, Alchemist, Druid, Baker, Bishop, Empress
- Outcasts: Wretch, Plague_Doctor
- Minions: Puppeteer, Shaman
- Demons: Lilis

### [20:20:24] Revealed #1 Bishop
Info: {'targets': [4, 6, 7], 'types': ['Outcast', 'Villager', 'Minion']}

### [20:20:30] Revealed #2 Hunter
Info: {'distance': 1}

### [20:20:30] Revealed #3 Alchemist
Info: {'cured_count': 1}

### [20:20:30] Revealed #4 Knight
Info: {}

#### [20:20:45] Solver Output
Scenarios: 150/1120
Definite evil: ['#3']
Definite good: ['#7']
Evil probabilities: #4=53%, #1=43%, #5=40%, #6=40%, #9=35%, #10=35%, #2=28%, #8=27%
  Generated 1120 candidate scenarios
  150 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Shaman', 'Lilis'})
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 8, 9, 10]

#### [20:20:45] Recommendation
Action: **EXECUTE** #4
Reason: Knight free check: #4 is 53% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [20:22:48] Executed #4 -> GOOD (WRONG!)

### [20:22:54] Revealed #8 Baker
Info: {'original_role': 'original'}


---

# New Game — 2026-03-11 20:26:11
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Fortune_Teller, Alchemist, Knight, Jester, Oracle, Lover, Medium
- Outcasts: Plague_Doctor
- Minions: Minion, Poisoner
- Demons: Pooka

### [20:27:35] Revealed #1 Oracle
Info: {'targets': [3, 5], 'minion_role': '1'}

### [20:27:36] Revealed #2 Jester
Info: {}

### [20:27:36] Revealed #3 Knight
Info: {}

### [20:27:36] Revealed #4 Medium
Info: {'good_position': 3, 'good_role': 'Knight'}

### [20:27:36] Revealed #5 Alchemist
Info: {'cured_count': 2}

### [20:27:36] Revealed #6 Lover
Info: {'evil_adjacent': 0}

### [20:27:36] Revealed #7 Plague_Doctor
Info: {}

### [20:27:36] Revealed #8 Medium
Info: {'good_position': 3, 'good_role': 'Knight'}

### [20:28:09] Revealed #9 Fortune_Teller
Info: {}

### [20:28:09] Revealed #10 Enlightened
Info: {}

#### [20:28:10] Solver Output
Scenarios: 0/4210
  Generated 4210 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #1 Oracle: rejected 2914/4210 (69%)
    #8 Medium: rejected 2437/4210 (58%)
    #4 Medium: rejected 2314/4210 (55%)
    #6 Lover: rejected 1617/4210 (38%)
    #5 Alchemist: rejected 1180/4210 (28%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Oracle: 95 scenarios survive  <-- SUSPECT
    WITHOUT #2 Jester: still 0
    WITHOUT #4 Medium: 107 scenarios survive  <-- SUSPECT
    WITHOUT #5 Alchemist: still 0
    WITHOUT #6 Lover: still 0
    WITHOUT #8 Medium: 51 scenarios survive  <-- SUSPECT
    WITHOUT #10 Enlightened: still 0

#### [20:28:10] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data


---

# New Game — 2026-03-11 20:31:34
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Lover, Architect, Judge, Medium, Slayer
- Outcasts: Wretch, Doppelganger
- Minions: Minion
- Demons: Baa


---

# New Game — 2026-03-11 21:22:39
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Scout, Witness, Knitter, Confessor, Medium
- Outcasts: Wretch, Plague_Doctor
- Minions: Poisoner, Minion
- Demons: Baa

#### [21:22:55] Solver Output
Scenarios: 504/504
Evil probabilities: #1=33%, #2=33%, #3=33%, #4=33%, #5=33%, #6=33%, #7=33%, #8=33%, #9=33%
  Generated 504 candidate scenarios
  504 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:22:55] Recommendation
Action: **REVEAL** #4
Reason: #4: 33% evil, 6.009 bits (95 outcomes)

### [21:23:21] Revealed #4 Medium
Info: {'good_position': 5, 'good_role': 'Knitter'}

#### [21:23:26] Solver Output
Scenarios: 276/504
Evil probabilities: #1=36%, #2=36%, #6=36%, #7=36%, #8=36%, #9=36%, #5=35%, #3=33%, #4=15%
  Generated 504 candidate scenarios
  276 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:23:26] Recommendation
Action: **REVEAL** #9
Reason: #9: 36% evil, 5.972 bits (85 outcomes)

### [21:23:53] Revealed #9 Bard
Info: {'corruption_distance': 1}

#### [21:23:58] Solver Output
Scenarios: 144/504
Evil probabilities: #9=69%, #1=42%, #8=42%, #2=32%, #6=32%, #7=32%, #3=28%, #5=17%, #4=7%
  Generated 504 candidate scenarios
  144 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:23:58] Recommendation
Action: **REVEAL** #8
Reason: #8: 42% evil, 5.180 bits (48 outcomes)

### [21:24:22] Revealed #8 Bard
Info: {'corruption_distance': 2}

#### [21:24:27] Solver Output
Scenarios: 60/504
Definite evil: ['#8']
Evil probabilities: #9=50%, #1=37%, #2=27%, #6=27%, #7=27%, #3=23%, #5=7%, #4=3%
  Generated 504 candidate scenarios
  60 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Minion', 'Baa', 'Poisoner'})
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [21:24:27] Recommendation
Action: **REVEAL** #1
Reason: #1: 37% evil, 4.498 bits (27 outcomes)

#### [21:24:41] Solver Output
Scenarios: 60/504
Definite evil: ['#8']
Evil probabilities: #9=50%, #1=37%, #2=27%, #6=27%, #7=27%, #3=23%, #5=7%, #4=3%
  Generated 504 candidate scenarios
  60 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Baa', 'Poisoner', 'Minion'})
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [21:24:41] Recommendation
Action: **REVEAL** #1
Reason: #1: 37% evil, 4.498 bits (27 outcomes)

### [21:25:53] Executed #8 -> Poisoner (EVIL)

#### [21:25:58] Solver Output
Scenarios: 32/56
Definite evil: ['#8']
Evil probabilities: #1=31%, #2=31%, #3=31%, #6=31%, #7=31%, #9=31%, #4=6%, #5=6%
  Generated 56 candidate scenarios
  32 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [21:25:58] Recommendation
Action: **REVEAL** #1
Reason: #1: 31% evil, 4.312 bits (21 outcomes)

### [21:27:01] Revealed #1 Scout
Info: {'evil_role': 'Minion', 'distance': 1}

#### [21:27:06] Solver Output
Scenarios: 18/56
Definite evil: ['#8']
Evil probabilities: #7=39%, #3=33%, #1=28%, #6=28%, #9=28%, #2=22%, #4=11%, #5=11%
  Generated 56 candidate scenarios
  18 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [21:27:06] Recommendation
Action: **REVEAL** #3
Reason: #3: 33% evil, 3.725 bits (14 outcomes)

### [21:28:01] Revealed #3 Scout
Info: {'evil_role': 'Poisoner', 'distance': 1}

#### [21:28:06] Solver Output
Scenarios: 5/56
Definite evil: ['#8']
Definite good: ['#4', '#5', '#6', '#9']
Evil probabilities: #3=80%, #1=60%, #2=40%, #7=20%
  Generated 56 candidate scenarios
  5 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 7]

#### [21:28:06] Recommendation
Action: **REVEAL** #2
Reason: #2: 40% evil, 1.922 bits (4 outcomes)

### [21:29:05] Revealed #2 Witness
Info: {'affected_position': 9}

#### [21:29:11] Solver Output
Scenarios: 3/56
Definite evil: ['#1', '#8']
Definite good: ['#2', '#4', '#5', '#6', '#9']
Evil probabilities: #3=67%, #7=33%
  Generated 56 candidate scenarios
  3 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Baa', 'Minion'})
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 7]

#### [21:29:11] Recommendation
Action: **REVEAL** #5
Reason: #5: 0% evil, 0.918 bits (2 outcomes)

### [21:30:11] Executed #1 -> Minion (EVIL)

#### [21:30:18] Solver Output
Scenarios: 2/7
Definite evil: ['#1', '#8']
Definite good: ['#2', '#4', '#5', '#6', '#9']
Evil probabilities: #3=50%, #7=50%
  Generated 7 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion'})
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 7]

#### [21:30:18] Recommendation
Action: **REVEAL** #5
Reason: #5: 0% evil, 1.000 bits (2 outcomes)

### [21:30:55] Revealed #7 Wretch
Info: {}

#### [21:31:02] Solver Output
Scenarios: 2/7
Definite evil: ['#1', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#9']
Evil probabilities: #6=50%, #7=50%
  Generated 7 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion'})
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [6, 7]

#### [21:31:02] Recommendation
Action: **REVEAL** #5
Reason: #5: 0% evil, 1.000 bits (2 outcomes)

### [21:32:00] Revealed #6 Confessor
Info: {'dizzy': True}

#### [21:32:06] Solver Output
Scenarios: 1/7
Definite evil: ['#1', '#6', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#9']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion'})
    #6 is DEFINITELY EVIL (possible roles: {'Baa'})
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [21:32:06] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Baa'})

### [21:32:47] Executed #6 -> Baa (EVIL)

## [21:32:56] GAME OVER — WIN
Final HP: 10
Notes: Minimum flips strategy. Shaman duplicated Scout+Bard. Fingerprint entropy guided flips. 8/9 flipped, 10HP perfect.


---

# New Game — 2026-03-11 21:36:28
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Gemcrafter, Bishop, Bard, Medium, Fortune_Teller, Alchemist
- Outcasts: Wretch, Plague_Doctor, Doppelganger
- Minions: Chancellor
- Demons: Pooka

#### [21:36:34] Solver Output
Scenarios: 0/0
  Generated 0 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  No candidate scenarios were generated (structural constraint failure)
  Check: confirmed_evil/confirmed_good, deck composition, n_evil count

#### [21:36:34] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data


---

# New Game — 2026-03-11 21:40:52
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Gemcrafter, Bishop, Bard, Medium, Fortune_Teller, Alchemist
- Outcasts: Wretch, Plague_Doctor, Doppelganger
- Minions: Chancellor
- Demons: Pooka

#### [21:40:58] Solver Output
Scenarios: 504/504
Evil probabilities: #1=22%, #2=22%, #3=22%, #4=22%, #5=22%, #6=22%, #7=22%, #8=22%, #9=22%
  Generated 504 candidate scenarios
  504 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:40:58] Recommendation
Action: **REVEAL** #9
Reason: #9: 22% evil, 5.006 bits (36 outcomes)

### [21:41:28] Revealed #9 Bard
Info: {'corruption_distance': 1}

#### [21:41:28] Solver Output
Scenarios: 196/504
Evil probabilities: #9=57%, #1=32%, #8=32%, #2=13%, #3=13%, #4=13%, #5=13%, #6=13%, #7=13%
  Generated 504 candidate scenarios
  196 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:41:28] Recommendation
Action: **REVEAL** #4
Reason: #4: 13% evil, 4.441 bits (22 outcomes)

### [21:42:02] Revealed #4 Bishop
Info: {'targets': [1, 2, 9], 'types': ['Outcast', 'Minion', 'Villager']}

#### [21:42:02] Solver Output
Scenarios: 156/504
Evil probabilities: #9=69%, #1=31%, #8=23%, #4=17%, #3=12%, #5=12%, #6=12%, #7=12%, #2=12%
  Generated 504 candidate scenarios
  156 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:42:02] Recommendation
Action: **REVEAL** #5
Reason: #5: 12% evil, 4.180 bits (19 outcomes)

### [21:42:37] Revealed #5 Bishop
Info: {'targets': [2, 6, 8], 'types': ['Villager', 'Minion', 'Outcast']}

#### [21:42:37] Solver Output
Scenarios: 91/504
Definite good: ['#3', '#7']
Evil probabilities: #9=66%, #8=38%, #1=22%, #5=21%, #6=21%, #2=19%, #4=13%
  Generated 504 candidate scenarios
  91 scenarios survived validation
    #3 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 8, 9]

#### [21:42:37] Recommendation
Action: **REVEAL** #8
Reason: #8: 38% evil, 3.680 bits (13 outcomes)

### [21:43:11] Revealed #8 Medium
Info: {'good_position': 3, 'good_role': 'Gemcrafter'}

#### [21:43:11] Solver Output
Scenarios: 39/504
Definite good: ['#3']
Evil probabilities: #9=67%, #5=28%, #6=28%, #1=26%, #2=18%, #4=15%, #8=15%, #7=3%
  Generated 504 candidate scenarios
  39 scenarios survived validation
    #3 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7, 8, 9]

#### [21:43:11] Recommendation
Action: **REVEAL** #6
Reason: #6: 28% evil, 3.560 bits (13 outcomes)

### [21:43:49] Revealed #6 Plague_Doctor
Info: {}

#### [21:43:49] Solver Output
Scenarios: 145/1120
Evil probabilities: #5=39%, #1=32%, #4=30%, #9=24%, #3=19%, #8=17%, #2=17%, #7=15%, #6=8%
  Generated 1120 candidate scenarios
  145 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:43:49] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#9']
Reason: Entropy 2.454 (adjusted 2.454) | timing x0.86

### [21:46:19] Ability used at #6

#### [21:46:22] Solver Output
Scenarios: 63/1120
Evil probabilities: #9=44%, #5=35%, #3=30%, #4=25%, #1=19%, #2=19%, #7=14%, #6=6%, #8=6%
  Generated 1120 candidate scenarios
  63 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:46:22] Recommendation
Action: **REVEAL** #2
Reason: #2: 19% evil, 4.112 bits (21 outcomes)

### [21:47:34] Revealed #2 Alchemist
Info: {'cured_count': 1}

#### [21:47:39] Solver Output
Scenarios: 34/1650
Definite good: ['#6']
Evil probabilities: #9=62%, #5=53%, #2=35%, #3=24%, #4=9%, #1=6%, #7=6%, #8=6%
  Generated 1650 candidate scenarios
  34 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [21:47:39] Recommendation
Action: **REVEAL** #3
Reason: #3: 24% evil, 3.416 bits (12 outcomes)

### [21:48:08] Revealed #3 Gemcrafter
Info: {}

#### [21:48:11] Solver Output
Scenarios: 45/2065
Definite good: ['#6']
Evil probabilities: #9=69%, #5=44%, #2=29%, #4=22%, #3=18%, #7=7%, #8=7%, #1=4%
  Generated 2065 candidate scenarios
  45 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [21:48:11] Recommendation
Action: **REVEAL** #1
Reason: #1: 4% evil, 3.458 bits (12 outcomes)

### [21:48:48] Revealed #3 Gemcrafter
Info: {'good_position': 7}

### [21:48:51] Revealed #1 Fortune_Teller
Info: {}

#### [21:48:55] Solver Output
Scenarios: 25/2691
Definite good: ['#3', '#6', '#8']
Evil probabilities: #9=84%, #5=60%, #4=36%, #1=8%, #2=8%, #7=4%
  Generated 2691 candidate scenarios
  25 scenarios survived validation
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 7, 9]

#### [21:48:55] Recommendation
Action: **USE_ABILITY** #1 (Fortune Teller) -> targets ['#2', '#4']
Reason: Entropy 0.999 (adjusted 0.979) | timing x1.00
WARNING: Corruption risk: 4%

### [21:49:45] Revealed #1 Fortune Teller
Info: {'targets': [2, 4], 'has_evil': True}

### [21:49:49] Ability used at #1

#### [21:49:52] Solver Output
Scenarios: 12/2691
Definite good: ['#3', '#6', '#8']
Evil probabilities: #4=75%, #9=75%, #1=17%, #5=17%, #2=8%, #7=8%
  Generated 2691 candidate scenarios
  12 scenarios survived validation
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 7, 9]

#### [21:49:52] Recommendation
Action: **REVEAL** #7
Reason: #7: 8% evil, 1.141 bits (3 outcomes)

### [21:50:15] Revealed #7 Wretch
Info: {}

#### [21:50:18] Solver Output
Scenarios: 8/2362
Definite good: ['#1', '#5', '#6']
Evil probabilities: #3=62%, #2=38%, #8=38%, #4=25%, #9=25%, #7=12%
  Generated 2362 candidate scenarios
  8 scenarios survived validation
    #1 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 7, 8, 9]

#### [21:50:18] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (62% evil Pooka, 25% good Gemcrafter (corrupted), 12% good Doppelganger (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 62%, but all reveal branches still lead to a forced win.

### [21:50:50] Executed #3 -> Pooka (EVIL)

#### [21:50:53] Solver Output
Scenarios: 5/242
Definite evil: ['#3']
Definite good: ['#1', '#4', '#5', '#6', '#7', '#9']
Evil probabilities: #8=60%, #2=40%
  Generated 242 candidate scenarios
  5 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 8]

#### [21:50:53] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (60% evil Chancellor, 40% good Medium (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [22:01:12] Executed #8 -> Chancellor (EVIL)

## [22:01:21] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Fingerprint entropy guided selective reveals. PD check + 2 Bishops + Alchemist + Gemcrafter + FT. Exec lookahead found guaranteed win line at 8 scenarios.


---

# New Game — 2026-03-11 22:03:49
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Bishop, Scout, Jester, Dreamer, Empress, Gemcrafter
- Outcasts: Drunk, Bombardier
- Minions: Chancellor
- Demons: Pooka

#### [22:04:08] Solver Output
Scenarios: 336/336
Evil probabilities: #1=25%, #2=25%, #3=25%, #4=25%, #5=25%, #6=25%, #7=25%, #8=25%
  Generated 336 candidate scenarios
  336 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:04:08] Recommendation
Action: **REVEAL** #8
Reason: #8: 25% evil, 6.942 bits (132 outcomes)

### [22:04:51] Revealed #8 Gemcrafter
Info: {'good_position': 4}

#### [22:04:57] Solver Output
Scenarios: 134/336
Evil probabilities: #2=31%, #3=31%, #5=31%, #6=31%, #4=25%, #1=20%, #7=20%, #8=9%
  Generated 336 candidate scenarios
  134 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:04:57] Recommendation
Action: **REVEAL** #3
Reason: #3: 31% evil, 6.065 bits (75 outcomes)

### [22:05:33] Revealed #3 Dreamer
Info: {}

#### [22:05:45] Solver Output
Scenarios: 134/336
Evil probabilities: #2=31%, #3=31%, #5=31%, #6=31%, #4=25%, #1=20%, #7=20%, #8=9%
  Generated 336 candidate scenarios
  134 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:05:45] Recommendation
Action: **REVEAL** #5
Reason: #5: 31% evil, 6.018 bits (74 outcomes)

### [22:06:23] Revealed #5 Bishop
Info: {'targets': [2, 3, 4], 'types': ['Outcast', 'Minion', 'Villager']}

#### [22:06:29] Solver Output
Scenarios: 144/366
Evil probabilities: #4=34%, #5=29%, #6=29%, #3=28%, #2=25%, #1=22%, #7=22%, #8=12%
  Generated 366 candidate scenarios
  144 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:06:29] Recommendation
Action: **REVEAL** #4
Reason: #4: 34% evil, 5.636 bits (55 outcomes)

### [22:07:03] Revealed #4 Jester
Info: {}

#### [22:07:09] Solver Output
Scenarios: 118/366
Evil probabilities: #4=42%, #5=32%, #6=29%, #3=26%, #1=23%, #7=23%, #8=14%, #2=11%
  Generated 366 candidate scenarios
  118 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:07:09] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#5']
Reason: Entropy 2.186 (adjusted 1.899) | timing x0.75
WARNING: Corruption risk: 26%

### [22:08:18] Revealed #3 Dreamer
Info: {'target': 5, 'evil_role': 'Chancellor'}

### [22:08:25] Ability used at #3

#### [22:08:32] Solver Output
Scenarios: 92/366
Evil probabilities: #4=51%, #6=32%, #3=28%, #1=25%, #7=25%, #8=18%, #5=13%, #2=8%
  Generated 366 candidate scenarios
  92 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:08:32] Recommendation
Action: **REVEAL** #1
Reason: #1: 25% evil, 4.956 bits (41 outcomes)

### [22:09:51] Revealed #1 Empress
Info: {'targets': [2, 5, 7]}

#### [22:09:57] Solver Output
Scenarios: 67/396
Evil probabilities: #4=54%, #1=31%, #6=27%, #7=27%, #3=21%, #8=18%, #5=12%, #2=10%
  Generated 396 candidate scenarios
  67 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:09:57] Recommendation
Action: **REVEAL** #6
Reason: #6: 27% evil, 4.566 bits (28 outcomes)

### [22:10:34] Revealed #6 Bombardier
Info: {}

#### [22:10:45] Solver Output
Scenarios: 58/346
Evil probabilities: #4=52%, #1=31%, #6=31%, #7=26%, #3=19%, #8=17%, #2=12%, #5=12%
  Generated 346 candidate scenarios
  58 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:10:45] Recommendation
Action: **REVEAL** #2
Reason: #2: 12% evil, 4.418 bits (28 outcomes)

### [22:11:19] Revealed #2 Druid
Info: {}

#### [22:11:26] Solver Output
Scenarios: 51/398
Evil probabilities: #4=45%, #6=41%, #1=39%, #7=24%, #2=14%, #8=14%, #3=12%, #5=12%
  Generated 398 candidate scenarios
  51 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:11:26] Recommendation
Action: **USE_ABILITY** #2 (Druid) -> targets ['#1', '#3', '#6']
Reason: Entropy 1.942 (adjusted 1.637) | timing x1.00
WARNING: Corruption risk: 31%

### [22:12:59] Revealed #2 Druid
Info: {'targets': [1, 3, 6], 'found_outcast': 'Drunk'}

### [22:13:06] Ability used at #2

#### [22:13:14] Solver Output
Scenarios: 17/398
Definite good: ['#3']
Evil probabilities: #6=65%, #4=35%, #1=24%, #7=24%, #8=24%, #2=18%, #5=12%
  Generated 398 candidate scenarios
  17 scenarios survived validation
    #3 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7, 8]

#### [22:13:14] Recommendation
Action: **USE_ABILITY** #4 (Jester) -> targets ['#1', '#2', '#8']
Reason: Expected posterior 8.3 scenarios (adjusted 8.8, info gain 0.957 bits) | timing x1.00
WARNING: Corruption risk: 12%

### [22:14:29] Revealed #4 Jester
Info: {'targets': [1, 2, 8], 'evil_count': 3}

### [22:14:37] Ability used at #4

#### [22:14:45] Solver Output
Scenarios: 8/398
Definite good: ['#1', '#3']
Evil probabilities: #4=75%, #8=50%, #6=25%, #7=25%, #2=12%, #5=12%
  Generated 398 candidate scenarios
  8 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 6, 7, 8]

#### [22:14:45] Recommendation
Action: **REVEAL** #7
Reason: #7: 25% evil, 2.156 bits (5 outcomes)

### [22:15:38] Revealed #7 Scout
Info: {'evil_role': 'Pooka', 'distance': 3}

#### [22:15:46] Solver Output
Scenarios: 4/454
Definite good: ['#1', '#3', '#5', '#7']
Evil probabilities: #4=75%, #8=75%, #2=25%, #6=25%
  Generated 454 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 4, 6, 8]

#### [22:15:46] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (75% evil Chancellor, 25% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [22:16:55] Executed #4 -> Chancellor (EVIL)

#### [22:17:03] Solver Output
Scenarios: 3/62
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7']
  Generated 62 candidate scenarios
  3 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #8 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [22:17:03] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 3 scenarios (roles: {'Pooka'})

### [22:18:06] Executed #8 -> Pooka (EVIL)

## [22:18:16] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. All 8 flipped. Jester lie (3 evils among 3 targets) was key constraint. Druid found Drunk. Exec lookahead guaranteed win at 4 scenarios.


---

# New Game — 2026-03-11 22:22:41
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Alchemist, Confessor, Architect, Bard, Judge
- Outcasts: Drunk
- Minions: Plague_Doctor, Chancellor, Minion
- Demons: Pooka

### [22:26:00] Revealed #1 Bard
Info: {'corruption_distance': 2}

### [22:26:01] Revealed #2 Plague_Doctor
Info: {}

### [22:26:46] Revealed #3 Judge
Info: {}

### [22:26:47] Revealed #4 Empress
Info: {'targets': [2, 3, 9]}

### [22:26:48] Revealed #5 Alchemist
Info: {'cured_count': 1}

### [22:26:49] Revealed #6 Alchemist
Info: {'cured_count': 1}

### [22:26:49] Revealed #7 Confessor
Info: {'dizzy': True}

### [22:26:50] Revealed #8 Alchemist
Info: {'cured_count': 2}

### [22:26:50] Revealed #9 Architect
Info: {'side': 'Left'}

#### [22:27:01] Solver Output
Scenarios: 308/13884
Definite good: ['#9']
Evil probabilities: #7=67%, #6=53%, #5=49%, #8=37%, #2=34%, #4=32%, #3=21%, #1=5%
  Generated 13884 candidate scenarios
  308 scenarios survived validation
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:27:01] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#8']
Reason: Entropy 2.394 (adjusted 2.394) | timing x1.00

### [22:28:48] Ability used at #2

#### [22:28:52] Solver Output
Scenarios: 148/13884
Definite good: ['#9']
Evil probabilities: #7=58%, #8=53%, #6=51%, #2=47%, #5=46%, #4=28%, #1=11%, #3=5%
  Generated 13884 candidate scenarios
  148 scenarios survived validation
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [22:28:52] Recommendation
Action: **USE_ABILITY** #3 (Judge) -> targets ['#1']
Reason: Expected posterior 95.0 scenarios (adjusted 108.5, info gain 0.448 bits) | timing x1.00
WARNING: Corruption risk: 28% -- corrupted Judge results are unreliable

### [22:29:45] Revealed #3 Judge
Info: {'target': 1, 'is_lying': True}

### [22:29:45] Ability used at #3

#### [22:29:51] Solver Output
Scenarios: 96/13884
Definite good: ['#3', '#9']
Evil probabilities: #8=73%, #7=52%, #6=46%, #4=44%, #5=42%, #2=27%, #1=17%
  Generated 13884 candidate scenarios
  96 scenarios survived validation
    #3 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7, 8]

#### [22:29:51] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (29% evil Chancellor, 22% evil Minion, 22% evil Plague_Doctor).
WARNING: Execution lookahead override -- immediate hit chance is 73%, but all reveal branches still lead to a forced win.

### [22:30:32] Executed #8 -> Minion (EVIL)

#### [22:30:36] Solver Output
Scenarios: 21/1098
Definite evil: ['#8']
Definite good: ['#2', '#3', '#9']
Evil probabilities: #4=62%, #6=52%, #7=38%, #5=29%, #1=19%
  Generated 1098 candidate scenarios
  21 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7]

#### [22:30:36] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (33% evil Pooka, 29% good Empress (corrupted), 19% evil Chancellor).
WARNING: Execution lookahead override -- immediate hit chance is 62%, but all reveal branches still lead to a forced win.

### [22:31:16] Executed #4 -> Chancellor (EVIL)

#### [22:31:20] Solver Output
Scenarios: 4/124
Definite evil: ['#4', '#6', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#9']
  Generated 124 candidate scenarios
  4 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [22:31:20] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 4 scenarios (roles: {'Pooka'})

### [22:32:07] Executed #6 -> Pooka (EVIL)

## [22:32:14] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect. PD check + Judge ability. 3 corrupted (Judge, Drunk, Confessor). Minion disguised as Alchemist x3.


---

# New Game — 2026-03-11 22:38:51
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Poet, Scout, Slayer, Architect, Enlightened
- Outcasts: Doppelganger, Bombardier
- Minions: Shaman
- Demons: Baa

### [22:40:52] Revealed #1 Architect
Info: {'side': 'Right'}

### [22:40:53] Revealed #2 Scout
Info: {'evil_role': 'Baa', 'distance': 2}

### [22:40:58] Revealed #3 Medium
Info: {'good_position': 2, 'good_role': 'Scout'}

### [22:40:59] Revealed #4 Slayer
Info: {}

### [22:41:00] Revealed #5 Poet
Info: {'targets': [3, 5], 'minion_role': 'Shaman', 'copied_role': 'Oracle'}

### [22:41:01] Revealed #6 Enlightened
Info: {'direction': 'CCW'}

### [22:41:02] Revealed #7 Bombardier
Info: {}

### [22:41:02] Revealed #8 Poet
Info: {'good_position': 5, 'copied_role': 'Gemcrafter'}

#### [22:41:09] Solver Output
Scenarios: 6/350
Definite evil: ['#2', '#3']
Definite good: ['#1', '#4', '#5', '#6', '#7', '#8']
  Generated 350 candidate scenarios
  6 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #3 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [22:41:09] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 6 scenarios (roles: {'Baa'})

### [22:41:54] Executed #2 -> Baa (EVIL)

### [22:42:39] Executed #3 -> Shaman (EVIL)

## [22:42:47] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect. Both evils identified from first solve (6 scenarios). Shaman duped Poet role. Baa added fake Outcast to deck.


---

# New Game — 2026-03-11 22:47:37
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Gemcrafter, Scout, Dreamer, Knitter, Confessor
- Outcasts: Bombardier
- Minions: Puppeteer
- Demons: Lilis


---

# New Game — 2026-03-11 22:48:15
Cards: 8, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Gemcrafter, Scout, Dreamer, Knitter, Confessor
- Outcasts: Bombardier
- Minions: Puppeteer
- Demons: Lilis

### [22:50:13] Revealed #1 Bombardier
Info: {}

### [22:50:14] Revealed #2 Gemcrafter
Info: {'good_position': 7}

### [22:50:15] Revealed #3 Dreamer
Info: {}

### [22:50:16] Revealed #4 Confessor
Info: {'dizzy': False}

### [22:51:09] Revealed #5 Knitter
Info: {'evil_pairs': 1}

### [22:51:10] Revealed #7 Scout
Info: {'evil_role': 'Lilis', 'distance': 2}

### [22:51:11] Revealed #8 Bombardier
Info: {}

#### [22:51:19] Solver Output
Scenarios: 2/78
Definite evil: ['#2', '#8']
Definite good: ['#1', '#4', '#5', '#6']
Evil probabilities: #3=50%, #7=50%
  Generated 78 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Puppet', 'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Lilis'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [3, 7]

#### [22:51:19] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Puppet', 'Lilis'})

### [22:52:17] Executed #2 -> Lilis (EVIL)

#### [22:52:25] Solver Output
Scenarios: 1/11
Definite evil: ['#2', '#7', '#8']
Definite good: ['#1', '#3', '#4', '#5', '#6']
  Generated 11 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #8 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [22:52:25] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [22:53:18] Executed #7 -> Puppet (EVIL)

### [22:54:07] Executed #8 -> Puppeteer (EVIL)

## [22:54:17] GAME OVER — WIN
Final HP: 8
Notes: 8HP win. Lilis night-killed #6 Empress. Puppeteer created Puppet from Scout at #7. 2 scenarios after all reveals, 1 after first exec.


---

# New Game — 2026-03-11 22:56:56
Cards: 7, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Fortune_Teller, Slayer, Witness, Lover
- Outcasts: Wretch, Plague_Doctor
- Minions: Puppeteer
- Demons: Baa

### [22:58:36] Revealed #1 Slayer
Info: {}

### [22:58:38] Revealed #2 Plague_Doctor
Info: {}

### [22:58:39] Revealed #3 Lover
Info: {'evil_adjacent': 2}

### [22:58:40] Revealed #4 Witness
Info: {'affected_position': 4}

### [22:58:41] Revealed #5 Scout
Info: {'evil_role': 'Puppet', 'distance': 1}

### [22:58:42] Revealed #6 Fortune_Teller
Info: {}

### [22:58:43] Revealed #7 Lover
Info: {'evil_adjacent': 1}

#### [22:58:50] Solver Output
Scenarios: 11/140
Definite good: ['#2']
Evil probabilities: #3=73%, #4=73%, #7=64%, #6=36%, #1=27%, #5=27%
  Generated 140 candidate scenarios
  11 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7]

#### [22:58:50] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.278 (adjusted 1.278) | timing x1.00

### [23:01:47] Ability used at #2

#### [23:01:49] Solver Output
Scenarios: 8/140
Definite evil: ['#3']
Definite good: ['#2']
Evil probabilities: #4=62%, #7=50%, #6=38%, #1=25%, #5=25%
  Generated 140 candidate scenarios
  8 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Baa'})
    #2 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7]

#### [23:01:49] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 8 scenarios (roles: {'Puppeteer', 'Baa'})

### [23:02:17] Executed #3 -> Puppeteer (EVIL)

#### [23:02:20] Solver Output
Scenarios: 4/18
Definite evil: ['#3', '#4']
Definite good: ['#2', '#5']
Evil probabilities: #7=50%, #1=25%, #6=25%
  Generated 18 candidate scenarios
  4 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #4 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 6, 7]

#### [23:02:20] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 4 scenarios (roles: {'Puppet'})

### [23:02:53] Executed #4 -> Puppet (EVIL)

#### [23:02:56] Solver Output
Scenarios: 0/13
  Generated 13 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #4 Witness: rejected 13/13 (100%)
    #5 Scout: rejected 6/13 (46%)
    #7 Lover: rejected 5/13 (38%)
    #3 Lover: rejected 1/13 (8%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #3 Lover: still 0
    WITHOUT #4 Witness: 4 scenarios survive  <-- SUSPECT
    WITHOUT #5 Scout: still 0
    WITHOUT #7 Lover: still 0

#### [23:02:56] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [23:05:46] Solver Output
Scenarios: 4/13
Definite evil: ['#3', '#4']
Definite good: ['#2', '#5']
Evil probabilities: #7=50%, #1=25%, #6=25%
  Generated 13 candidate scenarios
  4 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #4 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 6, 7]

#### [23:05:46] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#1', '#7']
Reason: Entropy 1.000 (adjusted 0.750) | follow-up bonus 0.562 | timing x1.00
WARNING: Corruption risk: 50%

### [23:06:36] Revealed #6 Fortune Teller
Info: {'targets': [1, 7], 'has_evil': False}

### [23:06:40] Ability used at #6

#### [23:06:43] Solver Output
Scenarios: 2/13
Definite evil: ['#3', '#4']
Definite good: ['#2', '#5', '#6']
Evil probabilities: #1=50%, #7=50%
  Generated 13 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #4 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [23:06:43] Recommendation
Action: **USE_ABILITY** #1 (Slayer) -> targets ['#7']
Reason: Target #7 is 50% evil (adjusted 0.50)

## [23:07:28] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect. Puppet_pos bug fix (executed Puppet not in scenario.puppet_position). FT+Slayer combo. Corrupted: #6 FT.


---

# New Game — 2026-03-11 23:09:11
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Poet, Lover, Empress, Enlightened, Scout, Bard
- Outcasts: Plague_Doctor
- Minions: Twin_Minion
- Demons: Lilis

### [23:11:16] Revealed #1 Lover
Info: {'evil_adjacent': 2}

### [23:11:20] Revealed #2 Enlightened
Info: {'direction': 'ccw'}

### [23:11:24] Revealed #3 Scout
Info: {'evil_role': 'Lilis', 'distance': 3}

### [23:11:29] Revealed #4 Bard
Info: {'corruption_distance': -1}

### [23:12:44] Revealed #6 Poet
Info: {'good_position': 4, 'good_role': 'evil', 'copied_role': 'Medium'}

### [23:12:55] Revealed #7 Empress
Info: {'targets': [3, 5, 6]}

### [23:13:00] Revealed #8 Plague_Doctor
Info: {}

#### [23:13:03] Solver Output
Scenarios: 0/224
  Generated 224 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #6 Poet: rejected 204/224 (91%)
    #1 Lover: rejected 130/224 (58%)
    #4 Bard: rejected 122/224 (54%)
    #2 Enlightened: rejected 116/224 (52%)
    #3 Scout: rejected 112/224 (50%)
    #7 Empress: rejected 102/224 (46%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Lover: still 0
    WITHOUT #2 Enlightened: still 0
    WITHOUT #3 Scout: still 0
    WITHOUT #4 Bard: still 0
    WITHOUT #6 Poet: 4 scenarios survive  <-- SUSPECT
    WITHOUT #7 Empress: 2 scenarios survive  <-- SUSPECT

#### [23:13:03] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [23:14:30] Revealed #6 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 4}

#### [23:14:33] Solver Output
Scenarios: 4/224
Definite evil: ['#1']
Definite good: ['#2', '#3', '#5', '#7', '#8']
Evil probabilities: #4=50%, #6=50%
  Generated 224 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Lilis', 'Twin_Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [23:14:33] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Lilis', 'Twin_Minion'})

### [23:15:08] Executed #1 -> Twin_Minion (EVIL)

#### [23:15:12] Solver Output
Scenarios: 2/31
Definite evil: ['#1']
Definite good: ['#2', '#3', '#5', '#7', '#8']
Evil probabilities: #4=50%, #6=50%
  Generated 31 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [23:15:12] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#4']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [23:16:04] Ability used at #8

#### [23:16:09] Solver Output
Scenarios: 1/31
Definite evil: ['#1', '#4']
Definite good: ['#2', '#3', '#5', '#6', '#7', '#8']
  Generated 31 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [23:16:09] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [23:16:55] Executed #4 -> GOOD (WRONG!)

#### [23:17:04] Solver Output
Scenarios: 0/26
  Generated 26 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #4 Bard: rejected 20/26 (77%)
    #6 Poet: rejected 17/26 (65%)
    #3 Scout: rejected 14/26 (54%)
    #2 Enlightened: rejected 12/26 (46%)
    #7 Empress: rejected 8/26 (31%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Lover: 1 scenarios survive  <-- SUSPECT
    WITHOUT #2 Enlightened: 1 scenarios survive  <-- SUSPECT
    WITHOUT #3 Scout: 1 scenarios survive  <-- SUSPECT
    WITHOUT #4 Bard: 2 scenarios survive  <-- SUSPECT
    WITHOUT #6 Poet: 1 scenarios survive  <-- SUSPECT
    WITHOUT #7 Empress: 1 scenarios survive  <-- SUSPECT

#### [23:17:04] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [23:18:19] Executed #6 -> Lilis (EVIL)

## [23:18:27] GAME OVER — WIN
Final HP: 3
Notes: 3HP. Wrong exec #4 (Bard). Solver 0-scenario bug: missed #6=Lilis scenario. PD corruption + Bard no-corrupted + Alchemist cure at night-killed #5 not modeled. Poet bounty_hunter format.


---

# New Game — 2026-03-11 23:24:01
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Gemcrafter, Slayer, Judge, Dreamer, Confessor, Knitter
- Outcasts: Plague_Doctor, Drunk
- Minions: Poisoner, Witch
- Demons: Baa

### [23:26:00] Revealed #1 blocked
Info: {}

### [23:26:11] Revealed #2 Confessor
Info: {'dizzy': True}

### [23:26:17] Revealed #3 Knitter
Info: {'evil_pairs': 2}

### [23:26:23] Revealed #4 Judge
Info: {}

### [23:26:28] Revealed #5 Knitter
Info: {'evil_pairs': 1}

### [23:26:33] Revealed #6 Gemcrafter
Info: {'good_position': 8}

### [23:26:39] Revealed #7 Dreamer
Info: {}

### [23:26:44] Revealed #8 Bishop
Info: {'targets': [5, 6, 8], 'types': ['Outcast', 'Minion', 'Villager']}

### [23:26:49] Revealed #9 Slayer
Info: {}

#### [23:26:54] Solver Output
Scenarios: 96/4542
Evil probabilities: #3=68%, #2=57%, #7=35%, #9=33%, #8=31%, #1=24%, #5=23%, #4=18%, #6=10%
  Generated 4542 candidate scenarios
  96 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [23:26:54] Recommendation
Action: **USE_ABILITY** #7 (Dreamer) -> targets ['#3']
Reason: Entropy 2.919 (adjusted 2.888) | timing x1.00
WARNING: Corruption risk: 2%

### [23:27:46] Revealed #7 Dreamer
Info: {'target': 3, 'evil_role': 'Witch'}

### [23:27:52] Ability used at #7

#### [23:28:00] Solver Output
Scenarios: 59/4542
Evil probabilities: #2=71%, #3=47%, #7=44%, #8=34%, #1=25%, #5=25%, #9=25%, #4=15%, #6=12%
  Generated 4542 candidate scenarios
  59 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [23:28:00] Recommendation
Action: **USE_ABILITY** #4 (Judge) -> targets ['#7']
Reason: Expected posterior 31.5 scenarios (adjusted 32.6, info gain 0.857 bits) | timing x1.00
WARNING: Corruption risk: 7% -- corrupted Judge results are unreliable

### [23:28:53] Revealed #4 Judge
Info: {'target': 7, 'is_lying': False}

### [23:28:59] Ability used at #4

#### [23:29:06] Solver Output
Scenarios: 32/4542
Evil probabilities: #2=72%, #3=53%, #8=41%, #9=41%, #5=31%, #1=19%, #7=19%, #6=16%, #4=9%
  Generated 4542 candidate scenarios
  32 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [23:29:06] Recommendation
Action: **USE_ABILITY** #9 (Slayer) -> targets ['#2']
Reason: Target #2 is 72% evil (adjusted 0.72)

### [23:30:05] Ability used at #9

#### [23:30:11] Solver Output
Scenarios: 19/4542
Evil probabilities: #3=68%, #9=68%, #2=53%, #5=32%, #1=26%, #4=16%, #7=16%, #8=16%, #6=5%
  Generated 4542 candidate scenarios
  19 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [23:30:11] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 68% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 68% confident (budget: 2 wrong execs)

### [23:30:55] Executed #3 -> Witch (EVIL)

### [23:32:05] Revealed #1 Slayer
Info: {}

#### [23:32:12] Solver Output
Scenarios: 11/602
Definite evil: ['#3']
Definite good: ['#4', '#7']
Evil probabilities: #9=73%, #2=45%, #1=27%, #5=27%, #8=18%, #6=9%
  Generated 602 candidate scenarios
  11 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 8, 9]

#### [23:32:12] Recommendation
Action: **USE_ABILITY** #1 (Slayer) -> targets ['#9']
Reason: Target #9 is 73% evil (adjusted 0.53)
WARNING: Corruption risk: 27% -- Slayer ability disabled if corrupted

### [23:33:13] Ability used at #1

#### [23:33:20] Solver Output
Scenarios: 8/602
Definite evil: ['#3']
Definite good: ['#4', '#7']
Evil probabilities: #9=62%, #1=38%, #2=38%, #5=25%, #8=25%, #6=12%
  Generated 602 candidate scenarios
  8 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 8, 9]

#### [23:33:20] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (50% good Drunk (corrupted), 25% evil Baa, 25% good Bishop (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 25%, but all reveal branches still lead to a forced win.

### [23:34:11] Executed #8 -> GOOD (WRONG!)

#### [23:34:25] Solver Output
Scenarios: 6/448
Definite evil: ['#3']
Definite good: ['#4', '#6', '#7', '#8']
Evil probabilities: #9=83%, #1=50%, #2=50%, #5=17%
  Generated 448 candidate scenarios
  6 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 9]

#### [23:34:25] Recommendation
Action: **EXECUTE** #9
Reason: No reveals available. #9 is 83% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 83% confident (budget: 1 wrong execs)

### [23:35:05] Executed #9 -> Poisoner (EVIL)

#### [23:35:12] Solver Output
Scenarios: 3/77
Definite evil: ['#3', '#9']
Definite good: ['#4', '#5', '#6', '#7', '#8']
Evil probabilities: #2=67%, #1=33%
  Generated 77 candidate scenarios
  3 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #9 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2]

#### [23:35:12] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (67% evil Baa, 33% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [23:35:56] Executed #2 -> Baa (EVIL)

## [23:36:05] GAME OVER — WIN
Final HP: 5
Notes: 5HP. Wrong exec #8 (Bishop corrupted). Witch blocked #1 (not #9). Dreamer confirmed #3=Witch. Drunk-as-Slayer at #1 (both Slayer shots failed). Corrupted: #1 Drunk, #8 Bishop.


---

# New Game — 2026-03-11 23:38:13
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Gemcrafter, Bishop, Architect, Baker, Knitter, Bard, Slayer
- Outcasts: Bombardier, Drunk, Wretch
- Minions: Shaman
- Demons: Baa

### [23:41:30] Revealed #2 Bishop
Info: {'targets': [1, 4, 5], 'types': ['Outcast', 'Minion', 'Villager']}

### [23:41:34] Revealed #3 Slayer
Info: {}

### [23:41:34] Revealed #4 Slayer
Info: {}

### [23:41:38] Revealed #5 Baker
Info: {'original_role': 'original'}

### [23:41:38] Revealed #6 Bombardier
Info: {}

### [23:41:42] Revealed #7 Knitter
Info: {'evil_pairs': 1}

### [23:41:42] Revealed #8 Architect
Info: {'side': 'Equal'}

#### [23:41:45] Solver Output
Scenarios: 40/350
Evil probabilities: #2=45%, #7=42%, #5=30%, #1=20%, #3=20%, #8=18%, #4=15%, #6=10%
  Generated 350 candidate scenarios
  40 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [23:41:45] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#2']
Reason: Target #2 is 45% evil (adjusted 0.43)
WARNING: Corruption risk: 5% -- Slayer ability disabled if corrupted

### [23:43:24] Ability used at #3

#### [23:43:28] Solver Output
Scenarios: 7/43
Definite evil: ['#2']
Definite good: ['#3', '#4']
Evil probabilities: #7=43%, #1=14%, #5=14%, #6=14%, #8=14%
  Generated 43 candidate scenarios
  7 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 5, 6, 7, 8]

#### [23:43:28] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#7']
Reason: Target #7 is 43% evil (adjusted 0.37)
WARNING: Corruption risk: 14% -- Slayer ability disabled if corrupted

### [23:44:20] Ability used at #4

#### [23:44:24] Solver Output
Scenarios: 5/43
Definite evil: ['#2']
Definite good: ['#3', '#4']
Evil probabilities: #1=20%, #5=20%, #6=20%, #7=20%, #8=20%
  Generated 43 candidate scenarios
  5 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 5, 6, 7, 8]

#### [23:44:24] Recommendation
Action: **EXECUTE** #1
Reason: No reveals available. #1 is 20% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 20% confident (budget: 2 wrong execs)
WARNING: Low confidence (20%) -- consider gathering more info

### [23:47:24] Executed #1 -> GOOD (WRONG!)

#### [23:47:43] Solver Output
Scenarios: 4/37
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4']
Evil probabilities: #5=25%, #6=25%, #7=25%, #8=25%
  Generated 37 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [5, 6, 7, 8]

#### [23:47:43] Recommendation
Action: **EXECUTE** #5
Reason: No reveals available. #5 is 25% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 25% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #5 (25%) despite low confidence — Bombardier candidate(s) [6] risk instant game loss if executed first.

### [23:49:09] Executed #5 -> GOOD (WRONG!)

## [23:49:19] GAME OVER — LOSS
Final HP: 0
Notes: Loss: Shaman disguised as Bombardier. Bombardier protection blocked execution. 4 scenarios 25% each, gambled on #5 wrong.


---

# New Game — 2026-03-11 23:54:45
Cards: 8, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Fortune_Teller, Enlightened, Hunter, Lover, Baker, Scout
- Outcasts: Doppelganger
- Minions: Puppeteer
- Demons: Lilis

### [23:56:03] Revealed #1 Enlightened
Info: {'direction': 'CCW'}

### [23:56:08] Revealed #2 Scout
Info: {'evil_role': 'Puppeteer', 'distance': 1}

### [23:56:14] Revealed #3 Baker
Info: {'original_role': 'Lover'}

### [23:56:15] Revealed #4 Baker
Info: {'original_role': 'original'}

### [23:57:07] Revealed #5 Hunter
Info: {'distance': 2}

### [23:57:08] Revealed #7 Lover
Info: {'evil_adjacent': 0}

### [23:57:12] Revealed #8 Baker
Info: {'original_role': 'original'}

#### [23:57:18] Solver Output
Scenarios: 15/480
Definite evil: ['#3']
Definite good: ['#4', '#5', '#6']
Evil probabilities: #1=67%, #2=67%, #7=33%, #8=33%
  Generated 480 candidate scenarios
  15 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Lilis'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 7, 8]

#### [23:57:18] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 15 scenarios (roles: {'Puppeteer', 'Lilis'})

### [23:58:04] Executed #3 -> Puppeteer (EVIL)

#### [23:58:09] Solver Output
Scenarios: 5/60
Definite evil: ['#1', '#2', '#3']
Definite good: ['#4', '#5', '#6', '#7', '#8']
  Generated 60 candidate scenarios
  5 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #2 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #3 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [23:58:09] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 5 scenarios (roles: {'Lilis'})

### [23:58:47] Executed #1 -> Lilis (EVIL)

### [23:59:24] Executed #2 -> Puppet (EVIL)

## [23:59:32] GAME OVER — WIN
Final HP: 8
Notes: 8HP, Lilis+Puppeteer+Puppet. Solver found all 3 definite evil. Night kill #6.


---

# New Game — 2026-03-12 00:01:58
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Confessor, Judge, Bishop, Slayer, Alchemist, Architect
- Outcasts: Plague_Doctor, Bombardier
- Minions: Puppeteer
- Demons: Lilis

### [00:06:13] Revealed #1 Confessor
Info: {'dizzy': False}

### [00:06:20] Revealed #2 Judge
Info: {'target': 3, 'is_lying': False}

### [00:06:27] Revealed #3 Oracle
Info: {'targets': [6, 9], 'minion_role': 'Puppeteer'}

### [00:06:34] Revealed #4 Bombardier
Info: {}

### [00:06:34] Revealed #5 Architect
Info: {'side': 'Right'}

### [00:06:40] Revealed #6 Slayer
Info: {}

### [00:06:41] Revealed #7 Alchemist
Info: {'cured_count': 1}

### [00:06:48] Revealed #8 Bishop
Info: {'targets': [3, 6, 9], 'types': ['Outcast', 'Villager', 'Minion']}

### [00:06:55] Ability used at #2

#### [00:07:01] Solver Output
Scenarios: 0/184
  Generated 184 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #7 Alchemist: rejected 142/184 (77%)
    #3 Oracle: rejected 124/184 (67%)
    #5 Architect: rejected 94/184 (51%)
    #8 Bishop: rejected 77/184 (42%)
    #2 Judge: rejected 72/184 (39%)
    #1 Confessor: rejected 56/184 (30%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Confessor: still 0
    WITHOUT #2 Judge: 1 scenarios survive  <-- SUSPECT
    WITHOUT #3 Oracle: 3 scenarios survive  <-- SUSPECT
    WITHOUT #5 Architect: 2 scenarios survive  <-- SUSPECT
    WITHOUT #7 Alchemist: 4 scenarios survive  <-- SUSPECT
    WITHOUT #8 Bishop: still 0

#### [00:07:01] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [00:15:32] Solver Output
Scenarios: 11/764
Definite good: ['#1', '#4', '#8', '#9']
Evil probabilities: #2=73%, #3=73%, #6=27%, #5=18%, #7=9%
  Generated 764 candidate scenarios
  11 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 5, 6, 7]

#### [00:15:32] Recommendation
Action: **USE_ABILITY** #6 (Slayer) -> targets ['#2']
Reason: Target #2 is 73% evil (adjusted 0.73)

### [00:16:45] Ability used at #6

#### [00:16:45] Solver Output
Scenarios: 8/88
Definite evil: ['#2', '#3']
Definite good: ['#1', '#5', '#6', '#7', '#8', '#9']
Evil probabilities: #4=50%
  Generated 88 candidate scenarios
  8 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #3 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4]

#### [00:16:45] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 8 scenarios (roles: {'Puppeteer'})

### [00:18:56] Executed #3 -> Puppeteer (EVIL)

## [00:19:01] GAME OVER — WIN
Final HP: 6
Notes: 6HP. Hidden PD at #9 (night-killed) corrupted Alchemist. Slayer killed Lilis. Judge ability on #3 confirmed truth. Solver fix: hidden PD corruption for night-killed positions.


---

# New Game — 2026-03-12 00:21:41
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Hunter, Slayer, Scout, Knight, Gemcrafter, Empress
- Outcasts: Drunk
- Minions: Poisoner
- Demons: Lilis

### [00:22:49] Revealed #1 Knight
Info: {}

### [00:22:53] Revealed #2 Hunter
Info: {'distance': 1}

### [00:22:56] Revealed #3 Slayer
Info: {}

### [00:22:59] Revealed #4 Gemcrafter
Info: {'good_position': 6}

#### [00:23:04] Solver Output
Scenarios: 86/536
Definite good: ['#5']
Evil probabilities: #2=48%, #7=41%, #8=37%, #3=27%, #1=22%, #6=21%, #4=5%
  Generated 536 candidate scenarios
  86 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8]

#### [00:23:04] Recommendation
Action: **REVEAL** #8
Reason: #8: 37% evil, 4.247 bits (26 outcomes)

### [00:23:59] Revealed #6 Empress
Info: {'targets': [2, 3, 5]}

### [00:24:04] Revealed #7 Knight
Info: {}

### [00:24:13] Revealed #8 Scout
Info: {'evil_role': 'Lilis', 'distance': 2}

#### [00:24:18] Solver Output
Scenarios: 25/566
Definite good: ['#5']
Evil probabilities: #7=52%, #3=36%, #8=32%, #2=24%, #1=20%, #6=20%, #4=16%
  Generated 566 candidate scenarios
  25 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8]

#### [00:24:18] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#8']
Reason: Target #8 is 32% evil (adjusted 0.27)
WARNING: Corruption risk: 16% -- Slayer ability disabled if corrupted

### [00:26:12] Ability used at #3

#### [00:26:18] Solver Output
Scenarios: 23/566
Definite good: ['#5']
Evil probabilities: #7=52%, #3=39%, #2=26%, #8=26%, #6=22%, #1=17%, #4=17%
  Generated 566 candidate scenarios
  23 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8]

#### [00:26:18] Recommendation
Action: **ERROR** #7
Reason: #7 is 52% likely evil but budget=1 requires >=80% confidence (HP=8, cost=5).
WARNING: Probabilistic execution -- 52% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 52% < 80% threshold. Consider manual override if you have extra information.

### [00:27:24] Executed #7 -> Poisoner (EVIL)

#### [00:27:29] Solver Output
Scenarios: 10/72
Definite evil: ['#7']
Definite good: ['#4', '#5', '#6', '#8']
Evil probabilities: #2=40%, #1=30%, #3=30%
  Generated 72 candidate scenarios
  10 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3]

#### [00:27:29] Recommendation
Action: **ERROR** #2
Reason: #2 is 40% likely evil but budget=1 requires >=80% confidence (HP=8, cost=5).
WARNING: Probabilistic execution -- 40% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 40% < 80% threshold. Consider manual override if you have extra information.

#### [00:29:22] Knight #1 execution blocked by immunity - confirmed good


#### [00:29:36] Solver Output
Scenarios: 7/60
Definite evil: ['#7']
Definite good: ['#1', '#4', '#5', '#6', '#8']
Evil probabilities: #2=57%, #3=43%
  Generated 60 candidate scenarios
  7 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3]

#### [00:29:36] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (57% evil Lilis, 43% good Hunter).
WARNING: Execution lookahead override -- immediate hit chance is 57%, but all reveal branches still lead to a forced win.

### [00:30:20] Executed #2 -> Lilis (EVIL)

## [00:30:27] GAME OVER — WIN
Final HP: 8
Notes: 8HP. Knight trick confirmed #1 good (free exec check). Lilis night-killed #5 (Baker). Scout #8 corrupted by Poisoner. Drunk #3 disguised as Slayer. Guaranteed win via lookahead.


---

# New Game — 2026-03-13 19:09:58
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Knight, Dreamer, Slayer, Empress, Scout
- Outcasts: Bombardier, Plague_Doctor
- Minions: Minion
- Demons: Pooka

### [19:11:21] Revealed #1 Bombardier
Info: {}

### [19:11:25] Revealed #2 Slayer
Info: {}

### [19:11:29] Revealed #3 Dreamer
Info: {}

### [19:11:32] Revealed #4 Alchemist
Info: {'cured_count': 2}

### [19:11:35] Revealed #5 Plague_Doctor
Info: {}

### [19:11:39] Revealed #6 Empress
Info: {'targets': [1, 3, 5]}

### [19:11:42] Revealed #7 Knight
Info: {}

### [19:11:54] Revealed #8 Scout
Info: {'evil_role': 'Minion', 'distance': 3}

#### [19:11:58] Solver Output
Scenarios: 14/236
Definite evil: ['#1']
Definite good: ['#3', '#5', '#6']
Evil probabilities: #7=36%, #4=29%, #8=29%, #2=7%
  Generated 236 candidate scenarios
  14 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka', 'Minion'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [2, 4, 7, 8]

#### [19:11:58] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 14 scenarios (roles: {'Pooka', 'Minion'})

### [19:12:33] Executed #1 -> Minion (EVIL)

#### [19:12:36] Solver Output
Scenarios: 7/37
Definite evil: ['#1']
Definite good: ['#2', '#3', '#5', '#6']
Evil probabilities: #4=57%, #7=29%, #8=14%
  Generated 37 candidate scenarios
  7 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [4, 7, 8]

#### [19:12:36] Recommendation
Action: **USE_ABILITY** #5 (Plague Doctor) -> targets ['#7']
Reason: Entropy 1.149 (adjusted 1.149) | timing x1.00

### [19:13:20] Ability used at #5

#### [19:13:24] Solver Output
Scenarios: 2/37
Definite evil: ['#1']
Definite good: ['#2', '#3', '#5', '#6', '#7']
Evil probabilities: #4=50%, #8=50%
  Generated 37 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [4, 8]

#### [19:13:24] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#2']
Reason: Entropy 1.000 (adjusted 0.750) | timing x1.00
WARNING: Corruption risk: 50%

### [19:14:24] Revealed #3 Dreamer
Info: {'target': 2, 'evil_role': 'Pooka'}

### [19:14:27] Ability used at #3

#### [19:14:31] Solver Output
Scenarios: 2/37
Definite evil: ['#1']
Definite good: ['#2', '#3', '#5', '#6', '#7']
Evil probabilities: #4=50%, #8=50%
  Generated 37 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [4, 8]

#### [19:14:31] Recommendation
Action: **USE_ABILITY** #2 (Slayer) -> targets ['#4']
Reason: Target #4 is 50% evil (adjusted 0.50)

### [19:15:07] Revealed #2 Slayer Result
Info: {}

## [19:15:13] GAME OVER — WIN
Final HP: 10
Notes: perfect 10HP, PD+Slayer combo


---

# New Game — 2026-03-13 19:17:35
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Druid, Hunter, Witness, Knitter, Lover
- Outcasts: Plague_Doctor, Drunk
- Minions: Witch
- Demons: Baa

### [19:19:32] Revealed #1 Knitter
Info: {'evil_pairs': 0}

### [19:19:38] Revealed #2 Witness
Info: {'affected_position': 1}

### [19:19:42] Revealed #3 Bard
Info: {'corruption_distance': 3}

### [19:19:46] Revealed #4 Druid
Info: {}

### [19:19:50] Revealed #5 Lover
Info: {'evil_adjacent': 0}

### [19:19:54] Revealed #6 Hunter
Info: {'distance': 3}

#### [19:20:25] Solver Output
Scenarios: 26/972
Definite evil: ['#2']
Definite good: ['#4', '#5', '#6']
Evil probabilities: #1=46%, #3=46%, #7=8%
  Generated 972 candidate scenarios
  26 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa', 'Witch'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 3, 7]

#### [19:20:25] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 26 scenarios (roles: {'Baa', 'Witch'})

### [19:21:05] Executed #2 -> Baa (EVIL)

#### [19:21:10] Solver Output
Scenarios: 13/156
Definite evil: ['#2']
Definite good: ['#4', '#5', '#6']
Evil probabilities: #1=46%, #3=46%, #7=8%
  Generated 156 candidate scenarios
  13 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 3, 7]

#### [19:21:10] Recommendation
Action: **USE_ABILITY** #4 (Druid) -> targets ['#1', '#3', '#5']
Reason: Entropy 0.619 (adjusted 0.572) | timing x1.00
WARNING: Corruption risk: 15%

### [19:22:07] Revealed #4 Druid
Info: {'targets': [1, 3, 5], 'found_outcast': None}

### [19:22:12] Ability used at #4

#### [19:22:16] Solver Output
Scenarios: 9/156
Definite evil: ['#2']
Definite good: ['#4', '#5', '#6']
Evil probabilities: #1=56%, #3=33%, #7=11%
  Generated 156 candidate scenarios
  9 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 3, 7]

#### [19:22:16] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (56% evil Witch, 22% good Knitter (corrupted), 11% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 56%, but all reveal branches still lead to a forced win.

### [19:22:59] Executed #1 -> Witch (EVIL)

## [19:23:06] GAME OVER — WIN
Final HP: 10
Notes: perfect 10HP, Druid found no outcasts narrowed Witch to #1


---

# New Game — 2026-03-13 19:25:31
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Dreamer, Confessor, Medium, Bishop, Fortune_Teller, Druid, Knight
- Outcasts: Plague_Doctor, Doppelganger
- Minions: Chancellor, Shaman
- Demons: Lilis

### [19:27:56] Revealed #1 Fortune_Teller
Info: {}

### [19:28:01] Revealed #2 Druid
Info: {}

### [19:28:14] Revealed #3 Confessor
Info: {'dizzy': True}

### [19:28:20] Revealed #4 Medium
Info: {'good_position': 8, 'good_role': 'Confessor'}

### [19:29:32] Revealed #6 Fortune_Teller
Info: {}

### [19:29:38] Revealed #7 Bishop
Info: {'targets': [3, 5, 6], 'types': ['Villager', 'Minion', 'Outcast']}

### [19:29:42] Revealed #8 Confessor
Info: {'dizzy': True}

### [19:29:47] Revealed #9 Plague_Doctor
Info: {}

#### [19:29:54] Solver Output
Scenarios: 475/10292
Definite good: ['#5', '#9']
Evil probabilities: #3=88%, #8=63%, #4=40%, #7=35%, #6=29%, #1=25%, #2=20%
  Generated 10292 candidate scenarios
  475 scenarios survived validation
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8]

#### [19:29:54] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#8']
Reason: Entropy 1.517 (adjusted 1.517) | timing x1.00

### [19:30:43] Ability used at #9

#### [19:30:50] Solver Output
Scenarios: 300/10292
Definite evil: ['#8']
Definite good: ['#5', '#9']
Evil probabilities: #3=81%, #4=56%, #7=22%, #6=20%, #1=13%, #2=9%
  Generated 10292 candidate scenarios
  300 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Shaman', 'Lilis', 'Chancellor'})
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7]

#### [19:30:50] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 300 scenarios (roles: {'Shaman', 'Lilis', 'Chancellor'})

### [19:31:34] Executed #8 -> Chancellor (EVIL)

#### [19:31:40] Solver Output
Scenarios: 90/984
Definite evil: ['#8']
Definite good: ['#5', '#9']
Evil probabilities: #3=79%, #4=61%, #7=21%, #6=19%, #1=10%, #2=10%
  Generated 984 candidate scenarios
  90 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7]

#### [19:31:40] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#4', '#9']
Reason: Entropy 1.000 (adjusted 0.950) | timing x1.00
WARNING: Corruption risk: 10%

### [19:32:26] Revealed #6 Fortune Teller
Info: {'targets': [4, 9], 'has_evil': True}

### [19:32:32] Ability used at #6

#### [19:32:38] Solver Output
Scenarios: 45/984
Definite evil: ['#8']
Definite good: ['#1', '#2', '#5', '#9']
Evil probabilities: #4=82%, #3=78%, #7=22%, #6=18%
  Generated 984 candidate scenarios
  45 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 4, 6, 7]

#### [19:32:38] Recommendation
Action: **USE_ABILITY** #1 (Fortune Teller) -> targets ['#2', '#3']
Reason: Entropy 0.982 (adjusted 0.884) | timing x1.00
WARNING: Corruption risk: 20%

### [19:33:27] Revealed #1 Fortune Teller
Info: {'targets': [2, 3], 'has_evil': False}

### [19:33:33] Ability used at #1

#### [19:33:39] Solver Output
Scenarios: 19/984
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#5', '#6', '#9']
Evil probabilities: #7=53%, #3=47%
  Generated 984 candidate scenarios
  19 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Shaman', 'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 7]

#### [19:33:39] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 19 scenarios (roles: {'Shaman', 'Lilis'})

### [19:34:27] Executed #4 -> Lilis (EVIL)

#### [19:34:32] Solver Output
Scenarios: 10/131
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#5', '#6', '#9']
Evil probabilities: #3=50%, #7=50%
  Generated 131 candidate scenarios
  10 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 7]

#### [19:34:32] Recommendation
Action: **USE_ABILITY** #2 (Druid) -> targets ['#1', '#3', '#6']
Reason: Entropy 0.000 (adjusted 0.000) | timing x1.00

### [19:36:02] Revealed #2 Druid
Info: {'targets': [1, 3, 6], 'found_outcast': None}

### [19:36:08] Ability used at #2

#### [19:36:14] Solver Output
Scenarios: 5/131
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#5', '#6', '#9']
Evil probabilities: #3=60%, #7=40%
  Generated 131 candidate scenarios
  5 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 7]

#### [19:36:14] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (60% evil Shaman, 40% good Confessor (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [19:37:05] Executed #3 -> GOOD (WRONG!)

#### [19:37:17] Solver Output
Scenarios: 2/111
Definite evil: ['#4', '#7', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#9']
  Generated 111 candidate scenarios
  2 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [19:37:17] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Shaman'})

### [19:38:07] Executed #7 -> Shaman (EVIL)

## [19:38:15] GAME OVER — WIN
Final HP: 1
Notes: 1HP survive, Lilis game, wrong exec on #3 corrupted Confessor


---

# New Game — 2026-03-13 19:41:17
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Druid, Poet, Fortune_Teller, Scout, Judge
- Outcasts: Doppelganger
- Minions: Minion
- Demons: Lilis

### [19:42:44] Revealed #1 Poet
Info: {'side': 'right', 'copied_role': 'Architect'}

### [19:42:51] Revealed #2 Scout
Info: {'evil_role': 'Minion', 'distance': 3}

### [19:42:58] Revealed #3 Fortune_Teller
Info: {}

### [19:42:58] Revealed #4 Druid
Info: {}

### [19:43:45] Revealed #6 Judge
Info: {}

### [19:43:45] Revealed #7 Confessor
Info: {'dizzy': False}

### [19:43:45] Revealed #8 Judge
Info: {}

#### [19:43:52] Solver Output
Scenarios: 60/336
Definite good: ['#5', '#7']
Evil probabilities: #2=60%, #3=40%, #8=40%, #1=20%, #4=20%, #6=20%
  Generated 336 candidate scenarios
  60 scenarios survived validation
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 8]

#### [19:43:52] Recommendation
Action: **USE_ABILITY** #3 (Fortune Teller) -> targets ['#2', '#8']
Reason: Entropy 0.971 (adjusted 0.971) | follow-up bonus 0.192 | timing x1.00

### [19:44:45] Revealed #3 Fortune Teller
Info: {'targets': [2, 8], 'has_evil': False}

### [19:44:45] Ability used at #3

#### [19:44:52] Solver Output
Scenarios: 36/336
Definite good: ['#4', '#5', '#7']
Evil probabilities: #3=67%, #1=33%, #2=33%, #6=33%, #8=33%
  Generated 336 candidate scenarios
  36 scenarios survived validation
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 6, 8]

#### [19:44:52] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#2']
Reason: Expected posterior 20.0 scenarios (adjusted 20.0, info gain 0.848 bits) | timing x1.00

### [19:45:38] Revealed #6 Judge
Info: {'target': 2, 'is_lying': False}

### [19:45:38] Ability used at #6

#### [19:45:44] Solver Output
Scenarios: 12/336
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7']
  Generated 336 candidate scenarios
  12 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [19:45:44] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 12 scenarios (roles: {'Minion', 'Lilis'})

### [19:46:36] Executed #3 -> Lilis (EVIL)

### [19:47:21] Executed #8 -> Minion (EVIL)

## [19:47:30] GAME OVER — WIN
Final HP: 8
Notes: 8HP Lilis game, FT+Judge combo nailed both evils


---

# New Game — 2026-03-13 19:51:40
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Dreamer, Judge, Enlightened, Poet, Hunter, Scout, Knitter, Gemcrafter, Baker
- Outcasts: Drunk
- Minions: Poisoner, Puppeteer
- Demons: Lilis

### [19:53:25] Revealed #1 Judge
Info: {}

### [19:53:25] Revealed #2 Enlightened
Info: {'direction': 'CW'}

### [19:53:26] Revealed #3 Knitter
Info: {'evil_pairs': 2}

### [19:53:26] Revealed #4 Poet
Info: {'targets': [8, 9, 10], 'types': ['Outcast', 'Minion', 'Villager'], 'copied_role': 'Bishop'}

### [19:54:31] Revealed #6 Scout
Info: {'evil_role': 'Puppeteer', 'distance': 1}

### [19:54:31] Revealed #7 Gemcrafter
Info: {'good_position': 6}

### [19:54:31] Revealed #8 Judge
Info: {}

### [19:54:32] Revealed #9 Baker
Info: {'original_role': 'original'}

#### [19:54:39] Solver Output
Scenarios: 144/10080
Definite good: ['#5', '#10']
Evil probabilities: #9=69%, #1=66%, #8=54%, #7=53%, #3=51%, #2=39%, #4=36%, #6=32%
  Generated 10080 candidate scenarios
  144 scenarios survived validation
    #5 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:54:39] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#3']
Reason: Expected posterior 81.0 scenarios (adjusted 85.8, info gain 0.747 bits) | timing x1.00
WARNING: Corruption risk: 12% -- corrupted Judge results are unreliable

### [19:55:28] Revealed #1 Judge
Info: {'target': 3, 'is_lying': True}

### [19:55:28] Ability used at #1

#### [19:55:36] Solver Output
Scenarios: 87/10080
Definite good: ['#5', '#10']
Evil probabilities: #9=72%, #7=61%, #2=60%, #1=53%, #6=45%, #3=38%, #8=38%, #4=33%
  Generated 10080 candidate scenarios
  87 scenarios survived validation
    #5 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:55:36] Recommendation
Action: **USE_ABILITY** #8 (Judge) -> targets ['#6']
Reason: Expected posterior 54.1 scenarios (adjusted 60.6, info gain 0.522 bits) | timing x1.00
WARNING: Corruption risk: 24% -- corrupted Judge results are unreliable

### [19:56:20] Revealed #8 Judge
Info: {'target': 6, 'is_lying': False}

### [19:56:21] Ability used at #8

#### [19:56:32] Solver Output
Scenarios: 56/10080
Definite good: ['#5', '#10']
Evil probabilities: #9=88%, #2=61%, #6=55%, #7=55%, #3=54%, #1=46%, #4=38%, #8=4%
  Generated 10080 candidate scenarios
  56 scenarios survived validation
    #5 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [19:56:32] Recommendation
Action: **EXECUTE** #9
Reason: No reveals available. #9 is 88% likely evil (HP=6, budget=1 wrong execs)
WARNING: Probabilistic execution -- 88% confident (budget: 1 wrong execs)

### [19:57:25] Executed #9 -> GOOD (WRONG!)

#### [19:57:33] Solver Output
Scenarios: 7/6088
Definite evil: ['#1', '#2']
Definite good: ['#5', '#8', '#9', '#10']
Evil probabilities: #3=86%, #4=86%, #6=14%, #7=14%
  Generated 6088 candidate scenarios
  7 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Poisoner', 'Puppet'})
    #2 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Lilis'})
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [3, 4, 6, 7]

#### [19:57:33] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 7 scenarios (roles: {'Poisoner', 'Puppet'})

### [19:58:23] Executed #1 -> Poisoner (EVIL)

#### [19:58:31] Solver Output
Scenarios: 1/632
Definite evil: ['#1', '#2', '#6', '#7']
Definite good: ['#3', '#4', '#5', '#8', '#9', '#10']
  Generated 632 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [19:58:31] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [19:59:21] Executed #2 -> Lilis (EVIL)

### [20:00:09] Executed #6 -> Puppet (EVIL)

### [20:01:01] Executed #7 -> Puppeteer (EVIL)

## [20:01:16] GAME OVER — WIN
Final HP: 1
Notes: 1HP survive, wrong exec Baker #9, Puppeteer created Puppet from Scout


---

# New Game — 2026-03-13 20:04:20
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Dreamer, Judge, Alchemist, Jester, Enlightened, Confessor
- Outcasts: Drunk, Doppelganger
- Minions: Minion
- Demons: Lilis

### [20:05:19] Revealed #1 Jester
Info: {}

### [20:05:19] Revealed #2 Judge
Info: {}

### [20:05:19] Revealed #3 Alchemist
Info: {'cured_count': 0}

### [20:05:19] Revealed #4 Dreamer
Info: {}

### [20:06:16] Revealed #5 Enlightened
Info: {'direction': 'CCW'}

### [20:06:16] Revealed #6 Enlightened
Info: {'direction': 'CCW'}

### [20:06:16] Revealed #7 Dreamer
Info: {}

### [20:06:17] Revealed #8 Confessor
Info: {'dizzy': False}

#### [20:06:24] Solver Output
Scenarios: 132/3024
Definite good: ['#3', '#8', '#9']
Evil probabilities: #4=64%, #1=45%, #2=36%, #7=27%, #6=18%, #5=9%
  Generated 3024 candidate scenarios
  132 scenarios survived validation
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7]

#### [20:06:24] Recommendation
Action: **USE_ABILITY** #4 (Dreamer) -> targets ['#1']
Reason: Entropy 2.391 (adjusted 2.391) | timing x1.00

### [20:07:16] Revealed #4 Dreamer
Info: {'target': 1, 'evil_role': 'Minion'}

### [20:07:16] Ability used at #4

#### [20:07:23] Solver Output
Scenarios: 102/3024
Definite good: ['#3', '#8', '#9']
Evil probabilities: #4=65%, #2=41%, #7=35%, #1=29%, #6=18%, #5=12%
  Generated 3024 candidate scenarios
  102 scenarios survived validation
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7]

#### [20:07:23] Recommendation
Action: **USE_ABILITY** #7 (Dreamer) -> targets ['#4']
Reason: Entropy 2.514 (adjusted 2.292) | timing x1.00
WARNING: Corruption risk: 18%

### [20:08:06] Revealed #7 Dreamer
Info: {'target': 4, 'evil_role': 'Lilis'}

### [20:08:07] Ability used at #7

#### [20:08:15] Solver Output
Scenarios: 66/3024
Definite good: ['#3', '#8', '#9']
Evil probabilities: #4=45%, #7=45%, #2=36%, #1=27%, #6=27%, #5=18%
  Generated 3024 candidate scenarios
  66 scenarios survived validation
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7]

#### [20:08:15] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#2', '#4', '#5']
Reason: Expected posterior 28.7 scenarios (adjusted 30.0, info gain 1.135 bits) | timing x1.00
WARNING: Corruption risk: 9%

### [20:09:18] Revealed #1 Jester
Info: {'targets': [2, 4, 5], 'evil_count': 3}

### [20:09:18] Ability used at #1

#### [20:09:26] Solver Output
Scenarios: 24/3024
Definite good: ['#3', '#5', '#7', '#8', '#9']
Evil probabilities: #1=75%, #2=50%, #4=50%, #6=25%
  Generated 3024 candidate scenarios
  24 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6]

#### [20:09:26] Recommendation
Action: **USE_ABILITY** #2 (Judge) -> targets ['#1']
Reason: Expected posterior 12.0 scenarios (adjusted 12.0, info gain 1.000 bits) | timing x1.00

### [20:10:10] Revealed #2 Judge
Info: {'target': 1, 'is_lying': True}

### [20:10:11] Ability used at #2

#### [20:10:19] Solver Output
Scenarios: 12/3024
Definite evil: ['#1']
Definite good: ['#2', '#3', '#5', '#7', '#8', '#9']
Evil probabilities: #4=50%, #6=50%
  Generated 3024 candidate scenarios
  12 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [20:10:19] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 12 scenarios (roles: {'Minion', 'Lilis'})

### [20:11:20] Executed #1 -> Minion (EVIL)

#### [20:11:20] Solver Output
Scenarios: 6/336
Definite evil: ['#1', '#6']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#8', '#9']
  Generated 336 candidate scenarios
  6 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion'})
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [20:11:20] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 6 scenarios (roles: {'Lilis'})

### [20:12:22] Executed #6 -> Lilis (EVIL)

## [20:12:23] GAME OVER — WIN
Final HP: 6
Notes: 6HP Lilis game, Jester impossible 3-evil claim exposed by Judge


---

# New Game — 2026-03-13 20:15:04
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Poet, Druid, Gemcrafter, Oracle, Empress, Knitter
- Outcasts: Wretch, Plague_Doctor
- Minions: Poisoner, Chancellor
- Demons: Lilis

### [20:16:07] Revealed #1 Knitter
Info: {'evil_pairs': 3}

### [20:16:07] Revealed #2 Druid
Info: {}

### [20:16:07] Revealed #3 Poet
Info: {'side': 'equal', 'copied_role': 'Architect'}

### [20:16:07] Revealed #4 Oracle
Info: {'targets': [3, 10], 'minion_role': 'Chancellor'}

### [20:17:12] Revealed #5 Empress
Info: {'targets': [2, 3, 4]}

### [20:17:12] Revealed #6 Empress
Info: {'targets': [2, 9, 10]}

### [20:17:13] Revealed #7 Gemcrafter
Info: {'good_position': 6}

### [20:17:13] Revealed #8 Wretch
Info: {}

#### [20:17:21] Solver Output
Scenarios: 31/5398
Definite good: ['#9', '#10']
Evil probabilities: #4=61%, #6=61%, #1=45%, #5=42%, #2=26%, #8=26%, #3=23%, #7=16%
  Generated 5398 candidate scenarios
  31 scenarios survived validation
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [20:17:21] Recommendation
Action: **USE_ABILITY** #2 (Druid) -> targets ['#1', '#3', '#4']
Reason: Entropy 0.999 (adjusted 0.886) | timing x1.00
WARNING: Corruption risk: 23%

### [20:18:16] Revealed #2 Druid
Info: {'targets': [1, 3, 4], 'found_outcast': None}

### [20:18:16] Ability used at #2

#### [20:18:17] Solver Output
Scenarios: 16/5398
Definite good: ['#2', '#9', '#10']
Evil probabilities: #6=75%, #8=50%, #1=44%, #3=38%, #4=31%, #5=31%, #7=31%
  Generated 5398 candidate scenarios
  16 scenarios survived validation
    #2 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8]

#### [20:18:17] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (38% evil Lilis, 31% evil Poisoner, 25% good Empress (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [20:19:15] Executed #6 -> Lilis (EVIL)

#### [20:19:16] Solver Output
Scenarios: 6/496
Definite evil: ['#6']
Definite good: ['#1', '#2', '#9', '#10']
Evil probabilities: #3=67%, #7=67%, #8=33%, #4=17%, #5=17%
  Generated 496 candidate scenarios
  6 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [3, 4, 5, 7, 8]

#### [20:19:16] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (33% evil Chancellor, 33% good Gemcrafter (corrupted), 33% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [20:20:17] Executed #7 -> Chancellor (EVIL)

#### [20:20:17] Solver Output
Scenarios: 2/47
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#8', '#9', '#10']
Evil probabilities: #4=50%, #5=50%
  Generated 47 candidate scenarios
  2 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [4, 5]

#### [20:20:17] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (50% good Oracle (corrupted), 50% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [20:21:15] Executed #4 -> GOOD (WRONG!)

#### [20:21:15] Solver Output
Scenarios: 1/39
Definite evil: ['#5', '#6', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#8', '#9', '#10']
  Generated 39 candidate scenarios
  1 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [20:21:15] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [20:22:21] Executed #5 -> Poisoner (EVIL)

## [20:22:21] GAME OVER — WIN
Final HP: 1
Notes: 1HP survive, wrong exec corrupted Oracle, Asc38 complete


---

# New Game — 2026-03-13 20:29:20
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Fortune_Teller, Lover, Slayer, Knitter, Oracle
- Outcasts: Bombardier
- Minions: 
- Demons: Pooka

### [20:30:27] Revealed #1 Oracle
Info: {'targets': [1, 4], 'minion_role': 'Minion'}

### [20:30:28] Revealed #2 Druid
Info: {}

### [20:30:28] Revealed #3 Fortune_Teller
Info: {}

### [20:30:28] Revealed #4 Bombardier
Info: {}

### [20:30:28] Revealed #5 Knitter
Info: {'evil_pairs': 0}

### [20:30:28] Revealed #6 Slayer
Info: {}

### [20:30:28] Revealed #7 Lover
Info: {'evil_adjacent': 0}

#### [20:30:38] Solver Output
Scenarios: 1/7
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [20:30:38] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [20:31:46] Executed #2 -> Pooka (EVIL)

## [20:31:47] GAME OVER — WIN
Final HP: 10
Notes: perfect 10HP, 1-scenario solve


---

# New Game — 2026-03-13 20:56:12
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Bishop, Gemcrafter, Scout, Bard, Druid, Baker
- Outcasts: Drunk, Bombardier
- Minions: Chancellor
- Demons: Pooka


---

# New Game — 2026-03-13 20:57:11
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Jester, Fortune_Teller, Slayer, Baker, Lover
- Outcasts: Plague_Doctor, Drunk, Wretch, Bombardier
- Minions: Chancellor
- Demons: Baa

### [20:57:51] Revealed #1 Bombardier
Info: {}

### [20:57:51] Revealed #2 Fortune_Teller
Info: {}

### [20:57:51] Revealed #3 Wretch
Info: {}

### [20:57:52] Revealed #4 Baker
Info: {'original_role': 'original'}

### [20:57:52] Revealed #5 Slayer
Info: {}

### [20:57:52] Revealed #6 Knight
Info: {}

### [20:57:52] Revealed #7 Slayer
Info: {}

### [20:57:52] Revealed #8 Baker
Info: {'original_role': 'Lover'}

#### [20:58:01] Solver Output
Scenarios: 255/451
Evil probabilities: #5=32%, #7=31%, #3=27%, #6=27%, #1=26%, #8=25%, #2=22%, #4=9%
  Generated 451 candidate scenarios
  255 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [20:58:01] Recommendation
Action: **EXECUTE** #6
Reason: Knight check: #6 is 27% evil, 17% corruption risk. Expected HP cost: 1.1 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 17% -- corrupted Knight loses immunity + 4 extra damage

### [20:59:22] Executed #6 -> GOOD (WRONG!)

#### [20:59:23] Solver Output
Scenarios: 44/338
Definite good: ['#6']
Evil probabilities: #5=41%, #7=39%, #8=32%, #3=30%, #1=27%, #2=27%, #4=5%
  Generated 338 candidate scenarios
  44 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8]

#### [20:59:23] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#4', '#8']
Reason: Entropy 1.000 (adjusted 1.000) | follow-up bonus 0.028 | timing x1.00

### [21:00:13] Revealed #2 Fortune Teller
Info: {'targets': [4, 8], 'has_evil': False}

### [21:00:13] Ability used at #2

#### [21:00:13] Solver Output
Scenarios: 22/338
Definite good: ['#4', '#6']
Evil probabilities: #5=55%, #7=55%, #1=36%, #3=36%, #2=9%, #8=9%
  Generated 338 candidate scenarios
  22 scenarios survived validation
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 7, 8]

#### [21:00:13] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#7']
Reason: Target #7 is 55% evil (adjusted 0.55)

### [21:01:01] Revealed #5 Slayer Result
Info: {}

### [21:01:01] Ability used at #5

#### [21:01:01] Solver Output
Scenarios: 26/298
Definite good: ['#4', '#6']
Evil probabilities: #1=46%, #3=46%, #5=46%, #7=46%, #2=8%, #8=8%
  Generated 298 candidate scenarios
  26 scenarios survived validation
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 7, 8]

#### [21:01:01] Recommendation
Action: **USE_ABILITY** #7 (Slayer) -> targets ['#5']
Reason: Target #5 is 46% evil (adjusted 0.46)

### [21:01:49] Revealed #7 Slayer Result
Info: {}

### [21:01:50] Ability used at #7

#### [21:01:50] Solver Output
Scenarios: 26/258
Definite good: ['#4', '#6']
Evil probabilities: #1=46%, #3=46%, #5=46%, #7=46%, #2=8%, #8=8%
  Generated 258 candidate scenarios
  26 scenarios survived validation
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 7, 8]

#### [21:01:50] Recommendation
Action: **ERROR** #5
Reason: #5 is 46% likely evil but HP too low to risk (HP=4, cost=5). Need more info.
WARNING: Probabilistic execution -- 46% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=4, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [21:03:13] Executed #5 -> Baa (EVIL)

#### [21:03:13] Solver Output
Scenarios: 6/42
Definite evil: ['#5']
Definite good: ['#2', '#4', '#6', '#8']
Evil probabilities: #1=33%, #3=33%, #7=33%
  Generated 42 candidate scenarios
  6 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 7]

#### [21:03:13] Recommendation
Action: **ERROR** #7
Reason: #7 is 33% likely evil but HP too low to risk (HP=4, cost=5). Need more info.
WARNING: Probabilistic execution -- 33% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=4, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [21:04:28] Executed #7 -> Chancellor (EVIL)

## [21:04:28] GAME OVER — WIN
Final HP: 4
Notes: 4HP, corrupted Knight check lost 6HP, both Slayers were evil


---

# New Game — 2026-03-13 21:07:08
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Hunter, Druid, Slayer, Enlightened, Oracle
- Outcasts: Plague_Doctor, Bombardier
- Minions: Minion, Shaman
- Demons: Lilis

### [21:08:22] Revealed #1 Hunter
Info: {'distance': 1}

### [21:08:22] Revealed #2 Enlightened
Info: {'direction': 'CCW'}

### [21:08:22] Revealed #3 Bombardier
Info: {}

### [21:08:22] Revealed #4 Bard
Info: {'corruption_distance': -1}

### [21:09:31] Revealed #5 Plague_Doctor
Info: {}

### [21:09:32] Revealed #7 Enlightened
Info: {'direction': 'CW'}

### [21:09:32] Revealed #8 Enlightened
Info: {'direction': 'CCW'}

### [21:09:32] Revealed #9 Druid
Info: {}

#### [21:09:43] Solver Output
Scenarios: 54/1974
Definite good: ['#4', '#6']
Evil probabilities: #2=67%, #5=56%, #8=56%, #3=44%, #7=44%, #1=22%, #9=11%
  Generated 1974 candidate scenarios
  54 scenarios survived validation
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 7, 8, 9]

#### [21:09:43] Recommendation
Action: **USE_ABILITY** #5 (Plague Doctor) -> targets ['#4']
Reason: Entropy 1.436 (adjusted 1.436) | timing x1.00

### [21:10:47] Ability used at #5

#### [21:10:47] Solver Output
Scenarios: 30/1974
Definite evil: ['#5']
Definite good: ['#3', '#4', '#6']
Evil probabilities: #8=60%, #1=40%, #2=40%, #7=40%, #9=20%
  Generated 1974 candidate scenarios
  30 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Shaman', 'Minion', 'Lilis'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 7, 8, 9]

#### [21:10:47] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 30 scenarios (roles: {'Shaman', 'Minion', 'Lilis'})

### [21:12:06] Executed #5 -> Shaman (EVIL)

#### [21:12:06] Solver Output
Scenarios: 10/56
Definite evil: ['#5']
Definite good: ['#3', '#4', '#6']
Evil probabilities: #8=60%, #1=40%, #2=40%, #7=40%, #9=20%
  Generated 56 candidate scenarios
  10 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 7, 8, 9]

#### [21:12:06] Recommendation
Action: **USE_ABILITY** #9 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.722 (adjusted 0.722) | timing x1.00

### [21:13:20] Revealed #9 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [21:13:20] Ability used at #9

#### [21:13:20] Solver Output
Scenarios: 2/56
Definite evil: ['#5', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#7']
  Generated 56 candidate scenarios
  2 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #9 is DEFINITELY EVIL (possible roles: {'Minion', 'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [21:13:20] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 2 scenarios (roles: {'Minion', 'Lilis'})

### [21:14:35] Executed #8 -> Minion (EVIL)

### [21:15:49] Executed #9 -> Lilis (EVIL)

## [21:15:50] GAME OVER — WIN
Final HP: 6
Notes: 6HP Lilis game, evil PD lie exposed Shaman


---

# New Game — 2026-03-13 21:18:30
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Baker, Knitter, Empress, Bard, Bishop
- Outcasts: Doppelganger
- Minions: 
- Demons: Pooka

### [21:19:34] Revealed #1 Bishop
Info: {'targets': [1, 4, 5], 'types': ['Outcast', 'Villager', 'Demon']}

### [21:19:34] Revealed #2 Bard
Info: {'corruption_distance': 1}

### [21:19:34] Revealed #3 Baker
Info: {'original_role': 'original'}

### [21:19:34] Revealed #4 Baker
Info: {'original_role': 'Empress'}

### [21:19:34] Revealed #5 Baker
Info: {'original_role': 'Knitter'}

### [21:19:35] Revealed #6 Empress
Info: {'targets': [1, 3, 7]}

### [21:19:35] Revealed #7 Alchemist
Info: {'cured_count': 2}

#### [21:19:47] Solver Output
Scenarios: 2/42
Definite good: ['#2', '#3', '#4', '#5', '#6']
Evil probabilities: #1=50%, #7=50%
  Generated 42 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [21:19:47] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% good Bishop (corrupted), 50% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [21:20:57] Executed #1 -> GOOD (WRONG!)

### [21:22:02] Executed #7 -> Pooka (EVIL)

## [21:22:03] GAME OVER — WIN
Final HP: 5
Notes: 5HP, wrong exec corrupted Bishop, Pooka at #7


---

# New Game — 2026-03-13 21:28:20
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Scout, Medium, Bard, Alchemist, Judge, Druid
- Outcasts: Drunk, Bombardier
- Minions: Poisoner, Chancellor
- Demons: Lilis

### [21:29:39] Revealed #1 Medium
Info: {'good_position': 6, 'good_role': 'Bard'}

### [21:29:39] Revealed #2 Alchemist
Info: {'cured_count': 2}

### [21:29:39] Revealed #3 Druid
Info: {}

### [21:29:39] Revealed #4 Judge
Info: {}

### [21:30:58] Revealed #6 Bard
Info: {'corruption_distance': 3}

### [21:30:58] Revealed #7 Druid
Info: {}

### [21:30:59] Revealed #8 Alchemist
Info: {'cured_count': 2}

### [21:30:59] Revealed #9 Bombardier
Info: {}

#### [21:31:12] Solver Output
Scenarios: 159/4727
Definite good: ['#5']
Evil probabilities: #2=70%, #8=70%, #6=48%, #1=34%, #7=31%, #9=19%, #4=14%, #3=13%
  Generated 4727 candidate scenarios
  159 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [21:31:12] Recommendation
Action: **USE_ABILITY** #7 (Druid) -> targets ['#1', '#2', '#9']
Reason: Entropy 1.632 (adjusted 1.539) | timing x1.00
WARNING: Corruption risk: 11%

### [21:32:26] Revealed #7 Druid
Info: {'targets': [1, 2, 9], 'found_outcast': None}

### [21:32:26] Ability used at #7

#### [21:32:26] Solver Output
Scenarios: 81/4727
Definite good: ['#5']
Evil probabilities: #2=74%, #8=69%, #7=56%, #6=38%, #1=25%, #9=23%, #4=9%, #3=6%
  Generated 4727 candidate scenarios
  81 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [21:32:26] Recommendation
Action: **USE_ABILITY** #3 (Druid) -> targets ['#1', '#2', '#9']
Reason: Entropy 1.646 (adjusted 1.463) | timing x1.00
WARNING: Corruption risk: 22%

### [21:33:40] Revealed #3 Druid
Info: {'targets': [1, 2, 9], 'found_outcast': 'Drunk'}

### [21:33:40] Ability used at #3

#### [21:33:40] Solver Output
Scenarios: 23/4727
Definite good: ['#3', '#5']
Evil probabilities: #8=78%, #6=65%, #2=61%, #7=35%, #1=26%, #9=26%, #4=9%
  Generated 4727 candidate scenarios
  23 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6, 7, 8, 9]

#### [21:33:40] Recommendation
Action: **USE_ABILITY** #4 (Judge) -> targets ['#9']
Reason: Expected posterior 12.6 scenarios (adjusted 12.6, info gain 0.872 bits) | timing x1.00

### [21:34:50] Revealed #4 Judge
Info: {'target': 9, 'is_lying': False}

### [21:34:50] Ability used at #4

#### [21:34:50] Solver Output
Scenarios: 15/4727
Definite evil: ['#6']
Definite good: ['#3', '#4', '#5', '#9']
Evil probabilities: #8=67%, #2=60%, #1=40%, #7=33%
  Generated 4727 candidate scenarios
  15 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Lilis', 'Chancellor', 'Poisoner'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 7, 8]

#### [21:34:50] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 15 scenarios (roles: {'Lilis', 'Chancellor', 'Poisoner'})

### [21:35:59] Executed #6 -> Lilis (EVIL)

#### [21:35:59] Solver Output
Scenarios: 6/500
Definite evil: ['#6']
Definite good: ['#3', '#4', '#5', '#9']
Evil probabilities: #2=67%, #7=50%, #8=50%, #1=33%
  Generated 500 candidate scenarios
  6 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 7, 8]

#### [21:35:59] Recommendation
Action: **ERROR** #2
Reason: #2 is 67% likely evil but budget=1 requires >=80% confidence (HP=6, cost=5).
WARNING: Probabilistic execution -- 67% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 67% < 80% threshold. Consider manual override if you have extra information.

### [21:37:09] Executed #2 -> Chancellor (EVIL)

#### [21:37:10] Solver Output
Scenarios: 4/88
Definite evil: ['#2', '#6']
Definite good: ['#1', '#3', '#4', '#5', '#9']
Evil probabilities: #7=50%, #8=50%
  Generated 88 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [7, 8]

#### [21:37:10] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% good Druid (corrupted), 50% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [21:38:22] Executed #7 -> Poisoner (EVIL)

## [21:38:22] GAME OVER — WIN
Final HP: 6
Notes: 6HP Lilis, dual-Druid disagreement, Judge confirmed #9 truthful


---

# New Game — 2026-03-13 21:41:27
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Confessor, Judge, Empress, Knight, Scout
- Outcasts: Bombardier
- Minions: Poisoner
- Demons: Lilis

### [21:43:15] Revealed #1 Knight
Info: {}

### [21:43:15] Revealed #2 Scout
Info: {'evil_role': 'Lilis', 'distance': 3}

### [21:43:15] Revealed #3 Empress
Info: {'targets': [2, 4, 8]}

### [21:43:16] Revealed #4 Hunter
Info: {'distance': 1}

### [21:44:53] Revealed #6 Bombardier
Info: {}

### [21:44:53] Revealed #7 Bombardier
Info: {}

### [21:44:53] Revealed #8 Confessor
Info: {'dizzy': False}

#### [21:44:53] Solver Output
Scenarios: 3/76
Definite good: ['#1', '#2', '#5', '#8']
Evil probabilities: #3=67%, #7=67%, #4=33%, #6=33%
  Generated 76 candidate scenarios
  3 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 6, 7]

#### [21:44:53] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (33% good Empress, 33% evil Lilis, 33% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [21:46:36] Executed #3 -> Lilis (EVIL)

#### [21:46:36] Solver Output
Scenarios: 1/8
Definite evil: ['#3', '#6']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#8']
  Generated 8 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [21:46:36] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [21:48:21] Executed #6 -> Poisoner (EVIL)

## [21:48:21] GAME OVER — WIN
Final HP: 8
Notes: 8HP Lilis, Scout+Empress+Hunter nailed it to 3 scenarios


---

# New Game — 2026-03-13 21:51:41
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Architect, Dreamer, Bishop, Slayer
- Outcasts: Wretch
- Minions: 
- Demons: Pooka

### [21:53:24] Revealed #1 Jester
Info: {}

### [21:53:24] Revealed #2 Architect
Info: {'side': 'Left'}

### [21:53:24] Revealed #3 Dreamer
Info: {}

### [21:53:25] Revealed #4 Slayer
Info: {}

### [21:53:25] Revealed #5 Bishop
Info: {'targets': [1, 6], 'types': ['Minion', 'Villager']}

### [21:53:25] Revealed #6 Wretch
Info: {}

#### [21:53:25] Solver Output
Scenarios: 2/6
Definite good: ['#1', '#4', '#5', '#6']
Evil probabilities: #2=50%, #3=50%
  Generated 6 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [2, 3]

#### [21:53:25] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#2']
Reason: Entropy 1.000 (adjusted 0.750) | timing x1.00
WARNING: Corruption risk: 50%

### [21:55:12] Revealed #3 Dreamer
Info: {'target': 2, 'evil_role': 'Pooka'}

### [21:55:12] Ability used at #3

#### [21:55:13] Solver Output
Scenarios: 1/6
Definite evil: ['#3']
Definite good: ['#1', '#2', '#4', '#5', '#6']
  Generated 6 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [21:55:13] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [21:56:51] Executed #3 -> Pooka (EVIL)

## [21:56:51] GAME OVER — WIN
Final HP: 10
Notes: perfect 10HP, Dreamer Pooka-claim narrowed to 1 scenario, Asc39 complete


---

# New Game — 2026-03-14 13:17:50
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Baker, Lover, Knight, Oracle, Medium, Fortune_Teller
- Outcasts: Doppelganger, Drunk
- Minions: Shaman, Witch
- Demons: Baa

### [13:19:21] Revealed #1 Fortune_Teller
Info: {}

### [13:19:25] Revealed #2 Oracle
Info: {'targets': [1, 9], 'minion_role': 'Witch'}

### [13:19:29] Revealed #3 Lover
Info: {'evil_adjacent': 2}

### [13:19:34] Revealed #4 Lover
Info: {'evil_adjacent': 2}

### [13:19:40] Revealed #5 Slayer
Info: {}

### [13:19:44] Revealed #6 Knight
Info: {}

### [13:19:49] Revealed #7 Slayer
Info: {}

### [13:19:53] Revealed #8 Medium
Info: {'good_position': 9, 'good_role': 'Doppelganger'}

#### [13:19:59] Solver Output
Scenarios: 1036/21672
Evil probabilities: #2=68%, #4=64%, #3=48%, #5=41%, #6=27%, #7=27%, #1=14%, #8=6%, #9=3%
  Generated 21672 candidate scenarios
  1036 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [13:19:59] Recommendation
Action: **EXECUTE** #6
Reason: Knight check: #6 is 27% evil, 13% corruption risk. Expected HP cost: 0.8 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 13% -- corrupted Knight loses immunity + 4 extra damage

### [13:20:46] Executed #6 -> GOOD (WRONG!)

#### [13:21:02] Solver Output
Scenarios: 754/14448
Definite good: ['#6']
Evil probabilities: #2=69%, #4=62%, #5=51%, #3=51%, #7=37%, #1=16%, #8=8%, #9=5%
  Generated 14448 candidate scenarios
  754 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [13:21:02] Recommendation
Action: **USE_ABILITY** #1 (Fortune Teller) -> targets ['#3', '#7']
Reason: Entropy 0.995 (adjusted 0.924) | follow-up bonus 0.310 | timing x1.00
WARNING: Corruption risk: 14%

### [13:22:57] Executed #1 -> Witch (EVIL)

### [13:24:34] Revealed #9 Slayer
Info: {}

### [13:24:42] Revealed #7 Slayer Result
Info: {}

### [13:24:46] Ability used at #7

#### [13:24:53] Solver Output
Scenarios: 10/1446
Definite evil: ['#1']
Definite good: ['#2', '#5', '#6', '#8', '#9']
Evil probabilities: #3=80%, #4=80%, #7=40%
  Generated 1446 candidate scenarios
  10 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 4, 7]

#### [13:24:53] Recommendation
Action: **USE_ABILITY** #9 (Slayer) -> targets ['#3']
Reason: Target #3 is 80% evil (adjusted 0.80)

### [13:27:29] Revealed #9 Slayer Result
Info: {}

### [13:27:36] Ability used at #9

### [13:27:45] Executed #3 -> Baa (EVIL)

#### [13:27:53] Solver Output
Scenarios: 0/146
  Generated 146 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #8 Medium: rejected 141/146 (97%)
    #4 Lover: rejected 90/146 (62%)
    #2 Oracle: rejected 43/146 (29%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #2 Oracle: still 0
    WITHOUT #3 Lover: still 0
    WITHOUT #4 Lover: still 0
    WITHOUT #8 Medium: still 0

#### [13:27:53] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [13:28:28] Solver Output
Scenarios: 0/146
  Generated 146 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #8 Medium: rejected 141/146 (97%)
    #4 Lover: rejected 90/146 (62%)
    #2 Oracle: rejected 43/146 (29%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #2 Oracle: still 0
    WITHOUT #3 Lover: still 0
    WITHOUT #4 Lover: still 0
    WITHOUT #8 Medium: still 0

#### [13:28:28] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [13:29:24] Solver Output
Scenarios: 8/258
Definite evil: ['#1', '#3']
Definite good: ['#2', '#6', '#8', '#9']
Evil probabilities: #4=50%, #5=38%, #7=12%
  Generated 258 candidate scenarios
  8 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Witch'})
    #3 is DEFINITELY EVIL (possible roles: {'Baa'})
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 5, 7]

#### [13:29:24] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#4']
Reason: Target #4 is 50% evil (adjusted 0.44)
WARNING: Corruption risk: 12% -- Slayer ability disabled if corrupted

## [13:30:31] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect. Knight check + 3 Slayers (Dopp copy). Accidental #1 execution was lucky (Witch). Slayer9->3 Baa, Slayer5->4 Shaman.


---

# New Game — 2026-03-14 13:34:08
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Lover, Confessor, Slayer, Scout, Knitter, Baker
- Outcasts: Plague_Doctor, Bombardier
- Minions: Minion, Puppeteer
- Demons: Pooka

### [13:35:23] Revealed #1 Lover
Info: {'evil_adjacent': 1}

### [13:35:23] Revealed #2 Slayer
Info: {}

### [13:35:23] Revealed #3 Plague_Doctor
Info: {}

### [13:35:23] Revealed #4 Knitter
Info: {'evil_pairs': 0}

### [13:35:24] Revealed #5 Scout
Info: {'evil_role': 'Puppeteer', 'distance': 2}

### [13:35:46] Revealed #6 Baker
Info: {'original_role': 'Scout'}

### [13:35:46] Revealed #7 Bombardier
Info: {}

### [13:35:46] Revealed #8 Architect
Info: {'side': 'Left'}

### [13:35:47] Revealed #9 Confessor
Info: {'dizzy': True}

### [13:35:47] Revealed #10 Scout
Info: {'evil_role': 'Minion', 'distance': 1}

#### [13:35:54] Solver Output
Scenarios: 11/3476
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3']
Evil probabilities: #10=82%, #5=73%, #9=73%, #6=27%, #8=27%, #4=18%
  Generated 3476 candidate scenarios
  11 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Minion', 'Puppeteer'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 8, 9, 10]

#### [13:35:54] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 11 scenarios (roles: {'Minion', 'Puppeteer'})

### [13:36:50] Executed #7 -> Puppeteer (EVIL)

#### [13:36:51] Solver Output
Scenarios: 4/532
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3']
Evil probabilities: #5=75%, #10=75%, #6=50%, #8=50%, #4=25%, #9=25%
  Generated 532 candidate scenarios
  4 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 8, 9, 10]

#### [13:36:51] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#4']
Reason: Entropy 1.500 (adjusted 1.500) | timing x1.00

### [13:38:05] Ability used at #3

#### [13:38:14] Solver Output
Scenarios: 2/532
Definite evil: ['#5', '#7', '#10']
Definite good: ['#1', '#2', '#3', '#4', '#9']
Evil probabilities: #6=50%, #8=50%
  Generated 532 candidate scenarios
  2 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #10 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [6, 8]

#### [13:38:14] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [13:39:00] Executed #5 -> Pooka (EVIL)

### [13:39:47] Executed #10 -> Minion (EVIL)

#### [13:39:55] Solver Output
Scenarios: 2/12
Definite evil: ['#5', '#7', '#10']
Definite good: ['#1', '#2', '#3', '#4', '#9']
Evil probabilities: #6=50%, #8=50%
  Generated 12 candidate scenarios
  2 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #10 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [6, 8]

#### [13:39:55] Recommendation
Action: **USE_ABILITY** #2 (Slayer) -> targets ['#6']
Reason: Target #6 is 50% evil (adjusted 0.50)

### [13:41:41] Executed #8 -> Puppet (EVIL)

## [13:41:41] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect. PD check revealed #10 evil + #4 corrupted. Slayer eliminated #6, confirmed #8 Puppet.


---

# New Game — 2026-03-14 13:45:26
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Medium, Oracle, Slayer, Gemcrafter, Alchemist, Knight, Architect
- Outcasts: Drunk, Doppelganger
- Minions: Poisoner, Shaman
- Demons: Baa

### [13:46:43] Revealed #1 Hunter
Info: {'distance': 1}

### [13:46:44] Revealed #2 Architect
Info: {'side': 'Left'}

### [13:46:44] Revealed #3 Knight
Info: {}

### [13:46:44] Revealed #4 Hunter
Info: {'distance': 2}

### [13:46:44] Revealed #5 Gemcrafter
Info: {'good_position': 4}

### [13:46:44] Revealed #6 Hunter
Info: {'distance': 1}

### [13:46:45] Revealed #7 Medium
Info: {'good_position': 6, 'good_role': 'Hunter'}

### [13:46:45] Revealed #8 Alchemist
Info: {'cured_count': 0}

### [13:46:45] Revealed #9 Oracle
Info: {'targets': [1, 8], 'minion_role': 'Shaman'}

#### [13:47:14] Solver Output
Scenarios: 78/31722
Definite good: ['#8']
Evil probabilities: #2=72%, #9=69%, #4=46%, #5=44%, #6=36%, #7=21%, #3=10%, #1=3%
  Generated 31722 candidate scenarios
  78 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [13:47:14] Recommendation
Action: **EXECUTE** #9
Reason: Execution lookahead: #9 guarantees a win across all reveal branches with current HP budget (29% evil Baa, 29% evil Shaman, 28% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 69%, but all reveal branches still lead to a forced win.

### [13:48:02] Executed #9 -> GOOD (WRONG!)

#### [13:48:10] Solver Output
Scenarios: 24/21224
Definite good: ['#6', '#8', '#9']
Evil probabilities: #2=92%, #4=83%, #5=83%, #3=17%, #7=17%, #1=8%
  Generated 21224 candidate scenarios
  24 scenarios survived validation
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7]

#### [13:48:10] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (42% evil Poisoner, 25% evil Baa, 25% evil Shaman).
WARNING: Execution lookahead override -- immediate hit chance is 92%, but all reveal branches still lead to a forced win.

### [13:48:55] Executed #2 -> Poisoner (EVIL)

#### [13:48:55] Solver Output
Scenarios: 10/2518
Definite evil: ['#2', '#4', '#5']
Definite good: ['#1', '#3', '#6', '#7', '#8', '#9']
  Generated 2518 candidate scenarios
  10 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #4 is DEFINITELY EVIL (possible roles: {'Baa', 'Shaman'})
    #5 is DEFINITELY EVIL (possible roles: {'Shaman', 'Baa'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [13:48:55] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 10 scenarios (roles: {'Baa', 'Shaman'})

### [13:49:38] Executed #4 -> Shaman (EVIL)

### [13:50:16] Executed #5 -> Baa (EVIL)

## [13:50:16] GAME OVER — WIN
Final HP: 8
Notes: 8HP, wrong exec on Drunk #9 (corrupted). Lookahead guaranteed win path.


---

# New Game — 2026-03-14 13:53:43
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Gemcrafter, Druid, Empress, Judge, Medium
- Outcasts: Doppelganger, Wretch
- Minions: Chancellor
- Demons: Lilis

### [13:54:54] Revealed #1 Medium
Info: {'good_position': 3, 'good_role': 'Enlightened'}

### [13:54:54] Revealed #2 Gemcrafter
Info: {'good_position': 3}

### [13:54:54] Revealed #3 Enlightened
Info: {'direction': 'Equidistant'}

### [13:54:54] Revealed #4 Medium
Info: {'good_position': 1, 'good_role': 'Doppelganger'}

### [13:56:03] Revealed #6 Wretch
Info: {}

### [13:56:03] Revealed #7 Enlightened
Info: {'direction': 'CW'}

### [13:56:03] Revealed #8 Judge
Info: {}

#### [13:56:03] Solver Output
Scenarios: 2/398
Definite evil: ['#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']
  Generated 398 candidate scenarios
  2 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Lilis', 'Chancellor'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis', 'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [13:56:03] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Lilis', 'Chancellor'})

### [13:56:51] Executed #7 -> Chancellor (EVIL)

### [13:57:25] Executed #8 -> Lilis (EVIL)

## [13:57:26] GAME OVER — WIN
Final HP: 8
Notes: 8HP Lilis game. Night killed Druid #5. Solver found both evils with 2 scenarios.


---

# New Game — 2026-03-14 14:00:35
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Empress, Baker, Hunter, Druid, Medium, Knight
- Outcasts: Wretch
- Minions: Poisoner, Shaman
- Demons: Lilis

### [14:01:39] Revealed #1 Confessor
Info: {'dizzy': True}

### [14:01:39] Revealed #2 Empress
Info: {'targets': [1, 4, 7]}

### [14:01:39] Revealed #3 Wretch
Info: {}

### [14:01:39] Revealed #4 Baker
Info: {'original_role': 'original'}

### [14:02:40] Revealed #5 Druid
Info: {}

### [14:02:40] Revealed #6 Medium
Info: {'good_position': 7, 'good_role': 'Baker'}

### [14:02:41] Revealed #7 Baker
Info: {'original_role': 'Empress'}

### [14:02:41] Revealed #9 Knight
Info: {}

#### [14:02:41] Solver Output
Scenarios: 22/694
Definite good: ['#4', '#8']
Evil probabilities: #9=64%, #2=55%, #1=45%, #3=45%, #5=45%, #7=27%, #6=18%
  Generated 694 candidate scenarios
  22 scenarios survived validation
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 6, 7, 9]

#### [14:02:41] Recommendation
Action: **USE_ABILITY** #5 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.994 (adjusted 0.994) | timing x1.00

### [14:03:44] Revealed #5 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Wretch'}

### [14:03:45] Ability used at #5

#### [14:03:45] Solver Output
Scenarios: 10/694
Definite evil: ['#5']
Definite good: ['#4', '#6', '#7', '#8']
Evil probabilities: #2=60%, #9=60%, #1=40%, #3=40%
  Generated 694 candidate scenarios
  10 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Lilis', 'Shaman'})
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 9]

#### [14:03:45] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 10 scenarios (roles: {'Lilis', 'Shaman'})

### [14:04:29] Executed #5 -> Lilis (EVIL)

#### [14:04:30] Solver Output
Scenarios: 5/79
Definite evil: ['#5']
Definite good: ['#4', '#6', '#7', '#8']
Evil probabilities: #2=60%, #9=60%, #1=40%, #3=40%
  Generated 79 candidate scenarios
  5 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 9]

#### [14:04:30] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (40% good Empress, 40% evil Poisoner, 20% evil Shaman).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [14:05:21] Executed #2 -> Poisoner (EVIL)

#### [14:05:21] Solver Output
Scenarios: 2/7
Definite evil: ['#2', '#5']
Definite good: ['#1', '#4', '#6', '#7', '#8']
Evil probabilities: #3=50%, #9=50%
  Generated 7 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 9]

#### [14:05:21] Recommendation
Action: **EXECUTE** #9
Reason: Knight free check: #9 is 50% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [14:06:15] Executed #9 -> Shaman (EVIL)

## [14:06:15] GAME OVER — WIN
Final HP: 6
Notes: 6HP Lilis. Night killed #8 Empress. Druid found Wretch, solver locked Lilis. Knight check on #9 = Shaman.


---

# New Game — 2026-03-14 14:09:45
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Gemcrafter, Dreamer, Druid, Jester, Alchemist, Bishop
- Outcasts: Wretch, Bombardier
- Minions: Minion
- Demons: Baa

### [14:11:02] Revealed #1 Bombardier
Info: {}

### [14:11:02] Revealed #2 Gemcrafter
Info: {'good_position': 8}

### [14:11:02] Revealed #3 Druid
Info: {}

### [14:11:03] Revealed #4 Dreamer
Info: {}

### [14:11:03] Revealed #5 Alchemist
Info: {'cured_count': 0}

### [14:11:03] Revealed #6 Bishop
Info: {'targets': [4, 5, 8], 'types': ['Outcast', 'Villager', 'Minion']}

### [14:11:03] Revealed #7 Bombardier
Info: {}

### [14:11:03] Revealed #8 Jester
Info: {}

#### [14:11:03] Solver Output
Scenarios: 4/56
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5', '#8']
Evil probabilities: #1=50%, #7=50%
  Generated 56 candidate scenarios
  4 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Minion', 'Baa'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [14:11:03] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 4 scenarios (roles: {'Minion', 'Baa'})

### [14:11:51] Executed #6 -> Baa (EVIL)

#### [14:11:51] Solver Output
Scenarios: 2/7
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5', '#8']
Evil probabilities: #1=50%, #7=50%
  Generated 7 candidate scenarios
  2 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Baa'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [14:11:51] Recommendation
Action: **USE_ABILITY** #3 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [14:12:44] Executed #1 -> Minion (EVIL)

## [14:12:44] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect. Bishop lied (Baa), two Bombardiers = one fake. Clean solve.


---

# New Game — 2026-03-14 14:16:00
Cards: 8, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Fortune_Teller, Knitter, Judge, Witness, Scout
- Outcasts: Plague_Doctor, Bombardier
- Minions: Puppeteer
- Demons: Baa

### [14:17:19] Revealed #1 Fortune_Teller
Info: {}

### [14:17:19] Revealed #2 Scout
Info: {'evil_role': 'Puppeteer', 'distance': 1}

### [14:17:19] Revealed #3 Enlightened
Info: {'direction': 'CW'}

### [14:17:19] Revealed #4 Fortune_Teller
Info: {}

### [14:17:34] Revealed #5 Witness
Info: {'affected_position': 1}

### [14:17:34] Revealed #6 Knitter
Info: {'evil_pairs': 0}

### [14:17:34] Revealed #7 Judge
Info: {}

### [14:17:35] Revealed #8 Plague_Doctor
Info: {}

#### [14:17:35] Solver Output
Scenarios: 8/324
Definite evil: ['#4']
Definite good: ['#2', '#8']
Evil probabilities: #5=75%, #6=62%, #3=25%, #7=25%, #1=12%
  Generated 324 candidate scenarios
  8 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Puppet', 'Baa', 'Puppeteer'})
    #2 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 6, 7]

#### [14:17:35] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 8 scenarios (roles: {'Puppet', 'Baa', 'Puppeteer'})

### [14:18:29] Executed #4 -> Puppeteer (EVIL)

#### [14:18:29] Solver Output
Scenarios: 2/52
Definite evil: ['#3', '#4']
Definite good: ['#1', '#2', '#7', '#8']
Evil probabilities: #5=50%, #6=50%
  Generated 52 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 6]

#### [14:18:29] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Puppet'})

### [14:19:19] Executed #3 -> GOOD (WRONG!)

#### [14:19:19] Solver Output
Scenarios: 0/21
  Generated 21 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #5 Witness: rejected 21/21 (100%)
    #6 Knitter: rejected 13/21 (62%)
    #2 Scout: rejected 8/21 (38%)
    #3 Enlightened: rejected 7/21 (33%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #2 Scout: still 0
    WITHOUT #3 Enlightened: still 0
    WITHOUT #5 Witness: 5 scenarios survive  <-- SUSPECT
    WITHOUT #6 Knitter: still 0
    WITHOUT #7 Judge: still 0

#### [14:19:19] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [14:20:23] Solver Output
Scenarios: 5/21
Definite evil: ['#4', '#5']
Definite good: ['#2', '#3', '#8']
Evil probabilities: #6=60%, #1=20%, #7=20%
  Generated 21 candidate scenarios
  5 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 6, 7]

#### [14:20:23] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 5 scenarios (roles: {'Puppet'})

### [14:21:13] Executed #5 -> Puppet (EVIL)

#### [14:21:13] Solver Output
Scenarios: 5/21
Definite evil: ['#4', '#5']
Definite good: ['#2', '#3', '#8']
Evil probabilities: #6=60%, #1=20%, #7=20%
  Generated 21 candidate scenarios
  5 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 6, 7]

#### [14:21:13] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.371 (adjusted 1.371) | timing x1.00

### [14:23:18] Ability used at #8

#### [14:23:18] Solver Output
Scenarios: 3/21
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#7', '#8']
  Generated 21 candidate scenarios
  3 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #6 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [14:23:18] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 3 scenarios (roles: {'Baa'})

### [14:24:12] Executed #6 -> Baa (EVIL)

## [14:24:12] GAME OVER — WIN
Final HP: 5
Notes: 5HP, wrong exec on #3 Enlightened (bad Witness data entry). PD check confirmed #6 Baa. Asc40 complete 7/7!


---

# New Game — 2026-03-28 14:10:37
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Architect, Baker, Knitter, Hunter, Medium
- Outcasts: Bombardier
- Minions: Poisoner, Twin_Minion
- Demons: Lilis

### [14:11:50] Revealed #1 Medium
Info: {'good_position': 3, 'good_role': 'Knitter'}

### [14:11:53] Revealed #2 Architect
Info: {'side': 'left'}

### [14:11:56] Revealed #3 Knitter
Info: {'evil_pairs': 1}

### [14:12:00] Revealed #4 Bombardier
Info: {}

### [14:13:14] Revealed #6 Architect
Info: {'side': 'right'}

### [14:13:20] Revealed #7 Baker
Info: {'original_role': 'Knitter'}

### [14:13:24] Revealed #8 Bishop
Info: {'targets': [1, 3, 5], 'types': ['Outcast', 'Villager', 'Minion']}

### [14:13:27] Revealed #9 Hunter
Info: {'distance': 1}

#### [14:13:33] Solver Output
Scenarios: 4/674
Definite evil: ['#6', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']
  Generated 674 candidate scenarios
  4 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Lilis', 'Twin_Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Lilis', 'Twin_Minion', 'Poisoner'})
    #9 is DEFINITELY EVIL (possible roles: {'Lilis', 'Twin_Minion', 'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [14:13:33] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 4 scenarios (roles: {'Lilis', 'Twin_Minion'})

### [14:14:25] Executed #6 -> Twin_Minion (EVIL)

#### [14:14:29] Solver Output
Scenarios: 2/76
Definite evil: ['#6', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']
  Generated 76 candidate scenarios
  2 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Lilis', 'Poisoner'})
    #9 is DEFINITELY EVIL (possible roles: {'Lilis', 'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [14:14:29] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Lilis', 'Poisoner'})

### [14:15:06] Executed #7 -> Poisoner (EVIL)

#### [14:15:10] Solver Output
Scenarios: 1/7
Definite evil: ['#6', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #9 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [14:15:10] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [14:15:54] Executed #9 -> Lilis (EVIL)

## [14:15:59] GAME OVER — WIN
Final HP: 6
Notes: Lilis game, all 3 evils locked by solver. Poisoner corrupted Bishop#8. Night killed Baker#5.


---

# New Game — 2026-03-28 14:22:15
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Gemcrafter, Witness, Slayer, Baker, Bishop, Oracle
- Outcasts: Plague_Doctor, Wretch
- Minions: Twin_Minion, Minion
- Demons: Baa

### [14:23:33] Revealed #1 Gemcrafter
Info: {'good_position': 9}

### [14:23:36] Revealed #2 Plague_Doctor
Info: {}

### [14:23:40] Revealed #3 Bishop
Info: {'targets': [1, 3, 9], 'types': ['Outcast', 'Villager', 'Minion']}

### [14:23:44] Revealed #4 Plague_Doctor
Info: {}

### [14:23:48] Revealed #5 Oracle
Info: {'targets': [2, 3], 'minion_role': 'Minion'}

### [14:23:52] Revealed #6 Baker
Info: {'original_role': 'original'}

### [14:23:56] Revealed #7 Oracle
Info: {'targets': [3, 9], 'minion_role': 'Twin_Minion'}

### [14:24:01] Revealed #8 Slayer
Info: {}

### [14:24:05] Revealed #9 Baker
Info: {'original_role': 'Witness'}

#### [14:24:09] Solver Output
Scenarios: 8/2604
Definite evil: ['#7']
Definite good: ['#1', '#3', '#6', '#9']
Evil probabilities: #4=75%, #5=75%, #2=25%, #8=25%
  Generated 2604 candidate scenarios
  8 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Baa', 'Minion', 'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 8]

#### [14:24:09] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 8 scenarios (roles: {'Baa', 'Minion', 'Twin_Minion'})

### [14:25:03] Executed #7 -> Baa (EVIL)

#### [14:25:06] Solver Output
Scenarios: 3/296
Definite evil: ['#7']
Definite good: ['#1', '#3', '#6', '#9']
Evil probabilities: #4=67%, #5=67%, #2=33%, #8=33%
  Generated 296 candidate scenarios
  3 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 8]

#### [14:25:06] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#1']
Reason: Entropy 0.918 (adjusted 0.918) | timing x1.00

### [14:26:05] Ability used at #2

#### [14:26:08] Solver Output
Scenarios: 2/296
Definite evil: ['#4', '#5', '#7']
Definite good: ['#1', '#2', '#3', '#6', '#8', '#9']
  Generated 296 candidate scenarios
  2 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [14:26:08] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Twin_Minion', 'Minion'})

### [14:26:52] Executed #4 -> Twin_Minion (EVIL)

#### [14:26:58] Solver Output
Scenarios: 1/37
Definite evil: ['#4', '#5', '#7']
Definite good: ['#1', '#2', '#3', '#6', '#8', '#9']
  Generated 37 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [14:26:58] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Minion'})

### [14:27:46] Executed #5 -> Minion (EVIL)

## [14:27:51] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. PD check on #1 narrowed to 2 scenarios. Bishop#3 corrupted by PD#2.


---

# New Game — 2026-03-28 14:33:12
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Bard, Empress, Slayer, Scout, Architect, Oracle
- Outcasts: Wretch, Doppelganger, Drunk
- Minions: Shaman, Chancellor
- Demons: Baa

### [14:35:14] Revealed #1 Bard
Info: {'corruption_distance': 1}

### [14:35:20] Revealed #2 Bard
Info: {'corruption_distance': 1}

### [14:35:25] Revealed #3 Slayer
Info: {}

### [14:35:25] Revealed #4 Architect
Info: {'side': 'right'}

### [14:35:25] Revealed #5 Wretch
Info: {}

### [14:35:30] Revealed #6 Scout
Info: {'evil_role': 'Chancellor', 'distance': 1}

### [14:35:30] Revealed #7 Scout
Info: {'evil_role': 'Chancellor', 'distance': 1}

### [14:35:30] Revealed #8 Scout
Info: {'evil_role': 'Baa', 'distance': 1}

### [14:35:35] Revealed #9 Oracle
Info: {'targets': [4, 5], 'minion_role': 'Shaman'}

#### [14:35:42] Solver Output
Scenarios: 108/24600
Definite good: ['#6', '#7', '#9']
Evil probabilities: #1=59%, #8=59%, #2=50%, #4=50%, #5=44%, #3=37%
  Generated 24600 candidate scenarios
  108 scenarios survived validation
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 8]

#### [14:35:42] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#1']
Reason: Target #1 is 59% evil (adjusted 0.54)
WARNING: Corruption risk: 9% -- Slayer ability disabled if corrupted

### [14:36:49] Ability used at #3

#### [14:36:55] Solver Output
Scenarios: 101/7144
Definite evil: ['#1']
Definite good: ['#3', '#9']
Evil probabilities: #2=70%, #5=41%, #4=30%, #8=30%, #6=20%, #7=10%
  Generated 7144 candidate scenarios
  101 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #3 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 6, 7, 8]

#### [14:36:55] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (30% good Drunk (corrupted), 28% evil Baa, 27% evil Chancellor).
WARNING: Execution lookahead override -- immediate hit chance is 70%, but all reveal branches still lead to a forced win.

### [14:37:44] Executed #2 -> Baa (EVIL)

#### [14:37:49] Solver Output
Scenarios: 28/563
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#8', '#9']
Evil probabilities: #5=64%, #6=18%, #7=18%
  Generated 563 candidate scenarios
  28 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [5, 6, 7]

#### [14:37:49] Recommendation
Action: **EXECUTE** #6
Reason: No reveals available. #6 is 18% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 18% confident (budget: 2 wrong execs)
WARNING: Low confidence (18%) -- consider gathering more info

### [14:38:54] Executed #6 -> GOOD (WRONG!)

#### [14:39:02] Solver Output
Scenarios: 23/501
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#6', '#8', '#9']
Evil probabilities: #5=78%, #7=22%
  Generated 501 candidate scenarios
  23 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Unknown'})
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [5, 7]

#### [14:39:02] Recommendation
Action: **ERROR** #7
Reason: #7 is 22% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 22% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 22% < 80% threshold. Consider manual override if you have extra information.

### [14:40:42] Executed #7 -> GOOD (WRONG!)

## [14:40:53] GAME OVER — LOSS
Final HP: 0
Notes: SOLVER BUG: #9 Chancellor disguised as Oracle, claimed #4/#5 is Shaman. But Slayer confirmed #1=Shaman. Oracle info was a lie but solver marked #9 DEFINITELY GOOD. Solver failed to propagate confirmed-Shaman-at-#1. Wrong exec #6 (good Scout) then #7 (good Scout) killed us.


---

# New Game — 2026-03-28 15:14:27
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Bishop, Jester, Baker, Poet, Bard
- Outcasts: Plague_Doctor, Wretch
- Minions: Puppeteer, Poisoner
- Demons: Lilis

### [15:15:40] Revealed #1 Jester
Info: {}

### [15:15:45] Revealed #2 Wretch
Info: {}

### [15:15:51] Revealed #3 Poet
Info: {'good_position': 7, 'good_role': 'Plague_Doctor', 'copied_role': 'Medium'}

### [15:15:56] Revealed #4 Baker
Info: {'original_role': 'original'}

### [15:16:58] Revealed #6 Bishop
Info: {'targets': [4, 7], 'types': ['Villager', 'Minion']}

### [15:16:58] Revealed #7 Plague_Doctor
Info: {}

### [15:16:58] Revealed #8 Jester
Info: {}

### [15:16:59] Revealed #9 Baker
Info: {'original_role': 'Jester'}

#### [15:17:05] Solver Output
Scenarios: 75/2290
Definite good: ['#5']
Evil probabilities: #2=92%, #1=63%, #9=61%, #8=60%, #6=59%, #3=35%, #4=20%, #7=11%
  Generated 2290 candidate scenarios
  75 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [15:17:05] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.976 (adjusted 1.976) | timing x1.00

### [15:18:14] Ability used at #7

#### [15:18:20] Solver Output
Scenarios: 21/2290
Definite good: ['#5']
Evil probabilities: #2=71%, #4=62%, #1=57%, #8=57%, #9=57%, #3=48%, #7=38%, #6=10%
  Generated 2290 candidate scenarios
  21 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [15:18:20] Recommendation
Action: **USE_ABILITY** #8 (Jester) -> targets ['#1', '#3', '#6']
Reason: Expected posterior 9.8 scenarios (adjusted 10.5, info gain 1.005 bits) | timing x1.00
WARNING: Corruption risk: 14%

### [15:19:31] Revealed #8 Jester
Info: {'targets': [1, 3, 6], 'evil_count': 0}

### [15:19:36] Ability used at #8

#### [15:19:41] Solver Output
Scenarios: 10/2290
Definite good: ['#5']
Evil probabilities: #2=70%, #8=70%, #4=60%, #9=60%, #3=50%, #1=40%, #7=40%, #6=10%
  Generated 2290 candidate scenarios
  10 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [15:19:41] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#3', '#6', '#8']
Reason: Expected posterior 3.6 scenarios (adjusted 3.8, info gain 1.415 bits) | timing x1.00
WARNING: Corruption risk: 10%

### [15:20:40] Revealed #1 Jester
Info: {'targets': [3, 6, 8], 'evil_count': 3}

### [15:20:46] Ability used at #1

#### [15:20:46] Solver Output
Scenarios: 3/2290
Definite good: ['#5']
Evil probabilities: #3=67%, #7=67%, #8=67%, #9=67%, #1=33%, #2=33%, #4=33%, #6=33%
  Generated 2290 candidate scenarios
  3 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [15:20:46] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (67% evil Lilis, 33% good Poet).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [15:21:28] Executed #3 -> Lilis (EVIL)

#### [15:21:29] Solver Output
Scenarios: 2/265
Definite evil: ['#3', '#7']
Definite good: ['#2', '#4', '#5']
Evil probabilities: #1=50%, #6=50%, #8=50%, #9=50%
  Generated 265 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner', 'Puppeteer'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 6, 8, 9]

#### [15:21:29] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Poisoner', 'Puppeteer'})

### [15:22:12] Executed #7 -> Puppeteer (EVIL)

#### [15:22:12] Solver Output
Scenarios: 1/14
Definite evil: ['#3', '#6', '#7', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#9']
  Generated 14 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [15:22:12] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [15:22:53] Executed #6 -> Puppet (EVIL)

#### [15:22:53] Solver Output
Scenarios: 1/7
Definite evil: ['#3', '#6', '#7', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#9']
  Generated 7 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [15:22:53] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [15:23:33] Executed #8 -> Poisoner (EVIL)

## [15:23:41] GAME OVER — WIN
Final HP: 6
Notes: 4 evils with Puppeteer+Puppet. Lilis killed #5. Jester#1 found 3 evils at 3,6,8. Poisoner corrupted Baker#9.


---

# New Game — 2026-03-28 15:27:55
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Poet, Gemcrafter, Scout, Hunter, Oracle
- Outcasts: Plague_Doctor
- Minions: 
- Demons: Pooka

### [15:29:33] Revealed #1 Architect
Info: {'side': 'left'}

### [15:29:33] Revealed #2 Poet
Info: {'targets': [1, 4, 6], 'types': ['Demon', 'Villager', 'Outcast'], 'copied_role': 'Bishop'}

### [15:29:34] Revealed #3 Oracle
Info: {'targets': [3, 5], 'minion_role': 'Witch'}

### [15:29:42] Revealed #4 Scout
Info: {'evil_role': 'Pooka', 'distance': 0}

### [15:29:48] Revealed #5 Hunter
Info: {'distance': 1}

### [15:29:48] Revealed #6 Gemcrafter
Info: {'good_position': 7}

### [15:29:48] Revealed #7 Plague_Doctor
Info: {}

#### [15:29:55] Solver Output
Scenarios: 1/31
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7']
  Generated 31 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [15:29:55] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [15:30:46] Executed #2 -> Pooka (EVIL)

## [15:30:46] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP, 1-evil game. Solver locked Pooka at #2 in 1 scenario. Pooka corrupted #1,#3,#5.


---

# New Game — 2026-03-28 15:34:48
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Scout, Architect, Confessor, Judge, Bard
- Outcasts: Drunk, Bombardier, Wretch
- Minions: Chancellor
- Demons: Baa

### [15:36:22] Revealed #1 Wretch
Info: {}

### [15:36:23] Revealed #2 Scout
Info: {'evil_role': 'Chancellor', 'distance': 2}

### [15:36:23] Revealed #3 Judge
Info: {}

### [15:36:23] Revealed #4 Bombardier
Info: {}

### [15:36:23] Revealed #5 Bombardier
Info: {}

### [15:36:30] Revealed #6 Confessor
Info: {'dizzy': False}

### [15:36:30] Revealed #7 Druid
Info: {}

### [15:36:30] Revealed #8 Bard
Info: {'corruption_distance': -1}

### [15:36:40] Revealed #6 Confessor
Info: {'dizzy': True}

#### [15:36:46] Solver Output
Scenarios: 4/316
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8']
  Generated 316 candidate scenarios
  4 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Baa'})
    #6 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [15:36:46] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 4 scenarios (roles: {'Chancellor', 'Baa'})

### [15:37:27] Executed #4 -> Chancellor (EVIL)

#### [15:37:27] Solver Output
Scenarios: 2/37
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8']
  Generated 37 candidate scenarios
  2 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #6 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [15:37:27] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Baa'})

### [15:38:14] Executed #6 -> Baa (EVIL)

## [15:38:14] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Solver locked both evils in 4 scenarios. Confessor dizzy = Baa disguise.


---

# New Game — 2026-03-28 15:42:53
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Knitter, Enlightened, Slayer, Confessor, Jester, Fortune_Teller
- Outcasts: Wretch, Plague_Doctor
- Minions: Chancellor, Poisoner
- Demons: Lilis

### [15:43:53] Revealed #1 Jester
Info: {}

### [15:43:53] Revealed #2 Enlightened
Info: {'direction': 'ccw'}

### [15:43:54] Revealed #3 Knitter
Info: {'evil_pairs': 2}

### [15:43:54] Revealed #4 Enlightened
Info: {'direction': 'ccw'}

### [15:44:43] Revealed #5 Wretch
Info: {}

### [15:44:43] Revealed #6 Fortune_Teller
Info: {}

### [15:44:44] Revealed #7 Confessor
Info: {'dizzy': True}

### [15:44:44] Revealed #8 Plague_Doctor
Info: {}

#### [15:44:51] Solver Output
Scenarios: 109/2218
Definite good: ['#8', '#9']
Evil probabilities: #4=72%, #2=70%, #7=63%, #6=31%, #3=23%, #1=22%, #5=19%
  Generated 2218 candidate scenarios
  109 scenarios survived validation
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [15:44:51] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.615 (adjusted 1.615) | timing x1.00

### [15:45:46] Ability used at #8

#### [15:45:46] Solver Output
Scenarios: 56/2218
Definite good: ['#3', '#8', '#9']
Evil probabilities: #4=89%, #7=88%, #2=64%, #1=29%, #5=27%, #6=4%
  Generated 2218 candidate scenarios
  56 scenarios survived validation
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7]

#### [15:45:46] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#3', '#6', '#7']
Reason: Expected posterior 25.6 scenarios (adjusted 27.2, info gain 1.040 bits) | timing x1.00
WARNING: Corruption risk: 12%

### [15:46:49] Revealed #1 Jester
Info: {'targets': [3, 6, 7], 'evil_count': 2}

### [15:46:49] Ability used at #1

#### [15:46:50] Solver Output
Scenarios: 25/2218
Definite evil: ['#4']
Definite good: ['#3', '#8', '#9']
Evil probabilities: #7=92%, #1=64%, #2=28%, #5=8%, #6=8%
  Generated 2218 candidate scenarios
  25 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Lilis', 'Poisoner'})
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 7]

#### [15:46:50] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 25 scenarios (roles: {'Chancellor', 'Lilis', 'Poisoner'})

### [15:47:30] Executed #4 -> Poisoner (EVIL)

#### [15:47:31] Solver Output
Scenarios: 16/213
Definite evil: ['#4']
Definite good: ['#3', '#8', '#9']
Evil probabilities: #7=88%, #1=69%, #2=19%, #5=12%, #6=12%
  Generated 213 candidate scenarios
  16 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 7]

#### [15:47:31] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#2', '#3']
Reason: Entropy 1.000 (adjusted 0.906) | timing x1.00
WARNING: Corruption risk: 19%

### [15:48:27] Revealed #6 Fortune Teller
Info: {'targets': [2, 3], 'has_evil': True}

### [15:48:28] Ability used at #6

#### [15:48:28] Solver Output
Scenarios: 8/213
Definite evil: ['#4', '#7']
Definite good: ['#3', '#5', '#8', '#9']
Evil probabilities: #1=38%, #2=38%, #6=25%
  Generated 213 candidate scenarios
  8 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Lilis'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 6]

#### [15:48:28] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 8 scenarios (roles: {'Chancellor', 'Lilis'})

### [15:49:07] Executed #7 -> Chancellor (EVIL)

#### [15:49:08] Solver Output
Scenarios: 3/25
Definite evil: ['#4', '#7']
Definite good: ['#3', '#5', '#8', '#9']
Evil probabilities: #1=33%, #2=33%, #6=33%
  Generated 25 candidate scenarios
  3 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 6]

#### [15:49:08] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (33% good Jester, 33% good Jester (corrupted), 33% evil Lilis).
WARNING: Execution lookahead override -- immediate hit chance is 33%, but all reveal branches still lead to a forced win.

### [15:50:00] Executed #1 -> GOOD (WRONG!)

#### [15:50:00] Solver Output
Scenarios: 1/21
Definite evil: ['#2', '#4', '#7']
Definite good: ['#1', '#3', '#5', '#6', '#8', '#9']
  Generated 21 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [15:50:00] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [15:50:46] Executed #2 -> Lilis (EVIL)

## [15:50:46] GAME OVER — WIN
Final HP: 1
Notes: 1HP clutch win. Lilis night killed #9. Poisoner corrupted #3(Knitter)+#1(Jester). Wrong exec #1(Jester corrupted) left 1HP, solver narrowed Lilis to #2.


---

# New Game — 2026-03-28 15:55:04
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Enlightened, Witness, Bishop, Lover, Gemcrafter, Architect, Jester
- Outcasts: Drunk, Doppelganger, Wretch
- Minions: Chancellor, Minion
- Demons: Baa

### [15:56:51] Revealed #1 Jester
Info: {}

### [15:56:52] Revealed #2 Lover
Info: {'evil_adjacent': 2}

### [15:56:52] Revealed #3 Alchemist
Info: {'cured_count': 0}

### [15:56:52] Revealed #4 Architect
Info: {'side': 'left'}

### [15:56:52] Revealed #5 Witness
Info: {'affected_position': 6}

### [15:56:52] Revealed #6 Wretch
Info: {}

### [15:56:52] Revealed #7 Lover
Info: {'evil_adjacent': 0}

### [15:56:53] Revealed #8 Enlightened
Info: {'direction': 'cw'}

### [15:56:53] Revealed #9 Gemcrafter
Info: {'good_position': 1}

#### [15:57:01] Solver Output
Scenarios: 160/24600
Definite good: ['#1', '#3', '#4', '#6', '#9']
Evil probabilities: #2=81%, #5=75%, #8=75%, #7=69%
  Generated 24600 candidate scenarios
  160 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 5, 7, 8]

#### [15:57:01] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#2', '#3', '#7']
Reason: Expected posterior 80.0 scenarios (adjusted 80.0, info gain 1.000 bits) | timing x1.00

### [15:57:53] Revealed #1 Jester
Info: {'targets': [2, 3, 7], 'evil_count': 2}

### [15:57:53] Ability used at #1

#### [15:57:55] Solver Output
Scenarios: 80/24600
Definite evil: ['#2', '#7']
Definite good: ['#1', '#3', '#4', '#6', '#9']
Evil probabilities: #5=50%, #8=50%
  Generated 24600 candidate scenarios
  80 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa', 'Chancellor', 'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Baa', 'Minion', 'Chancellor'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [5, 8]

#### [15:57:55] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 80 scenarios (roles: {'Baa', 'Chancellor', 'Minion'})

### [15:58:45] Executed #2 -> Baa (EVIL)

#### [15:58:45] Solver Output
Scenarios: 20/2620
Definite evil: ['#2', '#7']
Definite good: ['#1', '#3', '#4', '#6', '#9']
Evil probabilities: #5=50%, #8=50%
  Generated 2620 candidate scenarios
  20 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [5, 8]

#### [15:58:45] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 20 scenarios (roles: {'Chancellor', 'Minion'})

### [15:59:28] Executed #7 -> Chancellor (EVIL)

#### [15:59:28] Solver Output
Scenarios: 10/229
Definite evil: ['#2', '#7']
Definite good: ['#1', '#3', '#4', '#6', '#9']
Evil probabilities: #5=50%, #8=50%
  Generated 229 candidate scenarios
  10 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [5, 8]

#### [15:59:28] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (50% good Drunk (corrupted), 50% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [16:00:13] Executed #5 -> Minion (EVIL)

## [16:00:14] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Jester found 2 evils at #2,#7. 50/50 between #5/#8 for last evil, hit #5=Minion. Asc41 COMPLETE (7/7 villages).


---

# New Game — 2026-03-28 16:14:26
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Enlightened, Knitter, Judge, Gemcrafter, Bishop, Fortune_Teller
- Outcasts: Wretch, Drunk, Plague_Doctor
- Minions: Chancellor
- Demons: Baa

### [16:15:41] Revealed #1 Judge
Info: {}

### [16:15:41] Revealed #2 Gemcrafter
Info: {'good_position': 5}

### [16:15:41] Revealed #3 Plague_Doctor
Info: {}

### [16:15:41] Revealed #4 Gemcrafter
Info: {'good_position': 1}

### [16:15:42] Revealed #5 Fortune_Teller
Info: {}

### [16:15:42] Revealed #6 Knitter
Info: {'evil_pairs': 1}

### [16:15:42] Revealed #7 Bishop
Info: {'targets': [2, 7, 8], 'types': ['Outcast', 'Minion', 'Villager']}

### [16:15:42] Revealed #8 Slayer
Info: {}

#### [16:15:49] Solver Output
Scenarios: 80/2400
Evil probabilities: #8=51%, #6=40%, #7=34%, #1=32%, #5=18%, #3=10%, #2=8%, #4=8%
  Generated 2400 candidate scenarios
  80 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [16:15:49] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#7']
Reason: Entropy 2.101 (adjusted 2.101) | timing x1.00

### [16:16:28] Ability used at #3

#### [16:16:29] Solver Output
Scenarios: 41/2400
Definite good: ['#4']
Evil probabilities: #7=59%, #8=59%, #1=24%, #6=24%, #5=22%, #3=10%, #2=2%
  Generated 2400 candidate scenarios
  41 scenarios survived validation
    #4 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 6, 7, 8]

#### [16:16:29] Recommendation
Action: **USE_ABILITY** #5 (Fortune Teller) -> targets ['#6', '#7']
Reason: Entropy 1.000 (adjusted 0.829) | follow-up bonus 0.382 | timing x1.00
WARNING: Corruption risk: 34%

### [16:17:10] Revealed #5 Fortune Teller
Info: {'targets': [6, 7], 'has_evil': True}

### [16:17:10] Ability used at #5

#### [16:17:10] Solver Output
Scenarios: 21/2400
Definite good: ['#4']
Evil probabilities: #7=48%, #8=48%, #1=38%, #6=29%, #3=19%, #5=14%, #2=5%
  Generated 2400 candidate scenarios
  21 scenarios survived validation
    #4 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 6, 7, 8]

#### [16:17:10] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#7']
Reason: Target #7 is 48% evil (adjusted 0.45)
WARNING: Corruption risk: 5% -- Slayer ability disabled if corrupted

### [16:17:46] Ability used at #8

#### [16:17:47] Solver Output
Scenarios: 15/2400
Definite good: ['#4']
Evil probabilities: #8=67%, #6=40%, #3=27%, #7=27%, #5=20%, #1=13%, #2=7%
  Generated 2400 candidate scenarios
  15 scenarios survived validation
    #4 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 6, 7, 8]

#### [16:17:47] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#2']
Reason: Expected posterior 10.5 scenarios (adjusted 12.6, info gain 0.248 bits) | timing x1.00
WARNING: Corruption risk: 40% -- corrupted Judge results are unreliable

### [16:18:25] Revealed #1 Judge
Info: {'target': 2, 'is_lying': True}

### [16:18:25] Ability used at #1

#### [16:18:26] Solver Output
Scenarios: 11/2400
Definite good: ['#3', '#4']
Evil probabilities: #8=91%, #7=36%, #5=27%, #1=18%, #6=18%, #2=9%
  Generated 2400 candidate scenarios
  11 scenarios survived validation
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 7, 8]

#### [16:18:26] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (73% evil Chancellor, 18% evil Baa, 9% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 91%, but all reveal branches still lead to a forced win.

### [16:19:16] Executed #8 -> GOOD (WRONG!)

#### [16:19:17] Solver Output
Scenarios: 1/1724
Definite evil: ['#2', '#5']
Definite good: ['#1', '#3', '#4', '#6', '#7', '#8']
  Generated 1724 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [16:19:17] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Chancellor'})

### [16:20:06] Executed #2 -> Chancellor (EVIL)

### [16:21:00] Executed #5 -> Baa (EVIL)

## [16:21:00] GAME OVER — WIN
Final HP: 8
Notes: 8HP. Drunk wrong exec cost only 2HP. Wrong exec revealed corruption, solver locked both evils in 1 scenario.


---

# New Game — 2026-03-28 16:44:35
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Scout, Judge, Empress, Bishop, Baker, Alchemist
- Outcasts: Wretch
- Minions: Witch
- Demons: Pooka

### [16:45:52] Revealed #1 Medium
Info: {'good_position': 3, 'good_role': 'Bishop'}

### [16:45:52] Revealed #2 Judge
Info: {}

### [16:45:52] Revealed #3 Bishop
Info: {'targets': [5, 6, 8], 'types': ['Minion', 'Villager', 'Outcast']}

### [16:45:53] Revealed #4 Alchemist
Info: {'cured_count': 2}

### [16:45:53] Revealed #5 Baker
Info: {'original_role': 'original'}

### [16:45:53] Revealed #6 Empress
Info: {'targets': [1, 4, 8]}

### [16:45:53] Revealed #7 Wretch
Info: {}

#### [16:46:01] Solver Output
Scenarios: 5/56
Definite good: ['#8']
Evil probabilities: #5=80%, #1=20%, #2=20%, #3=20%, #4=20%, #6=20%, #7=20%
  Generated 56 candidate scenarios
  5 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [16:46:01] Recommendation
Action: **USE_ABILITY** #2 (Judge) -> targets ['#3']
Reason: Expected posterior 3.0 scenarios (adjusted 3.3, info gain 0.599 bits) | timing x1.00
WARNING: Corruption risk: 20% -- corrupted Judge results are unreliable

### [16:46:38] Revealed #2 Judge
Info: {'target': 3, 'is_lying': False}

### [16:46:38] Ability used at #2

#### [16:46:38] Solver Output
Scenarios: 3/56
Definite good: ['#2', '#4', '#8']
Evil probabilities: #5=67%, #1=33%, #3=33%, #6=33%, #7=33%
  Generated 56 candidate scenarios
  3 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 6, 7]

#### [16:46:38] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (67% evil Pooka, 33% good Baker).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [16:47:20] Executed #5 -> GOOD (WRONG!)

#### [16:47:20] Solver Output
Scenarios: 1/42
Definite evil: ['#1', '#3']
Definite good: ['#2', '#4', '#5', '#6', '#7', '#8']
  Generated 42 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Witch'})
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [16:47:20] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Witch'})

### [16:48:02] Executed #1 -> Witch (EVIL)

### [16:48:47] Executed #3 -> Pooka (EVIL)

## [16:48:47] GAME OVER — WIN
Final HP: 5
Notes: 5HP. Witch blocked #8. Wrong exec Baker#5 then locked Witch#1+Pooka#3. Pooka corrupted Judge#2+Alchemist#4.


---

# New Game — 2026-03-28 16:52:52
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Lover, Poet, Architect, Scout, Alchemist
- Outcasts: Doppelganger, Bombardier
- Minions: Witch
- Demons: Pooka

### [16:53:48] Revealed #1 Bombardier
Info: {}

### [16:53:49] Revealed #2 Enlightened
Info: {'direction': 'ccw'}

### [16:53:49] Revealed #3 Alchemist
Info: {'cured_count': 2}

### [16:53:49] Revealed #4 Lover
Info: {'evil_adjacent': 1}

### [16:53:49] Revealed #5 Scout
Info: {'evil_role': 'Witch', 'distance': 1}

### [16:53:49] Revealed #6 Architect
Info: {'side': 'right'}

### [16:53:49] Revealed #7 Scout
Info: {'evil_role': 'Pooka', 'distance': 3}

#### [16:53:57] Solver Output
Scenarios: 4/350
Definite evil: ['#1', '#4']
Definite good: ['#2', '#3', '#5', '#6', '#7', '#8']
  Generated 350 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Witch'})
    #4 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [16:53:57] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Witch'})

### [16:54:40] Executed #1 -> Witch (EVIL)

### [16:55:16] Executed #4 -> Pooka (EVIL)

## [16:55:17] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Both evils locked in 4 scenarios from card info alone.


---

# New Game — 2026-03-28 16:59:26
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Medium, Lover, Scout, Hunter, Druid, Alchemist
- Outcasts: Plague_Doctor, Bombardier
- Minions: Minion
- Demons: Baa

### [17:00:42] Revealed #1 Hunter
Info: {'distance': 2}

### [17:00:42] Revealed #2 Plague_Doctor
Info: {}

### [17:00:42] Revealed #3 Lover
Info: {'evil_adjacent': 2}

### [17:00:42] Revealed #4 Medium
Info: {'good_position': 5, 'good_role': 'Druid'}

### [17:00:42] Revealed #5 Druid
Info: {}

### [17:00:43] Revealed #6 Confessor
Info: {'dizzy': True}

### [17:00:43] Revealed #7 Alchemist
Info: {'cured_count': 1}

### [17:00:43] Revealed #8 Scout
Info: {'evil_role': 'Baa', 'distance': 3}

#### [17:00:50] Solver Output
Scenarios: 8/266
Definite evil: ['#3', '#6']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#8']
  Generated 266 candidate scenarios
  8 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Baa', 'Minion'})
    #6 is DEFINITELY EVIL (possible roles: {'Baa', 'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [17:00:50] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 8 scenarios (roles: {'Baa', 'Minion'})

### [17:01:48] Executed #3 -> Baa (EVIL)

### [17:02:45] Executed #6 -> Minion (EVIL)

## [17:02:46] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Both evils locked in 8 scenarios from card info alone. Confessor dizzy = Minion disguise.


---

# New Game — 2026-03-28 17:07:00
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Knitter, Baker, Judge, Confessor, Empress
- Outcasts: Bombardier, Doppelganger
- Minions: Poisoner
- Demons: Lilis

### [17:08:07] Revealed #1 Baker
Info: {'original_role': 'original'}

### [17:08:07] Revealed #2 Baker
Info: {'original_role': 'Knitter'}

### [17:08:07] Revealed #3 Bombardier
Info: {}

### [17:08:07] Revealed #4 Bombardier
Info: {}

### [17:09:15] Revealed #5 Judge
Info: {}

### [17:09:15] Revealed #6 Scout
Info: {'evil_role': 'Poisoner', 'distance': 2}

### [17:09:16] Revealed #7 Knitter
Info: {'evil_pairs': 0}

### [17:09:16] Revealed #8 Empress
Info: {'targets': [3, 4, 9]}

#### [17:09:16] Solver Output
Scenarios: 34/654
Definite good: ['#7', '#8', '#9']
Evil probabilities: #3=76%, #1=35%, #2=24%, #4=24%, #6=24%, #5=18%
  Generated 654 candidate scenarios
  34 scenarios survived validation
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6]

#### [17:09:16] Recommendation
Action: **USE_ABILITY** #5 (Judge) -> targets ['#6']
Reason: Expected posterior 20.5 scenarios (adjusted 22.6, info gain 0.588 bits) | timing x1.00
WARNING: Corruption risk: 21% -- corrupted Judge results are unreliable

### [17:10:08] Revealed #5 Judge
Info: {'target': 6, 'is_lying': True}

### [17:10:08] Ability used at #5

#### [17:10:08] Solver Output
Scenarios: 20/654
Definite good: ['#1', '#7', '#8', '#9']
Evil probabilities: #3=70%, #6=40%, #2=30%, #4=30%, #5=30%
  Generated 654 candidate scenarios
  20 scenarios survived validation
    #1 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6]

#### [17:10:08] Recommendation
Action: **EXECUTE** #6
Reason: No reveals available. #6 is 40% likely evil (HP=6, budget=1 wrong execs)
WARNING: Probabilistic execution -- 40% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #6 (40%) despite low confidence — Bombardier candidate(s) [3, 4] risk instant game loss if executed first.

### [17:11:06] Executed #6 -> GOOD (WRONG!)

#### [17:11:06] Solver Output
Scenarios: 12/498
Definite good: ['#1', '#6', '#7', '#8', '#9']
Evil probabilities: #2=50%, #3=50%, #4=50%, #5=50%
  Generated 498 candidate scenarios
  12 scenarios survived validation
    #1 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5]

#### [17:11:06] Recommendation
Action: **ERROR** #2
Reason: #2 is 50% likely evil but HP too low to risk (HP=1, cost=5). Need more info.
WARNING: Probabilistic execution -- 50% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [17:12:22] Executed #2 -> GOOD (WRONG!)

## [17:12:23] GAME OVER — LOSS
Final HP: 0
Notes: 50/50 coin flip loss. Poisoner at #3 disguised as Bombardier. Lilis at #5 disguised as Judge. Solver had 0 budget, 50% on #2 which was wrong.


---

# New Game — 2026-03-28 19:05:36
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Gemcrafter, Scout, Architect, Bard, Bishop, Medium, Poet
- Outcasts: Bombardier, Plague_Doctor
- Minions: Poisoner, Minion
- Demons: Pooka

### [19:07:13] Revealed #1 Medium
Info: {'good_position': 7, 'good_role': 'Bombardier'}

### [19:07:13] Revealed #2 Architect
Info: {'side': 'left'}

### [19:07:13] Revealed #3 Bard
Info: {'corruption_distance': 2}

### [19:07:14] Revealed #4 Scout
Info: {'evil_role': 'Pooka', 'distance': 1}

### [19:07:14] Revealed #5 Gemcrafter
Info: {'good_position': 7}

### [19:07:14] Revealed #6 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 7}

### [19:07:14] Revealed #7 Bombardier
Info: {}

### [19:07:15] Revealed #8 Bishop
Info: {'targets': [4, 9, 10], 'types': ['Outcast', 'Villager', 'Minion']}

### [19:07:15] Revealed #9 Plague_Doctor
Info: {}

### [19:07:15] Revealed #10 Scout
Info: {'evil_role': 'Poisoner', 'distance': 1}

#### [19:07:27] Solver Output
Scenarios: 6/4632
Definite good: ['#5']
Evil probabilities: #4=83%, #7=83%, #1=33%, #2=17%, #3=17%, #6=17%, #8=17%, #9=17%, #10=17%
  Generated 4632 candidate scenarios
  6 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9, 10]

#### [19:07:27] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#1']
Reason: Entropy 2.252 (adjusted 2.252) | timing x1.00

### [19:08:29] Ability used at #9

#### [19:08:30] Solver Output
Scenarios: 2/4632
Definite evil: ['#1', '#4', '#7']
Definite good: ['#2', '#3', '#5', '#6', '#8', '#9', '#10']
  Generated 4632 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #7 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [19:08:30] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [19:09:06] Executed #1 -> Pooka (EVIL)

### [19:09:26] Executed #4 -> Poisoner (EVIL)

### [19:10:38] Executed #7 -> Minion (EVIL)

## [19:10:38] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. PD check on #1 cracked it — all 3 evils locked in 2 scenarios.


---

# New Game — 2026-03-28 19:15:22
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Dreamer, Architect, Scout, Knight, Alchemist, Baker
- Outcasts: Plague_Doctor
- Minions: Minion, Poisoner
- Demons: Pooka

### [19:16:55] Revealed #1 Scout
Info: {'evil_role': 'Minion', 'distance': 3}

### [19:16:55] Revealed #2 Architect
Info: {'side': 'left'}

### [19:16:55] Revealed #3 Plague_Doctor
Info: {}

### [19:16:56] Revealed #4 Knight
Info: {}

### [19:16:56] Revealed #5 Alchemist
Info: {'cured_count': 1}

### [19:16:56] Revealed #6 Architect
Info: {'side': 'right'}

### [19:16:56] Revealed #7 Baker
Info: {'original_role': 'original'}

### [19:16:57] Revealed #8 Bard
Info: {'corruption_distance': -1}

### [19:16:57] Revealed #9 Dreamer
Info: {}

#### [19:17:17] Solver Output
Scenarios: 175/2465
Definite good: ['#3']
Evil probabilities: #2=67%, #9=51%, #1=43%, #6=38%, #7=30%, #8=30%, #4=27%, #5=14%
  Generated 2465 candidate scenarios
  175 scenarios survived validation
    #3 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7, 8, 9]

#### [19:17:17] Recommendation
Action: **EXECUTE** #4
Reason: Knight check: #4 is 27% evil, 11% corruption risk. Expected HP cost: 0.7 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 11% -- corrupted Knight loses immunity + 4 extra damage

### [19:18:38] Executed #4 -> GOOD (WRONG!)

#### [19:18:38] Solver Output
Scenarios: 128/1604
Definite good: ['#3', '#4']
Evil probabilities: #2=62%, #9=58%, #1=55%, #6=45%, #7=32%, #8=30%, #5=17%
  Generated 1604 candidate scenarios
  128 scenarios survived validation
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 7, 8, 9]

#### [19:18:38] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#8']
Reason: Entropy 2.072 (adjusted 2.072) | timing x1.00

### [19:19:43] Ability used at #3

#### [19:19:44] Solver Output
Scenarios: 40/1604
Definite evil: ['#1']
Definite good: ['#3', '#4', '#8']
Evil probabilities: #2=80%, #9=50%, #7=42%, #6=20%, #5=8%
  Generated 1604 candidate scenarios
  40 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka', 'Minion', 'Poisoner'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 5, 6, 7, 9]

#### [19:19:44] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 40 scenarios (roles: {'Pooka', 'Minion', 'Poisoner'})

### [19:20:50] Executed #1 -> Pooka (EVIL)

#### [19:20:50] Solver Output
Scenarios: 11/220
Definite evil: ['#1']
Definite good: ['#3', '#4', '#8']
Evil probabilities: #2=82%, #9=55%, #7=36%, #6=18%, #5=9%
  Generated 220 candidate scenarios
  11 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 5, 6, 7, 9]

#### [19:20:50] Recommendation
Action: **USE_ABILITY** #9 (Dreamer) -> targets ['#7']
Reason: Entropy 0.946 (adjusted 0.731) | timing x1.00
WARNING: Corruption risk: 45%

### [19:21:49] Revealed #9 Dreamer
Info: {'target': 7, 'evil_role': 'Pooka'}

### [19:21:50] Ability used at #9

#### [19:21:50] Solver Output
Scenarios: 11/220
Definite evil: ['#1']
Definite good: ['#3', '#4', '#8']
Evil probabilities: #2=82%, #9=55%, #7=36%, #6=18%, #5=9%
  Generated 220 candidate scenarios
  11 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 5, 6, 7, 9]

#### [19:21:50] Recommendation
Action: **EXECUTE** #2
Reason: No reveals available. #2 is 82% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 82% confident (budget: 2 wrong execs)

### [19:22:54] Executed #2 -> Minion (EVIL)

#### [19:22:54] Solver Output
Scenarios: 9/41
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#6', '#8']
Evil probabilities: #7=44%, #9=44%, #5=11%
  Generated 41 candidate scenarios
  9 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY EVIL (possible roles: {'Minion'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 7, 9]

#### [19:22:54] Recommendation
Action: **EXECUTE** #7
Reason: No reveals available. #7 is 44% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 44% confident (budget: 2 wrong execs)
WARNING: Low confidence (44%) -- consider gathering more info

#### [00:02:40] Solver Output
Scenarios: 9/41
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#6', '#8']
Evil probabilities: #7=44%, #9=44%, #5=11%
  Generated 41 candidate scenarios
  9 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY EVIL (possible roles: {'Minion'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 7, 9]

#### [00:02:40] Recommendation
Action: **EXECUTE** #7
Reason: No reveals available. #7 is 44% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 44% confident (budget: 2 wrong execs)
WARNING: Low confidence (44%) -- consider gathering more info

## [00:04:09] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect. Knight immunity saved wrong exec. Corrupted #8 Bard + #9 Dreamer.

## [00:06:18] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect. Knight immunity saved wrong exec. Corrupted #8 Bard + #9 Dreamer.


---

# New Game — 2026-03-29 00:14:02
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Druid, Poet, Knight, Gemcrafter, Alchemist, Baker, Fortune_Teller
- Outcasts: Doppelganger, Wretch
- Minions: Chancellor, Puppeteer
- Demons: Lilis

### [00:15:51] Revealed #1 Gemcrafter
Info: {'good_position': 7}

### [00:15:54] Revealed #2 Wretch
Info: {}

### [00:15:57] Revealed #3 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 9}

### [00:16:00] Revealed #4 Baker
Info: {'original_role': 'Fortune_Teller'}

### [00:18:15] Revealed #5 Alchemist
Info: {'cured_count': 0}

### [00:18:19] Revealed #7 Empress
Info: {'targets': [2, 6, 9]}

### [00:18:23] Revealed #8 Fortune_Teller
Info: {}

### [00:18:26] Revealed #9 Empress
Info: {'targets': [1, 5, 7]}

#### [00:18:41] Solver Output
Scenarios: 22/3596
Definite evil: ['#3', '#4']
Definite good: ['#6', '#7', '#9']
Evil probabilities: #5=77%, #8=55%, #2=45%, #1=23%
  Generated 3596 candidate scenarios
  22 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis', 'Chancellor'})
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer', 'Chancellor'})
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 8]

#### [00:18:41] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 22 scenarios (roles: {'Lilis', 'Chancellor'})

### [00:19:47] Executed #3 -> Lilis (EVIL)

#### [00:19:50] Solver Output
Scenarios: 18/396
Definite evil: ['#3', '#4']
Definite good: ['#6', '#7', '#9']
Evil probabilities: #5=72%, #2=56%, #8=44%, #1=28%
  Generated 396 candidate scenarios
  18 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Puppeteer'})
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 8]

#### [00:19:50] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 18 scenarios (roles: {'Chancellor', 'Puppeteer'})

### [00:20:29] Executed #4 -> Puppeteer (EVIL)

#### [00:20:32] Solver Output
Scenarios: 13/33
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#6', '#7', '#9']
Evil probabilities: #8=62%, #2=38%
  Generated 33 candidate scenarios
  13 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #1 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 8]

#### [00:20:32] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 13 scenarios (roles: {'Puppet'})

### [00:21:13] Executed #5 -> Puppet (EVIL)

#### [00:21:15] Solver Output
Scenarios: 13/33
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#6', '#7', '#9']
Evil probabilities: #8=62%, #2=38%
  Generated 33 candidate scenarios
  13 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #1 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 8]

#### [00:21:15] Recommendation
Action: **USE_ABILITY** #8 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.961 (adjusted 0.961) | timing x1.00

### [00:22:53] Revealed #8 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [00:22:57] Ability used at #8

#### [00:23:00] Solver Output
Scenarios: 8/33
Definite evil: ['#3', '#4', '#5', '#8']
Definite good: ['#1', '#2', '#6', '#7', '#9']
  Generated 33 candidate scenarios
  8 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [00:23:00] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 8 scenarios (roles: {'Chancellor'})

### [00:23:47] Executed #8 -> Chancellor (EVIL)

## [00:23:54] GAME OVER — WIN
Final HP: 6
Notes: 6HP. Lilis game with Puppeteer creating Puppet. FT ability resolved last evil. Night killed #6, -4HP total from nights.


---

# New Game — 2026-04-04 00:36:28
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Knitter, Dreamer, Baker, Fortune_Teller, Scout, Poet
- Outcasts: Doppelganger, Bombardier
- Minions: Witch, Puppeteer
- Demons: Baa

### [00:43:35] Revealed #1 Knight
Info: {}

### [00:43:35] Revealed #2 Knitter
Info: {'evil_pairs': 3}

### [00:43:35] Revealed #3 Poet
Info: {'evil_role': 'Puppeteer', 'distance': 1, 'copied_role': 'Scout'}

### [00:43:38] Revealed #4 Dreamer
Info: {}

### [00:43:39] Revealed #5 Dreamer
Info: {}

### [00:43:42] Revealed #6 Scout
Info: {'evil_role': 'Puppet', 'distance': 3}

### [00:43:43] Revealed #7 Poet
Info: {'targets': [1, 2, 5], 'types': ['Minion', 'Outcast', 'Villager'], 'copied_role': 'Bishop'}

### [00:43:43] Revealed #8 Fortune_Teller
Info: {}

#### [00:44:17] Solver Output
Scenarios: 138/4536
Definite evil: ['#6']
Evil probabilities: #7=88%, #8=49%, #5=45%, #2=42%, #4=26%, #9=26%, #3=16%, #1=7%
  Generated 4536 candidate scenarios
  138 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Baa', 'Puppeteer', 'Witch'})
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [00:44:17] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 138 scenarios (roles: {'Baa', 'Puppeteer', 'Witch'})

### [00:45:51] Revealed #4 Dreamer
Info: {'target': 7, 'evil_role': 'Puppeteer'}

### [00:45:52] Ability used at #4

### [00:46:29] Revealed #5 Dreamer
Info: {'target': 7, 'evil_role': 'Puppeteer'}

### [00:46:30] Ability used at #5

### [00:47:20] Revealed #8 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [00:47:21] Ability used at #8

#### [00:47:25] Solver Output
Scenarios: 18/4536
Definite evil: ['#6']
Definite good: ['#1']
Evil probabilities: #8=78%, #7=67%, #9=56%, #3=33%, #2=22%, #4=22%, #5=22%
  Generated 4536 candidate scenarios
  18 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Baa', 'Puppeteer', 'Witch'})
    #1 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 7, 8, 9]

#### [00:47:25] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 18 scenarios (roles: {'Baa', 'Puppeteer', 'Witch'})

### [00:48:26] Executed #6 -> Witch (EVIL)

### [00:51:50] Revealed #9 Baker
Info: {'original_role': 'Knitter'}

#### [00:51:53] Solver Output
Scenarios: 5/504
Definite evil: ['#6', '#7', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5']
  Generated 504 candidate scenarios
  5 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Witch'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #8 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #9 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD

#### [00:51:53] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 5 scenarios (roles: {'Puppeteer'})

### [00:52:46] Executed #7 -> Puppeteer (EVIL)

#### [00:52:50] Solver Output
Scenarios: 5/36
Definite evil: ['#6', '#7', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5']
  Generated 36 candidate scenarios
  5 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Witch'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #8 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #9 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD

#### [00:52:50] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 5 scenarios (roles: {'Puppet'})

### [00:53:24] Executed #8 -> Puppet (EVIL)

### [00:54:08] Executed #9 -> Baa (EVIL)

## [00:54:13] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Witch blocked #9, Doppelganger+Dreamer+FT confirmed Puppeteer at #7, Baa faked Baker chain


---

# New Game — 2026-04-04 01:00:31
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Confessor, Hunter, Druid, Jester, Bard, Fortune_Teller
- Outcasts: Drunk, Bombardier
- Minions: Minion
- Demons: Lilis

### [01:02:35] Revealed #1 Confessor
Info: {'dizzy': False}

### [01:02:35] Revealed #2 Baker
Info: {'original_role': 'original'}

### [01:02:36] Revealed #3 Hunter
Info: {'distance': 1}

### [01:02:36] Revealed #4 Bard
Info: {'corruption_distance': 1}

### [01:04:03] Revealed #6 Hunter
Info: {'distance': 1}

### [01:04:03] Revealed #7 Druid
Info: {}

### [01:04:03] Revealed #8 Baker
Info: {'original_role': 'Fortune_Teller'}

### [01:04:04] Revealed #9 Bombardier
Info: {}

#### [01:04:12] Solver Output
Scenarios: 26/448
Definite good: ['#1', '#2', '#5']
Evil probabilities: #6=46%, #4=38%, #7=38%, #3=31%, #8=23%, #9=23%
  Generated 448 candidate scenarios
  26 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [3, 4, 6, 7, 8, 9]

#### [01:04:12] Recommendation
Action: **USE_ABILITY** #7 (Druid) -> targets ['#1', '#2', '#9']
Reason: Entropy 1.760 (adjusted 1.693) | timing x1.00
WARNING: Corruption risk: 8%

### [01:05:10] Revealed #7 Druid
Info: {'targets': [1, 2, 9], 'found_outcast': None}

### [01:05:10] Ability used at #7

#### [01:05:15] Solver Output
Scenarios: 14/448
Definite good: ['#1', '#2', '#5']
Evil probabilities: #7=57%, #4=43%, #3=29%, #6=29%, #9=29%, #8=14%
  Generated 448 candidate scenarios
  14 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [3, 4, 6, 7, 8, 9]

#### [01:05:15] Recommendation
Action: **EXECUTE** #7
Reason: No reveals available. #7 is 57% likely evil (HP=6, budget=1 wrong execs)
WARNING: Probabilistic execution -- 57% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #7 (57%) despite low confidence — Bombardier candidate(s) [9] risk instant game loss if executed first.

### [01:06:43] Executed #7 -> Lilis (EVIL)

#### [01:07:02] Solver Output
Scenarios: 4/49
Definite evil: ['#7']
Definite good: ['#1', '#2', '#5', '#6', '#9']
Evil probabilities: #3=50%, #4=25%, #8=25%
  Generated 49 candidate scenarios
  4 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 4, 8]

#### [01:07:02] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% evil Minion, 25% good Drunk (corrupted), 25% good Hunter).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [01:08:13] Executed #3 -> Minion (EVIL)

## [01:08:19] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis killed Drunk#5 night1, dual Hunter distance+Druid lie locked evils at #3+#7


---

# New Game — 2026-04-04 01:11:49
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Confessor, Lover, Dreamer, Medium, Judge, Druid
- Outcasts: Wretch, Drunk
- Minions: Puppeteer, Shaman
- Demons: Baa

### [01:15:19] Revealed #1 Knitter
Info: {'evil_pairs': 1}

### [01:15:20] Revealed #2 Judge
Info: {}

### [01:15:20] Revealed #3 Wretch
Info: {}

### [01:15:21] Revealed #4 Druid
Info: {}

### [01:15:21] Revealed #5 Lover
Info: {'evil_adjacent': 0}

### [01:15:21] Revealed #6 Druid
Info: {}

### [01:15:22] Revealed #7 Medium
Info: {'good_position': 8, 'good_role': 'Lover'}

### [01:15:22] Revealed #8 Lover
Info: {'evil_adjacent': 2}

### [01:15:23] Revealed #9 Confessor
Info: {'dizzy': False}

#### [01:15:29] Solver Output
Scenarios: 16/3612
Definite evil: ['#2']
Definite good: ['#4', '#5', '#6', '#9']
Evil probabilities: #1=88%, #7=75%, #8=75%, #3=62%
  Generated 3612 candidate scenarios
  16 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 7, 8]

#### [01:15:29] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 16 scenarios (roles: {'Puppet'})

### [01:16:45] Revealed #2 Judge
Info: {'target': 3, 'is_lying': False}

### [01:16:45] Ability used at #2

### [01:17:19] Executed #2 -> Puppet (EVIL)

#### [01:17:23] Solver Output
Scenarios: 6/474
Definite evil: ['#1', '#2', '#7', '#8']
Definite good: ['#3', '#4', '#5', '#6', '#9']
  Generated 474 candidate scenarios
  6 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #2 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #7 is DEFINITELY EVIL (possible roles: {'Shaman', 'Baa'})
    #8 is DEFINITELY EVIL (possible roles: {'Shaman', 'Baa'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [01:17:23] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 6 scenarios (roles: {'Puppeteer'})

### [01:17:59] Executed #1 -> Puppeteer (EVIL)

### [01:18:28] Executed #7 -> Baa (EVIL)

### [01:19:09] Executed #8 -> Shaman (EVIL)

## [01:19:17] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Puppet-Judge truthfully checked Wretch, Lover+Medium+Knitter locked all evils


---

# New Game — 2026-04-04 01:22:56
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Hunter, Lover, Bishop, Druid, Slayer
- Outcasts: Doppelganger
- Minions: 
- Demons: Pooka

### [01:24:55] Revealed #1 Lover
Info: {'evil_adjacent': 0}

### [01:24:56] Revealed #2 Lover
Info: {'evil_adjacent': 0}

### [01:24:56] Revealed #3 Bishop
Info: {'targets': [1, 6, 7], 'types': ['Demon', 'Outcast', 'Villager']}

### [01:24:56] Revealed #4 Druid
Info: {}

### [01:24:57] Revealed #5 Slayer
Info: {}

### [01:24:57] Revealed #6 Hunter
Info: {'distance': 3}

### [01:24:58] Revealed #7 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 7}

#### [01:25:02] Solver Output
Scenarios: 1/42
Definite evil: ['#6']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#7']
  Generated 42 candidate scenarios
  1 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [01:25:02] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [01:25:57] Executed #6 -> Pooka (EVIL)

## [01:26:03] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, dual Lover+Bishop locked Pooka in 1 scenario, corrupted Poet self-call confirmed


---

# New Game — 2026-04-04 01:30:33
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Jester, Knitter, Bard, Baker, Knight, Gemcrafter, Alchemist
- Outcasts: Doppelganger, Wretch, Drunk
- Minions: Chancellor, Minion
- Demons: Baa

### [01:33:07] Revealed #1 Alchemist
Info: {'cured_count': 0}

### [01:33:07] Revealed #2 Jester
Info: {}

### [01:33:08] Revealed #3 Gemcrafter
Info: {'good_position': 7}

### [01:33:08] Revealed #4 Knight
Info: {}

### [01:33:08] Revealed #5 Baker
Info: {'original_role': 'Knight'}

### [01:33:09] Revealed #6 Bard
Info: {'corruption_distance': 3}

### [01:33:09] Revealed #7 Knight
Info: {}

### [01:33:09] Revealed #8 Baker
Info: {'original_role': 'original'}

### [01:33:10] Revealed #9 Knitter
Info: {'evil_pairs': 2}

#### [01:33:17] Solver Output
Scenarios: 936/32508
Definite good: ['#1']
Evil probabilities: #5=79%, #9=67%, #6=49%, #4=31%, #2=28%, #8=23%, #7=15%, #3=8%
  Generated 32508 candidate scenarios
  936 scenarios survived validation
    #1 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6, 7, 8, 9]

#### [01:33:18] Recommendation
Action: **EXECUTE** #4
Reason: Knight check: #4 is 31% evil, 5% corruption risk. Expected HP cost: 0.3 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 5% -- corrupted Knight loses immunity + 4 extra damage

### [01:36:20] Revealed #2 Jester
Info: {'targets': [3, 5, 7], 'evil_count': 0}

### [01:36:20] Ability used at #2

#### [01:36:27] Solver Output
Scenarios: 384/32508
Definite good: ['#1', '#3', '#7']
Evil probabilities: #5=69%, #9=62%, #2=50%, #6=50%, #4=38%, #8=31%
  Generated 32508 candidate scenarios
  384 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 6, 8, 9]

#### [01:36:27] Recommendation
Action: **EXECUTE** #4
Reason: Knight free check: #4 is 38% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [01:37:19] Executed #4 -> GOOD (WRONG!)

#### [01:37:54] Solver Output
Scenarios: 240/21672
Definite good: ['#1', '#3', '#4', '#7']
Evil probabilities: #5=80%, #9=70%, #2=60%, #8=50%, #6=40%
  Generated 21672 candidate scenarios
  240 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 5, 6, 8, 9]

#### [01:37:54] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (28% evil Baa, 28% evil Minion, 25% evil Chancellor).
WARNING: Execution lookahead override -- immediate hit chance is 80%, but all reveal branches still lead to a forced win.

### [01:38:37] Executed #5 -> Chancellor (EVIL)

#### [01:38:37] Solver Output
Scenarios: 60/3096
Definite evil: ['#5']
Definite good: ['#1', '#3', '#4', '#7']
Evil probabilities: #2=80%, #9=60%, #8=40%, #6=20%
  Generated 3096 candidate scenarios
  60 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 6, 8, 9]

#### [01:38:37] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (40% evil Baa, 40% evil Minion, 20% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 80%, but all reveal branches still lead to a forced win.

### [01:40:02] Executed #2 -> Minion (EVIL)

#### [01:40:03] Solver Output
Scenarios: 24/473
Definite evil: ['#2', '#5']
Definite good: ['#1', '#3', '#4', '#6', '#7']
Evil probabilities: #8=50%, #9=50%
  Generated 473 candidate scenarios
  24 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [8, 9]

#### [01:40:03] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 42% good Baker, 8% good Doppelganger).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [01:40:47] Executed #8 -> Baa (EVIL)

## [01:41:48] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Knight check free info, Jester+Baker chain locked evils


---

# New Game — 2026-04-04 01:45:54
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Lover, Knitter, Knight, Confessor, Jester, Slayer
- Outcasts: Drunk
- Minions: 
- Demons: Pooka

### [01:48:03] Revealed #1 Alchemist
Info: {'cured_count': 2}

### [01:48:04] Revealed #2 Slayer
Info: {}

### [01:48:04] Revealed #3 Jester
Info: {}

### [01:48:04] Revealed #4 Knitter
Info: {'evil_pairs': 0}

### [01:48:04] Revealed #5 Lover
Info: {'evil_adjacent': 0}

### [01:48:05] Revealed #6 Knight
Info: {}

### [01:48:05] Revealed #7 Confessor
Info: {'dizzy': False}

#### [01:48:13] Solver Output
Scenarios: 3/42
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7']
  Generated 42 candidate scenarios
  3 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [01:48:13] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 3 scenarios (roles: {'Pooka'})

### [01:50:16] Executed #2 -> Pooka (EVIL)

## [01:50:16] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Alchemist+Knitter+Lover locked Pooka in 3 scenarios


---

# New Game — 2026-04-04 01:53:53
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Poet, Bard, Knitter, Baker, Empress, Gemcrafter
- Outcasts: Drunk
- Minions: Poisoner
- Demons: Pooka

### [01:55:58] Revealed #1 Medium
Info: {'good_position': 4, 'good_role': 'Knitter'}

### [01:55:58] Revealed #2 Baker
Info: {'original_role': 'Bard'}

### [01:55:59] Revealed #3 Empress
Info: {'targets': [1, 5, 7]}

### [01:55:59] Revealed #4 Knitter
Info: {'evil_pairs': 1}

### [01:55:59] Revealed #5 Bard
Info: {'corruption_distance': 3}

### [01:56:00] Revealed #6 Poet
Info: {'targets': [2, 4, 7], 'types': ['Outcast', 'Minion', 'Villager'], 'copied_role': 'Bishop'}

### [01:56:00] Revealed #7 Gemcrafter
Info: {'good_position': 3}

### [01:56:00] Revealed #8 Poet
Info: {'side': 'ccw', 'copied_role': 'Architect'}

#### [01:56:05] Solver Output
Scenarios: 3/560
Definite good: ['#3', '#5', '#7', '#8']
Evil probabilities: #2=67%, #4=67%, #1=33%, #6=33%
  Generated 560 candidate scenarios
  3 scenarios survived validation
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 6]

#### [01:56:05] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (33% good Baker (corrupted), 33% evil Poisoner, 33% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [01:56:55] Executed #2 -> GOOD (WRONG!)

#### [01:56:56] Solver Output
Scenarios: 1/420
Definite evil: ['#1', '#6']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#8']
  Generated 420 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [01:56:56] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [01:59:02] Executed #6 -> Poisoner (EVIL)

#### [01:59:02] Solver Output
Scenarios: 1/58
Definite evil: ['#1', '#6']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#8']
  Generated 58 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [01:59:02] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [02:05:14] Revealed #1 Medium
Info: {'good_position': 4, 'good_role': 'Knitter'}

#### [02:05:18] Solver Output
Scenarios: 1/58
Definite evil: ['#1', '#6']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#8']
  Generated 58 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [02:05:18] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [02:09:03] Executed #3 -> Pooka (EVIL)

## [02:09:11] GAME OVER — WIN
Final HP: 5
Notes: 5HP, SOLVER BUG: Architect validator uses side-count not closest-distance, overrode solver with memory reader to execute #3 not #1, wrong exec on Baker#2, ascension 43 complete


---

# New Game — 2026-04-04 12:07:23
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Oracle, Slayer, Confessor, Enlightened, Bishop, Baker
- Outcasts: Alchemist, Bombardier
- Minions: Twin_Minion, Minion
- Demons: Lilis

## Deck
- Villagers: Hunter, Oracle, Slayer, Confessor, Enlightened, Bishop, Baker, Alchemist
- Outcasts: Bombardier
- Minions: Twin_Minion, Minion
- Demons: Lilis

#### [12:08:14] Claude Reasoning


### [12:11:23] Revealed #1 Enlightened
Info: {'direction': 'cw'}

### [12:11:30] Revealed #2 Confessor
Info: {'dizzy': True}

### [12:11:35] Revealed #3 Confessor
Info: {'dizzy': True}

### [12:12:00] Revealed #4 Slayer
Info: {}

#### [12:12:24] Solver Output
Scenarios: 30/720
Definite evil: ['#2', '#3']
Definite good: ['#1', '#7', '#10']
Evil probabilities: #4=20%, #5=20%, #6=20%, #8=20%, #9=20%
  Generated 720 candidate scenarios
  30 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Lilis', 'Minion'})
    #3 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Lilis', 'Minion'})
    #1 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 8, 9]

#### [12:12:24] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 30 scenarios (roles: {'Twin_Minion', 'Lilis', 'Minion'})

### [12:19:27] Revealed #8 Alchemist
Info: {'cured_count': 0}

### [12:19:32] Executed #4 -> GOOD (WRONG!)

#### [12:19:47] Solver Output
Scenarios: 18/504
Definite evil: ['#2', '#3']
Definite good: ['#1', '#4', '#7', '#8', '#10']
Evil probabilities: #5=33%, #6=33%, #9=33%
  Generated 504 candidate scenarios
  18 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Minion', 'Lilis'})
    #3 is DEFINITELY EVIL (possible roles: {'Minion', 'Twin_Minion', 'Lilis'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [5, 6, 9]

#### [12:19:47] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 18 scenarios (roles: {'Twin_Minion', 'Minion', 'Lilis'})


---

# New Game — 2026-04-09 14:59:49
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Druid, Bishop, Bard, Poet, Judge, Fortune_Teller
- Outcasts: Wretch
- Minions: Twin_Minion
- Demons: Pooka

### [15:00:42] Revealed #1 Poet
Info: {'corruption_distance': 1, 'copied_role': 'Bard'}

### [15:00:43] Revealed #2 Knitter
Info: {'evil_pairs': 0}

### [15:00:43] Revealed #3 Wretch
Info: {}

### [15:00:43] Revealed #4 Fortune_Teller
Info: {}

### [15:00:43] Revealed #5 Bishop
Info: {'targets': [1, 2, 4], 'types': ['Villager', 'Minion', 'Outcast']}

### [15:00:43] Revealed #6 Druid
Info: {}

### [15:00:43] Revealed #7 Judge
Info: {}

### [15:00:43] Revealed #8 Bard
Info: {'corruption_distance': -1}

#### [15:00:48] Solver Output
Scenarios: 2/56
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4', '#6']
Evil probabilities: #7=50%, #8=50%

#### [15:00:48] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 2 scenarios (roles: {'Twin_Minion'})

### [15:01:25] Executed #5 -> Twin_Minion (EVIL)

#### [15:01:29] Solver Output
Scenarios: 2/7
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4', '#6']
Evil probabilities: #7=50%, #8=50%

#### [15:01:29] Recommendation
Action: **USE_ABILITY** #4 (Fortune Teller) -> targets ['#1', '#7']
Reason: Entropy 1.000 (adjusted 1.000) | follow-up bonus 0.500 | timing x1.00

### [15:02:20] Revealed #4 Fortune Teller
Info: {'targets': [1, 7], 'has_evil': True}

### [15:02:21] Ability used at #4

#### [15:02:21] Solver Output
Scenarios: 1/7
Definite evil: ['#5', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#8']

#### [15:02:21] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [15:02:53] Executed #7 -> Pooka (EVIL)

## [15:02:53] GAME OVER — WIN
Final HP: 10
Notes: perfect 10HP, memory reader clue test game


---

# New Game — 2026-04-09 15:19:37
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Slayer, Hunter, Knitter, Enlightened, Oracle, Dreamer, Scout
- Outcasts: Plague_Doctor
- Minions: Poisoner, Minion
- Demons: Lilis

### [15:20:26] Revealed #1 Plague_Doctor
Info: {}

### [15:20:26] Revealed #3 Enlightened
Info: {'direction': 'CW'}

### [15:20:26] Revealed #4 Slayer
Info: {}

### [15:20:33] Revealed #2 Scout
Info: {'evil_role': 'Lilis', 'distance': 2}

### [15:21:15] Revealed #7 Hunter
Info: {'distance': 1}

### [15:21:15] Revealed #8 Slayer
Info: {}

### [15:21:15] Revealed #9 Oracle
Info: {'targets': [1, 5], 'minion_role': 'Poisoner'}

### [15:21:21] Revealed #6 Medium
Info: {'good_position': 9, 'good_role': 'Oracle'}

#### [15:21:26] Solver Output
Scenarios: 81/4548
Definite good: ['#5']
Evil probabilities: #4=59%, #8=56%, #9=49%, #6=32%, #3=31%, #10=25%, #2=22%, #7=20%, #1=6%

#### [15:21:26] Recommendation
Action: **USE_ABILITY** #1 (Plague Doctor) -> targets ['#9']
Reason: Entropy 1.958 (adjusted 1.958) | timing x1.00

### [15:22:21] Ability used at #1

#### [15:22:21] Solver Output
Scenarios: 25/4548
Definite good: ['#2', '#5', '#6', '#9']
Evil probabilities: #8=84%, #10=80%, #4=56%, #3=44%, #1=20%, #7=16%

#### [15:22:21] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#10']
Reason: Target #10 is 80% evil (adjusted 0.80)

### [15:23:12] Ability used at #8

#### [15:23:12] Solver Output
Scenarios: 23/4548
Definite good: ['#2', '#5', '#6', '#9']
Evil probabilities: #8=91%, #10=78%, #4=52%, #3=48%, #1=22%, #7=9%

#### [15:23:12] Recommendation
Action: **REVEAL** #10
Reason: #10: 78% evil, 2.062 bits (4 outcomes)

### [15:23:32] Revealed #10 Knitter
Info: {'evil_pairs': 2}

#### [15:23:32] Solver Output
Scenarios: 19/4602
Definite evil: ['#8']
Definite good: ['#2', '#5', '#6', '#7', '#9']
Evil probabilities: #10=95%, #3=58%, #4=42%, #1=5%

#### [15:23:32] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 19 scenarios (roles: {'Minion', 'Lilis', 'Poisoner'})

### [15:24:18] Executed #8 -> Minion (EVIL)

#### [15:24:18] Solver Output
Scenarios: 5/480
Definite evil: ['#8']
Definite good: ['#2', '#5', '#6', '#7', '#9']
Evil probabilities: #10=80%, #3=60%, #4=40%, #1=20%

#### [15:24:18] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#10']
Reason: Target #10 is 80% evil (adjusted 0.64)
WARNING: Corruption risk: 20% -- Slayer ability disabled if corrupted

### [15:24:55] Ability used at #4

#### [15:24:55] Solver Output
Scenarios: 4/480
Definite evil: ['#8']
Definite good: ['#2', '#5', '#6', '#7', '#9']
Evil probabilities: #10=75%, #3=50%, #4=50%, #1=25%

#### [15:24:55] Recommendation
Action: **EXECUTE** #10
Reason: Execution lookahead: #10 guarantees a win across all reveal branches with current HP budget (50% evil Lilis, 25% good Knitter (corrupted), 25% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [15:25:30] Executed #10 -> Lilis (EVIL)

#### [15:25:30] Solver Output
Scenarios: 2/67
Definite evil: ['#8', '#10']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#9']
Evil probabilities: #3=50%, #4=50%

#### [15:25:30] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% good Enlightened, 50% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:26:04] Executed #3 -> GOOD (WRONG!)

#### [15:26:05] Solver Output
Scenarios: 1/55
Definite evil: ['#4', '#8', '#10']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7', '#9']

#### [15:26:05] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [15:26:38] Executed #4 -> Poisoner (EVIL)

## [15:26:38] GAME OVER — WIN
Final HP: 1
Notes: 1HP win, auto_card test, Lilis game, both Slayers were evil/corrupted


---

# New Game — 2026-04-09 15:37:20
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Slayer, Dreamer, Baker, Lover, Oracle, Jester
- Outcasts: Plague_Doctor, Wretch
- Minions: Shaman, Twin_Minion
- Demons: Lilis

### [15:37:59] Revealed #1 Lover
Info: {'evil_adjacent': 0}

### [15:37:59] Revealed #3 Wretch
Info: {}

### [15:37:59] Revealed #4 Oracle
Info: {'targets': [4, 9], 'minion_role': 'Twin Minion'}

### [15:38:09] Revealed #2 Jester
Info: {}

### [15:38:42] Revealed #5 Medium
Info: {'good_position': 10, 'good_role': 'Dreamer'}

### [15:38:42] Revealed #6 Oracle
Info: {'targets': [1, 8], 'minion_role': 'Shaman'}

### [15:38:42] Revealed #8 Lover
Info: {'evil_adjacent': 0}

### [15:38:49] Revealed #7 Slayer
Info: {}

#### [15:38:55] Solver Output
Scenarios: 0/270

#### [15:38:55] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [15:40:10] Solver Output
Scenarios: 1/270
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#7', '#10']

#### [15:40:10] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [15:40:18] Executed #8 -> Shaman (EVIL)

#### [15:40:40] Solver Output
Scenarios: 1/18
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#7', '#10']

#### [15:40:40] Recommendation
Action: **WIN**
Reason: All evil characters have been executed!

#### [15:40:56] Solver Output
Scenarios: 1/18
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#7', '#10']

#### [15:40:56] Recommendation
Action: **WIN**
Reason: All evil characters have been executed!

### [15:41:38] Executed #7 -> Lilis (EVIL)

## [15:41:38] GAME OVER — WIN
Final HP: 6
Notes: auto_next test, auto-executed #8 Shaman, solver WIN bug when 3rd evil at night-killed pos


---

# New Game — 2026-04-09 15:46:14
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Dreamer, Bard, Scout, Medium, Oracle
- Outcasts: Bombardier
- Minions: 
- Demons: Pooka

### [15:46:28] Revealed #1 Bard
Info: {'corruption_distance': 3}

### [15:46:28] Revealed #2 Medium
Info: {'good_position': 6, 'good_role': 'Dreamer'}

### [15:46:28] Revealed #3 Lover
Info: {'evil_adjacent': 0}

### [15:46:28] Revealed #5 Bombardier
Info: {}

### [15:46:28] Revealed #6 Dreamer
Info: {}

### [15:46:28] Revealed #7 Oracle
Info: {'targets': [2, 4], 'minion_role': 'Twin_Minion'}

### [15:46:50] Revealed #4 Scout
Info: {}

#### [15:46:50] Solver Output
Scenarios: 1/7
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']

#### [15:46:50] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [15:46:58] Executed #7 -> Pooka (EVIL)

## [15:47:19] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, auto_next one-shot win, 6/7 cards auto-entered


---

# New Game — 2026-04-09 15:50:46
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Hunter, Enlightened, Poet, Bard, Scout, Medium
- Outcasts: Bombardier, Drunk
- Minions: Puppeteer
- Demons: Pooka

### [15:50:58] Revealed #1 Enlightened
Info: {'direction': 'CW'}

### [15:50:58] Revealed #2 Medium
Info: {'good_position': 3, 'good_role': 'Bard'}

### [15:50:58] Revealed #3 Bard
Info: {'corruption_distance': 3}

### [15:50:58] Revealed #6 Scout
Info: {'evil_role': 'Puppeteer', 'distance': 1}

### [15:50:58] Revealed #7 Hunter
Info: {'distance': 3}

### [15:51:11] Revealed #4 Bombardier
Info: {}

### [15:51:12] Revealed #5 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 2}

### [15:51:12] Revealed #8 Slayer
Info: {}

#### [15:51:12] Solver Output
Scenarios: 12/794
Evil probabilities: #3=50%, #4=50%, #1=25%, #2=25%, #5=25%, #6=25%, #7=25%, #8=25%

#### [15:51:12] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#3']
Reason: Target #3 is 50% evil (adjusted 0.50)

### [15:51:41] Ability used at #8

#### [15:51:41] Solver Output
Scenarios: 4/99
Definite evil: ['#3']
Definite good: ['#7', '#8']
Evil probabilities: #2=50%, #1=25%, #4=25%, #5=25%, #6=25%

#### [15:51:41] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (50% good Medium (corrupted), 50% evil Puppeteer).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:52:11] Executed #2 -> GOOD (WRONG!)

#### [15:52:11] Solver Output
Scenarios: 2/83
Definite evil: ['#3']
Definite good: ['#1', '#2', '#7', '#8']
Evil probabilities: #4=50%, #5=50%, #6=50%

#### [15:52:11] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (50% good Drunk (corrupted), 50% evil Puppeteer).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:52:53] Executed #5 -> GOOD (WRONG!)

#### [15:52:54] Solver Output
Scenarios: 1/61
Definite evil: ['#3', '#4']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8']

#### [15:52:54] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Puppeteer'})

### [15:53:01] Executed #4 -> Puppeteer (EVIL)

## [15:53:09] GAME OVER — WIN
Final HP: 3
Notes: 3HP, auto_next executed Puppeteer, Slayer killed Pooka, Drunk exec cost 2HP not 5


---

# New Game — 2026-04-09 15:57:16
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Slayer, Hunter, Poet, Druid
- Outcasts: Drunk, Doppelganger
- Minions: Poisoner
- Demons: Baa

### [15:57:39] Revealed #1 Lover
Info: {'evil_adjacent': 1}

### [15:57:39] Revealed #2 Poet
Info: {'distance': 2, 'copied_role': 'Hunter'}

### [15:57:39] Revealed #5 Lover
Info: {'evil_adjacent': 0}

### [15:57:39] Revealed #6 Hunter
Info: {'distance': 1}

### [15:57:50] Revealed #3 Poet
Info: {}

### [15:57:50] Revealed #4 Druid
Info: {}

### [15:57:50] Revealed #7 Slayer
Info: {}

#### [15:57:51] Solver Output
Scenarios: 96/2114
Evil probabilities: #6=47%, #2=39%, #3=34%, #7=32%, #4=25%, #1=17%, #5=6%

#### [15:57:51] Recommendation
Action: **USE_ABILITY** #4 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.980 (adjusted 0.898) | timing x1.00
WARNING: Corruption risk: 17%

### [15:58:29] Revealed #4 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [15:58:29] Ability used at #4

#### [15:58:29] Solver Output
Scenarios: 56/2114
Evil probabilities: #6=41%, #4=38%, #3=32%, #7=30%, #2=29%, #1=20%, #5=11%

#### [15:58:29] Recommendation
Action: **USE_ABILITY** #7 (Slayer) -> targets ['#6']
Reason: Target #6 is 41% evil (adjusted 0.38)
WARNING: Corruption risk: 7% -- Slayer ability disabled if corrupted

### [15:58:56] Ability used at #7

#### [15:58:57] Solver Output
Scenarios: 37/2114
Evil probabilities: #3=49%, #7=46%, #4=41%, #1=22%, #2=16%, #5=16%, #6=11%

#### [15:58:57] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 49% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 49% confident (budget: 2 wrong execs)
WARNING: Low confidence (49%) -- consider gathering more info

### [15:59:25] Executed #3 -> Poisoner (EVIL)

#### [15:59:25] Solver Output
Scenarios: 14/302
Definite evil: ['#3']
Definite good: ['#4', '#5', '#6']
Evil probabilities: #7=57%, #1=29%, #2=14%

#### [15:59:25] Recommendation
Action: **EXECUTE** #7
Reason: No reveals available. #7 is 57% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 57% confident (budget: 2 wrong execs)

### [16:00:02] Executed #7 -> Baa (EVIL)

## [16:00:03] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, card1 reflip needed, Druid+Slayer abilities


---

# New Game — 2026-04-09 16:00:59
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Druid, Empress, Bard, Gemcrafter
- Outcasts: Wretch
- Minions: 
- Demons: Pooka

### [16:01:12] Revealed #1 Bard
Info: {'corruption_distance': 3}

### [16:01:12] Revealed #2 Wretch
Info: {}

### [16:01:12] Revealed #3 Druid
Info: {}

### [16:01:12] Revealed #4 Gemcrafter
Info: {'good_position': 1}

### [16:01:12] Revealed #5 Medium
Info: {'good_position': 6, 'good_role': 'Empress'}

### [16:01:12] Revealed #6 Empress
Info: {'targets': [1, 3, 4]}

#### [16:01:12] Solver Output
Scenarios: 1/6
Definite evil: ['#6']
Definite good: ['#1', '#2', '#3', '#4', '#5']

#### [16:01:12] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [16:01:20] Executed #6 -> Pooka (EVIL)

## [16:01:28] GAME OVER — WIN
Final HP: 10
Notes: 10HP PERFECT, FULLY AUTOMATED: auto_card 6/6 + auto_next one-shot, ascension 44 complete


---

# New Game — 2026-04-09 16:05:03
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Empress, Poet, Druid, Knitter, Oracle
- Outcasts: Plague_Doctor
- Minions: 
- Demons: Pooka

### [16:05:37] Revealed #1 Oracle
Info: {'targets': [5, 7], 'minion_role': 'Minion'}

### [16:05:37] Revealed #3 Medium
Info: {'good_position': 7, 'good_role': 'Druid'}

### [16:05:37] Revealed #5 Knitter
Info: {'evil_pairs': 0}

### [16:05:37] Revealed #6 Empress
Info: {'targets': [2, 3, 7]}

#### [16:05:37] Solver Output
Scenarios: 1/24
Definite evil: ['#1']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7']

#### [16:05:37] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [16:06:30] Revealed #2 Plague_Doctor
Info: {}

### [16:06:30] Revealed #4 Poet
Info: {}

### [16:06:30] Revealed #7 Druid
Info: {}

#### [16:06:30] Solver Output
Scenarios: 1/31
Definite evil: ['#1']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7']

#### [16:06:30] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [16:07:14] Executed #1 -> Pooka (EVIL)

## [16:07:14] GAME OVER — WIN
Final HP: 10
Notes: 10HP, auto_exec failed on Oracle active ability, manual exec. Time: ~1775768834s


---

# New Game — 2026-04-09 16:09:09
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Baker, Judge, Medium, Bard, Oracle, Slayer
- Outcasts: Plague_Doctor
- Minions: Poisoner, Twin_Minion
- Demons: Lilis

### [16:09:37] Revealed #3 Oracle
Info: {'targets': [2, 9], 'minion_role': 'Poisoner'}

### [16:09:46] Revealed #1 Judge
Info: {}

### [16:09:46] Revealed #2 Baker
Info: {'original_role': 'original'}

### [16:09:46] Revealed #4 Plague_Doctor
Info: {}

### [16:10:05] Revealed #6 Bard
Info: {'corruption_distance': 1}

### [16:10:16] Revealed #5 Plague_Doctor
Info: {}

### [16:10:16] Revealed #7 Slayer
Info: {}

### [16:10:17] Revealed #9 Poet
Info: {'targets': [3, 5, 6], 'types': ['Outcast', 'Villager', 'Minion'], 'copied_role': 'Bishop'}

#### [16:10:17] Solver Output
Scenarios: 134/5070
Definite good: ['#8']
Evil probabilities: #4=57%, #9=55%, #5=43%, #6=37%, #3=25%, #2=22%, #10=21%, #1=20%, #7=20%

#### [16:10:17] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#6']
Reason: Entropy 2.750 (adjusted 2.750) | timing x1.00

### [16:10:49] Ability used at #5

#### [16:10:49] Solver Output
Scenarios: 74/5070
Definite good: ['#8']
Evil probabilities: #4=65%, #9=46%, #6=42%, #5=35%, #1=23%, #3=23%, #10=23%, #2=22%, #7=22%

#### [16:10:49] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#10']
Reason: Entropy 2.156 (adjusted 2.156) | timing x1.00

### [16:11:18] Ability used at #4

#### [16:11:18] Solver Output
Scenarios: 35/5070
Definite evil: ['#4']
Definite good: ['#5', '#8']
Evil probabilities: #6=66%, #10=31%, #9=29%, #2=20%, #7=20%, #1=17%, #3=17%

#### [16:11:18] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 35 scenarios (roles: {'Lilis', 'Twin_Minion', 'Poisoner'})

### [16:11:26] Executed #4 -> Poisoner (EVIL)

#### [16:11:35] Solver Output
Scenarios: 15/352
Definite evil: ['#4']
Definite good: ['#2', '#5', '#8', '#9']
Evil probabilities: #6=87%, #10=33%, #1=27%, #3=27%, #7=27%

#### [16:11:35] Recommendation
Action: **REVEAL** #10
Reason: #10: 33% evil, 3.640 bits (13 outcomes)

### [16:11:48] Revealed #10 Bard
Info: {'corruption_distance': 3}

#### [16:11:48] Solver Output
Scenarios: 5/352
Definite evil: ['#4', '#6']
Definite good: ['#2', '#3', '#5', '#8', '#9']
Evil probabilities: #10=60%, #1=20%, #7=20%

#### [16:11:48] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 5 scenarios (roles: {'Twin_Minion', 'Lilis'})

### [16:11:56] Executed #6 -> Twin Minion (EVIL)

#### [16:12:04] Solver Output
Scenarios: 8/86
Definite evil: ['#4', '#6']
Definite good: ['#2', '#3', '#5', '#8', '#9']
Evil probabilities: #10=50%, #1=25%, #7=25%

#### [16:12:04] Recommendation
Action: **USE_ABILITY** #7 (Slayer) -> targets ['#10']
Reason: Target #10 is 50% evil (adjusted 0.50)

### [16:12:30] Ability used at #7

## [16:12:30] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, auto_exec x2, time=219s


---

# New Game — 2026-04-09 16:22:59
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Bishop, Bard, Fortune_Teller, Medium, Lover
- Outcasts: Doppelganger
- Minions: 
- Demons: Pooka

### [16:23:06] Revealed #1 Lover
Info: {'evil_adjacent': 0}

### [16:23:06] Revealed #3 Bard
Info: {'corruption_distance': -1}

### [16:23:06] Revealed #4 Medium
Info: {'good_position': 6, 'good_role': 'Fortune Teller'}

### [16:23:06] Revealed #5 Bishop
Info: {'targets': [5, 1, 2], 'types': ['Villager', 'Outcast', 'Demon']}

#### [16:23:06] Solver Output
Scenarios: 2/42
Definite good: ['#1', '#3', '#5', '#6', '#7']
Evil probabilities: #2=50%, #4=50%

#### [16:23:06] Recommendation
Action: **REVEAL** #2
Reason: #2: 50% evil, 1.100 bits (2 outcomes)

### [16:23:17] Revealed #2 Fortune_Teller
Info: {}

### [16:23:17] Revealed #6 Fortune_Teller
Info: {}

### [16:23:18] Revealed #7 Judge
Info: {}

#### [16:23:18] Solver Output
Scenarios: 2/42
Definite good: ['#1', '#3', '#5', '#6', '#7']
Evil probabilities: #2=50%, #4=50%

#### [16:23:18] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#1', '#3']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

#### [16:23:25] Solver Output
Scenarios: 2/42
Definite good: ['#1', '#3', '#5', '#6', '#7']
Evil probabilities: #2=50%, #4=50%

#### [16:23:25] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#1', '#3']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [16:23:53] Revealed #2 Fortune Teller
Info: {'targets': [1, 3], 'has_evil': False}

### [16:23:53] Ability used at #2

#### [16:23:53] Solver Output
Scenarios: 1/42
Definite evil: ['#4']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7']

#### [16:23:53] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [16:24:01] Executed #4 -> Pooka (EVIL)

## [16:24:09] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, FT resolved, auto_exec Pooka


---

# New Game — 2026-04-09 16:27:14
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Bard, Druid, Alchemist, Gemcrafter, Jester
- Outcasts: Plague_Doctor, Doppelganger
- Minions: Twin_Minion
- Demons: Baa

### [16:27:21] Revealed #3 Bard
Info: {'corruption_distance': 3}

### [16:27:21] Revealed #4 Confessor
Info: {'dizzy': True}

### [16:27:21] Revealed #7 Confessor
Info: {'dizzy': False}

### [16:27:21] Revealed #8 Gemcrafter
Info: {'good_position': 6}

### [16:27:44] Revealed #1 Druid
Info: {}

### [16:27:45] Revealed #2 Plague_Doctor
Info: {}

### [16:27:45] Revealed #5 Jester
Info: {}

### [16:27:45] Revealed #6 Alchemist
Info: {'cured_count': 2}

#### [16:27:45] Solver Output
Scenarios: 36/1610
Definite evil: ['#4']
Definite good: ['#2', '#3', '#7', '#8']
Evil probabilities: #1=33%, #5=33%, #6=33%

#### [16:27:45] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 36 scenarios (roles: {'Twin_Minion', 'Baa'})

### [16:27:53] Executed #4 -> Twin Minion (EVIL)

#### [16:28:00] Solver Output
Scenarios: 18/223
Definite evil: ['#4']
Definite good: ['#2', '#3', '#7', '#8']
Evil probabilities: #1=33%, #5=33%, #6=33%

#### [16:28:00] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.585 (adjusted 1.585) | timing x1.00

#### [16:28:06] Solver Output
Scenarios: 18/223
Definite evil: ['#4']
Definite good: ['#2', '#3', '#7', '#8']
Evil probabilities: #1=33%, #5=33%, #6=33%

#### [16:28:06] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.585 (adjusted 1.585) | timing x1.00

### [16:28:30] Ability used at #2

#### [16:28:30] Solver Output
Scenarios: 6/223
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8']

#### [16:28:30] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 6 scenarios (roles: {'Baa'})

### [16:28:38] Executed #6 -> Baa (EVIL)

## [16:28:48] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, 2 auto-execs, PD resolved, time=111s


---

# New Game — 2026-04-09 16:29:46
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Medium, Druid, Bishop, Fortune_Teller, Knight, Oracle
- Outcasts: Doppelganger, Drunk
- Minions: Poisoner
- Demons: Lilis

### [16:30:11] Revealed #1 Fortune_Teller
Info: {}

### [16:30:11] Revealed #2 Fortune_Teller
Info: {}

### [16:30:11] Revealed #3 Judge
Info: {}

### [16:30:11] Revealed #4 Bishop
Info: {'targets': [1, 6, 3], 'types': ['Villager', 'Outcast', 'Minion']}

### [16:30:59] Revealed #5 Oracle
Info: {'targets': [2, 6], 'minion_role': 'Poisoner'}

### [16:30:59] Revealed #7 Knight
Info: {}

### [16:30:59] Revealed #8 Druid
Info: {}

### [16:30:59] Revealed #9 Judge
Info: {}

#### [16:30:59] Solver Output
Scenarios: 1077/5064
Definite good: ['#6']
Evil probabilities: #5=49%, #4=38%, #1=28%, #3=27%, #8=15%, #7=15%, #2=15%, #9=14%

#### [16:31:00] Recommendation
Action: **USE_ABILITY** #1 (Fortune Teller) -> targets ['#2', '#5']
Reason: Entropy 1.000 (adjusted 0.928) | follow-up bonus 0.275 | timing x1.00
WARNING: Corruption risk: 14%

#### [16:31:07] Solver Output
Scenarios: 1077/5064
Definite good: ['#6']
Evil probabilities: #5=49%, #4=38%, #1=28%, #3=27%, #8=15%, #7=15%, #2=15%, #9=14%

#### [16:31:08] Recommendation
Action: **USE_ABILITY** #1 (Fortune Teller) -> targets ['#2', '#5']
Reason: Entropy 1.000 (adjusted 0.928) | follow-up bonus 0.275 | timing x1.00
WARNING: Corruption risk: 14%

### [16:33:27] Revealed #1 Fortune Teller
Info: {'targets': [2, 5], 'has_evil': True}

### [16:33:27] Ability used at #1

#### [16:33:27] Solver Output
Scenarios: 544/5064
Definite good: ['#6']
Evil probabilities: #5=60%, #4=33%, #3=28%, #1=23%, #8=15%, #7=15%, #9=14%, #2=11%

#### [16:33:28] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#3', '#4']
Reason: Entropy 1.000 (adjusted 0.905) | follow-up bonus 0.281 | timing x1.00
WARNING: Corruption risk: 19%

### [16:34:25] Revealed #2 Fortune Teller
Info: {'targets': [3, 4], 'has_evil': False}

### [16:34:25] Ability used at #2

#### [16:34:25] Solver Output
Scenarios: 274/5064
Definite good: ['#6']
Evil probabilities: #5=65%, #8=22%, #7=22%, #1=21%, #3=20%, #4=20%, #9=16%, #2=14%

#### [16:34:26] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.950 (adjusted 0.879) | timing x1.00
WARNING: Corruption risk: 15%

### [16:35:58] Revealed #8 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [16:35:59] Ability used at #8

### [16:35:59] Revealed #3 Judge
Info: {'target': 3, 'is_lying': False}

### [16:35:59] Ability used at #3

#### [16:35:59] Solver Output
Scenarios: 123/5064
Definite good: ['#6']
Evil probabilities: #5=62%, #8=27%, #1=25%, #7=23%, #3=20%, #9=15%, #4=15%, #2=13%

#### [16:35:59] Recommendation
Action: **USE_ABILITY** #9 (Judge) -> targets ['#8']
Reason: Expected posterior 73.1 scenarios (adjusted 80.0, info gain 0.621 bits) | timing x1.00
WARNING: Corruption risk: 19% -- corrupted Judge results are unreliable

#### [16:36:43] Solver Output
Scenarios: 123/5064
Definite good: ['#6']
Evil probabilities: #5=62%, #8=27%, #1=25%, #7=23%, #3=20%, #9=15%, #4=15%, #2=13%

#### [16:36:43] Recommendation
Action: **USE_ABILITY** #9 (Judge) -> targets ['#8']
Reason: Expected posterior 73.1 scenarios (adjusted 80.0, info gain 0.621 bits) | timing x1.00
WARNING: Corruption risk: 19% -- corrupted Judge results are unreliable

## [16:45:23] GAME OVER — LOSS
Final HP: 6
Notes: GAME STUCK: Lilis night animation loops forever when all cards revealed, no kill target. Game appears frozen.

### [18:01:07] Executed #5 -> GOOD (WRONG!)

#### [18:01:07] Solver Output
Scenarios: 47/3982
Definite good: ['#5', '#6']
Evil probabilities: #1=66%, #4=38%, #2=34%, #7=21%, #3=19%, #8=15%, #9=6%

#### [18:01:07] Recommendation
Action: **USE_ABILITY** #9 (Judge) -> targets ['#4']
Reason: Expected posterior 29.0 scenarios (adjusted 32.4, info gain 0.535 bits) | timing x1.00
WARNING: Corruption risk: 23% -- corrupted Judge results are unreliable

### [18:01:53] Revealed #9 Judge
Info: {'target': 4, 'is_lying': True}

### [18:01:53] Ability used at #9

#### [18:01:54] Solver Output
Scenarios: 28/3982
Definite good: ['#3', '#5', '#6']
Evil probabilities: #4=64%, #2=54%, #1=46%, #8=14%, #7=11%, #9=11%

#### [18:01:54] Recommendation
Action: **ERROR** #4
Reason: #4 is 64% likely evil but HP too low to risk (HP=1, cost=5). Need more info.
WARNING: Probabilistic execution -- 64% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [18:02:25] Executed #4 -> Lilis (EVIL)

#### [18:02:25] Solver Output
Scenarios: 18/524
Definite evil: ['#4']
Definite good: ['#3', '#5', '#6', '#7', '#8', '#9']
Evil probabilities: #2=83%, #1=17%

#### [18:02:25] Recommendation
Action: **ERROR** #2
Reason: #2 is 83% likely evil but HP too low to risk (HP=1, cost=5). Need more info.
WARNING: Probabilistic execution -- 83% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [18:03:06] Executed #2 -> Poisoner (EVIL)

## [18:03:06] GAME OVER — WIN
Final HP: 1
Notes: 1HP clutch, wrong coords caused stuck state, judge resolved, 83pct gamble


---

# New Game — 2026-04-09 18:11:34
Cards: 7, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Bishop, Knitter, Oracle, Poet, Slayer
- Outcasts: Doppelganger, Wretch
- Minions: Puppeteer
- Demons: Baa

### [18:12:43] Revealed #1 Bishop
Info: {'targets': [2, 3, 6], 'types': ['Villager', 'Outcast', 'Minion']}

### [18:12:43] Revealed #2 Bishop
Info: {'targets': [2, 3, 6], 'types': ['Villager', 'Outcast', 'Minion']}

### [18:12:43] Revealed #3 Knitter
Info: {'evil_pairs': 2}

### [18:12:43] Revealed #6 Oracle
Info: {'targets': [3, 6], 'minion_role': 'Puppet'}

### [18:15:08] Revealed #4 Slayer
Info: {}

### [18:15:12] Revealed #5 Baker
Info: {'original_role': 'Poet'}

### [18:15:18] Revealed #7 Poet
Info: {'good_position': 1, 'good_role': 'Bishop', 'copied_role': 'Medium'}

#### [18:15:32] Solver Output
Scenarios: 4/350
Definite evil: ['#4', '#5']
Definite good: ['#1', '#2', '#7']
Evil probabilities: #3=50%, #6=50%

#### [18:15:32] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 4 scenarios (roles: {'Puppeteer', 'Baa'})

### [18:16:25] Executed #4 -> Baa (EVIL)

#### [18:16:28] Solver Output
Scenarios: 2/50
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#7']

#### [18:16:28] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 2 scenarios (roles: {'Puppeteer'})

### [18:17:04] Executed #5 -> Puppeteer (EVIL)

#### [18:17:07] Solver Output
Scenarios: 2/5
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#7']

#### [18:17:07] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Puppet'})

### [18:17:44] Executed #6 -> Puppet (EVIL)

## [18:17:48] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, 3 auto-execs, Puppeteer+Puppet+Baa, solver had only 4 scenarios from start


---

# New Game — 2026-04-09 18:20:02
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Medium, Baker, Bishop, Poet, Scout, Knight
- Outcasts: Plague_Doctor
- Minions: Minion, Witch
- Demons: Lilis

### [18:23:33] Revealed #1 Baker
Info: {'original_role': 'Hunter'}

### [18:23:37] Revealed #2 Knight
Info: {}

### [18:23:40] Revealed #3 Plague_Doctor
Info: {}

### [18:23:43] Revealed #4 Knight
Info: {}

### [18:24:56] Revealed #5 Scout
Info: {'evil_role': 'Lilis', 'distance': 1}

### [18:24:59] Revealed #6 Bishop
Info: {'targets': [4, 8, 9], 'types': ['Villager', 'Minion', 'Outcast']}

### [18:25:02] Revealed #7 Medium
Info: {'good_position': 2, 'good_role': 'Knight'}

#### [18:25:10] Solver Output
Scenarios: 50/1848
Definite good: ['#9']
Evil probabilities: #1=80%, #4=60%, #2=48%, #8=32%, #6=28%, #7=24%, #5=20%, #3=8%

#### [18:25:10] Recommendation
Action: **EXECUTE** #4
Reason: Knight free check: #4 is 60% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [18:26:25] Executed #4 -> GOOD (WRONG!)

#### [18:27:05] Solver Output
Scenarios: 20/1176
Definite evil: ['#2']
Definite good: ['#3', '#4', '#5', '#9']
Evil probabilities: #1=60%, #7=60%, #6=40%, #8=40%

#### [18:27:05] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 20 scenarios (roles: {'Minion', 'Witch', 'Lilis'})

### [18:27:40] Executed #2 -> Witch (EVIL)

### [18:28:21] Revealed #8 Poet
Info: {'corruption_distance': 2, 'copied_role': 'Bard'}

#### [18:28:24] Solver Output
Scenarios: 4/162
Definite evil: ['#2']
Definite good: ['#3', '#4', '#5', '#9']
Evil probabilities: #7=75%, #1=50%, #6=50%, #8=25%

#### [18:28:24] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#1']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [18:29:14] Ability used at #3

#### [18:29:21] Solver Output
Scenarios: 2/162
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#5', '#6', '#9']
Evil probabilities: #7=50%, #8=50%

#### [18:29:21] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Lilis'})

### [18:30:12] Executed #1 -> Lilis (EVIL)

#### [18:30:15] Solver Output
Scenarios: 2/26
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#5', '#6', '#9']
Evil probabilities: #7=50%, #8=50%

#### [18:30:15] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% good Medium (corrupted), 50% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [18:31:00] Executed #7 -> Minion (EVIL)

## [18:31:06] GAME OVER — WIN
Final HP: 6
Notes: Asc45 complete! Lilis+Witch+Minion, Knight free check, PD clean check, 50-50 hit on Minion, 6HP


---

# New Game — 2026-04-09 21:04:25
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Druid, Enlightened, Poet, Baker, Fortune_Teller, Confessor
- Outcasts: Plague_Doctor
- Minions: Minion, Shaman
- Demons: Pooka

### [21:06:24] Revealed #1 Baker
Info: {'original_role': 'Fortune Teller'}

### [21:06:24] Revealed #3 Druid
Info: {}

### [21:06:24] Revealed #4 Baker
Info: {'original_role': 'original'}

### [21:06:24] Revealed #5 Confessor
Info: {'dizzy': True}

### [21:06:24] Revealed #6 Plague_Doctor
Info: {}

### [21:06:24] Revealed #7 Baker
Info: {'original_role': 'Enlightened'}

### [21:06:24] Revealed #8 Bishop
Info: {'targets': [6, 3, 1], 'types': ['Villager', 'Outcast', 'Minion']}

### [21:06:24] Revealed #9 Baker
Info: {'original_role': 'Confessor'}

### [21:06:46] Revealed #2 Poet
Info: {'targets': [3], 'has_evil': False, 'copied_role': 'Fortune Teller'}

#### [21:06:51] Claude Reasoning


### [21:08:35] Revealed #4 Baker
Info: {'original_role': 'none'}

#### [21:08:45] Solver Output
Scenarios: 26/1848
Definite good: ['#2', '#3', '#6']
Evil probabilities: #5=92%, #9=77%, #7=62%, #8=38%, #1=15%, #4=15%

#### [21:08:45] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#1']
Reason: Entropy 2.377 (adjusted 2.377) | timing x1.00

### [21:09:35] Revealed #6 Pd Check
Info: {}

### [21:09:39] Ability used at #6

#### [21:09:43] Solver Output
Scenarios: 4/504
Definite evil: ['#5']
Definite good: ['#2', '#3', '#4', '#6']
Evil probabilities: #1=50%, #7=50%, #8=50%, #9=50%

#### [21:09:43] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 4 scenarios (roles: {'Minion', 'Shaman'})

### [21:10:25] Executed #5 -> Shaman (EVIL)

#### [21:10:29] Solver Output
Scenarios: 2/56
Definite evil: ['#5']
Definite good: ['#2', '#3', '#4', '#6']
Evil probabilities: #1=50%, #7=50%, #8=50%, #9=50%

#### [21:10:29] Recommendation
Action: **USE_ABILITY** #3 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 0.000 (adjusted 0.000) | timing x1.00

### [21:13:37] Revealed #3 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': 'Wretch'}

### [21:13:41] Ability used at #3

#### [21:13:46] Solver Output
Scenarios: 0/56

#### [21:13:46] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [21:13:57] Revealed #3 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': None}

#### [21:14:00] Solver Output
Scenarios: 2/56
Definite evil: ['#5']
Definite good: ['#2', '#3', '#4', '#6']
Evil probabilities: #1=50%, #7=50%, #8=50%, #9=50%

#### [21:14:00] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% good Baker (corrupted), 50% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [21:14:45] Executed #1 -> GOOD (WRONG!)

#### [21:14:53] Solver Output
Scenarios: 1/42
Definite evil: ['#5', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#8']

#### [21:14:53] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Minion'})

## [21:17:31] GAME OVER — LOSS
Final HP: 5
Notes: Solver bug: Shaman-swapped Bakers' 'I was a [role]' misinterpreted as Baker ability (previous card role reveal) instead of self-identification of original role. Solver wrongly concluded #7 was evil Minion with 100% confidence in 1 scenario.


---

# New Game — 2026-04-09 21:38:08
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Confessor, Architect, Knight, Judge, Slayer
- Outcasts: Bombardier
- Minions: Twin_Minion
- Demons: Pooka

### [21:39:06] Revealed #1 Slayer
Info: {}

### [21:39:06] Revealed #2 Architect
Info: {'side': 'Left'}

### [21:39:06] Revealed #3 Bombardier
Info: {}

### [21:39:06] Revealed #4 Knight
Info: {}

### [21:39:06] Revealed #5 Knight
Info: {}

### [21:39:06] Revealed #6 Bishop
Info: {'targets': [8, 3, 5], 'types': ['Villager', 'Outcast', 'Minion']}

### [21:39:06] Revealed #7 Judge
Info: {}

### [21:39:06] Revealed #8 Confessor
Info: {'dizzy': False}

#### [21:39:26] Solver Output
Scenarios: 8/56
Definite good: ['#1', '#3', '#8']
Evil probabilities: #5=75%, #4=50%, #6=38%, #2=25%, #7=12%

#### [21:39:26] Recommendation
Action: **EXECUTE** #5
Reason: Knight check: #5 is 75% evil, 25% corruption risk. Expected HP cost: 0.6 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 25% -- corrupted Knight loses immunity + 4 extra damage

### [21:40:10] Executed #5 -> Twin_Minion (EVIL)

#### [21:40:15] Solver Output
Scenarios: 2/7
Definite evil: ['#5']
Definite good: ['#1', '#3', '#6', '#7', '#8']
Evil probabilities: #2=50%, #4=50%

#### [21:40:15] Recommendation
Action: **EXECUTE** #4
Reason: Knight free check: #4 is 50% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [21:41:06] Executed #4 -> Pooka (EVIL)

## [21:41:15] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Two Knights = both evil. Bishop confirmed #5 Minion. Knight free check on #4.


---

# New Game — 2026-04-09 21:42:45
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Gemcrafter, Baker, Scout, Bard, Slayer, Oracle, Dreamer
- Outcasts: Plague_Doctor
- Minions: Minion, Twin_Minion
- Demons: Pooka

### [21:44:05] Revealed #1 Gemcrafter
Info: {'good_position': 6}

### [21:44:05] Revealed #2 Baker
Info: {'original_role': 'original'}

### [21:44:05] Revealed #5 Baker
Info: {'original_role': 'Oracle'}

### [21:44:05] Revealed #6 Knitter
Info: {'evil_pairs': 0}

### [21:44:05] Revealed #8 Bard
Info: {'corruption_distance': 2}

### [21:44:05] Revealed #9 Baker
Info: {'original_role': 'Scout'}

### [21:45:16] Revealed #3 Dreamer
Info: {}

### [21:45:16] Revealed #4 Plague_Doctor
Info: {}

### [21:45:16] Revealed #7 Slayer
Info: {}

#### [21:45:41] Solver Output
Scenarios: 96/1848
Definite good: ['#4']
Evil probabilities: #6=73%, #5=48%, #7=46%, #9=46%, #8=40%, #1=25%, #3=17%, #2=6%

#### [21:45:41] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#6']
Reason: Entropy 2.761 (adjusted 2.617) | timing x1.00
WARNING: Corruption risk: 10%

### [21:46:41] Revealed #3 Dreamer
Info: {'target': 6, 'evil_role': 'Twin_Minion'}

### [21:46:48] Ability used at #3

#### [21:46:48] Solver Output
Scenarios: 55/1848
Definite good: ['#4']
Evil probabilities: #5=55%, #8=55%, #6=53%, #7=45%, #9=40%, #3=25%, #1=16%, #2=11%

#### [21:46:48] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#9']
Reason: Entropy 1.972 (adjusted 1.972) | timing x1.00

### [21:47:27] Ability used at #4

#### [21:47:34] Solver Output
Scenarios: 12/1848
Definite evil: ['#7']
Definite good: ['#2', '#4', '#9']
Evil probabilities: #8=92%, #5=50%, #3=33%, #6=17%, #1=8%

#### [21:47:34] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 12 scenarios (roles: {'Pooka', 'Twin_Minion', 'Minion'})

### [21:48:09] Executed #7 -> Pooka (EVIL)

#### [21:48:09] Solver Output
Scenarios: 5/224
Definite evil: ['#7']
Definite good: ['#2', '#4', '#9']
Evil probabilities: #8=80%, #3=40%, #5=40%, #1=20%, #6=20%

#### [21:48:09] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (40% evil Minion, 40% evil Twin_Minion, 20% good Bard (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 80%, but all reveal branches still lead to a forced win.

### [21:48:46] Executed #8 -> Twin_Minion (EVIL)

#### [21:48:46] Solver Output
Scenarios: 2/31
Definite evil: ['#7', '#8']
Definite good: ['#1', '#2', '#4', '#6', '#9']
Evil probabilities: #3=50%, #5=50%

#### [21:48:46] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% good Dreamer, 50% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [21:49:31] Executed #3 -> Minion (EVIL)

## [21:49:31] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. PD confirmed #7 evil + #9 corrupted. Baker chain, no Shaman.


---

# New Game — 2026-04-09 21:50:42
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Jester, Dreamer, Fortune_Teller, Lover, Scout
- Outcasts: Wretch, Bombardier
- Minions: Twin_Minion
- Demons: Pooka

### [21:51:21] Revealed #2 Scout
Info: {'evil_role': 'Twin Minion', 'distance': 1}

### [21:51:21] Revealed #4 Scout
Info: {'evil_role': 'Twin Minion', 'distance': 3}

### [21:51:21] Revealed #7 Lover
Info: {'evil_adjacent': 0}

### [21:51:21] Revealed #8 Alchemist
Info: {'cured_count': 1}

### [21:51:21] Revealed #9 Wretch
Info: {}

### [21:51:58] Revealed #1 Jester
Info: {}

### [21:51:58] Revealed #3 Bombardier
Info: {}

### [21:51:59] Revealed #5 Fortune_Teller
Info: {}

### [21:51:59] Revealed #6 Dreamer
Info: {}

#### [21:52:05] Solver Output
Scenarios: 6/72
Definite good: ['#1', '#3', '#6', '#7', '#9']
Evil probabilities: #2=67%, #4=50%, #5=50%, #8=33%

#### [21:52:05] Recommendation
Action: **USE_ABILITY** #6 (Dreamer) -> targets ['#2']
Reason: Entropy 1.585 (adjusted 1.585) | timing x1.00

### [21:52:42] Revealed #6 Dreamer
Info: {'target': 2, 'evil_role': 'Pooka'}

### [21:52:43] Ability used at #6

#### [21:52:43] Solver Output
Scenarios: 4/72
Definite good: ['#1', '#3', '#6', '#7', '#9']
Evil probabilities: #4=75%, #2=50%, #5=50%, #8=25%

#### [21:52:43] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#2', '#3', '#5']
Reason: Expected posterior 1.5 scenarios (adjusted 1.5, info gain 1.415 bits) | timing x1.00

### [21:53:27] Revealed #1 Jester
Info: {'targets': [2, 3, 5], 'evil_count': 1}

### [21:53:27] Ability used at #1

#### [21:53:27] Solver Output
Scenarios: 2/72
Definite evil: ['#4']
Definite good: ['#1', '#3', '#6', '#7', '#8', '#9']
Evil probabilities: #2=50%, #5=50%

#### [21:53:27] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Twin_Minion'})

### [21:54:05] Executed #4 -> Twin_Minion (EVIL)

#### [21:54:05] Solver Output
Scenarios: 2/8
Definite evil: ['#4']
Definite good: ['#1', '#3', '#6', '#7', '#8', '#9']
Evil probabilities: #2=50%, #5=50%

#### [21:54:05] Recommendation
Action: **USE_ABILITY** #5 (Fortune Teller) -> targets ['#1', '#3']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [21:55:35] Revealed #5 Fortune Teller
Info: {'targets': [1, 3], 'has_evil': False}

### [21:55:42] Ability used at #5

#### [21:55:42] Solver Output
Scenarios: 1/8
Definite evil: ['#2', '#4']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8', '#9']

#### [21:55:42] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [21:56:24] Executed #2 -> Pooka (EVIL)

## [21:56:24] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Dreamer found Pooka at #2, FT cleared #1+#3, Jester confirmed 1 evil in group.


---

# New Game — 2026-04-09 21:57:44
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Hunter, Oracle, Bishop, Poet, Alchemist
- Outcasts: Bombardier, Wretch, Plague_Doctor
- Minions: Witch, Chancellor
- Demons: Baa

### [21:58:20] Revealed #2 Oracle
Info: {'targets': [6, 9], 'minion_role': 'Witch'}

### [21:58:20] Revealed #6 Oracle
Info: {'targets': [2, 8], 'minion_role': 'Chancellor'}

### [21:58:20] Revealed #7 Hunter
Info: {'distance': 3}

### [21:59:09] Revealed #1 Bombardier
Info: {}

### [21:59:09] Revealed #3 Bombardier
Info: {}

### [21:59:09] Revealed #4 Slayer
Info: {}

### [21:59:09] Revealed #5 Alchemist
Info: {'cured_count': 2}

### [21:59:10] Revealed #8 Plague_Doctor
Info: {}

#### [21:59:15] Solver Output
Scenarios: 22/2064
Definite good: ['#8', '#9']
Evil probabilities: #3=55%, #6=55%, #1=45%, #2=45%, #5=45%, #7=36%, #4=18%

#### [21:59:15] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.154 (adjusted 2.154) | timing x1.00

### [21:59:56] Ability used at #8

#### [21:59:57] Solver Output
Scenarios: 10/2064
Definite evil: ['#5']
Definite good: ['#4', '#7', '#8', '#9']
Evil probabilities: #6=60%, #1=50%, #3=50%, #2=40%

#### [21:59:57] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 10 scenarios (roles: {'Baa', 'Chancellor', 'Witch'})

### [22:00:36] Executed #5 -> Baa (EVIL)

#### [22:00:36] Solver Output
Scenarios: 6/243
Definite evil: ['#5']
Definite good: ['#4', '#7', '#8', '#9']
Evil probabilities: #6=67%, #1=50%, #3=50%, #2=33%

#### [22:00:36] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#6']
Reason: Target #6 is 67% evil (adjusted 0.67)

### [22:01:17] Ability used at #4

#### [22:01:18] Solver Output
Scenarios: 2/243
Definite evil: ['#2', '#5']
Definite good: ['#4', '#6', '#7', '#8', '#9']
Evil probabilities: #1=50%, #3=50%

#### [22:01:18] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Chancellor'})

### [22:01:54] Executed #2 -> Chancellor (EVIL)

#### [22:01:54] Solver Output
Scenarios: 2/27
Definite evil: ['#2', '#5']
Definite good: ['#4', '#6', '#7', '#8', '#9']
Evil probabilities: #1=50%, #3=50%

#### [22:01:54] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 0% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 0% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Low confidence (0% < 50%) -- consider gathering more info

### [22:03:55] Revealed #1 Jester
Info: {'targets': [2, 3, 5], 'evil_count': 1}

#### [22:03:55] Solver Output
Scenarios: 1/31
Definite evil: ['#1', '#2', '#5']
Definite good: ['#3', '#4', '#6', '#7', '#8', '#9']

#### [22:03:55] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Witch'})

### [22:04:37] Executed #1 -> Witch (EVIL)

## [22:04:37] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Jester lie caught Witch among Bombardier twins. PD clean check + Oracle + Slayer narrowed field.


---

# New Game — 2026-04-09 22:05:50
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Gemcrafter, Scout, Lover, Knight, Slayer, Poet
- Outcasts: Plague_Doctor, Wretch
- Minions: Minion
- Demons: Baa

### [22:06:16] Revealed #3 Wretch
Info: {}

### [22:06:16] Revealed #4 Gemcrafter
Info: {'good_position': 8}

### [22:06:16] Revealed #5 Slayer
Info: {}

### [22:06:16] Revealed #6 Scout
Info: {'evil_role': 'Baa', 'distance': 1}

### [22:06:16] Revealed #7 Lover
Info: {'evil_adjacent': 0}

### [22:06:45] Revealed #1 Poet
Info: {'targets': [5, 7, 8], 'types': ['Villager', 'Outcast', 'Minion'], 'copied_role': 'Bishop'}

### [22:06:45] Revealed #2 Slayer
Info: {}

### [22:06:45] Revealed #8 Knight
Info: {}

#### [22:06:52] Solver Output
Scenarios: 2/56
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#5', '#6', '#7', '#8']

#### [22:06:52] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Baa', 'Minion'})

### [22:07:30] Executed #1 -> Baa (EVIL)

#### [22:07:31] Solver Output
Scenarios: 1/7
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#5', '#6', '#7', '#8']

#### [22:07:31] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Minion'})

### [22:08:15] Executed #2 -> Minion (EVIL)

## [22:08:15] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Poet-Bishop + Scout + Lover + Gemcrafter narrowed to 2 scenarios, both had #1 evil.


---

# New Game — 2026-04-09 22:09:34
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Enlightened, Scout, Knight, Oracle, Slayer, Bard
- Outcasts: Plague_Doctor, Wretch
- Minions: Poisoner
- Demons: Baa

### [22:09:55] Revealed #2 Baker
Info: {'original_role': 'Scout'}

### [22:09:55] Revealed #3 Plague_Doctor
Info: {}

### [22:09:55] Revealed #5 Enlightened
Info: {'direction': 'CW'}

### [22:09:55] Revealed #6 Oracle
Info: {'targets': [4, 5], 'minion_role': 'Poisoner'}

### [22:09:55] Revealed #7 Bard
Info: {'corruption_distance': 1}

### [22:09:55] Revealed #8 Scout
Info: {'evil_role': 'Poisoner', 'distance': 3}

### [22:10:20] Revealed #1 Slayer
Info: {}

### [22:10:20] Revealed #4 Knight
Info: {}

#### [22:10:20] Solver Output
Scenarios: 10/396
Definite good: ['#3']
Evil probabilities: #6=60%, #2=40%, #1=30%, #8=30%, #4=20%, #5=10%, #7=10%

#### [22:10:20] Recommendation
Action: **EXECUTE** #4
Reason: Knight check: #4 is 20% evil, 10% corruption risk. Expected HP cost: 0.7 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 10% -- corrupted Knight loses immunity + 4 extra damage

### [22:11:12] Ability used at #3

#### [22:11:12] Solver Output
Scenarios: 4/396
Definite good: ['#3', '#5']
Evil probabilities: #2=50%, #4=50%, #1=25%, #6=25%, #7=25%, #8=25%

#### [22:11:12] Recommendation
Action: **EXECUTE** #4
Reason: Knight free check: #4 is 50% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [22:11:52] Executed #4 -> Poisoner (EVIL)

#### [22:11:52] Solver Output
Scenarios: 2/37
Definite evil: ['#4']
Definite good: ['#1', '#3', '#5', '#6', '#8']
Evil probabilities: #2=50%, #7=50%

#### [22:11:52] Recommendation
Action: **USE_ABILITY** #1 (Slayer) -> targets ['#2']
Reason: Target #2 is 50% evil (adjusted 0.50)

### [22:12:31] Ability used at #1

#### [22:12:31] Solver Output
Scenarios: 1/37
Definite evil: ['#4', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#8']

#### [22:12:31] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Baa'})

### [22:13:15] Executed #7 -> Baa (EVIL)

## [22:13:15] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Knight free check caught Poisoner, Slayer cleared #2, solver found Baa at #7.


---

# New Game — 2026-04-09 22:14:39
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Gemcrafter, Dreamer, Fortune_Teller, Medium, Baker, Knight, Jester
- Outcasts: Plague_Doctor
- Minions: Minion, Witch
- Demons: Lilis

### [22:16:46] Revealed #1 Gemcrafter
Info: {'good_position': 8}

### [22:16:46] Revealed #2 Medium
Info: {'good_position': 5, 'good_role': 'Dreamer'}

### [22:16:46] Revealed #3 Baker
Info: {'original_role': 'Fortune Teller'}

### [22:16:46] Revealed #7 Baker
Info: {'original_role': 'original'}

### [22:16:58] Revealed #4 Plague_Doctor
Info: {}

### [22:16:58] Revealed #5 Dreamer
Info: {}

### [22:16:58] Revealed #6 Fortune_Teller
Info: {}

#### [22:16:58] Solver Output
Scenarios: 114/1848
Definite good: ['#9']
Evil probabilities: #3=79%, #6=42%, #7=42%, #5=37%, #8=37%, #1=26%, #2=26%, #4=11%

#### [22:16:58] Recommendation
Action: **USE_ABILITY** #5 (Dreamer) -> targets ['#3']
Reason: Entropy 2.991 (adjusted 2.834) | timing x1.00
WARNING: Corruption risk: 11%

### [22:17:40] Revealed #5 Dreamer
Info: {'target': 3, 'evil_role': 'Witch'}

### [22:17:40] Ability used at #5

#### [22:17:40] Solver Output
Scenarios: 68/1848
Definite good: ['#9']
Evil probabilities: #3=65%, #5=47%, #6=41%, #7=41%, #2=35%, #8=35%, #1=29%, #4=6%

#### [22:17:40] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#3']
Reason: Entropy 2.106 (adjusted 2.106) | timing x1.00

### [22:18:16] Ability used at #4

#### [22:18:16] Solver Output
Scenarios: 40/1848
Definite evil: ['#3']
Definite good: ['#4', '#9']
Evil probabilities: #5=50%, #6=35%, #7=35%, #2=30%, #8=30%, #1=20%

#### [22:18:16] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 40 scenarios (roles: {'Minion', 'Witch', 'Lilis'})

### [22:18:56] Executed #3 -> Witch (EVIL)

### [22:19:33] Revealed #8 Knight
Info: {}

#### [22:19:33] Solver Output
Scenarios: 8/62
Definite evil: ['#3', '#8']
Definite good: ['#2', '#4', '#5', '#9']
Evil probabilities: #1=50%, #6=25%, #7=25%

#### [22:19:33] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 8 scenarios (roles: {'Minion', 'Lilis'})

### [22:20:19] Executed #8 -> Lilis (EVIL)

#### [22:20:19] Solver Output
Scenarios: 4/31
Definite evil: ['#3', '#8']
Definite good: ['#2', '#4', '#5', '#9']
Evil probabilities: #1=50%, #6=25%, #7=25%

#### [22:20:19] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 1.000 (adjusted 0.875) | timing x1.00
WARNING: Corruption risk: 25%

### [22:21:09] Revealed #6 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [22:21:09] Ability used at #6

#### [22:21:09] Solver Output
Scenarios: 2/31
Definite evil: ['#3', '#8']
Definite good: ['#2', '#4', '#5', '#6', '#9']
Evil probabilities: #1=50%, #7=50%

#### [22:21:09] Recommendation
Action: **ERROR** #7
Reason: #7 is 50% likely evil but HP too low to risk (HP=4, cost=5, threshold=95%). Need more info.
WARNING: Probabilistic execution -- 50% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=4, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [22:22:55] Executed #1 -> Minion (EVIL)

## [22:22:56] GAME OVER — WIN
Final HP: 4
Notes: 4HP. Lilis+Witch+Minion. Dreamer found Witch, PD clean checks, night_no_kill confirmed Lilis. Solver 50/50 on last exec but Gemcrafter lie on #8 proved #1 evil.


---

# New Game — 2026-04-09 22:25:27
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Oracle, Confessor, Empress, Baker, Judge
- Outcasts: Doppelganger
- Minions: Witch, Twin_Minion
- Demons: Lilis

### [22:27:18] Revealed #2 Oracle
Info: {'targets': [3, 9], 'minion_role': 'Twin_Minion'}

### [22:27:18] Revealed #3 Hunter
Info: {'distance': 1}

### [22:27:18] Revealed #4 Confessor
Info: {'dizzy': False}

### [22:27:18] Revealed #8 Confessor
Info: {'dizzy': True}

### [22:27:33] Revealed #1 Judge
Info: {}

### [22:27:33] Revealed #6 Judge
Info: {}

### [22:27:33] Revealed #7 Judge
Info: {}

#### [22:27:33] Solver Output
Scenarios: 144/3024
Definite evil: ['#8']
Definite good: ['#4', '#5', '#9']
Evil probabilities: #2=75%, #1=33%, #6=33%, #7=33%, #3=25%

#### [22:27:33] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 144 scenarios (roles: {'Lilis', 'Witch', 'Twin_Minion'})

### [22:28:18] Executed #8 -> Twin_Minion (EVIL)

#### [22:28:18] Solver Output
Scenarios: 36/336
Definite evil: ['#2', '#8']
Definite good: ['#3', '#4', '#5', '#9']
Evil probabilities: #1=33%, #6=33%, #7=33%

#### [22:28:18] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 36 scenarios (roles: {'Witch', 'Lilis'})

### [22:28:59] Executed #2 -> Lilis (EVIL)

#### [22:29:00] Solver Output
Scenarios: 18/42
Definite evil: ['#2', '#8']
Definite good: ['#3', '#4', '#5', '#9']
Evil probabilities: #1=33%, #6=33%, #7=33%

#### [22:29:00] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#3']
Reason: Expected posterior 10.0 scenarios (adjusted 10.0, info gain 0.848 bits) | timing x1.00

### [22:30:13] Revealed #1 Judge
Info: {'target': 3, 'is_lying': False}

### [22:30:13] Ability used at #1

#### [22:30:13] Solver Output
Scenarios: 12/42
Definite evil: ['#2', '#8']
Definite good: ['#1', '#3', '#4', '#5', '#9']
Evil probabilities: #6=50%, #7=50%

#### [22:30:13] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#1']
Reason: Expected posterior 6.0 scenarios (adjusted 6.0, info gain 1.000 bits) | timing x1.00

### [22:30:55] Revealed #6 Judge
Info: {'target': 1, 'is_lying': False}

### [22:30:55] Ability used at #6

#### [22:30:55] Solver Output
Scenarios: 6/42
Definite evil: ['#2', '#7', '#8']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#9']

#### [22:30:55] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 6 scenarios (roles: {'Witch'})

### [22:31:46] Executed #7 -> Witch (EVIL)

## [22:31:46] GAME OVER — WIN
Final HP: 6
Notes: 6HP. Lilis+Witch+Twin_Minion. 3 Judges (1 pool) = duplicates reveal. Judge truth checks narrowed Witch to #7.


---

# New Game — 2026-04-09 22:33:29
Cards: 8, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Knitter, Confessor, Architect, Poet, Medium
- Outcasts: Plague_Doctor
- Minions: Puppeteer
- Demons: Lilis

### [22:34:46] Revealed #1 Confessor
Info: {'dizzy': True}

### [22:34:46] Revealed #2 Knitter
Info: {'evil_pairs': 2}

### [22:34:46] Revealed #3 Architect
Info: {'side': 'Equal'}

### [22:34:46] Revealed #7 Medium
Info: {'good_position': 8, 'good_role': 'Knitter'}

### [22:34:46] Revealed #8 Knitter
Info: {'evil_pairs': 0}

### [22:35:13] Revealed #4 Jester
Info: {}

### [22:35:13] Revealed #5 Plague_Doctor
Info: {}

#### [22:35:13] Solver Output
Scenarios: 4/264
Definite evil: ['#1']
Definite good: ['#4', '#5', '#6']
Evil probabilities: #2=50%, #3=50%, #7=50%, #8=50%

#### [22:35:13] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Puppeteer', 'Puppet', 'Lilis'})

### [22:35:56] Executed #1 -> Puppet (EVIL)

#### [22:35:57] Solver Output
Scenarios: 2/42
Definite evil: ['#1', '#7', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#6']

#### [22:35:57] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Lilis'})

### [22:36:40] Executed #7 -> Lilis (EVIL)

#### [22:36:40] Solver Output
Scenarios: 2/8
Definite evil: ['#1', '#7', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#6']

#### [22:36:40] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 2 scenarios (roles: {'Puppeteer'})

### [22:37:25] Executed #8 -> Puppeteer (EVIL)

## [22:37:25] GAME OVER — WIN
Final HP: 8
Notes: 8HP. Puppeteer+Puppet+Lilis. Duplicate Knitter + dizzy Confessor = Puppet/Puppeteer confirmed. Lilis night killed Poet.


---

# New Game — 2026-04-09 22:38:41
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Bard, Dreamer, Alchemist, Medium, Lover
- Outcasts: Plague_Doctor, Doppelganger
- Minions: Puppeteer
- Demons: Pooka

### [22:39:08] Revealed #1 Lover
Info: {'evil_adjacent': 2}

### [22:39:08] Revealed #2 Bard
Info: {'corruption_distance': -1}

### [22:39:08] Revealed #3 Bard
Info: {'corruption_distance': 1}

### [22:39:08] Revealed #4 Oracle
Info: {'targets': [4, 8], 'minion_role': 'Puppet'}

### [22:39:08] Revealed #5 Lover
Info: {'evil_adjacent': 0}

### [22:39:08] Revealed #7 Alchemist
Info: {'cured_count': 0}

### [22:39:08] Revealed #8 Medium
Info: {'good_position': 6, 'good_role': 'Dreamer'}

### [22:39:34] Revealed #7 Alchemist
Info: {'cured_count': 2}

### [22:39:34] Revealed #6 Dreamer
Info: {}

### [22:39:34] Revealed #9 Plague_Doctor
Info: {}

#### [22:39:35] Solver Output
Scenarios: 23/2268
Definite good: ['#3', '#6', '#9']
Evil probabilities: #7=78%, #8=78%, #1=57%, #2=43%, #4=22%, #5=22%

#### [22:39:35] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.742 (adjusted 1.742) | timing x1.00

### [22:40:15] Ability used at #9

#### [22:40:15] Solver Output
Scenarios: 10/2268
Definite evil: ['#1']
Definite good: ['#2', '#3', '#6', '#9']
Evil probabilities: #7=70%, #8=70%, #4=30%, #5=30%

#### [22:40:15] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 10 scenarios (roles: {'Pooka'})

### [22:41:03] Executed #1 -> Pooka (EVIL)

#### [22:41:03] Solver Output
Scenarios: 10/306
Definite evil: ['#1']
Definite good: ['#2', '#3', '#6', '#9']
Evil probabilities: #7=70%, #8=70%, #4=30%, #5=30%

#### [22:41:03] Recommendation
Action: **USE_ABILITY** #6 (Dreamer) -> targets ['#4']
Reason: Entropy 1.571 (adjusted 1.335) | timing x1.00
WARNING: Corruption risk: 30%

### [22:41:41] Revealed #6 Dreamer
Info: {'target': 4, 'evil_role': 'Puppet'}

### [22:41:41] Ability used at #6

#### [22:41:42] Solver Output
Scenarios: 6/306
Definite evil: ['#1']
Definite good: ['#2', '#3', '#6', '#9']
Evil probabilities: #4=50%, #5=50%, #7=50%, #8=50%

#### [22:41:42] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (50% evil Puppet, 33% good Oracle, 17% good Doppelganger).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [22:42:25] Executed #4 -> Puppet (EVIL)

#### [22:42:25] Solver Output
Scenarios: 3/50
Definite evil: ['#1', '#4', '#5']
Definite good: ['#2', '#3', '#6', '#7', '#8', '#9']

#### [22:42:25] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 3 scenarios (roles: {'Puppeteer'})

### [22:43:14] Executed #5 -> Puppeteer (EVIL)

## [22:43:15] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. PD clean + Dreamer found Puppet. Duplicate Lover + Oracle Puppet detection. All 100% confident execs.


---

# New Game — 2026-04-09 22:44:36
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Oracle, Gemcrafter, Alchemist, Scout, Architect, Judge
- Outcasts: Drunk, Wretch
- Minions: Chancellor, Shaman
- Demons: Lilis

### [22:46:15] Revealed #1 Judge
Info: {}

### [22:46:15] Revealed #2 Scout
Info: {'evil_role': 'Lilis', 'distance': 3}

### [22:46:15] Revealed #3 Wretch
Info: {}

### [22:46:15] Revealed #4 Judge
Info: {}

### [22:46:15] Revealed #5 Gemcrafter
Info: {'good_position': 3}

### [22:46:15] Revealed #7 Architect
Info: {'side': 'Right'}

### [22:46:15] Revealed #8 Gemcrafter
Info: {'good_position': 3}

### [22:46:29] Revealed #6 Medium
Info: {'good_position': 5, 'good_role': 'Drunk'}

#### [22:46:29] Solver Output
Scenarios: 43/5496
Definite good: ['#9', '#10']
Evil probabilities: #8=86%, #2=77%, #5=37%, #1=28%, #4=23%, #3=21%, #6=14%, #7=14%

#### [22:46:29] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#6']
Reason: Expected posterior 21.5 scenarios (adjusted 21.5, info gain 0.999 bits) | timing x1.00

### [22:47:23] Revealed #1 Judge
Info: {'target': 6, 'is_lying': True}

### [22:47:23] Ability used at #1

#### [22:47:23] Solver Output
Scenarios: 22/5496
Definite good: ['#7', '#9', '#10']
Evil probabilities: #8=73%, #2=68%, #5=59%, #1=41%, #6=27%, #4=18%, #3=14%

#### [22:47:23] Recommendation
Action: **USE_ABILITY** #4 (Judge) -> targets ['#1']
Reason: Expected posterior 11.4 scenarios (adjusted 11.4, info gain 0.953 bits) | timing x1.00

### [22:48:03] Revealed #4 Judge
Info: {'target': 1, 'is_lying': False}

### [22:48:04] Ability used at #4

#### [22:48:04] Solver Output
Scenarios: 13/5496
Definite good: ['#3', '#7', '#9', '#10']
Evil probabilities: #2=85%, #5=85%, #8=54%, #6=46%, #1=15%, #4=15%

#### [22:48:04] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (38% evil Lilis, 23% evil Chancellor, 23% evil Shaman).
WARNING: Execution lookahead override -- immediate hit chance is 85%, but all reveal branches still lead to a forced win.

### [22:49:01] Executed #2 -> GOOD (WRONG!)

#### [22:49:01] Solver Output
Scenarios: 2/3970
Definite evil: ['#1', '#4', '#8']
Definite good: ['#2', '#3', '#5', '#6', '#7', '#9', '#10']

#### [22:49:01] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Chancellor', 'Shaman'})

### [22:49:56] Executed #1 -> Shaman (EVIL)

#### [22:49:56] Solver Output
Scenarios: 1/485
Definite evil: ['#1', '#4', '#8']
Definite good: ['#2', '#3', '#5', '#6', '#7', '#9', '#10']

#### [22:49:56] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Chancellor'})

### [22:50:44] Executed #4 -> Chancellor (EVIL)

#### [22:50:44] Solver Output
Scenarios: 1/43
Definite evil: ['#1', '#4', '#8']
Definite good: ['#2', '#3', '#5', '#6', '#7', '#9', '#10']

#### [22:50:44] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [22:51:35] Executed #8 -> Lilis (EVIL)

## [22:51:35] GAME OVER — WIN
Final HP: 1
Notes: 1HP clutch! Shaman+Chancellor+Lilis. 10-card game. Wrong exec on #2 (85% gamble) but solver lookahead guaranteed win. Judge truth checks + Architect narrowed remaining evils to 100%.


---

# New Game — 2026-04-09 22:53:12
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Hunter, Gemcrafter, Alchemist, Baker, Druid, Poet
- Outcasts: Doppelganger, Drunk
- Minions: Minion
- Demons: Baa

### [22:53:37] Revealed #2 Druid
Info: {}

### [22:53:37] Revealed #3 Baker
Info: {'original_role': 'original'}

### [22:53:37] Revealed #4 Baker
Info: {'original_role': 'Gemcrafter'}

### [22:53:37] Revealed #5 Hunter
Info: {'distance': 1}

### [22:53:37] Revealed #6 Medium
Info: {'good_position': 5, 'good_role': 'Hunter'}

### [22:53:37] Revealed #7 Baker
Info: {'original_role': 'Alchemist'}

### [22:53:57] Revealed #1 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 3}

#### [22:53:58] Solver Output
Scenarios: 74/1302
Evil probabilities: #4=70%, #1=59%, #6=19%, #2=14%, #5=14%, #7=14%, #3=11%

#### [22:53:58] Recommendation
Action: **USE_ABILITY** #2 (Druid) -> targets ['#1', '#3', '#4']
Reason: Entropy 0.842 (adjusted 0.785) | timing x1.00
WARNING: Corruption risk: 14%

### [22:54:48] Revealed #2 Druid
Info: {'targets': [1, 3, 4], 'found_outcast': None}

### [22:54:48] Ability used at #2

#### [22:54:49] Solver Output
Scenarios: 42/1302
Definite good: ['#7']
Evil probabilities: #4=71%, #1=62%, #2=24%, #3=14%, #5=14%, #6=14%

#### [22:54:49] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (36% evil Baa, 36% evil Minion, 29% good Baker).
WARNING: Execution lookahead override -- immediate hit chance is 71%, but all reveal branches still lead to a forced win.

### [22:55:39] Executed #4 -> GOOD (WRONG!)

#### [22:55:39] Solver Output
Scenarios: 12/930
Definite good: ['#3', '#4', '#7']
Evil probabilities: #1=83%, #5=50%, #6=50%, #2=17%

#### [22:55:39] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (42% evil Baa, 42% evil Minion, 17% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 83%, but all reveal branches still lead to a forced win.

### [22:56:29] Executed #1 -> Baa (EVIL)

#### [22:56:29] Solver Output
Scenarios: 5/155
Definite evil: ['#1']
Definite good: ['#2', '#3', '#4', '#7']
Evil probabilities: #5=60%, #6=40%

#### [22:56:29] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (60% good Drunk (corrupted), 40% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 40%, but all reveal branches still lead to a forced win.

### [22:57:25] Executed #6 -> GOOD (WRONG!)

#### [22:57:25] Solver Output
Scenarios: 3/124
Definite evil: ['#1', '#5']
Definite good: ['#2', '#3', '#4', '#6', '#7']

#### [22:57:25] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 3 scenarios (roles: {'Minion'})

### [22:58:17] Executed #5 -> Minion (EVIL)

## [22:58:17] GAME OVER — WIN
Final HP: 3
Notes: 3HP. Poet bounty_hunter + Baker chain + Druid narrowed. Two wrong execs (Baker, Drunk) but lookahead guaranteed win.


---

# New Game — 2026-04-09 22:59:44
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Jester, Druid, Lover, Judge, Dreamer, Gemcrafter
- Outcasts: Plague_Doctor
- Minions: Minion, Twin_Minion
- Demons: Pooka

### [23:00:06] Revealed #1 Oracle
Info: {'targets': [2, 9], 'minion_role': 'Minion'}

### [23:00:06] Revealed #5 Gemcrafter
Info: {'good_position': 1}

### [23:00:06] Revealed #7 Gemcrafter
Info: {'good_position': 1}

### [23:00:06] Revealed #8 Lover
Info: {'evil_adjacent': 1}

### [23:00:40] Revealed #2 Druid
Info: {}

### [23:00:40] Revealed #3 Judge
Info: {}

### [23:00:40] Revealed #4 Plague_Doctor
Info: {}

### [23:00:41] Revealed #6 Jester
Info: {}

### [23:00:41] Revealed #9 Dreamer
Info: {}

#### [23:00:41] Solver Output
Scenarios: 46/1848
Definite evil: ['#1']
Definite good: ['#2', '#4', '#9']
Evil probabilities: #5=70%, #7=65%, #8=30%, #6=26%, #3=9%

#### [23:00:41] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 46 scenarios (roles: {'Minion', 'Twin_Minion', 'Pooka'})

### [23:01:34] Executed #1 -> Pooka (EVIL)

#### [23:01:34] Solver Output
Scenarios: 14/224
Definite evil: ['#1']
Definite good: ['#2', '#4', '#9']
Evil probabilities: #7=86%, #5=71%, #3=14%, #6=14%, #8=14%

#### [23:01:34] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#2']
Reason: Entropy 1.592 (adjusted 1.592) | timing x1.00

### [23:02:21] Ability used at #4

#### [23:02:22] Solver Output
Scenarios: 14/224
Definite evil: ['#1']
Definite good: ['#2', '#4', '#9']
Evil probabilities: #7=86%, #5=71%, #3=14%, #6=14%, #8=14%

#### [23:02:22] Recommendation
Action: **USE_ABILITY** #6 (Jester) -> targets ['#2', '#3', '#7']
Reason: Expected posterior 5.6 scenarios (adjusted 6.0, info gain 1.213 bits) | timing x1.00
WARNING: Corruption risk: 14%

### [23:03:16] Revealed #6 Jester
Info: {'targets': [2, 5, 7], 'evil_count': 2}

### [23:03:16] Ability used at #6

#### [23:03:16] Solver Output
Scenarios: 8/224
Definite evil: ['#1', '#7']
Definite good: ['#2', '#3', '#4', '#8', '#9']
Evil probabilities: #5=75%, #6=25%

#### [23:03:16] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 8 scenarios (roles: {'Twin_Minion', 'Minion'})

### [23:04:04] Executed #7 -> Minion (EVIL)

#### [23:04:04] Solver Output
Scenarios: 4/31
Definite evil: ['#1', '#7']
Definite good: ['#2', '#3', '#4', '#8', '#9']
Evil probabilities: #5=75%, #6=25%

#### [23:04:04] Recommendation
Action: **USE_ABILITY** #3 (Judge) -> targets ['#6']
Reason: Expected posterior 2.6 scenarios (adjusted 2.9, info gain 0.452 bits) | timing x1.00
WARNING: Corruption risk: 25% -- corrupted Judge results are unreliable

### [23:04:50] Revealed #3 Judge
Info: {'target': 6, 'is_lying': True}

### [23:04:50] Ability used at #3

#### [23:04:50] Solver Output
Scenarios: 2/31
Definite evil: ['#1', '#7']
Definite good: ['#2', '#3', '#4', '#8', '#9']
Evil probabilities: #5=50%, #6=50%

#### [23:04:50] Recommendation
Action: **USE_ABILITY** #9 (Dreamer) -> targets ['#5']
Reason: Entropy 1.000 (adjusted 0.500) | timing x1.00
WARNING: Corruption risk: 100%

### [23:05:41] Revealed #9 Dreamer
Info: {'target': 5, 'evil_role': 'Pooka'}

### [23:05:42] Ability used at #9

#### [23:05:42] Solver Output
Scenarios: 2/31
Definite evil: ['#1', '#7']
Definite good: ['#2', '#3', '#4', '#8', '#9']
Evil probabilities: #5=50%, #6=50%

#### [23:05:42] Recommendation
Action: **USE_ABILITY** #2 (Druid) -> targets ['#3', '#4', '#5']
Reason: Entropy 0.000 (adjusted 0.000) | timing x1.00
WARNING: Corruption risk: 100%

### [23:06:45] Executed #5 -> Twin_Minion (EVIL)

## [23:06:45] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Duplicate Gemcrafters + Oracle + Jester + PD corruption check + Judge truth. 50/50 final exec hit.


---

# New Game — 2026-04-09 23:08:13
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Knight, Enlightened, Scout, Oracle, Dreamer
- Outcasts: Wretch, Bombardier
- Minions: Chancellor
- Demons: Pooka

### [23:08:34] Revealed #1 Baker
Info: {'original_role': 'original'}

### [23:08:34] Revealed #3 Wretch
Info: {}

### [23:08:34] Revealed #4 Baker
Info: {'original_role': 'Enlightened'}

### [23:08:34] Revealed #6 Enlightened
Info: {'direction': 'CCW'}

### [23:08:34] Revealed #7 Oracle
Info: {'targets': [5, 6], 'minion_role': 'Chancellor'}

### [23:08:34] Revealed #8 Baker
Info: {'original_role': 'Dreamer'}

### [23:09:07] Revealed #2 Bombardier
Info: {}

### [23:09:07] Revealed #5 Knight
Info: {}

### [23:09:07] Revealed #8 Baker
Info: {'original_role': 'Enlightened'}

#### [23:09:07] Solver Output
Scenarios: 1/76
Definite evil: ['#4', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#8']

#### [23:09:07] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Chancellor'})

### [23:10:02] Executed #4 -> Chancellor (EVIL)

#### [23:10:02] Solver Output
Scenarios: 1/7
Definite evil: ['#4', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#8']

#### [23:10:02] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [23:10:56] Executed #7 -> Pooka (EVIL)

## [23:10:56] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Asc47 complete! Baker chain + duplicate original_role + Enlightened + Oracle narrowed to 1 scenario immediately. Both execs 100% confident.


---

# New Game — 2026-04-09 23:31:28
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Bishop, Scout, Architect, Poet, Bard
- Outcasts: Plague_Doctor, Bombardier, Wretch
- Minions: Chancellor, Minion
- Demons: Pooka

### [23:32:11] Revealed #2 Bishop
Info: {'targets': [1, 4, 7], 'types': ['Villager', 'Outcast', 'Minion']}

### [23:32:11] Revealed #3 Enlightened
Info: {'direction': 'CW'}

### [23:32:11] Revealed #6 Wretch
Info: {}

### [23:32:11] Revealed #7 Scout
Info: {'evil_role': 'Minion', 'distance': 1}

### [23:32:11] Revealed #8 Bard
Info: {'corruption_distance': 2}

### [23:35:10] Revealed #1 Plague_Doctor
Info: {}

### [23:35:10] Revealed #4 Bombardier
Info: {}

### [23:35:10] Revealed #5 Poet
Info: {'good_position': 9, 'good_role': 'Bombardier', 'copied_role': 'Medium'}

### [23:35:10] Revealed #9 Bombardier
Info: {}

#### [23:35:23] Solver Output
Scenarios: 21/1710
Definite good: ['#1', '#2', '#3']
Evil probabilities: #4=90%, #6=62%, #9=62%, #8=48%, #5=33%, #7=5%

#### [23:35:23] Recommendation
Action: **USE_ABILITY** #1 (Plague Doctor) -> targets ['#8']
Reason: Entropy 1.805 (adjusted 1.805) | timing x1.00

### [23:36:20] Ability used at #1

#### [23:36:20] Solver Output
Scenarios: 10/1710
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#7']
Evil probabilities: #6=70%, #9=30%

#### [23:36:20] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 10 scenarios (roles: {'Chancellor', 'Minion'})

### [23:36:59] Executed #4 -> Minion (EVIL)

#### [23:36:59] Solver Output
Scenarios: 4/228
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#7']
Evil probabilities: #6=75%, #9=25%

#### [23:36:59] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 4 scenarios (roles: {'Pooka'})

### [23:37:37] Executed #8 -> Pooka (EVIL)

#### [23:37:38] Solver Output
Scenarios: 4/32
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#7']
Evil probabilities: #6=75%, #9=25%

#### [23:37:38] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (75% evil Chancellor, 25% good Wretch).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [23:38:44] Executed #6 -> GOOD (WRONG!)

#### [23:38:44] Solver Output
Scenarios: 1/22
Definite evil: ['#4', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7']

#### [23:38:44] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Chancellor'})

### [23:39:28] Executed #9 -> Chancellor (EVIL)

## [23:39:28] GAME OVER — WIN
Final HP: 5
Notes: 5HP. Memory-reader-first approach! Only 1 verification screenshot. PD clean check + Bishop + Enlightened + Scout narrowed. Wrong exec on Wretch #6 but lookahead guaranteed win.


---

# New Game — 2026-04-09 23:56:00
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Fortune_Teller, Knitter, Hunter, Enlightened, Architect
- Outcasts: Wretch
- Minions: 
- Demons: Pooka

#### [23:58:33] Claude Reasoning


#### [23:58:37] Solver Output
Scenarios: 7/7
Evil probabilities: #1=14%, #2=14%, #3=14%, #4=14%, #5=14%, #6=14%, #7=14%

#### [23:58:37] Recommendation
Action: **REVEAL** #1
Reason: #1: 14% evil, 2.807 bits (7 outcomes)


---

# New Game — 2026-04-10 00:05:51
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Dreamer, Fortune_Teller, Hunter, Gemcrafter, Architect, Slayer
- Outcasts: Drunk
- Minions: Puppeteer
- Demons: Pooka


---

# New Game — 2026-04-10 00:06:47
Cards: 8, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Dreamer, Fortune_Teller, Hunter, Gemcrafter, Architect, Slayer
- Outcasts: Drunk
- Minions: Puppeteer
- Demons: Pooka

### [00:09:34] Revealed #2 Hunter
Info: {'distance': 2}

### [00:09:34] Revealed #4 Gemcrafter
Info: {'good_position': 6}

### [00:09:34] Revealed #5 Architect
Info: {'side': 'Left'}

### [00:09:34] Revealed #7 Enlightened
Info: {'direction': 'CCW'}

### [00:09:34] Revealed #8 Gemcrafter
Info: {'good_position': 6}

### [00:10:12] Revealed #1 Dreamer
Info: {}

### [00:10:14] Revealed #3 Slayer
Info: {}

### [00:11:11] Revealed #6 Fortune_Teller
Info: {}

#### [00:11:41] Solver Output
Scenarios: 12/480
Definite good: ['#2']
Evil probabilities: #5=67%, #6=58%, #3=50%, #4=42%, #7=33%, #1=25%, #8=25%

#### [00:11:41] Recommendation
Action: **USE_ABILITY** #1 (Dreamer) -> targets ['#5']
Reason: Entropy 2.689 (adjusted 2.353) | timing x1.00
WARNING: Corruption risk: 25%

### [00:12:42] Revealed #1 Dreamer
Info: {'target': 5, 'evil_role': 'Puppeteer'}

### [00:12:44] Ability used at #1

#### [00:12:47] Solver Output
Scenarios: 8/480
Definite good: ['#2']
Evil probabilities: #6=62%, #5=50%, #7=50%, #1=38%, #3=38%, #8=38%, #4=25%

#### [00:12:47] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#2', '#5']
Reason: Entropy 0.954 (adjusted 0.895) | follow-up bonus 0.750 | timing x1.00
WARNING: Corruption risk: 12%

### [00:13:29] Revealed #6 Fortune Teller
Info: {'targets': [2, 5], 'has_evil': False}

### [00:13:33] Ability used at #6

#### [00:13:33] Solver Output
Scenarios: 5/480
Definite good: ['#2']
Evil probabilities: #5=60%, #6=60%, #8=60%, #1=40%, #7=40%, #3=20%, #4=20%

#### [00:13:33] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#5']
Reason: Target #5 is 60% evil (adjusted 0.60)

### [00:14:33] Ability used at #3

#### [00:14:33] Solver Output
Scenarios: 2/60
Definite evil: ['#5', '#6']
Definite good: ['#2', '#3', '#4', '#7']
Evil probabilities: #1=50%, #8=50%

#### [00:14:33] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Puppeteer'})

### [00:15:16] Executed #6 -> Puppeteer (EVIL)

#### [00:15:19] Solver Output
Scenarios: 2/30
Definite evil: ['#5', '#6']
Definite good: ['#2', '#3', '#4', '#7']
Evil probabilities: #1=50%, #8=50%

#### [00:15:19] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% good Dreamer (corrupted), 50% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [00:16:00] Executed #1 -> GOOD (WRONG!)

#### [00:16:09] Solver Output
Scenarios: 1/25
Definite evil: ['#5', '#6', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#7']

#### [00:16:09] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [00:16:53] Executed #8 -> Pooka (EVIL)

## [00:16:56] GAME OVER — WIN
Final HP: 5
Notes: 5HP, Dreamer+FT+Slayer abilities, 1 wrong exec on corrupted Dreamer


---

# New Game — 2026-04-10 00:20:59
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Dreamer, Oracle, Judge, Fortune_Teller, Baker, Knight
- Outcasts: Drunk, Plague_Doctor, Doppelganger, Wretch
- Minions: Chancellor
- Demons: Baa

### [00:23:34] Revealed #2 Knight
Info: {}

### [00:23:34] Revealed #3 Oracle
Info: {'targets': [6, 8], 'minion_role': 'Chancellor'}

### [00:23:34] Revealed #5 Baker
Info: {'original_role': 'Judge'}

### [00:23:34] Revealed #7 Baker
Info: {'original_role': 'Fortune Teller'}

### [00:23:34] Revealed #8 Wretch
Info: {}

### [00:23:44] Revealed #1 Dreamer
Info: {}

### [00:23:44] Revealed #4 Plague_Doctor
Info: {}

### [00:23:44] Revealed #6 Judge
Info: {}

#### [00:23:50] Solver Output
Scenarios: 1303/7858
Evil probabilities: #5=52%, #7=49%, #8=25%, #2=25%, #1=17%, #6=16%, #3=14%, #4=2%

#### [00:23:50] Recommendation
Action: **EXECUTE** #2
Reason: Knight check: #2 is 25% evil, 17% corruption risk. Expected HP cost: 1.1 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 17% -- corrupted Knight loses immunity + 4 extra damage

#### [00:24:43] Execution Blocked
#2 Knight immunity — confirmed good, no HP loss

#### [00:24:46] Solver Output
Scenarios: 977/5520
Definite good: ['#2']
Evil probabilities: #5=58%, #7=54%, #8=30%, #1=21%, #6=18%, #3=17%, #4=3%

#### [00:24:46] Recommendation
Action: **USE_ABILITY** #1 (Dreamer) -> targets ['#5']
Reason: Entropy 2.526 (adjusted 2.283) | timing x1.00
WARNING: Corruption risk: 19%

### [00:25:22] Revealed #1 Dreamer
Info: {'target': 5, 'evil_role': 'Chancellor'}

### [00:25:22] Ability used at #1

#### [00:25:23] Solver Output
Scenarios: 682/5520
Definite good: ['#2']
Evil probabilities: #7=61%, #5=39%, #8=30%, #1=25%, #6=22%, #3=19%, #4=3%

#### [00:25:23] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.174 (adjusted 2.174) | timing x1.00

### [00:26:04] Ability used at #4

#### [00:26:04] Solver Output
Scenarios: 279/5520
Definite good: ['#2']
Evil probabilities: #5=95%, #7=43%, #8=20%, #1=13%, #3=13%, #6=10%, #4=5%

#### [00:26:04] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#1']
Reason: Expected posterior 171.0 scenarios (adjusted 190.3, info gain 0.552 bits) | timing x1.00
WARNING: Corruption risk: 23% -- corrupted Judge results are unreliable

### [00:26:56] Revealed #6 Judge
Info: {'target': 1, 'is_lying': False}

### [00:26:56] Ability used at #6

#### [00:26:56] Solver Output
Scenarios: 172/5520
Definite good: ['#2']
Evil probabilities: #5=91%, #7=53%, #8=21%, #3=17%, #4=9%, #1=5%, #6=5%

#### [00:26:56] Recommendation
Action: **EXECUTE** #5
Reason: No reveals available. #5 is 91% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 91% confident (budget: 2 wrong execs)

### [00:28:05] Executed #5 -> GOOD (WRONG!)

#### [00:28:05] Solver Output
Scenarios: 15/4060
Definite evil: ['#4', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#8']

#### [00:28:05] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 15 scenarios (roles: {'Chancellor', 'Baa'})

### [00:28:44] Executed #4 -> Chancellor (EVIL)

#### [00:28:44] Solver Output
Scenarios: 10/303
Definite evil: ['#4', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#8']

#### [00:28:44] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 10 scenarios (roles: {'Baa'})

### [00:29:24] Executed #7 -> Baa (EVIL)

## [00:29:32] GAME OVER — WIN
Final HP: 8
Notes: 8HP, 1 wrong exec Drunk (2HP), Knight trick, Dreamer+PD+Judge


---

# New Game — 2026-04-10 00:32:34
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Baker, Bard, Oracle, Slayer, Scout
- Outcasts: Bombardier, Doppelganger, Wretch
- Minions: Poisoner
- Demons: Baa

### [00:33:47] Revealed #1 Scout
Info: {'evil_role': 'Baa', 'distance': 2}

### [00:33:47] Revealed #2 Oracle
Info: {'targets': [3, 5], 'minion_role': 'Poisoner'}

### [00:33:47] Revealed #4 Baker
Info: {'original_role': 'Bard'}

### [00:33:47] Revealed #5 Confessor
Info: {'dizzy': True}

### [00:33:47] Revealed #6 Bard
Info: {'corruption_distance': 1}

### [00:33:47] Revealed #8 Wretch
Info: {}

### [00:33:56] Revealed #3 Slayer
Info: {}

### [00:33:56] Revealed #7 Slayer
Info: {}

#### [00:34:15] Solver Output
Scenarios: 4/540
Definite evil: ['#2', '#4']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8']

#### [00:34:15] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Baa'})

### [00:34:50] Executed #2 -> Baa (EVIL)

#### [00:34:50] Solver Output
Scenarios: 4/68
Definite evil: ['#2', '#4']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8']

#### [00:34:50] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 4 scenarios (roles: {'Poisoner'})

### [00:35:27] Executed #4 -> Poisoner (EVIL)

## [00:35:32] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, no abilities needed, pure deduction from Scout+Oracle+Confessor+Bard clues


---

# New Game — 2026-04-10 00:38:29
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Scout, Bishop, Dreamer, Fortune_Teller, Knight
- Outcasts: Bombardier, Wretch
- Minions: Poisoner
- Demons: Lilis

### [00:39:50] Revealed #1 Bombardier
Info: {}

### [00:39:50] Revealed #3 Bard
Info: {'corruption_distance': 3}

### [00:39:50] Revealed #4 Bishop
Info: {'targets': [3, 5, 1], 'types': ['Villager', 'Outcast', 'Minion']}

### [00:40:15] Revealed #2 Dreamer
Info: {}

### [00:42:05] Revealed #6 Bard
Info: {'corruption_distance': 4}

### [00:42:05] Revealed #8 Knight
Info: {}

### [00:42:05] Revealed #9 Wretch
Info: {}

### [00:42:14] Revealed #7 Fortune_Teller
Info: {}

#### [00:42:18] Solver Output
Scenarios: 10/98
Definite good: ['#2', '#5']
Evil probabilities: #3=90%, #6=40%, #1=20%, #4=20%, #7=10%, #8=10%, #9=10%

#### [00:42:18] Recommendation
Action: **EXECUTE** #8
Reason: Knight free check: #8 is 10% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

#### [00:43:07] Execution Blocked
#8 Knight immunity — confirmed good, no HP loss

#### [00:43:07] Solver Output
Scenarios: 9/78
Definite good: ['#2', '#5', '#8']
Evil probabilities: #3=89%, #6=44%, #1=22%, #4=22%, #7=11%, #9=11%

#### [00:43:07] Recommendation
Action: **USE_ABILITY** #2 (Dreamer) -> targets ['#3']
Reason: Entropy 2.059 (adjusted 1.487) | timing x1.00
WARNING: Corruption risk: 56%

### [00:43:44] Revealed #2 Dreamer
Info: {'target': 3, 'evil_role': 'Lilis'}

### [00:43:44] Ability used at #2

#### [00:43:44] Solver Output
Scenarios: 7/78
Definite good: ['#2', '#5', '#8']
Evil probabilities: #3=86%, #6=43%, #4=29%, #1=14%, #7=14%, #9=14%

#### [00:43:44] Recommendation
Action: **USE_ABILITY** #7 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.985 (adjusted 0.915) | timing x1.00
WARNING: Corruption risk: 14%

### [00:44:28] Revealed #7 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': True}

### [00:44:28] Ability used at #7

#### [00:44:29] Solver Output
Scenarios: 3/78
Definite evil: ['#3']
Definite good: ['#2', '#4', '#5', '#8', '#9']
Evil probabilities: #1=33%, #6=33%, #7=33%

#### [00:44:29] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 3 scenarios (roles: {'Lilis', 'Poisoner'})

### [00:45:08] Executed #3 -> Poisoner (EVIL)

#### [00:45:08] Solver Output
Scenarios: 2/12
Definite evil: ['#3']
Definite good: ['#2', '#4', '#5', '#6', '#8', '#9']
Evil probabilities: #1=50%, #7=50%

#### [00:45:08] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% good Fortune Teller, 50% evil Lilis).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [00:45:48] Executed #7 -> Lilis (EVIL)

## [00:45:55] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis night killed #5 Scout, Knight trick, Dreamer+FT abilities


---

# New Game — 2026-04-10 00:48:43
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Enlightened, Druid, Jester, Bard, Architect
- Outcasts: Wretch
- Minions: Twin_Minion
- Demons: Lilis

### [00:49:57] Revealed #1 Architect
Info: {'side': 'Left'}

### [00:49:57] Revealed #2 Wretch
Info: {}

### [00:49:57] Revealed #3 Enlightened
Info: {'direction': 'CCW'}

### [00:49:57] Revealed #4 Bard
Info: {'corruption_distance': -1}

### [00:50:37] Revealed #5 Bard
Info: {'corruption_distance': 1}

### [00:50:37] Revealed #7 Hunter
Info: {'distance': 2}

### [00:50:45] Revealed #6 Jester
Info: {}

#### [00:50:46] Solver Output
Scenarios: 2/56
Definite evil: ['#1', '#5']
Definite good: ['#2', '#3', '#4', '#6', '#7', '#8']

#### [00:50:46] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Lilis', 'Twin_Minion'})

### [00:51:28] Executed #1 -> Lilis (EVIL)

#### [00:51:28] Solver Output
Scenarios: 1/7
Definite evil: ['#1', '#5']
Definite good: ['#2', '#3', '#4', '#6', '#7', '#8']

#### [00:51:28] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [00:52:13] Executed #5 -> Twin_Minion (EVIL)

## [00:52:13] GAME OVER — WIN
Final HP: 8
Notes: 8HP, Lilis night killed Druid, no abilities needed, 2 scenarios from passive clues


---

# New Game — 2026-04-10 00:55:04
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Hunter, Enlightened, Gemcrafter, Oracle
- Outcasts: Doppelganger, Drunk, Wretch, Bombardier
- Minions: Chancellor
- Demons: Baa

### [00:55:40] Revealed #1 Oracle
Info: {'targets': [3, 6], 'minion_role': 'Chancellor'}

### [00:55:40] Revealed #4 Gemcrafter
Info: {'good_position': 7}

### [00:55:40] Revealed #5 Bombardier
Info: {}

### [00:55:40] Revealed #6 Wretch
Info: {}

### [00:55:40] Revealed #7 Gemcrafter
Info: {'good_position': 6}

### [00:55:40] Revealed #8 Enlightened
Info: {'direction': 'CCW'}

### [00:55:51] Revealed #2 Enlightened
Info: {'direction': 'cw'}

### [00:55:51] Revealed #3 Jester
Info: {}

#### [00:55:57] Solver Output
Scenarios: 98/1960
Definite good: ['#1']
Evil probabilities: #7=49%, #3=37%, #2=29%, #4=24%, #5=22%, #6=22%, #8=16%

#### [00:55:57] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#4', '#5']
Reason: Expected posterior 46.6 scenarios (adjusted 48.5, info gain 1.013 bits) | timing x1.00
WARNING: Corruption risk: 8%

### [00:57:03] Revealed #3 Jester
Info: {'targets': [1, 4, 5], 'evil_count': 1}

### [00:57:03] Ability used at #3

#### [00:57:04] Solver Output
Scenarios: 50/1960
Definite good: ['#1', '#2', '#8']
Evil probabilities: #7=56%, #3=48%, #6=44%, #4=32%, #5=20%

#### [00:57:04] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (44% good Drunk (corrupted), 32% evil Baa, 24% evil Chancellor).
WARNING: Execution lookahead override -- immediate hit chance is 56%, but all reveal branches still lead to a forced win.

### [00:57:43] Executed #7 -> Baa (EVIL)

#### [00:57:44] Solver Output
Scenarios: 16/230
Definite evil: ['#7']
Definite good: ['#1', '#2', '#5', '#6', '#8']
Evil probabilities: #3=50%, #4=50%

#### [00:57:44] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% evil Chancellor, 38% good Jester, 12% good Doppelganger).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [00:58:27] Executed #3 -> GOOD (WRONG!)

#### [00:58:27] Solver Output
Scenarios: 8/188
Definite evil: ['#4', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#8']

#### [00:58:27] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 8 scenarios (roles: {'Chancellor'})

### [00:59:04] Executed #4 -> Chancellor (EVIL)

## [00:59:04] GAME OVER — WIN
Final HP: 5
Notes: 5HP, Jester ability + 1 wrong exec, Asc48 complete 7/7!


---

# New Game — 2026-04-10 11:40:23
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Druid, Oracle, Jester, Gemcrafter, Knitter, Fortune_Teller, Architect
- Outcasts: Wretch
- Minions: Minion, Witch
- Demons: Lilis

### [11:43:16] Revealed #1 Architect
Info: {'side': 'Left'}

### [11:43:16] Revealed #2 Knitter
Info: {'evil_pairs': 0}

### [11:43:16] Revealed #3 Jester
Info: {}

### [11:43:16] Revealed #4 Wretch
Info: {}

### [11:43:16] Revealed #6 Fortune_Teller
Info: {}

### [11:43:16] Revealed #7 Bishop
Info: {'targets': [5, 1], 'types': ['Villager', 'Minion']}

### [11:43:16] Revealed #8 Druid
Info: {}

### [11:43:16] Revealed #9 Architect
Info: {'side': 'Right'}

#### [11:43:31] Solver Output
Scenarios: 6/270
Definite evil: ['#1', '#2', '#10']
Definite good: ['#3', '#4', '#5', '#6', '#7', '#8', '#9']

#### [11:43:31] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 6 scenarios (roles: {'Lilis', 'Minion', 'Witch'})

### [11:44:21] Executed #1 -> Minion (EVIL)

#### [11:44:24] Solver Output
Scenarios: 2/18
Definite evil: ['#1', '#2', '#10']
Definite good: ['#3', '#4', '#5', '#6', '#7', '#8', '#9']

#### [11:44:24] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Lilis', 'Witch'})

### [11:45:15] Executed #2 -> Lilis (EVIL)

## [11:45:21] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis killed Witch night2, two 100% confident executions, no abilities needed


---

# New Game — 2026-04-10 11:48:40
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Empress, Slayer, Lover, Bishop, Druid
- Outcasts: Wretch, Doppelganger
- Minions: Twin_Minion
- Demons: Pooka

### [11:50:21] Revealed #1 Druid
Info: {}

### [11:50:21] Revealed #2 Slayer
Info: {}

### [11:50:21] Revealed #3 Knight
Info: {}

### [11:50:21] Revealed #4 Empress
Info: {'targets': [1, 2, 8]}

### [11:50:21] Revealed #5 Wretch
Info: {}

### [11:50:21] Revealed #6 Slayer
Info: {}

### [11:50:21] Revealed #7 Bishop
Info: {'targets': [9, 8, 3], 'types': ['Villager', 'Outcast', 'Minion']}

### [11:50:21] Revealed #8 Lover
Info: {'evil_adjacent': 0}

### [11:50:21] Revealed #9 Empress
Info: {'targets': [6, 7, 8]}

#### [11:50:27] Solver Output
Scenarios: 25/448
Definite good: ['#3', '#5', '#8', '#9']
Evil probabilities: #6=60%, #2=40%, #4=40%, #7=40%, #1=20%

#### [11:50:27] Recommendation
Action: **USE_ABILITY** #1 (Druid) -> targets ['#2', '#3', '#4']
Reason: Entropy 0.722 (adjusted 0.722) | timing x1.00

### [11:51:32] Revealed #1 Druid
Info: {'targets': [2, 3, 4], 'found_outcast': 'Doppelganger'}

### [11:51:35] Ability used at #1

#### [11:51:38] Solver Output
Scenarios: 10/448
Definite good: ['#3', '#5', '#8', '#9']
Evil probabilities: #6=60%, #2=40%, #4=40%, #7=40%, #1=20%

#### [11:51:38] Recommendation
Action: **USE_ABILITY** #2 (Slayer) -> targets ['#6']
Reason: Target #6 is 60% evil (adjusted 0.60)

### [11:52:30] Ability used at #2

#### [11:52:34] Solver Output
Scenarios: 6/448
Definite good: ['#1', '#3', '#5', '#8', '#9']
Evil probabilities: #2=67%, #7=67%, #4=33%, #6=33%

#### [11:52:34] Recommendation
Action: **USE_ABILITY** #6 (Slayer) -> targets ['#2']
Reason: Target #2 is 67% evil (adjusted 0.22)
WARNING: Corruption risk: 67% -- Slayer ability disabled if corrupted

### [11:53:07] Ability used at #6

#### [11:53:10] Solver Output
Scenarios: 6/448
Definite good: ['#1', '#3', '#5', '#8', '#9']
Evil probabilities: #2=67%, #7=67%, #4=33%, #6=33%

#### [11:53:10] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (67% evil Twin_Minion, 17% good Doppelganger, 17% good Slayer).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [11:53:48] Executed #2 -> Twin_Minion (EVIL)

#### [11:53:52] Solver Output
Scenarios: 4/49
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4', '#5', '#8', '#9']
Evil probabilities: #6=50%, #7=50%

#### [11:53:52] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (50% evil Pooka, 50% good Slayer (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [11:54:42] Executed #6 -> Pooka (EVIL)

## [11:54:47] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Druid found Doppelganger, both Slayer abilities failed (both evil fakes), lookahead forced win


---

# New Game — 2026-04-10 11:56:24
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Medium, Knitter, Bard, Hunter, Empress
- Outcasts: Doppelganger, Bombardier
- Minions: Twin_Minion
- Demons: Pooka

### [11:57:45] Revealed #1 Bombardier
Info: {}

### [11:57:45] Revealed #2 Knitter
Info: {'evil_pairs': 1}

### [11:57:45] Revealed #3 Bard
Info: {'corruption_distance': 4}

### [11:57:45] Revealed #4 Bard
Info: {'corruption_distance': 3}

### [11:57:45] Revealed #5 Knitter
Info: {'evil_pairs': 0}

### [11:57:45] Revealed #6 Hunter
Info: {'distance': 4}

### [11:57:45] Revealed #7 Medium
Info: {'good_position': 5, 'good_role': 'Knitter'}

### [11:57:45] Revealed #8 Empress
Info: {'targets': [1, 3, 5]}

### [11:57:45] Revealed #9 Baker
Info: {'original_role': 'original'}

#### [11:57:51] Solver Output
Scenarios: 8/448
Definite evil: ['#6']
Definite good: ['#1', '#2', '#4', '#7', '#8', '#9']
Evil probabilities: #5=62%, #3=38%

#### [11:57:51] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 8 scenarios (roles: {'Pooka', 'Twin_Minion'})

### [11:58:33] Executed #6 -> Pooka (EVIL)

#### [11:58:37] Solver Output
Scenarios: 5/49
Definite evil: ['#5', '#6']
Definite good: ['#1', '#2', '#3', '#4', '#7', '#8', '#9']

#### [11:58:37] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 5 scenarios (roles: {'Twin_Minion'})

### [11:59:16] Executed #5 -> Twin_Minion (EVIL)

## [11:59:23] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, two 100% confident executions, no abilities needed, corrupted Medium


---

# New Game — 2026-04-10 12:00:58
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Slayer, Hunter, Lover, Bishop, Oracle
- Outcasts: Plague_Doctor, Doppelganger, Wretch
- Minions: Chancellor
- Demons: Lilis

### [12:03:26] Revealed #1 Bishop
Info: {'targets': [1, 3, 5], 'types': ['Villager', 'Outcast', 'Minion']}

### [12:03:26] Revealed #2 Confessor
Info: {'dizzy': False}

### [12:03:26] Revealed #3 Confessor
Info: {'dizzy': False}

### [12:03:26] Revealed #6 Hunter
Info: {'distance': 1}

### [12:03:26] Revealed #7 Lover
Info: {'evil_adjacent': 0}

### [12:03:26] Revealed #8 Wretch
Info: {}

### [12:03:26] Revealed #9 Confessor
Info: {'dizzy': True}

### [12:03:55] Revealed #4 Plague_Doctor
Info: {}

#### [12:04:00] Solver Output
Scenarios: 10/1835
Definite evil: ['#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#8']

#### [12:04:00] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 10 scenarios (roles: {'Lilis', 'Chancellor'})

### [12:04:37] Executed #7 -> Lilis (EVIL)

#### [12:04:45] Solver Output
Scenarios: 5/201
Definite evil: ['#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#8']

#### [12:04:45] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 5 scenarios (roles: {'Chancellor'})

### [12:05:29] Executed #9 -> Chancellor (EVIL)

## [12:05:34] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis night killed Oracle, two 100% confident executions, PD corrupted Bishop


---

# New Game — 2026-04-10 12:07:15
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Bishop, Witness, Scout, Hunter, Architect, Oracle
- Outcasts: Doppelganger, Drunk
- Minions: Minion
- Demons: Baa

### [12:07:51] Revealed #1 Hunter
Info: {'distance': 2}

### [12:07:51] Revealed #2 Bishop
Info: {'targets': [1, 3, 7], 'types': ['Villager', 'Outcast', 'Minion']}

### [12:07:51] Revealed #3 Scout
Info: {'evil_role': 'Baa', 'distance': 2}

### [12:07:51] Revealed #4 Architect
Info: {'side': 'Equal'}

### [12:07:51] Revealed #6 Oracle
Info: {'targets': [2, 4], 'minion_role': 'Minion'}

### [12:07:51] Revealed #7 Hunter
Info: {'distance': 1}

### [12:07:51] Revealed #8 Medium
Info: {'good_position': 2, 'good_role': 'Bishop'}

### [12:08:13] Revealed #5 Witness
Info: {'affected_position': 0}

#### [12:08:17] Solver Output
Scenarios: 6/2408
Definite evil: ['#6']
Definite good: ['#2', '#4', '#5', '#7', '#8']
Evil probabilities: #1=67%, #3=33%

#### [12:08:17] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 6 scenarios (roles: {'Baa'})

### [12:09:06] Executed #6 -> Baa (EVIL)

#### [12:09:06] Solver Output
Scenarios: 6/301
Definite evil: ['#6']
Definite good: ['#2', '#4', '#5', '#7', '#8']
Evil probabilities: #1=67%, #3=33%

#### [12:09:06] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (67% evil Minion, 17% good Doppelganger, 17% good Hunter).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [12:09:52] Executed #1 -> GOOD (WRONG!)

#### [12:09:56] Solver Output
Scenarios: 2/258
Definite evil: ['#3', '#6']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#8']

#### [12:09:56] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Minion'})

### [12:10:31] Executed #3 -> Minion (EVIL)

## [12:10:39] GAME OVER — WIN
Final HP: 5
Notes: 5HP, wrong exec on Doppelganger#1, lookahead forced win, no abilities used


---

# New Game — 2026-04-10 12:12:26
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Slayer, Medium, Architect, Judge, Baker
- Outcasts: Bombardier, Wretch
- Minions: Shaman
- Demons: Baa

### [12:13:21] Revealed #1 Poet
Info: {'distance': 2, 'copied_role': 'Hunter'}

### [12:13:21] Revealed #2 Baker
Info: {'original_role': 'Judge'}

### [12:13:21] Revealed #3 Medium
Info: {'good_position': 1, 'good_role': 'Poet'}

### [12:13:21] Revealed #4 Wretch
Info: {}

### [12:13:21] Revealed #5 Slayer
Info: {}

### [12:13:21] Revealed #6 Judge
Info: {}

### [12:13:43] Revealed #7 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 1}

### [12:14:14] Revealed #7 Poet
Info: {}

#### [12:14:20] Solver Output
Scenarios: 4/42
Definite evil: ['#6']
Definite good: ['#1', '#2', '#3', '#7']
Evil probabilities: #4=50%, #5=50%

#### [12:14:20] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 4 scenarios (roles: {'Shaman', 'Baa'})

### [12:16:11] Revealed #1 Poet
Info: {'evil_role': 'Baa', 'distance': 2, 'copied_role': 'Scout'}

#### [12:16:16] Solver Output
Scenarios: 7/42
Definite good: ['#1', '#3', '#7']
Evil probabilities: #2=71%, #4=57%, #6=57%, #5=14%

#### [12:16:16] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#3']
Reason: Expected posterior 3.6 scenarios (adjusted 3.6, info gain 0.971 bits) | timing x1.00

### [12:16:54] Revealed #6 Judge
Info: {'target': 3, 'is_lying': False}

### [12:17:00] Ability used at #6

#### [12:17:00] Solver Output
Scenarios: 3/42
Definite evil: ['#2']
Definite good: ['#1', '#3', '#6', '#7']
Evil probabilities: #4=67%, #5=33%

#### [12:17:00] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 3 scenarios (roles: {'Shaman', 'Baa'})

### [12:17:36] Executed #2 -> Baa (EVIL)

#### [12:17:37] Solver Output
Scenarios: 2/6
Definite evil: ['#2']
Definite good: ['#1', '#3', '#6', '#7']
Evil probabilities: #4=50%, #5=50%

#### [12:17:37] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#4']
Reason: Target #4 is 50% evil (adjusted 0.25)
WARNING: Wretch kill risk: 50% -- costs 5 HP

### [12:18:33] Ability used at #5

#### [12:18:33] Solver Output
Scenarios: 1/6
Definite evil: ['#2', '#5']
Definite good: ['#1', '#3', '#4', '#6', '#7']

#### [12:18:33] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [12:19:13] Executed #5 -> Shaman (EVIL)

## [12:19:20] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, caught auto_card misparse (Poet Scout->Hunter), Judge confirmed Medium truthful, Slayer fail narrowed to Shaman


---

# New Game — 2026-04-10 12:20:49
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Knitter, Druid, Bishop, Bard, Scout, Architect
- Outcasts: Drunk, Bombardier, Plague_Doctor
- Minions: Chancellor
- Demons: Baa

### [12:21:28] Revealed #1 Bombardier
Info: {}

### [12:21:28] Revealed #4 Architect
Info: {'side': 'Right'}

### [12:21:28] Revealed #5 Bishop
Info: {'targets': [2, 3, 7], 'types': ['Villager', 'Outcast', 'Minion']}

### [12:21:28] Revealed #6 Bard
Info: {'corruption_distance': 3}

### [12:21:28] Revealed #7 Scout
Info: {'evil_role': 'Baa', 'distance': 2}

### [12:21:28] Revealed #8 Knitter
Info: {'evil_pairs': 1}

### [12:21:50] Revealed #2 Druid
Info: {}

### [12:21:50] Revealed #3 Druid
Info: {}

#### [12:21:55] Solver Output
Scenarios: 2/540
Definite evil: ['#2', '#8']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7']

#### [12:21:55] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Chancellor'})

### [12:22:34] Executed #2 -> Chancellor (EVIL)

#### [12:22:34] Solver Output
Scenarios: 2/43
Definite evil: ['#2', '#8']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7']

#### [12:22:34] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 2 scenarios (roles: {'Baa'})

### [12:23:17] Executed #8 -> Baa (EVIL)

## [12:23:25] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, two 100% confident executions, Drunk disguised as Druid, no abilities needed. Asc49 complete! 7/7 perfect run


---

# New Game — 2026-04-10 13:26:53
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Alchemist, Jester, Medium, Baker, Gemcrafter, Slayer, Druid
- Outcasts: Drunk, Bombardier, Doppelganger
- Minions: Puppeteer, Chancellor
- Demons: Baa

### [13:27:23] Revealed #2 Alchemist
Info: {'cured_count': 0}

### [13:27:23] Revealed #3 Baker
Info: {'original_role': 'Slayer'}

### [13:27:23] Revealed #4 Bombardier
Info: {}

### [13:27:23] Revealed #5 Gemcrafter
Info: {'good_position': 6}

### [13:27:23] Revealed #7 Poet
Info: {'distance': 2, 'copied_role': 'Hunter'}

### [13:27:43] Revealed #1 Jester
Info: {}

### [13:27:43] Revealed #6 Druid
Info: {}

### [13:27:43] Revealed #8 Druid
Info: {}

### [13:27:43] Revealed #9 Slayer
Info: {}

#### [13:27:44] Solver Output
Scenarios: 2708/46680
Evil probabilities: #3=82%, #7=60%, #4=54%, #1=44%, #9=34%, #8=33%, #2=16%, #5=15%, #6=13%

#### [13:27:45] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 1.928 (adjusted 1.808) | timing x1.00
WARNING: Corruption risk: 12%

### [13:28:50] Revealed #1 Jester
Info: {'targets': [2, 3, 7], 'evil_count': 1}

### [13:28:50] Ability used at #1

#### [13:28:51] Solver Output
Scenarios: 1380/46680
Evil probabilities: #3=79%, #7=59%, #4=52%, #1=46%, #9=38%, #8=32%, #2=17%, #5=17%, #6=12%

#### [13:28:51] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 1.949 (adjusted 1.823) | timing x1.00
WARNING: Corruption risk: 13%

### [13:29:27] Revealed #8 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': 'Bombardier'}

### [13:29:27] Ability used at #8

#### [13:29:27] Solver Output
Scenarios: 584/46680
Evil probabilities: #3=84%, #7=71%, #1=55%, #4=28%, #8=27%, #9=25%, #5=22%, #6=20%, #2=16%

#### [13:29:27] Recommendation
Action: **USE_ABILITY** #6 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 1.537 (adjusted 1.408) | timing x1.00
WARNING: Corruption risk: 17%

### [13:30:02] Revealed #6 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': None}

### [13:30:02] Ability used at #6

#### [13:30:02] Solver Output
Scenarios: 317/46680
Evil probabilities: #3=78%, #7=55%, #4=52%, #1=38%, #8=37%, #9=26%, #5=25%, #6=17%, #2=11%

#### [13:30:02] Recommendation
Action: **USE_ABILITY** #9 (Slayer) -> targets ['#3']
Reason: Target #3 is 78% evil (adjusted 0.78)

#### [13:30:43] Solver Output
Scenarios: 138/46680
Evil probabilities: #4=74%, #9=55%, #7=51%, #3=49%, #8=41%, #5=28%, #6=17%, #1=14%, #2=9%

#### [13:30:43] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (51% good Drunk (corrupted), 19% evil Baa, 14% evil Chancellor).
WARNING: Execution lookahead override -- immediate hit chance is 49%, but all reveal branches still lead to a forced win.

### [13:30:58] Executed #3 -> GOOD (WRONG!)

### [13:31:37] Executed #7 -> Baa (EVIL)

#### [13:31:46] Solver Output
Scenarios: 31/3622
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3']
Evil probabilities: #4=68%, #8=68%, #5=42%, #6=32%, #9=19%

#### [13:31:46] Recommendation
Action: **EXECUTE** #8
Reason: No reveals available. #8 is 68% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 68% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #8 (68%) despite low confidence — Bombardier candidate(s) [4] risk instant game loss if executed first.

#### [13:32:07] Solver Output
Scenarios: 31/3622
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3']
Evil probabilities: #4=68%, #8=68%, #5=42%, #6=32%, #9=19%

#### [13:32:07] Recommendation
Action: **EXECUTE** #8
Reason: No reveals available. #8 is 68% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 68% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #8 (68%) despite low confidence — Bombardier candidate(s) [4] risk instant game loss if executed first.

### [13:32:33] Executed #5 -> Chancellor (EVIL)

#### [13:32:33] Solver Output
Scenarios: 5/324
Definite evil: ['#5', '#6', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#8', '#9']

#### [13:32:33] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 5 scenarios (roles: {'Puppeteer'})

### [13:33:03] Executed #6 -> Puppeteer (EVIL)

## [13:33:03] GAME OVER — WIN
Final HP: 5
Notes: 5HP, auto_loop stress test, forced_safe exec on Drunk, Knight check fix verified


---

# New Game — 2026-04-10 13:39:04
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Jester, Architect, Dreamer, Medium, Lover, Hunter, Bard
- Outcasts: Drunk, Wretch, Plague_Doctor
- Minions: Chancellor, Shaman
- Demons: Baa

### [13:40:00] Revealed #1 Lover
Info: {'evil_adjacent': 1}

### [13:40:00] Revealed #3 Baker
Info: {'original_role': 'Bard'}

### [13:40:00] Revealed #5 Baker
Info: {'original_role': 'original'}

### [13:40:00] Revealed #6 Hunter
Info: {'distance': 4}

### [13:40:00] Revealed #7 Hunter
Info: {'distance': 2}

### [13:40:00] Revealed #9 Bard
Info: {'corruption_distance': -1}

### [13:45:29] Revealed #2 Plague_Doctor
Info: {}

### [13:45:32] Revealed #4 Dreamer
Info: {}

### [13:45:36] Revealed #8 Jester
Info: {}

#### [13:45:43] Solver Output
Scenarios: 702/17748
Evil probabilities: #9=72%, #6=57%, #7=40%, #4=38%, #5=24%, #8=24%, #1=23%, #3=20%, #2=3%

#### [13:45:43] Recommendation
Action: **USE_ABILITY** #4 (Dreamer) -> targets ['#9']
Reason: Entropy 2.957 (adjusted 2.721) | timing x1.00
WARNING: Corruption risk: 16%

### [13:46:59] Revealed #4 Dreamer
Info: {'target': 9, 'evil_role': 'Baa'}

### [13:47:02] Ability used at #4

#### [13:47:06] Solver Output
Scenarios: 472/17748
Evil probabilities: #6=59%, #9=58%, #4=44%, #7=43%, #1=34%, #5=22%, #8=20%, #3=17%, #2=4%

#### [13:47:06] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#6']
Reason: Entropy 2.261 (adjusted 2.261) | timing x1.00

### [13:48:00] Ability used at #2

#### [13:48:05] Solver Output
Scenarios: 264/17748
Definite good: ['#3', '#5']
Evil probabilities: #6=100%, #7=54%, #9=53%, #1=38%, #4=35%, #8=20%, #2=0%

#### [13:48:05] Recommendation
Action: **USE_ABILITY** #8 (Jester) -> targets ['#4', '#6', '#9']
Reason: Expected posterior 118.4 scenarios (adjusted 129.7, info gain 1.026 bits) | timing x1.00
WARNING: Corruption risk: 19%

### [13:49:09] Revealed #8 Jester
Info: {'targets': [4, 6, 9], 'evil_count': 3}

### [13:49:14] Ability used at #8

#### [13:49:18] Solver Output
Scenarios: 127/17748
Definite evil: ['#6']
Definite good: ['#2', '#3', '#5']
Evil probabilities: #9=69%, #8=42%, #7=33%, #4=31%, #1=25%

#### [13:49:18] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 127 scenarios (roles: {'Chancellor', 'Baa', 'Shaman'})

### [13:50:04] Executed #6 -> Shaman (EVIL)

#### [13:50:08] Solver Output
Scenarios: 40/2098
Definite evil: ['#6']
Definite good: ['#2', '#3', '#5']
Evil probabilities: #9=65%, #7=42%, #8=42%, #1=25%, #4=25%

#### [13:50:08] Recommendation
Action: **EXECUTE** #9
Reason: Execution lookahead: #9 guarantees a win across all reveal branches with current HP budget (40% evil Chancellor, 25% evil Baa, 18% good Bard (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 65%, but all reveal branches still lead to a forced win.

### [13:50:53] Executed #9 -> Baa (EVIL)

#### [13:50:59] Solver Output
Scenarios: 10/266
Definite evil: ['#6', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5']
Evil probabilities: #7=50%, #8=50%

#### [13:50:59] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% evil Chancellor, 30% good Drunk (corrupted), 20% good Hunter (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [13:51:46] Executed #7 -> GOOD (WRONG!)

#### [13:52:02] Solver Output
Scenarios: 5/230
Definite evil: ['#6', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#7']

#### [13:52:02] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 5 scenarios (roles: {'Chancellor'})

### [13:52:48] Executed #8 -> Chancellor (EVIL)

## [13:53:00] GAME OVER — WIN
Final HP: 8
Notes: 8HP, Shaman+Chancellor+Baa, Drunk wrong exec cost only 2HP, Dreamer found Baa, PD checked #6 clean, Jester lied 3 evil


---

# New Game — 2026-04-10 13:54:50
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Baker, Poet, Lover, Slayer, Alchemist
- Outcasts: Bombardier
- Minions: 
- Demons: Pooka

### [13:55:45] Revealed #2 Bombardier
Info: {}

### [13:55:45] Revealed #3 Alchemist
Info: {'cured_count': 1}

### [13:55:45] Revealed #4 Poet
Info: {'distance': 3, 'copied_role': 'Hunter'}

### [13:55:45] Revealed #5 Baker
Info: {'original_role': 'original'}

### [13:55:45] Revealed #6 Baker
Info: {'original_role': 'Poet'}

### [13:55:45] Revealed #7 Lover
Info: {'evil_adjacent': 1}

### [13:56:13] Revealed #1 Slayer
Info: {}

#### [13:56:18] Solver Output
Scenarios: 1/7
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']

#### [13:56:18] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [13:56:59] Executed #7 -> Pooka (EVIL)

## [13:57:06] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, 1 scenario solve, Poet copied Hunter, Alchemist cured 1 corruption, corrupted Baker lied


---

# New Game — 2026-04-10 13:58:55
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Dreamer, Knitter, Lover, Baker, Poet, Druid
- Outcasts: Plague_Doctor, Doppelganger
- Minions: Minion, Chancellor
- Demons: Pooka

### [14:00:10] Revealed #2 Lover
Info: {'evil_adjacent': 2}

### [14:00:10] Revealed #3 Dreamer
Info: {}

### [14:00:10] Revealed #4 Bard
Info: {'corruption_distance': -1}

### [14:00:10] Revealed #5 Poet
Info: {'distance': 2, 'copied_role': 'Hunter'}

### [14:00:10] Revealed #6 Plague_Doctor
Info: {}

### [14:00:10] Revealed #8 Bard
Info: {'corruption_distance': 2}

### [14:00:10] Revealed #9 Baker
Info: {'original_role': 'original'}

### [14:00:10] Revealed #10 Baker
Info: {'original_role': 'Poet'}

### [14:01:34] Revealed #1 Druid
Info: {}

### [14:01:39] Revealed #7 Poet
Info: {'targets': [4, 5, 9], 'types': ['Villager', 'Minion', 'Outcast'], 'copied_role': 'Bishop'}

#### [14:01:46] Solver Output
Scenarios: 176/27552
Definite good: ['#6']
Evil probabilities: #7=79%, #2=57%, #3=39%, #1=27%, #10=27%, #4=22%, #8=19%, #9=17%, #5=12%

#### [14:01:46] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#7']
Reason: Entropy 2.859 (adjusted 2.729) | timing x1.00
WARNING: Corruption risk: 9%

### [14:02:34] Revealed #3 Dreamer
Info: {'target': 7, 'evil_role': 'Chancellor'}

### [14:02:34] Ability used at #3

#### [14:02:39] Solver Output
Scenarios: 94/27552
Definite good: ['#6']
Evil probabilities: #7=61%, #3=53%, #2=51%, #1=34%, #8=28%, #10=26%, #4=21%, #9=14%, #5=13%

#### [14:02:39] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#4']
Reason: Entropy 2.676 (adjusted 2.676) | timing x1.00

### [14:03:24] Ability used at #6

#### [14:03:29] Solver Output
Scenarios: 31/27552
Definite evil: ['#2']
Definite good: ['#4', '#6', '#10']
Evil probabilities: #7=61%, #5=39%, #1=35%, #3=26%, #8=23%, #9=16%

#### [14:03:29] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 31 scenarios (roles: {'Chancellor', 'Pooka', 'Minion'})

### [14:04:13] Executed #2 -> Minion (EVIL)

#### [14:04:13] Solver Output
Scenarios: 17/2918
Definite evil: ['#2']
Definite good: ['#4', '#6', '#10']
Evil probabilities: #7=65%, #1=53%, #5=35%, #9=29%, #8=12%, #3=6%

#### [14:04:13] Recommendation
Action: **USE_ABILITY** #1 (Druid) -> targets ['#3', '#4', '#5']
Reason: Entropy 0.998 (adjusted 0.998) | timing x1.00

### [14:05:20] Revealed #1 Druid
Info: {'targets': [3, 4, 5], 'found_outcast': 'Doppelganger'}

### [14:05:21] Ability used at #1

#### [14:05:21] Solver Output
Scenarios: 8/2918
Definite evil: ['#2']
Definite good: ['#4', '#6', '#8', '#10']
Evil probabilities: #7=75%, #1=50%, #9=38%, #5=25%, #3=12%

#### [14:05:21] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (62% evil Chancellor, 12% good Doppelganger, 12% good Poet).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [14:06:06] Executed #7 -> Chancellor (EVIL)

#### [14:06:06] Solver Output
Scenarios: 5/259
Definite evil: ['#2', '#7']
Definite good: ['#3', '#4', '#5', '#6', '#8', '#10']
Evil probabilities: #9=60%, #1=40%

#### [14:06:06] Recommendation
Action: **EXECUTE** #9
Reason: Execution lookahead: #9 guarantees a win across all reveal branches with current HP budget (60% evil Pooka, 20% good Baker, 20% good Doppelganger).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [14:06:53] Executed #9 -> GOOD (WRONG!)

#### [14:06:53] Solver Output
Scenarios: 2/223
Definite evil: ['#1', '#2', '#7']
Definite good: ['#3', '#4', '#5', '#6', '#8', '#9', '#10']

#### [14:06:53] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [14:07:38] Executed #1 -> Pooka (EVIL)

## [14:07:38] GAME OVER — WIN
Final HP: 5
Notes: 5HP, Dreamer found Chancellor, PD found #4 corrupted + #2 evil, wrong exec on #9 Baker


---

# New Game — 2026-04-10 14:09:59
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Empress, Bishop, Fortune_Teller, Architect, Enlightened, Poet
- Outcasts: Plague_Doctor
- Minions: Puppeteer, Shaman
- Demons: Lilis

### [14:14:08] Revealed #1 Empress
Info: {'targets': [4, 6, 8]}

### [14:14:08] Revealed #2 Confessor
Info: {'dizzy': True}

### [14:14:08] Revealed #4 Bishop
Info: {'targets': [8, 1, 3], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:14:08] Revealed #6 Architect
Info: {'side': 'Right'}

### [14:14:53] Revealed #3 Plague_Doctor
Info: {}

### [14:14:53] Revealed #7 Enlightened
Info: {'direction': 'Equidistant'}

### [14:14:53] Revealed #8 Plague_Doctor
Info: {}

### [14:14:53] Revealed #9 Fortune_Teller
Info: {}

#### [14:15:00] Solver Output
Scenarios: 33/4248
Definite evil: ['#2']
Definite good: ['#5', '#10']
Evil probabilities: #7=58%, #3=55%, #6=52%, #8=45%, #4=36%, #9=36%, #1=18%

#### [14:15:01] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 33 scenarios (roles: {'Lilis', 'Puppet', 'Puppeteer', 'Shaman'})

### [14:15:51] Executed #2 -> Lilis (EVIL)

#### [14:15:51] Solver Output
Scenarios: 10/455
Definite evil: ['#2']
Definite good: ['#1', '#5', '#10']
Evil probabilities: #7=70%, #8=60%, #9=50%, #3=40%, #4=40%, #6=40%

#### [14:15:51] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#1']
Reason: Entropy 1.961 (adjusted 1.961) | timing x1.00

### [14:16:21] Ability used at #8

#### [14:16:21] Solver Output
Scenarios: 8/455
Definite evil: ['#2']
Definite good: ['#1', '#5', '#10']
Evil probabilities: #7=62%, #3=50%, #4=50%, #6=50%, #8=50%, #9=38%

#### [14:16:21] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#7']
Reason: Entropy 1.406 (adjusted 1.406) | timing x1.00

### [14:17:04] Ability used at #3

#### [14:17:04] Solver Output
Scenarios: 3/455
Definite evil: ['#2', '#3', '#4', '#7']
Definite good: ['#1', '#5', '#6', '#8', '#9', '#10']

#### [14:17:04] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 3 scenarios (roles: {'Puppeteer'})

### [14:17:45] Executed #3 -> Puppeteer (EVIL)

#### [14:17:45] Solver Output
Scenarios: 3/31
Definite evil: ['#2', '#3', '#4', '#7']
Definite good: ['#1', '#5', '#6', '#8', '#9', '#10']

#### [14:17:45] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 3 scenarios (roles: {'Puppet'})

### [14:18:22] Executed #4 -> Puppet (EVIL)

#### [14:18:22] Solver Output
Scenarios: 3/31
Definite evil: ['#2', '#3', '#4', '#7']
Definite good: ['#1', '#5', '#6', '#8', '#9', '#10']

#### [14:18:22] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 3 scenarios (roles: {'Shaman'})

### [14:19:09] Executed #7 -> Shaman (EVIL)

## [14:19:09] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, night killed #5 #10 (both good), Confessor dizzy, PD #8 clean check, evil PD #3 caught lying, no wrong execs


---

# New Game — 2026-04-10 14:20:58
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Architect, Slayer, Bishop, Confessor, Empress
- Outcasts: Doppelganger, Bombardier
- Minions: Chancellor
- Demons: Pooka

### [14:21:28] Revealed #1 Bombardier
Info: {}

### [14:21:28] Revealed #2 Architect
Info: {'side': 'Left'}

### [14:21:28] Revealed #3 Confessor
Info: {'dizzy': True}

### [14:21:28] Revealed #4 Baker
Info: {'original_role': 'Architect'}

### [14:21:28] Revealed #5 Bishop
Info: {'targets': [5, 2, 6], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:21:28] Revealed #7 Empress
Info: {'targets': [1, 2, 5]}

### [14:21:36] Revealed #6 Slayer
Info: {}

### [14:21:36] Revealed #8 Slayer
Info: {}

#### [14:21:37] Solver Output
Scenarios: 14/454
Definite good: ['#1', '#6', '#8']
Evil probabilities: #4=71%, #7=43%, #2=36%, #3=29%, #5=21%

#### [14:21:37] Recommendation
Action: **USE_ABILITY** #6 (Slayer) -> targets ['#4']
Reason: Target #4 is 71% evil (adjusted 0.71)

### [14:22:17] Ability used at #6

#### [14:22:18] Solver Output
Scenarios: 14/454
Definite good: ['#1', '#6', '#8']
Evil probabilities: #4=71%, #7=43%, #2=36%, #3=29%, #5=21%

#### [14:22:18] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#4']
Reason: Target #4 is 71% evil (adjusted 0.71)

#### [14:22:32] Solver Output
Scenarios: 10/52
Definite evil: ['#4']
Definite good: ['#1', '#3', '#6', '#8']
Evil probabilities: #7=60%, #2=30%, #5=10%

#### [14:22:32] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#7']
Reason: Target #7 is 60% evil (adjusted 0.60)

### [14:23:17] Ability used at #8

## [14:23:17] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, both evils killed by Slayers (#6 killed Pooka, Doppelganger-Slayer #8 killed Chancellor), no executions needed


---

# New Game — 2026-04-10 14:25:08
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Hunter, Druid, Poet, Scout, Jester
- Outcasts: Doppelganger
- Minions: Minion
- Demons: Pooka

### [14:25:26] Revealed #1 Scout
Info: {'evil_role': 'Pooka', 'distance': 3}

### [14:25:26] Revealed #3 Architect
Info: {'side': 'Equal'}

### [14:25:26] Revealed #4 Poet
Info: {'evil_pairs': 1, 'copied_role': 'Knitter'}

### [14:25:26] Revealed #7 Hunter
Info: {'distance': 1}

### [14:25:35] Revealed #2 Jester
Info: {}

### [14:25:35] Revealed #5 Jester
Info: {}

### [14:25:35] Revealed #6 Druid
Info: {}

### [14:25:36] Revealed #8 Jester
Info: {}

#### [14:25:36] Solver Output
Scenarios: 6/336
Definite evil: ['#8']
Definite good: ['#1', '#2', '#5', '#6', '#7']
Evil probabilities: #3=83%, #4=17%

#### [14:25:36] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 6 scenarios (roles: {'Minion', 'Pooka'})

### [14:26:14] Executed #8 -> Minion (EVIL)

#### [14:26:14] Solver Output
Scenarios: 5/42
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7']

#### [14:26:14] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 5 scenarios (roles: {'Pooka'})

### [14:26:58] Executed #3 -> Pooka (EVIL)

## [14:26:58] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Scout+Hunter+Knitter(Poet) constrained solver to 6 scenarios instantly, clean 100% executions on both evils


---

# New Game — 2026-04-10 14:34:08
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Knight, Gemcrafter, Oracle, Jester, Bishop
- Outcasts: Doppelganger, Wretch
- Minions: Chancellor
- Demons: Pooka

### [14:35:17] Revealed #1 Scout
Info: {'evil_role': 'Chancellor', 'distance': 1}

### [14:35:17] Revealed #2 Oracle
Info: {'targets': [2, 5], 'minion_role': 'Chancellor'}

### [14:35:17] Revealed #3 Knight
Info: {}

### [14:35:17] Revealed #5 Wretch
Info: {}

### [14:35:17] Revealed #6 Oracle
Info: {'targets': [1, 3], 'minion_role': 'Chancellor'}

### [14:35:17] Revealed #7 Knight
Info: {}

### [14:35:17] Revealed #8 Bishop
Info: {'targets': [7, 5, 8], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:38:08] Revealed #4 Jester
Info: {}

#### [14:38:16] Solver Output
Scenarios: 46/454
Definite good: ['#2']
Evil probabilities: #3=52%, #1=35%, #6=28%, #7=28%, #8=28%, #4=24%, #5=4%

#### [14:38:16] Recommendation
Action: **EXECUTE** #3
Reason: Knight check: #3 is 52% evil, 7% corruption risk. Expected HP cost: 0.3 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 7% -- corrupted Knight loses immunity + 4 extra damage

#### [14:39:25] Execution Blocked
#3 Knight immunity — confirmed good, no HP loss

### [14:41:37] Executed #3 -> GOOD (WRONG!)

#### [14:41:52] Solver Output
Scenarios: 3/335
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8']

#### [14:41:52] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 3 scenarios (roles: {'Pooka'})

### [14:42:37] Executed #4 -> Pooka (EVIL)

#### [14:42:41] Solver Output
Scenarios: 3/51
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8']

#### [14:42:41] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 3 scenarios (roles: {'Chancellor'})

### [14:43:22] Executed #6 -> Chancellor (EVIL)

## [14:43:29] GAME OVER — WIN
Final HP: 1
Notes: 1HP win, corrupted Knight at #3 lost immunity (9HP cost from Pooka corruption), solver found both evils with 100% confidence after


---

# New Game — 2026-04-10 14:56:52
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Jester, Lover, Oracle, Architect, Slayer
- Outcasts: Bombardier
- Minions: Shaman
- Demons: Pooka

### [14:57:28] Revealed #1 Bombardier
Info: {}

### [14:57:28] Revealed #3 Architect
Info: {'side': 'Equal'}

### [14:57:28] Revealed #4 Bombardier
Info: {}

### [14:57:28] Revealed #5 Oracle
Info: {'targets': [1, 2], 'minion_role': 'Shaman'}

### [14:57:28] Revealed #6 Architect
Info: {'side': 'Left'}

### [14:57:28] Revealed #8 Lover
Info: {'evil_adjacent': 0}

### [14:58:25] Revealed #2 Jester
Info: {}

### [14:58:25] Revealed #7 Slayer
Info: {}

#### [14:58:30] Solver Output
Scenarios: 2/56
Definite good: ['#2', '#3', '#6', '#8']
Evil probabilities: #1=50%, #4=50%, #5=50%, #7=50%

#### [14:58:30] Recommendation
Action: **USE_ABILITY** #2 (Jester) -> targets ['#1', '#3', '#6']
Reason: Expected posterior 1.0 scenarios (adjusted 1.0, info gain 1.000 bits) | timing x1.00

### [14:59:34] Revealed #2 Jester
Info: {'targets': [1, 3, 6], 'evil_count': 1}

### [14:59:38] Ability used at #2

#### [14:59:42] Solver Output
Scenarios: 1/56
Definite evil: ['#1', '#7']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#8']

#### [14:59:42] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [15:00:25] Executed #1 -> Shaman (EVIL)

#### [15:00:29] Solver Output
Scenarios: 1/7
Definite evil: ['#1', '#7']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#8']

#### [15:00:29] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [15:01:00] Executed #7 -> Pooka (EVIL)

## [15:01:08] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Jester ability on #1,#3,#6 found 1 evil, solved to 1 scenario, clean executions


---

# New Game — 2026-04-10 15:02:58
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Empress, Oracle, Gemcrafter, Druid, Confessor
- Outcasts: Wretch
- Minions: 
- Demons: Pooka

### [15:03:31] Revealed #1 Gemcrafter
Info: {'good_position': 5}

### [15:03:31] Revealed #2 Confessor
Info: {'dizzy': True}

### [15:03:31] Revealed #3 Empress
Info: {'targets': [1, 4, 6]}

### [15:03:31] Revealed #5 Wretch
Info: {}

### [15:03:31] Revealed #6 Oracle
Info: {'targets': [5, 7], 'minion_role': 'Shaman'}

### [15:03:58] Revealed #4 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 5}

### [15:03:58] Revealed #7 Druid
Info: {}

#### [15:04:03] Solver Output
Scenarios: 2/7
Definite good: ['#3', '#4', '#5', '#6', '#7']
Evil probabilities: #1=50%, #2=50%

#### [15:04:03] Recommendation
Action: **USE_ABILITY** #7 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 1.000 (adjusted 0.750) | timing x1.00
WARNING: Corruption risk: 50%

### [15:05:04] Revealed #7 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [15:05:04] Ability used at #7

#### [15:05:09] Solver Output
Scenarios: 1/7
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7']

#### [15:05:09] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [15:05:48] Executed #2 -> Pooka (EVIL)

## [15:05:48] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, 1 evil only, Druid+Jester info solved to 1 scenario


---

# New Game — 2026-04-10 15:07:39
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Medium, Hunter, Confessor, Gemcrafter
- Outcasts: Plague_Doctor, Doppelganger, Wretch
- Minions: Chancellor
- Demons: Baa

### [15:08:10] Revealed #2 Hunter
Info: {'distance': 3}

### [15:08:10] Revealed #3 Gemcrafter
Info: {'good_position': 6}

### [15:08:10] Revealed #4 Confessor
Info: {'dizzy': False}

### [15:08:10] Revealed #5 Baker
Info: {'original_role': 'original'}

### [15:08:10] Revealed #6 Hunter
Info: {'distance': 2}

### [15:08:10] Revealed #7 Hunter
Info: {'distance': 1}

### [15:08:36] Revealed #1 Plague_Doctor
Info: {}

#### [15:08:41] Solver Output
Scenarios: 44/1170
Definite good: ['#4']
Evil probabilities: #6=55%, #2=45%, #3=36%, #1=23%, #7=23%, #5=18%

#### [15:08:41] Recommendation
Action: **USE_ABILITY** #1 (Plague Doctor) -> targets ['#2']
Reason: Entropy 1.544 (adjusted 1.544) | timing x1.00

### [15:09:37] Ability used at #1

#### [15:09:42] Solver Output
Scenarios: 26/1170
Definite good: ['#4', '#5', '#7']
Evil probabilities: #3=62%, #6=62%, #1=38%, #2=38%

#### [15:09:42] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (31% evil Baa, 31% evil Chancellor, 31% good Gemcrafter).
WARNING: Execution lookahead override -- immediate hit chance is 62%, but all reveal branches still lead to a forced win.

### [15:10:30] Executed #3 -> Baa (EVIL)

#### [15:10:31] Solver Output
Scenarios: 8/187
Definite evil: ['#3', '#6']
Definite good: ['#1', '#2', '#4', '#5', '#7']

#### [15:10:31] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 8 scenarios (roles: {'Chancellor'})

### [15:11:10] Executed #6 -> Chancellor (EVIL)

## [15:11:10] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, PD found #2 corrupted + #3 evil, clean executions both 100%


---

# New Game — 2026-04-10 15:13:12
Cards: 8, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Hunter, Knitter, Architect, Lover, Poet
- Outcasts: Bombardier, Drunk
- Minions: Puppeteer
- Demons: Baa

### [15:13:56] Revealed #1 Hunter
Info: {'distance': 2}

### [15:13:56] Revealed #2 Bombardier
Info: {}

### [15:13:56] Revealed #3 Bombardier
Info: {}

### [15:13:56] Revealed #4 Lover
Info: {'evil_adjacent': 1}

### [15:13:56] Revealed #5 Baker
Info: {'original_role': 'original'}

### [15:13:56] Revealed #6 Knitter
Info: {'evil_pairs': 0}

### [15:13:56] Revealed #7 Baker
Info: {'original_role': 'Poet'}

### [15:13:56] Revealed #8 Baker
Info: {'original_role': 'Architect'}

#### [15:14:15] Solver Output
Scenarios: 7/322
Definite evil: ['#3']
Definite good: ['#2', '#5']
Evil probabilities: #6=71%, #4=57%, #7=43%, #1=14%, #8=14%

#### [15:14:15] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 7 scenarios (roles: {'Puppeteer', 'Baa'})

### [15:14:57] Executed #3 -> Puppeteer (EVIL)

#### [15:14:57] Solver Output
Scenarios: 6/67
Definite evil: ['#3']
Definite good: ['#5', '#8']
Evil probabilities: #4=67%, #6=67%, #2=33%, #1=17%, #7=17%

#### [15:14:57] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (67% evil Puppet, 33% good Lover).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [15:15:35] Executed #4 -> Puppet (EVIL)

#### [15:15:36] Solver Output
Scenarios: 4/31
Definite evil: ['#3', '#4']
Definite good: ['#1', '#2', '#5', '#8']
Evil probabilities: #6=75%, #7=25%

#### [15:15:36] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (75% evil Baa, 25% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [15:16:16] Executed #6 -> Baa (EVIL)

## [15:16:16] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, 3 evils found without any wrong executions, Puppeteer+Puppet+Baa game


---

# New Game — 2026-04-10 15:18:23
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Knitter, Confessor, Druid, Bishop, Fortune_Teller
- Outcasts: Wretch, Drunk
- Minions: Witch
- Demons: Baa

### [15:18:57] Revealed #1 Bishop
Info: {'targets': [6, 5, 2], 'types': ['Villager', 'Outcast', 'Minion']}

### [15:18:57] Revealed #3 Confessor
Info: {'dizzy': True}

### [15:18:57] Revealed #4 Knitter
Info: {'evil_pairs': 1}

### [15:18:57] Revealed #5 Architect
Info: {'side': 'Right'}

### [15:20:21] Revealed #2 Fortune_Teller
Info: {}

### [15:20:21] Revealed #6 Druid
Info: {}

#### [15:20:26] Solver Output
Scenarios: 9/252
Definite good: ['#4', '#5', '#6']
Evil probabilities: #1=67%, #2=56%, #3=56%, #7=22%

#### [15:20:26] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#1', '#3']
Reason: Entropy 0.991 (adjusted 0.991) | timing x1.00

### [15:21:12] Revealed #2 Fortune Teller
Info: {'targets': [1, 3], 'has_evil': True}

### [15:21:12] Ability used at #2

#### [15:21:17] Solver Output
Scenarios: 4/252
Definite evil: ['#1']
Definite good: ['#2', '#4', '#5', '#6']
Evil probabilities: #3=50%, #7=50%

#### [15:21:17] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Baa', 'Witch'})

### [15:21:58] Executed #1 -> Witch (EVIL)

### [15:22:25] Revealed #7 Wretch
Info: {}

#### [15:22:26] Solver Output
Scenarios: 4/31
Definite evil: ['#1']
Definite good: ['#2', '#4', '#6']
Evil probabilities: #3=50%, #5=25%, #7=25%

#### [15:22:26] Recommendation
Action: **USE_ABILITY** #6 (Druid) -> targets ['#2', '#3', '#4']
Reason: Entropy 0.811 (adjusted 0.710) | timing x1.00
WARNING: Corruption risk: 25%

### [15:23:10] Revealed #6 Druid
Info: {'targets': [2, 3, 4], 'found_outcast': None}

### [15:23:10] Ability used at #6

#### [15:23:10] Solver Output
Scenarios: 1/31
Definite evil: ['#1', '#3']
Definite good: ['#2', '#4', '#5', '#6', '#7']

#### [15:23:10] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Baa'})

### [15:23:53] Executed #3 -> Baa (EVIL)

## [15:23:53] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Witch blocked #7, FT confirmed evil in #1/#3, Druid disambiguated to 1 scenario


---

# New Game — 2026-04-10 15:25:58
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Judge, Witness, Knitter, Enlightened, Hunter, Poet
- Outcasts: Drunk, Bombardier
- Minions: Twin_Minion
- Demons: Baa

### [15:26:27] Revealed #2 Knitter
Info: {'evil_pairs': 0}

### [15:26:27] Revealed #5 Hunter
Info: {'distance': 4}

### [15:26:27] Revealed #8 Baker
Info: {'original_role': 'original'}

### [15:28:04] Revealed #1 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 4}

### [15:28:04] Revealed #3 Poet
Info: {'targets': [1, 4], 'minion_role': 'Twin_Minion', 'copied_role': 'Oracle'}

### [15:28:04] Revealed #4 Witness
Info: {'affected_position': 0}

### [15:28:04] Revealed #6 Judge
Info: {}

### [15:28:04] Revealed #7 Enlightened
Info: {'direction': 'ccw'}

#### [15:28:10] Solver Output
Scenarios: 4/392
Definite good: ['#2', '#4', '#8']
Evil probabilities: #1=50%, #3=50%, #5=50%, #6=25%, #7=25%

#### [15:28:10] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#7']
Reason: Expected posterior 2.0 scenarios (adjusted 2.0, info gain 1.000 bits) | timing x1.00

### [15:28:54] Revealed #6 Judge
Info: {'target': 7, 'is_lying': True}

### [15:28:54] Ability used at #6

#### [15:28:54] Solver Output
Scenarios: 2/392
Definite evil: ['#1']
Definite good: ['#2', '#3', '#4', '#5', '#8']
Evil probabilities: #6=50%, #7=50%

#### [15:28:54] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Twin_Minion'})

### [15:29:35] Executed #1 -> Twin_Minion (EVIL)

#### [15:29:36] Solver Output
Scenarios: 2/49
Definite evil: ['#1']
Definite good: ['#2', '#3', '#4', '#5', '#8']
Evil probabilities: #6=50%, #7=50%

#### [15:29:36] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 50% good Judge).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:30:22] Executed #6 -> GOOD (WRONG!)

#### [15:30:23] Solver Output
Scenarios: 1/42
Definite evil: ['#1', '#7']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#8']

#### [15:30:23] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Baa'})

### [15:31:07] Executed #7 -> Baa (EVIL)

## [15:31:07] GAME OVER — WIN
Final HP: 5
Notes: 5HP, Judge found #7 lying, 50/50 on #6 was wrong exec (Judge good), lookahead guaranteed win, Asc51 complete 7/7


---

# New Game — 2026-04-10 15:42:37
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Oracle, Judge, Bishop, Dreamer, Baker, Lover, Slayer, Fortune_Teller
- Outcasts: Doppelganger
- Minions: Puppeteer, Shaman
- Demons: Lilis

### [15:44:03] Revealed #1 Bishop
Info: {'targets': [6, 2, 7], 'types': ['Villager', 'Outcast', 'Minion']}

### [15:44:03] Revealed #2 Knight
Info: {}

### [15:44:03] Revealed #4 Lover
Info: {'evil_adjacent': 2}

### [15:44:21] Revealed #3 Fortune_Teller
Info: {}

### [15:45:33] Revealed #5 Oracle
Info: {'targets': [6, 7], 'minion_role': 'Puppeteer'}

### [15:45:33] Revealed #8 Knight
Info: {}

### [15:45:33] Revealed #9 Bishop
Info: {'targets': [1, 2, 5], 'types': ['Villager', 'Outcast', 'Minion']}

### [15:45:48] Revealed #6 Judge
Info: {}

#### [15:45:49] Solver Output
Scenarios: 4/2016
Definite evil: ['#5', '#6', '#10']
Definite good: ['#1', '#2', '#7', '#8', '#9']
Evil probabilities: #3=50%, #4=50%

#### [15:45:49] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 4 scenarios (roles: {'Puppet'})

### [15:46:32] Executed #5 -> Puppet (EVIL)

#### [15:46:32] Solver Output
Scenarios: 4/644
Definite evil: ['#5', '#6', '#10']
Definite good: ['#1', '#2', '#7', '#8', '#9']
Evil probabilities: #3=50%, #4=50%

#### [15:46:32] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 4 scenarios (roles: {'Puppeteer'})

### [15:47:11] Executed #6 -> Puppeteer (EVIL)

#### [15:47:11] Solver Output
Scenarios: 4/98
Definite evil: ['#5', '#6', '#10']
Definite good: ['#1', '#2', '#7', '#8', '#9']
Evil probabilities: #3=50%, #4=50%

#### [15:47:11] Recommendation
Action: **USE_ABILITY** #3 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [15:48:06] Revealed #3 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [15:48:06] Ability used at #3

#### [15:48:06] Solver Output
Scenarios: 2/98
Definite evil: ['#4', '#5', '#6', '#10']
Definite good: ['#1', '#2', '#3', '#7', '#8', '#9']

#### [15:48:06] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Shaman'})

### [15:48:51] Executed #4 -> Lilis (EVIL)

## [15:48:51] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, night killed Slayer(#7) and Shaman(#10), FT confirmed #1/#2 clean, all 4 evils found at 100%


---

# New Game — 2026-04-10 15:52:34
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Gemcrafter, Slayer, Bishop, Empress, Oracle, Enlightened
- Outcasts: Wretch, Bombardier
- Minions: Puppeteer
- Demons: Pooka

### [15:52:54] Revealed #1 Gemcrafter
Info: {'good_position': 7}

### [15:52:54] Revealed #2 Empress
Info: {'targets': [5, 6, 7]}

### [15:52:54] Revealed #3 Enlightened
Info: {'direction': 'CCW'}

### [15:52:54] Revealed #4 Hunter
Info: {'distance': 2}

### [15:52:54] Revealed #5 Bombardier
Info: {}

### [15:52:54] Revealed #6 Bishop
Info: {'targets': [4, 5, 7], 'types': ['Villager', 'Outcast', 'Minion']}

### [15:52:54] Revealed #7 Wretch
Info: {}

### [15:52:54] Revealed #9 Oracle
Info: {'targets': [3, 5], 'minion_role': 'Puppeteer'}

### [15:53:03] Revealed #8 Slayer
Info: {}

#### [15:53:03] Solver Output
Scenarios: 3/112
Definite evil: ['#1']
Definite good: ['#3', '#4', '#5']
Evil probabilities: #2=67%, #6=33%, #7=33%, #8=33%, #9=33%

#### [15:53:03] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 3 scenarios (roles: {'Pooka', 'Puppeteer'})

### [15:53:48] Executed #1 -> Puppeteer (EVIL)

#### [15:53:48] Solver Output
Scenarios: 2/14
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#5', '#6', '#7']
Evil probabilities: #8=50%, #9=50%

#### [15:53:48] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Puppet'})

### [15:54:23] Executed #2 -> Puppet (EVIL)

#### [15:54:23] Solver Output
Scenarios: 2/7
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#5', '#6', '#7']
Evil probabilities: #8=50%, #9=50%

#### [15:54:23] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#9']
Reason: Target #9 is 50% evil (adjusted 0.25)
WARNING: Corruption risk: 50% -- Slayer ability disabled if corrupted

### [15:55:23] Revealed #8 Slayer
Info: {}

### [15:55:23] Ability used at #8

#### [15:55:23] Solver Output
Scenarios: 2/7
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#5', '#6', '#7']
Evil probabilities: #8=50%, #9=50%

#### [15:55:23] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (50% evil Pooka, 50% good Slayer (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:56:06] Executed #8 -> Pooka (EVIL)

## [15:56:07] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, all 3 evils at 100% confidence, Slayer couldn't kill #9 (Pooka disguised as Slayer)


---

# New Game — 2026-04-10 15:57:54
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Witness, Empress, Bard, Slayer, Lover
- Outcasts: Bombardier, Plague_Doctor
- Minions: Twin_Minion
- Demons: Baa

### [15:58:06] Revealed #1 Lover
Info: {'evil_adjacent': 0}

### [15:58:06] Revealed #3 Empress
Info: {'targets': [4, 5, 7]}

### [15:58:06] Revealed #6 Bard
Info: {'corruption_distance': 3}

### [15:58:23] Revealed #2 Plague_Doctor
Info: {}

### [15:58:23] Revealed #4 Slayer
Info: {}

### [15:58:23] Revealed #5 Plague_Doctor
Info: {}

### [15:58:23] Revealed #7 Witness
Info: {'affected_position': 0}

#### [15:58:31] Solver Output
Scenarios: 8/182
Definite good: ['#3', '#7']
Evil probabilities: #5=75%, #6=50%, #1=25%, #2=25%, #4=25%

#### [15:58:31] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.500 (adjusted 1.500) | timing x1.00

### [15:59:23] Ability used at #2

#### [15:59:23] Solver Output
Scenarios: 6/182
Definite good: ['#3', '#4', '#7']
Evil probabilities: #5=67%, #6=67%, #1=33%, #2=33%

#### [15:59:23] Recommendation
Action: **USE_ABILITY** #5 (Plague Doctor) -> targets ['#4']
Reason: Entropy 1.585 (adjusted 1.585) | timing x1.00

### [16:00:03] Ability used at #5

#### [16:00:03] Solver Output
Scenarios: 4/182
Definite good: ['#3', '#4', '#7']
Evil probabilities: #1=50%, #2=50%, #5=50%, #6=50%

#### [16:00:03] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#1']
Reason: Target #1 is 50% evil (adjusted 0.25)
WARNING: Corruption risk: 50% -- Slayer ability disabled if corrupted

### [16:00:39] Revealed #4 Slayer
Info: {}

### [16:00:39] Ability used at #4

#### [16:00:40] Solver Output
Scenarios: 4/182
Definite good: ['#3', '#4', '#7']
Evil probabilities: #1=50%, #2=50%, #5=50%, #6=50%

#### [16:00:40] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% good Lover, 25% evil Baa, 25% evil Twin_Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [16:00:53] Executed #1 -> Baa (EVIL)

#### [16:00:53] Solver Output
Scenarios: 1/26
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#5', '#6', '#7']

#### [16:00:53] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [16:01:34] Executed #2 -> Twin_Minion (EVIL)

## [16:01:35] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Slayer killed Baa, both PDs checked clean, solved to 1 scenario


---

# New Game — 2026-04-10 16:03:10
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Gemcrafter, Alchemist, Jester, Dreamer, Fortune_Teller
- Outcasts: Doppelganger, Plague_Doctor
- Minions: Twin_Minion
- Demons: Lilis

### [16:04:25] Revealed #1 Jester
Info: {}

### [16:04:26] Revealed #2 Dreamer
Info: {}

### [16:04:26] Revealed #3 Fortune_Teller
Info: {}

### [16:04:26] Revealed #4 Dreamer
Info: {}

### [16:05:03] Revealed #5 Knight
Info: {}

### [16:05:03] Revealed #6 Gemcrafter
Info: {'good_position': 5}

### [16:05:03] Revealed #7 Alchemist
Info: {'cured_count': 0}

### [16:05:14] Revealed #9 Jester
Info: {}

#### [16:05:14] Solver Output
Scenarios: 336/2716
Definite good: ['#5', '#6', '#7', '#8']
Evil probabilities: #9=50%, #1=38%, #2=38%, #3=38%, #4=38%

#### [16:05:14] Recommendation
Action: **USE_ABILITY** #2 (Dreamer) -> targets ['#9']
Reason: Entropy 2.406 (adjusted 2.105) | timing x1.00
WARNING: Corruption risk: 25%

### [16:06:03] Revealed #2 Dreamer
Info: {'target': 9, 'evil_role': 'Lilis'}

### [16:06:03] Ability used at #2

#### [16:06:03] Solver Output
Scenarios: 252/2716
Definite good: ['#5', '#6', '#7', '#8']
Evil probabilities: #1=42%, #2=42%, #3=42%, #4=42%, #9=33%

#### [16:06:03] Recommendation
Action: **USE_ABILITY** #4 (Dreamer) -> targets ['#1']
Reason: Entropy 2.302 (adjusted 2.015) | timing x1.00
WARNING: Corruption risk: 25%

### [16:06:49] Revealed #4 Dreamer
Info: {'target': 1, 'evil_role': 'Twin_Minion'}

### [16:06:49] Ability used at #4

#### [16:06:50] Solver Output
Scenarios: 196/2716
Definite good: ['#5', '#6', '#7', '#8']
Evil probabilities: #2=46%, #3=46%, #4=46%, #9=36%, #1=25%

#### [16:06:50] Recommendation
Action: **USE_ABILITY** #9 (Jester) -> targets ['#1', '#5', '#6']
Reason: Expected posterior 86.6 scenarios (adjusted 86.6, info gain 1.178 bits) | timing x1.00

### [16:07:36] Revealed #9 Jester
Info: {'targets': [1, 5, 6], 'evil_count': 0}

### [16:07:36] Ability used at #9

#### [16:07:36] Solver Output
Scenarios: 91/2716
Definite good: ['#5', '#6', '#7', '#8']
Evil probabilities: #2=62%, #3=62%, #4=62%, #1=8%, #9=8%

#### [16:07:36] Recommendation
Action: **USE_ABILITY** #3 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.996 (adjusted 0.881) | timing x1.00
WARNING: Corruption risk: 23%

### [16:08:22] Revealed #3 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [16:08:22] Ability used at #3

#### [16:08:23] Solver Output
Scenarios: 49/2716
Definite good: ['#5', '#6', '#7', '#8']
Evil probabilities: #2=86%, #3=57%, #4=29%, #1=14%, #9=14%

#### [16:08:23] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#2', '#3', '#9']
Reason: Expected posterior 23.2 scenarios (adjusted 26.5, info gain 0.889 bits) | timing x1.00
WARNING: Corruption risk: 29%

### [16:09:11] Revealed #1 Jester
Info: {'targets': [2, 3, 9], 'evil_count': 2}

### [16:09:11] Ability used at #1

#### [16:09:11] Solver Output
Scenarios: 21/2716
Definite good: ['#4', '#5', '#6', '#7', '#8']
Evil probabilities: #2=67%, #3=67%, #1=33%, #9=33%

#### [16:09:11] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (33% evil Lilis, 33% evil Twin_Minion, 29% good Dreamer).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [16:09:54] Executed #2 -> Twin_Minion (EVIL)

#### [16:09:54] Solver Output
Scenarios: 7/322
Definite evil: ['#2', '#3']
Definite good: ['#1', '#4', '#5', '#6', '#7', '#8', '#9']

#### [16:09:54] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 7 scenarios (roles: {'Lilis'})

### [16:10:37] Executed #3 -> Lilis (EVIL)

## [16:10:38] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, heavy ability usage (2 Dreamers, 2 Jesters, FT), night killed PD(#8), all abilities used to narrow 336->21 scenarios


---

# New Game — 2026-04-10 16:12:19
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Dreamer, Jester, Enlightened, Fortune_Teller, Judge, Architect, Alchemist
- Outcasts: Plague_Doctor
- Minions: Twin_Minion, Minion
- Demons: Pooka

### [16:12:33] Revealed #3 Enlightened
Info: {'direction': 'CW'}

### [16:12:33] Revealed #6 Architect
Info: {'side': 'Left'}

### [16:12:33] Revealed #7 Alchemist
Info: {'cured_count': 2}

### [16:12:33] Revealed #8 Bishop
Info: {'targets': [2, 3, 7], 'types': ['Villager', 'Outcast', 'Minion']}

### [16:12:58] Revealed #1 Plague_Doctor
Info: {}

### [16:12:58] Revealed #2 Judge
Info: {}

### [16:12:59] Revealed #4 Alchemist
Info: {'cured_count': 2}

### [16:12:59] Revealed #5 Jester
Info: {}

### [16:12:59] Revealed #9 Fortune_Teller
Info: {}

### [16:12:59] Revealed #10 Dreamer
Info: {}

#### [16:13:06] Solver Output
Scenarios: 90/3240
Definite good: ['#1']
Evil probabilities: #7=78%, #4=67%, #5=38%, #8=36%, #9=36%, #10=22%, #3=13%, #2=7%, #6=4%

#### [16:13:06] Recommendation
Action: **USE_ABILITY** #1 (Plague Doctor) -> targets ['#8']
Reason: Entropy 2.545 (adjusted 2.545) | timing x1.00

### [16:13:47] Ability used at #1

#### [16:13:47] Solver Output
Scenarios: 32/3240
Definite evil: ['#8']
Definite good: ['#1', '#3', '#6']
Evil probabilities: #4=75%, #7=50%, #5=31%, #9=31%, #2=6%, #10=6%

#### [16:13:47] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 32 scenarios (roles: {'Minion', 'Twin_Minion', 'Pooka'})

### [16:14:30] Executed #8 -> Pooka (EVIL)

#### [16:14:30] Solver Output
Scenarios: 22/352
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#6']
Evil probabilities: #4=91%, #7=45%, #9=36%, #5=18%, #10=9%

#### [16:14:30] Recommendation
Action: **USE_ABILITY** #10 (Dreamer) -> targets ['#7']
Reason: Entropy 2.278 (adjusted 2.071) | timing x1.00
WARNING: Corruption risk: 18%

### [16:15:07] Revealed #10 Dreamer
Info: {'target': 7, 'evil_role': 'Minion'}

### [16:15:07] Ability used at #10

#### [16:15:07] Solver Output
Scenarios: 17/352
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#6']
Evil probabilities: #4=94%, #9=47%, #7=29%, #5=18%, #10=12%

#### [16:15:07] Recommendation
Action: **USE_ABILITY** #5 (Jester) -> targets ['#1', '#2', '#10']
Reason: Expected posterior 7.5 scenarios (adjusted 8.1, info gain 1.062 bits) | timing x1.00
WARNING: Corruption risk: 18%

### [16:15:50] Revealed #5 Jester
Info: {'targets': [1, 2, 10], 'evil_count': 2}

### [16:15:50] Ability used at #5

#### [16:15:50] Solver Output
Scenarios: 6/352
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#6', '#10']
Evil probabilities: #4=83%, #5=50%, #7=33%, #9=33%

#### [16:15:50] Recommendation
Action: **USE_ABILITY** #2 (Judge) -> targets ['#6']
Reason: Expected posterior 3.3 scenarios (adjusted 3.3, info gain 0.848 bits) | timing x1.00

### [16:16:33] Revealed #2 Judge
Info: {'target': 6, 'is_lying': False}

### [16:16:33] Ability used at #2

#### [16:16:34] Solver Output
Scenarios: 4/352
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#6', '#10']
Evil probabilities: #4=75%, #7=50%, #9=50%, #5=25%

#### [16:16:34] Recommendation
Action: **USE_ABILITY** #9 (Fortune Teller) -> targets ['#1', '#7']
Reason: Entropy 1.000 (adjusted 0.750) | timing x1.00
WARNING: Corruption risk: 50%

### [16:17:13] Revealed #9 Fortune Teller
Info: {'targets': [1, 7], 'has_evil': True}

### [16:17:13] Ability used at #9

#### [16:17:13] Solver Output
Scenarios: 2/352
Definite evil: ['#4', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7', '#10']

#### [16:17:13] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Twin_Minion', 'Minion'})

### [16:17:57] Executed #4 -> Minion (EVIL)

#### [16:17:57] Solver Output
Scenarios: 1/43
Definite evil: ['#4', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7', '#10']

#### [16:17:57] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [16:18:40] Executed #9 -> Twin_Minion (EVIL)

## [16:18:40] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, 8 abilities used (PD+Dreamer+2Jesters+FT+Judge+Alchemist), solved 90 scenarios down to 1


---

# New Game — 2026-04-10 16:20:17
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Bishop, Baker, Bard, Empress, Poet, Knight
- Outcasts: Plague_Doctor
- Minions: Puppeteer, Minion
- Demons: Pooka

### [16:20:32] Revealed #1 Baker
Info: {'original_role': 'original'}

### [16:20:32] Revealed #2 Bard
Info: {'corruption_distance': 1}

### [16:20:32] Revealed #6 Knight
Info: {}

### [16:20:32] Revealed #7 Bishop
Info: {'targets': [1, 10, 7], 'types': ['Villager', 'Outcast', 'Minion']}

### [16:20:32] Revealed #10 Empress
Info: {'targets': [1, 7, 8]}

### [16:20:49] Revealed #3 Druid
Info: {}

### [16:20:50] Revealed #4 Druid
Info: {}

### [16:20:50] Revealed #5 Druid
Info: {}

### [16:20:50] Revealed #8 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 9}

### [16:20:50] Revealed #9 Plague_Doctor
Info: {}

#### [16:20:57] Solver Output
Scenarios: 97/3696
Definite good: ['#9', '#10']
Evil probabilities: #4=82%, #5=68%, #3=62%, #8=52%, #7=48%, #6=42%, #2=37%, #1=8%

#### [16:20:57] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#2']
Reason: Entropy 2.268 (adjusted 2.268) | timing x1.00

### [16:21:40] Ability used at #9

#### [16:21:40] Solver Output
Scenarios: 51/3696
Definite good: ['#1', '#9', '#10']
Evil probabilities: #4=84%, #2=71%, #5=61%, #3=55%, #8=51%, #7=49%, #6=29%

#### [16:21:40] Recommendation
Action: **USE_ABILITY** #5 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.993 (adjusted 0.915) | timing x1.00
WARNING: Corruption risk: 16%

### [16:22:26] Revealed #5 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [16:22:27] Ability used at #5

#### [16:22:27] Solver Output
Scenarios: 23/3696
Definite evil: ['#4']
Definite good: ['#1', '#9', '#10']
Evil probabilities: #2=70%, #3=52%, #8=52%, #5=48%, #7=48%, #6=30%

#### [16:22:27] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 23 scenarios (roles: {'Puppet', 'Pooka', 'Minion', 'Puppeteer'})

### [16:23:09] Executed #4 -> Puppeteer (EVIL)

#### [16:23:09] Solver Output
Scenarios: 11/448
Definite evil: ['#4']
Definite good: ['#1', '#9', '#10']
Evil probabilities: #2=73%, #5=64%, #8=55%, #7=45%, #3=36%, #6=27%

#### [16:23:09] Recommendation
Action: **USE_ABILITY** #3 (Druid) -> targets ['#1', '#2', '#5']
Reason: Entropy 0.845 (adjusted 0.730) | timing x1.00
WARNING: Corruption risk: 27%

### [16:25:05] Revealed #3 Druid
Info: {'targets': [1, 2, 5], 'found_outcast': 'Plague_Doctor'}

### [16:25:05] Ability used at #3

#### [16:25:05] Solver Output
Scenarios: 3/448
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#9', '#10']
Evil probabilities: #8=67%, #7=33%

#### [16:25:05] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 3 scenarios (roles: {'Puppet'})

### [16:25:52] Executed #5 -> Puppet (EVIL)

#### [16:25:52] Solver Output
Scenarios: 3/224
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#9', '#10']
Evil probabilities: #8=67%, #7=33%

#### [16:25:52] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 3 scenarios (roles: {'Pooka', 'Minion'})

#### [16:26:51] Execution Blocked
#6 Knight immunity — confirmed good, no HP loss

#### [16:26:51] Solver Output
Scenarios: 0/162

#### [16:26:51] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [16:27:10] Revealed #3 Druid
Info: {'targets': [1, 2, 5], 'found_outcast': 'Doppelganger'}

#### [16:27:10] Solver Output
Scenarios: 0/162

#### [16:27:10] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [16:27:50] Revealed #3 Druid
Info: {'targets': [1, 2, 5], 'found_outcast': None}

#### [16:27:51] Solver Output
Scenarios: 4/162
Definite evil: ['#2', '#4', '#5']
Definite good: ['#1', '#3', '#6', '#9', '#10']
Evil probabilities: #7=50%, #8=50%

#### [16:27:51] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Minion'})

### [16:28:56] Revealed #3 Druid
Info: {}

#### [16:28:56] Solver Output
Scenarios: 4/162
Definite evil: ['#2', '#4', '#5']
Definite good: ['#1', '#3', '#6', '#9', '#10']
Evil probabilities: #7=50%, #8=50%

#### [16:28:56] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Minion'})

### [16:30:03] Revealed #5 Druid
Info: {}

#### [16:30:03] Solver Output
Scenarios: 4/162
Definite evil: ['#2', '#4', '#5']
Definite good: ['#1', '#3', '#6', '#9', '#10']
Evil probabilities: #7=50%, #8=50%

#### [16:30:03] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Minion'})

### [16:31:30] Executed #2 -> GOOD (WRONG!)

#### [16:31:30] Solver Output
Scenarios: 0/110

#### [16:31:30] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [16:31:44] Revealed #7 Bishop
Info: {}

#### [16:31:44] Solver Output
Scenarios: 0/110

#### [16:31:44] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [16:32:45] Revealed #8 Poet
Info: {}

#### [16:32:45] Solver Output
Scenarios: 0/110

#### [16:32:45] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [16:32:51] Revealed #1 Baker
Info: {}

#### [16:32:51] Solver Output
Scenarios: 6/110
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#2', '#6', '#9']
Evil probabilities: #10=67%, #7=17%, #8=17%

#### [16:32:51] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 6 scenarios (roles: {'Minion'})

### [16:33:32] Executed #3 -> Minion (EVIL)

#### [16:33:32] Solver Output
Scenarios: 6/21
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#2', '#6', '#9']
Evil probabilities: #10=67%, #7=17%, #8=17%

#### [16:33:32] Recommendation
Action: **ERROR** #10
Reason: #10 is 67% likely evil but budget=1 requires >=85% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 67% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 67% < 85% threshold. Consider manual override if you have extra information.

### [16:33:57] Revealed #8 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 9}

#### [16:33:57] Solver Output
Scenarios: 3/21
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#2', '#6', '#9']
Evil probabilities: #7=33%, #8=33%, #10=33%

#### [16:33:57] Recommendation
Action: **ERROR** #10
Reason: #10 is 33% likely evil but budget=1 requires >=85% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 33% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 33% < 85% threshold. Consider manual override if you have extra information.

#### [16:34:13] Claude Reasoning


### [16:35:30] Revealed #3 Druid
Info: {'targets': [1, 2, 5], 'found_outcast': 'Plague_Doctor'}

#### [16:35:31] Solver Output
Scenarios: 3/21
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#2', '#6', '#9']
Evil probabilities: #7=33%, #8=33%, #10=33%

#### [16:35:31] Recommendation
Action: **ERROR** #10
Reason: #10 is 33% likely evil but budget=1 requires >=85% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 33% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 33% < 85% threshold. Consider manual override if you have extra information.

### [16:36:10] Revealed #7 Bishop
Info: {'targets': [1, 7, 10], 'types': ['Villager', 'Outcast', 'Minion']}

#### [16:36:10] Solver Output
Scenarios: 2/21
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#2', '#6', '#9', '#10']
Evil probabilities: #7=50%, #8=50%

#### [16:36:10] Recommendation
Action: **ERROR** #7
Reason: #7 is 50% likely evil but budget=1 requires >=85% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 50% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 50% < 85% threshold. Consider manual override if you have extra information.

### [16:37:28] Executed #7 -> GOOD (WRONG!)

## [16:37:28] GAME OVER — LOSS
Final HP: 0
Notes: LOSS - data entry errors caused solver degraded state. Druid #3 said 'Doppelganger' (out of pool) - solver couldnt handle. Multiple wrong execs from bad info: #2 Bard (wrong), #6 Knight (immune), #7 Bishop (wrong). #8 Pooka was last evil per memory reader.


---

# New Game — 2026-04-10 16:46:53
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Alchemist, Jester, Druid, Bard, Dreamer
- Outcasts: Wretch
- Minions: Poisoner
- Demons: Pooka

### [16:47:09] Revealed #3 Bard
Info: {'corruption_distance': 2}

### [16:47:09] Revealed #7 Wretch
Info: {}

### [16:47:09] Revealed #8 Poet
Info: {'distance': 4, 'copied_role': 'Hunter'}

### [16:47:21] Revealed #1 Alchemist
Info: {'cured_count': 2}

### [16:47:21] Revealed #2 Dreamer
Info: {}

### [16:47:21] Revealed #4 Druid
Info: {}

### [16:47:21] Revealed #5 Jester
Info: {}

### [16:47:21] Revealed #6 Alchemist
Info: {'cured_count': 1}

#### [16:47:27] Solver Output
Scenarios: 3/80
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5', '#7']
Evil probabilities: #8=67%, #1=33%

#### [16:47:27] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 3 scenarios (roles: {'Poisoner', 'Pooka'})

### [16:48:08] Executed #6 -> Pooka (EVIL)

#### [16:48:08] Solver Output
Scenarios: 2/10
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5', '#7']
Evil probabilities: #1=50%, #8=50%

#### [16:48:08] Recommendation
Action: **USE_ABILITY** #2 (Dreamer) -> targets ['#1']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [16:48:49] Revealed #2 Dreamer
Info: {'target': 1, 'evil_role': 'Poisoner'}

### [16:48:49] Ability used at #2

#### [16:48:49] Solver Output
Scenarios: 2/10
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5', '#7']
Evil probabilities: #1=50%, #8=50%

#### [16:48:49] Recommendation
Action: **USE_ABILITY** #4 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.000 (adjusted 0.000) | timing x1.00

#### [16:49:10] Solver Output
Scenarios: 2/10
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5', '#7']
Evil probabilities: #1=50%, #8=50%

#### [16:49:10] Recommendation
Action: **USE_ABILITY** #4 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.000 (adjusted 0.000) | timing x1.00

### [16:49:58] Revealed #4 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [16:49:58] Ability used at #4

#### [16:49:58] Solver Output
Scenarios: 2/10
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5', '#7']
Evil probabilities: #1=50%, #8=50%

#### [16:49:58] Recommendation
Action: **USE_ABILITY** #5 (Jester) -> targets ['#1', '#2', '#3']
Reason: Expected posterior 1.7 scenarios (adjusted 2.5, info gain 0.000 bits) | timing x1.00
WARNING: Corruption risk: 100%

### [16:50:53] Executed #1 -> Poisoner (EVIL)

## [16:50:53] GAME OVER — WIN
Final HP: 10
Notes: 10HP retry win after v6 loss, Dreamer pinpointed Poisoner, Pooka+Poisoner game with 2 Alchemist disguises


---

# New Game — 2026-04-10 16:52:29
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Medium, Knitter, Dreamer, Jester, Witness
- Outcasts: Doppelganger, Plague_Doctor, Wretch
- Minions: Minion, Witch
- Demons: Baa

### [16:52:51] Revealed #1 Witness
Info: {'affected_position': 6}

### [16:52:51] Revealed #3 Wretch
Info: {}

### [16:52:51] Revealed #4 Knitter
Info: {'evil_pairs': 2}

### [16:52:51] Revealed #6 Medium
Info: {'good_position': 7, 'good_role': 'Dreamer'}

### [16:53:32] Revealed #2 Witness
Info: {'affected_position': 0}

### [16:53:33] Revealed #5 Jester
Info: {}

### [16:53:33] Revealed #7 Dreamer
Info: {}

### [16:53:33] Revealed #8 Druid
Info: {}

#### [16:53:42] Solver Output
Scenarios: 636/13902
Definite good: ['#2']
Evil probabilities: #1=83%, #4=61%, #8=46%, #3=42%, #6=25%, #7=21%, #5=17%, #9=5%

#### [16:53:42] Recommendation
Action: **USE_ABILITY** #7 (Dreamer) -> targets ['#1']
Reason: Entropy 2.843 (adjusted 2.682) | timing x1.00
WARNING: Corruption risk: 11%

### [16:54:27] Revealed #7 Dreamer
Info: {'target': 1, 'evil_role': 'Minion'}

### [16:54:27] Ability used at #7

#### [16:54:27] Solver Output
Scenarios: 332/13902
Definite good: ['#2']
Evil probabilities: #1=67%, #4=60%, #8=46%, #3=37%, #6=36%, #7=33%, #5=18%, #9=3%

#### [16:54:27] Recommendation
Action: **USE_ABILITY** #5 (Jester) -> targets ['#1', '#6', '#7']
Reason: Expected posterior 128.4 scenarios (adjusted 133.0, info gain 1.320 bits) | timing x1.00
WARNING: Corruption risk: 7%

### [16:55:11] Revealed #5 Jester
Info: {'targets': [1, 6, 7], 'evil_count': 1}

### [16:55:11] Ability used at #5

#### [16:55:11] Solver Output
Scenarios: 160/13902
Definite good: ['#2']
Evil probabilities: #1=74%, #4=66%, #8=44%, #3=42%, #6=29%, #5=20%, #7=19%, #9=6%

#### [16:55:11] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 1.000 (adjusted 0.962) | timing x1.00
WARNING: Corruption risk: 8%

### [16:56:01] Revealed #8 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Doppelganger'}

### [16:56:01] Ability used at #8

#### [16:56:02] Solver Output
Scenarios: 86/13902
Definite good: ['#2']
Evil probabilities: #1=79%, #8=67%, #4=65%, #3=30%, #6=21%, #5=14%, #7=14%, #9=9%

#### [16:56:02] Recommendation
Action: **EXECUTE** #1
Reason: No reveals available. #1 is 79% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 79% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card

### [16:56:43] Executed #1 -> Minion (EVIL)

#### [16:56:43] Solver Output
Scenarios: 48/1670
Definite evil: ['#1']
Definite good: ['#2', '#5', '#6', '#7']
Evil probabilities: #8=75%, #4=62%, #3=46%, #9=17%

#### [16:56:43] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (38% evil Baa, 38% evil Witch, 17% good Druid (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [16:57:24] Executed #8 -> Baa (EVIL)

#### [16:57:24] Solver Output
Scenarios: 18/228
Definite evil: ['#1', '#8']
Definite good: ['#2', '#5', '#6', '#7']
Evil probabilities: #4=50%, #3=28%, #9=22%

#### [16:57:24] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (50% evil Witch, 22% good Knitter (corrupted), 17% good Knitter).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [16:58:15] Executed #4 -> GOOD (WRONG!)

#### [16:58:15] Solver Output
Scenarios: 9/192
Definite evil: ['#1', '#8']
Definite good: ['#2', '#4', '#5', '#6', '#7']
Evil probabilities: #3=56%, #9=44%

#### [16:58:15] Recommendation
Action: **ERROR** #9
Reason: #9 is 44% likely evil but budget=1 requires >=85% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 44% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: CAUTION: budget=1, confidence 44% < 85% threshold. Consider manual override if you have extra information.

### [16:59:05] Executed #9 -> Witch (EVIL)

## [16:59:05] GAME OVER — WIN
Final HP: 5
Notes: 5HP, FINAL village! Witch at #9 blocked itself, executed directly per CLAUDE.md. Asc52 complete!


---

# New Game — 2026-04-10 17:30:42
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Bard, Fortune_Teller, Druid, Medium, Baker
- Outcasts: Plague_Doctor, Bombardier
- Minions: Chancellor
- Demons: Pooka

### [17:31:00] Revealed #1 Bombardier
Info: {}

### [17:31:00] Revealed #2 Medium
Info: {'good_position': 6, 'good_role': 'Medium'}

### [17:31:00] Revealed #3 Baker
Info: {'original_role': 'Druid'}

### [17:31:00] Revealed #6 Medium
Info: {'good_position': 2, 'good_role': 'Medium'}

### [17:31:00] Revealed #7 Bard
Info: {'corruption_distance': 3}

### [17:31:00] Revealed #8 Bishop
Info: {'targets': [3, 1, 2], 'types': ['Villager', 'Outcast', 'Minion']}

### [17:31:15] Revealed #4 Fortune_Teller
Info: {}

### [17:31:15] Revealed #5 Plague_Doctor
Info: {}

#### [17:31:15] Solver Output
Scenarios: 0/256

#### [17:31:15] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [17:32:05] Revealed #3 Baker
Info: {}

#### [17:32:05] Solver Output
Scenarios: 1/256
Definite evil: ['#2', '#6']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [17:32:05] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Chancellor'})

### [17:32:47] Executed #2 -> Chancellor (EVIL)

#### [17:32:48] Solver Output
Scenarios: 1/26
Definite evil: ['#2', '#6']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [17:32:48] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [17:33:35] Executed #6 -> Pooka (EVIL)

## [17:33:35] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, 2 Mediums on board (both evil disguises), corrupted Baker false claim required reset


---

# New Game — 2026-04-10 17:35:14
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Hunter, Bard, Architect, Jester, Slayer
- Outcasts: Plague_Doctor, Wretch
- Minions: Witch
- Demons: Baa

### [17:35:32] Revealed #1 Wretch
Info: {}

### [17:35:32] Revealed #2 Architect
Info: {'side': 'Right'}

### [17:35:32] Revealed #4 Bard
Info: {'corruption_distance': 2}

### [17:35:32] Revealed #5 Lover
Info: {'evil_adjacent': 1}

### [17:35:32] Revealed #6 Hunter
Info: {'distance': 2}

### [17:35:42] Revealed #3 Slayer
Info: {}

#### [17:35:42] Solver Output
Scenarios: 6/162
Definite evil: ['#3']
Definite good: ['#1', '#2', '#6', '#7']
Evil probabilities: #4=67%, #5=33%

#### [17:35:42] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 6 scenarios (roles: {'Witch', 'Baa'})

### [17:36:23] Executed #3 -> Witch (EVIL)

### [17:37:00] Revealed #7 Jester
Info: {}

#### [17:37:00] Solver Output
Scenarios: 1/6
Definite evil: ['#3', '#4']
Definite good: ['#1', '#2', '#5', '#6', '#7']

#### [17:37:00] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Baa'})

### [17:37:45] Executed #4 -> Baa (EVIL)

## [17:37:45] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Witch+Baa, solver found Witch at 100%, then Baa at 100% after #7 unblocked


---

# New Game — 2026-04-10 17:39:26
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Enlightened, Alchemist, Bard, Scout, Fortune_Teller, Confessor
- Outcasts: Doppelganger, Plague_Doctor
- Minions: Poisoner
- Demons: Lilis

### [17:39:54] Revealed #1 Scout
Info: {'evil_role': 'Lilis', 'distance': 2}

### [17:39:54] Revealed #3 Enlightened
Info: {'direction': 'CCW'}

### [17:40:03] Revealed #2 Fortune_Teller
Info: {}

### [17:40:03] Revealed #4 Plague_Doctor
Info: {}

### [17:40:37] Revealed #5 Architect
Info: {'side': 'Equal'}

### [17:40:37] Revealed #6 Alchemist
Info: {'cured_count': 1}

### [17:40:37] Revealed #7 Bard
Info: {'corruption_distance': 2}

### [17:40:37] Revealed #8 Confessor
Info: {'dizzy': False}

#### [17:40:48] Solver Output
Scenarios: 135/3130
Definite good: ['#4', '#8', '#9']
Evil probabilities: #7=69%, #1=54%, #6=31%, #2=25%, #3=10%, #5=10%

#### [17:40:48] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.413 (adjusted 1.413) | timing x1.00

### [17:41:29] Ability used at #4

#### [17:41:29] Solver Output
Scenarios: 88/3130
Definite good: ['#4', '#8', '#9']
Evil probabilities: #1=58%, #7=52%, #6=48%, #2=27%, #3=10%, #5=5%

#### [17:41:29] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#4', '#6']
Reason: Entropy 1.000 (adjusted 0.835) | timing x1.00
WARNING: Corruption risk: 33%

### [17:42:20] Revealed #2 Fortune Teller
Info: {'targets': [4, 6], 'has_evil': False}

### [17:42:20] Ability used at #2

#### [17:42:20] Solver Output
Scenarios: 43/3130
Definite good: ['#4', '#8', '#9']
Evil probabilities: #1=58%, #6=58%, #7=42%, #2=23%, #3=9%, #5=9%

#### [17:42:20] Recommendation
Action: **ERROR** #6
Reason: #6 is 58% likely evil but budget=1 requires >=83% confidence (HP=6, cost=5).
WARNING: Probabilistic execution -- 58% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 58% < 83% threshold. Consider manual override if you have extra information.

### [17:43:18] Executed #6 -> GOOD (WRONG!)

#### [17:43:18] Solver Output
Scenarios: 18/2031
Definite evil: ['#7']
Definite good: ['#2', '#4', '#6', '#8', '#9']
Evil probabilities: #1=56%, #3=22%, #5=22%

#### [17:43:18] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 18 scenarios (roles: {'Poisoner', 'Lilis'})

### [17:44:01] Executed #7 -> Lilis (EVIL)

#### [17:44:01] Solver Output
Scenarios: 2/278
Definite evil: ['#1', '#7']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#8', '#9']

#### [17:44:01] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Poisoner'})

### [17:44:53] Executed #1 -> Poisoner (EVIL)

## [17:44:53] GAME OVER — WIN
Final HP: 1
Notes: 1HP clutch, Lilis game, night killed Doppelganger #9, wrong exec on Alchemist #6 at 58% (followed solver per CLAUDE.md), recovered with 100% picks for Lilis and Poisoner


---

# New Game — 2026-04-10 17:46:51
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Alchemist, Slayer, Judge, Scout, Poet, Knight
- Outcasts: Plague_Doctor
- Minions: Minion, Shaman
- Demons: Pooka

### [17:47:11] Revealed #1 Knight
Info: {}

### [17:47:11] Revealed #2 Poet
Info: {'distance': 5, 'copied_role': 'Hunter'}

### [17:47:11] Revealed #4 Confessor
Info: {'dizzy': False}

### [17:47:11] Revealed #5 Scout
Info: {'evil_role': 'Minion', 'distance': 1}

### [17:47:11] Revealed #7 Alchemist
Info: {'cured_count': 2}

### [17:47:11] Revealed #10 Poet
Info: {'good_position': 1, 'good_role': 'Knight', 'copied_role': 'Medium'}

### [17:47:24] Revealed #3 Slayer
Info: {}

### [17:47:25] Revealed #6 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 6}

### [17:47:25] Revealed #8 Slayer
Info: {}

### [17:47:25] Revealed #9 Plague_Doctor
Info: {}

#### [17:47:25] Solver Output
Scenarios: 18/3240
Definite evil: ['#2']
Definite good: ['#1', '#4', '#6', '#9', '#10']
Evil probabilities: #7=78%, #8=61%, #3=50%, #5=11%

#### [17:47:25] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 18 scenarios (roles: {'Minion', 'Pooka', 'Shaman'})

### [17:48:11] Executed #2 -> Shaman (EVIL)

#### [17:48:11] Solver Output
Scenarios: 9/352
Definite evil: ['#2']
Definite good: ['#1', '#4', '#6', '#9', '#10']
Evil probabilities: #7=78%, #8=67%, #3=44%, #5=11%

#### [17:48:11] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#6']
Reason: Entropy 0.986 (adjusted 0.986) | timing x1.00

### [17:49:03] Ability used at #9

#### [17:49:03] Solver Output
Scenarios: 4/352
Definite evil: ['#2', '#3']
Definite good: ['#1', '#4', '#5', '#6', '#9', '#10']
Evil probabilities: #7=75%, #8=25%

#### [17:49:03] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 4 scenarios (roles: {'Minion'})

### [17:49:48] Executed #3 -> Minion (EVIL)

#### [17:49:48] Solver Output
Scenarios: 4/43
Definite evil: ['#2', '#3']
Definite good: ['#1', '#4', '#5', '#6', '#9', '#10']
Evil probabilities: #7=75%, #8=25%

#### [17:49:48] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#7']
Reason: Target #7 is 75% evil (adjusted 0.19)
WARNING: Corruption risk: 75% -- Slayer ability disabled if corrupted

### [17:50:38] Revealed #8 Slayer
Info: {}

### [17:50:39] Ability used at #8

#### [17:50:39] Solver Output
Scenarios: 4/43
Definite evil: ['#2', '#3']
Definite good: ['#1', '#4', '#5', '#6', '#9', '#10']
Evil probabilities: #7=75%, #8=25%

#### [17:50:39] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (75% evil Pooka, 25% good Alchemist (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [17:51:27] Executed #7 -> GOOD (WRONG!)

#### [17:51:27] Solver Output
Scenarios: 1/37
Definite evil: ['#2', '#3', '#8']
Definite good: ['#1', '#4', '#5', '#6', '#7', '#9', '#10']

#### [17:51:27] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [17:52:15] Executed #8 -> Pooka (EVIL)

## [17:52:16] GAME OVER — WIN
Final HP: 5
Notes: 5HP, Shaman+Minion+Pooka, PD revealed Minion at #3, wrong exec on Alchemist #7 (lookahead 25%), recovered


---

# New Game — 2026-04-10 17:53:59
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Bard, Judge, Knight, Enlightened
- Outcasts: Drunk, Wretch, Doppelganger
- Minions: Poisoner
- Demons: Baa

### [17:54:14] Revealed #1 Bard
Info: {'corruption_distance': -1}

### [17:54:14] Revealed #2 Confessor
Info: {'dizzy': False}

### [17:54:14] Revealed #5 Enlightened
Info: {'direction': 'CCW'}

### [17:54:14] Revealed #6 Confessor
Info: {'dizzy': False}

### [17:54:14] Revealed #7 Knight
Info: {}

### [17:54:14] Revealed #8 Wretch
Info: {}

### [17:54:24] Revealed #3 Judge
Info: {}

### [17:54:25] Revealed #4 Enlightened
Info: {'direction': 'equidistant'}

#### [17:54:25] Solver Output
Scenarios: 79/2894
Definite good: ['#2', '#6']
Evil probabilities: #1=57%, #4=39%, #5=38%, #3=25%, #7=20%, #8=20%

#### [17:54:25] Recommendation
Action: **EXECUTE** #7
Reason: Knight check: #7 is 20% evil, 18% corruption risk. Expected HP cost: 1.3 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 18% -- corrupted Knight loses immunity + 4 extra damage

#### [17:55:16] Execution Blocked
#7 Knight immunity — confirmed good, no HP loss

#### [17:55:16] Solver Output
Scenarios: 63/2320
Definite good: ['#2', '#6', '#7']
Evil probabilities: #1=68%, #5=41%, #4=33%, #3=32%, #8=25%

#### [17:55:16] Recommendation
Action: **USE_ABILITY** #3 (Judge) -> targets ['#7']
Reason: Expected posterior 40.6 scenarios (adjusted 46.3, info gain 0.443 bits) | timing x1.00
WARNING: Corruption risk: 29% -- corrupted Judge results are unreliable

### [17:56:08] Revealed #3 Judge
Info: {'target': 7, 'is_lying': True}

### [17:56:08] Ability used at #3

#### [17:56:08] Solver Output
Scenarios: 42/2320
Definite good: ['#2', '#6', '#7']
Evil probabilities: #1=71%, #4=50%, #3=40%, #5=19%, #8=19%

#### [17:56:08] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (64% evil Baa, 14% good Drunk (corrupted), 10% good Bard (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 71%, but all reveal branches still lead to a forced win.

### [17:56:57] Executed #1 -> Baa (EVIL)

#### [17:56:57] Solver Output
Scenarios: 27/314
Definite evil: ['#1']
Definite good: ['#2', '#6', '#7', '#8']
Evil probabilities: #4=56%, #5=30%, #3=15%

#### [17:56:57] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (56% evil Poisoner, 30% good Enlightened (corrupted), 15% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 56%, but all reveal branches still lead to a forced win.

### [17:57:50] Executed #4 -> Poisoner (EVIL)

## [17:57:50] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Knight check on #7 succeeded (immunity blocked), then lookahead picks for Baa and Poisoner


---

# New Game — 2026-04-10 17:59:37
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Medium, Alchemist, Knitter, Scout, Enlightened
- Outcasts: Wretch, Bombardier, Doppelganger
- Minions: Chancellor
- Demons: Lilis

### [18:00:20] Revealed #1 Wretch
Info: {}

### [18:00:20] Revealed #2 Architect
Info: {'side': 'Equal'}

### [18:00:20] Revealed #3 Scout
Info: {'evil_role': 'Chancellor', 'distance': 1}

### [18:00:20] Revealed #4 Bombardier
Info: {}

### [18:00:56] Revealed #5 Medium
Info: {'good_position': 4, 'good_role': 'Bombardier'}

### [18:00:56] Revealed #6 Scout
Info: {'evil_role': 'Chancellor', 'distance': 1}

### [18:01:04] Revealed #8 Alchemist
Info: {'cured_count': 1}

### [18:01:04] Revealed #9 Enlightened
Info: {'direction': 'cw'}

#### [18:01:04] Solver Output
Scenarios: 10/498
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#7']

#### [18:01:04] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 10 scenarios (roles: {'Chancellor', 'Lilis'})

### [18:02:02] Executed #8 -> Lilis (EVIL)

#### [18:02:03] Solver Output
Scenarios: 5/54
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#7']

#### [18:02:03] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 5 scenarios (roles: {'Chancellor'})

### [18:02:54] Executed #9 -> Chancellor (EVIL)

## [18:02:54] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, night killed Knitter #7, both evils found at 100% confidence


---

# New Game — 2026-04-10 18:04:41
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Poet, Fortune_Teller, Lover, Alchemist, Oracle, Confessor
- Outcasts: Plague_Doctor
- Minions: Puppeteer, Minion
- Demons: Pooka

### [18:04:56] Revealed #2 Confessor
Info: {'dizzy': False}

### [18:04:56] Revealed #6 Confessor
Info: {'dizzy': True}

### [18:04:56] Revealed #7 Lover
Info: {'evil_adjacent': 0}

### [18:04:56] Revealed #8 Oracle
Info: {'targets': [2, 7], 'minion_role': 'Puppeteer'}

### [18:04:56] Revealed #9 Baker
Info: {'original_role': 'original'}

### [18:05:13] Revealed #1 Alchemist
Info: {'cured_count': 2}

### [18:05:13] Revealed #3 Plague_Doctor
Info: {}

### [18:05:13] Revealed #4 Poet
Info: {'targets': [1, 4], 'minion_role': 'Puppeteer', 'copied_role': 'Oracle'}

### [18:05:14] Revealed #5 Fortune_Teller
Info: {}

#### [18:05:14] Solver Output
Scenarios: 22/1932
Definite evil: ['#6']
Definite good: ['#2', '#3']
Evil probabilities: #8=59%, #9=55%, #4=50%, #7=50%, #5=45%, #1=41%

#### [18:05:14] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 22 scenarios (roles: {'Puppet', 'Pooka', 'Minion', 'Puppeteer'})

### [18:06:02] Executed #6 -> Minion (EVIL)

#### [18:06:03] Solver Output
Scenarios: 5/222
Definite evil: ['#6', '#8']
Definite good: ['#2', '#3']
Evil probabilities: #7=60%, #9=60%, #1=40%, #4=20%, #5=20%

#### [18:06:03] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 5 scenarios (roles: {'Puppet', 'Pooka'})

### [18:06:52] Executed #8 -> Pooka (EVIL)

#### [18:06:52] Solver Output
Scenarios: 2/26
Definite evil: ['#1', '#6', '#8', '#9']
Definite good: ['#2', '#3', '#4', '#5', '#7']

#### [18:06:52] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Puppeteer'})

### [18:07:35] Executed #1 -> Puppeteer (EVIL)

#### [18:07:36] Solver Output
Scenarios: 2/8
Definite evil: ['#1', '#6', '#8', '#9']
Definite good: ['#2', '#3', '#4', '#5', '#7']

#### [18:07:36] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 2 scenarios (roles: {'Puppet'})

### [18:08:30] Executed #9 -> Puppet (EVIL)

## [18:08:30] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, FINAL village! All 4 evils at 100% confidence, Asc53 complete 7/7!


---

# New Game — 2026-04-10 18:20:47
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Witness, Architect, Empress, Confessor, Hunter, Knight
- Outcasts: Drunk, Doppelganger
- Minions: Poisoner, Puppeteer
- Demons: Baa

### [18:21:50] Revealed #1 Knight
Info: {}

### [18:21:50] Revealed #2 Architect
Info: {'side': 'Equal'}

### [18:21:50] Revealed #3 Confessor
Info: {'dizzy': True}

### [18:21:50] Revealed #4 Empress
Info: {'targets': [2, 3, 8]}

### [18:21:50] Revealed #5 Empress
Info: {'targets': [1, 2, 9]}

### [18:21:50] Revealed #6 Hunter
Info: {'distance': 3}

### [18:21:50] Revealed #7 Confessor
Info: {'dizzy': True}

### [18:21:50] Revealed #8 Witness
Info: {'affected_position': 6}

### [18:21:50] Revealed #9 Witness
Info: {'affected_position': 6}

#### [18:22:02] Solver Output
Scenarios: 123/66618
Definite evil: ['#7']
Definite good: ['#8']
Evil probabilities: #3=90%, #5=46%, #4=39%, #2=26%, #9=23%, #1=22%, #6=9%

#### [18:22:02] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 123 scenarios (roles: {'Poisoner', 'Baa', 'Puppet', 'Puppeteer'})

### [18:22:26] Executed #7 -> Baa (EVIL)

#### [18:22:28] Solver Output
Scenarios: 22/7402
Definite evil: ['#7']
Definite good: ['#6', '#8']
Evil probabilities: #3=82%, #5=77%, #4=36%, #9=23%, #1=18%, #2=18%

#### [18:22:28] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (82% evil Puppeteer, 18% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 82%, but all reveal branches still lead to a forced win.

### [18:22:54] Executed #3 -> Puppeteer (EVIL)

#### [18:22:54] Solver Output
Scenarios: 18/912
Definite evil: ['#3', '#7']
Definite good: ['#1', '#2', '#6', '#8']
Evil probabilities: #5=72%, #4=44%, #9=28%

#### [18:22:54] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (72% evil Poisoner, 22% good Empress, 6% good Doppelganger).
WARNING: Execution lookahead override -- immediate hit chance is 72%, but all reveal branches still lead to a forced win.

### [18:23:16] Executed #5 -> Poisoner (EVIL)

#### [18:23:16] Solver Output
Scenarios: 13/175
Definite evil: ['#3', '#5', '#7']
Definite good: ['#1', '#2', '#6', '#8', '#9']
Evil probabilities: #4=62%

#### [18:23:16] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (62% evil Puppet, 31% good Empress, 8% good Doppelganger).
WARNING: Execution lookahead override -- immediate hit chance is 62%, but all reveal branches still lead to a forced win.

### [18:23:39] Executed #4 -> Puppet (EVIL)

## [18:23:44] GAME OVER — WIN
Final HP: 10

## [18:23:53] GAME OVER — WIN
Final HP: 10
Notes: Asc54 v1 perfect 10HP, 4 evils, 9-card board with Baa+Poisoner+Puppeteer


---

# New Game — 2026-04-10 18:26:09
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Alchemist, Knitter, Bard, Hunter, Confessor
- Outcasts: Bombardier, Plague_Doctor, Drunk
- Minions: Chancellor
- Demons: Baa

### [18:26:46] Revealed #2 Bombardier
Info: {}

### [18:26:46] Revealed #3 Knitter
Info: {'evil_pairs': 1}

### [18:26:46] Revealed #4 Bard
Info: {'corruption_distance': 3}

### [18:26:46] Revealed #5 Knitter
Info: {'evil_pairs': 0}

### [18:26:46] Revealed #6 Confessor
Info: {'dizzy': True}

### [18:27:16] Revealed #1 Alchemist
Info: {'cured_count': 2}

### [18:27:16] Revealed #7 Slayer
Info: {}

#### [18:27:20] Solver Output
Scenarios: 5/330
Definite evil: ['#6']
Definite good: ['#1', '#2', '#4', '#7']
Evil probabilities: #3=60%, #5=40%

#### [18:27:20] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 5 scenarios (roles: {'Chancellor', 'Baa'})

### [18:27:41] Executed #6 -> Baa (EVIL)

#### [18:27:41] Solver Output
Scenarios: 2/42
Definite evil: ['#6']
Definite good: ['#1', '#2', '#4', '#7']
Evil probabilities: #3=50%, #5=50%

#### [18:27:41] Recommendation
Action: **USE_ABILITY** #7 (Slayer) -> targets ['#3']
Reason: Target #3 is 50% evil (adjusted 0.50)

## [18:28:05] GAME OVER — WIN
Final HP: 10
Notes: Asc54 v2 perfect 10HP, Slayer kill on Chancellor

## [18:28:25] GAME OVER — WIN
Final HP: 10
Notes: Asc54 v2 perfect 10HP, Slayer kill on Chancellor


---

# New Game — 2026-04-10 18:30:01
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Hunter, Knitter, Bishop, Gemcrafter, Confessor
- Outcasts: Doppelganger
- Minions: Poisoner
- Demons: Pooka

### [18:30:19] Revealed #1 Bishop
Info: {'targets': [4, 2, 8], 'types': ['Villager', 'Outcast', 'Minion']}

### [18:30:19] Revealed #2 Confessor
Info: {'dizzy': False}

### [18:30:19] Revealed #3 Knitter
Info: {'evil_pairs': 0}

### [18:30:19] Revealed #4 Knitter
Info: {'evil_pairs': 0}

### [18:30:19] Revealed #5 Hunter
Info: {'distance': 4}

### [18:30:19] Revealed #6 Confessor
Info: {'dizzy': False}

### [18:30:19] Revealed #7 Scout
Info: {'evil_role': 'Poisoner', 'distance': 1}

### [18:30:19] Revealed #8 Gemcrafter
Info: {'good_position': 2}

#### [18:30:23] Solver Output
Scenarios: 1/560
Definite evil: ['#3', '#4']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8']

#### [18:30:23] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [18:30:46] Executed #3 -> Pooka (EVIL)

#### [18:30:46] Solver Output
Scenarios: 1/70
Definite evil: ['#3', '#4']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8']

#### [18:30:46] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [18:31:10] Executed #4 -> Poisoner (EVIL)

## [18:31:10] GAME OVER — WIN
Final HP: 10
Notes: Asc54 v3 perfect 10HP, both adjacent (Pooka+Poisoner), 1-scenario solver


---

# New Game — 2026-04-10 18:32:31
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Judge, Gemcrafter, Medium, Bard, Poet
- Outcasts: Wretch
- Minions: Minion, Poisoner
- Demons: Lilis

#### [18:33:54] Claude Reasoning


### [18:34:23] Revealed #1 Gemcrafter
Info: {'good_position': 4}

### [18:34:23] Revealed #2 Wretch
Info: {}

### [18:35:30] Revealed #3 Judge
Info: {}

### [18:35:31] Revealed #4 Poet
Info: {'targets': [5, 6], 'types': ['Minion', 'Villager'], 'copied_role': 'Bishop'}

### [18:36:31] Revealed #7 Gemcrafter
Info: {'good_position': 2}

### [18:36:31] Revealed #8 Medium
Info: {'good_position': 4, 'good_role': 'Poet'}

### [18:36:42] Revealed #6 Enlightened
Info: {'direction': 'ccw'}

### [18:36:42] Revealed #9 Enlightened
Info: {'direction': 'cw'}

#### [18:36:46] Solver Output
Scenarios: 2/694
Definite evil: ['#6', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']

#### [18:36:46] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Poisoner'})

### [18:37:12] Executed #6 -> Poisoner (EVIL)

#### [18:37:12] Solver Output
Scenarios: 2/76
Definite evil: ['#6', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']

#### [18:37:12] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Minion', 'Lilis'})

### [18:37:38] Executed #7 -> Minion (EVIL)

#### [18:37:38] Solver Output
Scenarios: 1/7
Definite evil: ['#6', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']

#### [18:37:38] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [18:38:08] Executed #9 -> Lilis (EVIL)

## [18:38:08] GAME OVER — WIN
Final HP: 6
Notes: Asc54 v4 Lilis 6HP, Bard night-killed, batched flips, 3 confident execs

## [18:39:12] GAME OVER — WIN
Final HP: 6
Notes: Asc54 v4 Lilis 6HP, Bard night-killed, batched flips, 3 confident execs


---

# New Game — 2026-04-10 18:41:16
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Poet, Lover, Dreamer, Bishop, Fortune_Teller, Jester
- Outcasts: Bombardier, Drunk
- Minions: Poisoner
- Demons: Baa

### [18:42:10] Revealed #3 Bishop
Info: {'targets': [6, 2, 3], 'types': ['Villager', 'Outcast', 'Minion']}

### [18:42:10] Revealed #4 Lover
Info: {'evil_adjacent': 1}

### [18:42:10] Revealed #5 Bombardier
Info: {}

### [18:42:10] Revealed #6 Baker
Info: {'original_role': 'original'}

### [18:42:10] Revealed #8 Baker
Info: {'original_role': 'Poet'}

### [18:43:03] Revealed #1 Jester
Info: {}

### [18:43:03] Revealed #2 Dreamer
Info: {}

### [18:43:04] Revealed #7 Fortune_Teller
Info: {}

#### [18:43:07] Solver Output
Scenarios: 86/540
Evil probabilities: #3=52%, #4=35%, #1=24%, #8=24%, #2=22%, #5=21%, #7=17%, #6=3%

#### [18:43:07] Recommendation
Action: **USE_ABILITY** #2 (Dreamer) -> targets ['#3']
Reason: Entropy 2.446 (adjusted 2.090) | timing x1.00
WARNING: Corruption risk: 29%

### [18:43:28] Revealed #2 Dreamer
Info: {'target': 3, 'evil_role': 'Baa'}

### [18:43:34] Ability used at #2

#### [18:43:34] Solver Output
Scenarios: 69/540
Evil probabilities: #4=43%, #3=41%, #8=28%, #1=23%, #2=22%, #7=20%, #5=19%, #6=4%

#### [18:43:34] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#3', '#5', '#8']
Reason: Expected posterior 32.8 scenarios (adjusted 36.3, info gain 0.925 bits) | timing x1.00
WARNING: Corruption risk: 22%

### [18:43:59] Revealed #1 Jester
Info: {'targets': [3, 5, 8], 'evil_count': 0}

### [18:43:59] Ability used at #1

#### [18:43:59] Solver Output
Scenarios: 34/540
Evil probabilities: #4=44%, #3=41%, #2=26%, #1=24%, #7=24%, #8=24%, #5=15%, #6=3%

#### [18:43:59] Recommendation
Action: **USE_ABILITY** #7 (Fortune Teller) -> targets ['#3', '#4']
Reason: Entropy 1.000 (adjusted 0.912) | timing x1.00
WARNING: Corruption risk: 18%

### [18:44:25] Revealed #7 Fortune Teller
Info: {'targets': [3, 4], 'has_evil': False}

### [18:44:25] Ability used at #7

#### [18:44:26] Solver Output
Scenarios: 17/540
Definite good: ['#6']
Evil probabilities: #4=41%, #7=41%, #3=35%, #1=24%, #5=24%, #2=18%, #8=18%

#### [18:44:26] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (41% good Jester, 24% good Drunk (corrupted), 18% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 24%, but all reveal branches still lead to a forced win.

### [18:45:04] Executed #1 -> GOOD (WRONG!)

#### [18:45:04] Solver Output
Scenarios: 13/404
Definite good: ['#1', '#6']
Evil probabilities: #4=54%, #7=54%, #3=31%, #2=23%, #8=23%, #5=15%

#### [18:45:04] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 54% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 54% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #4 (54%) despite low confidence — Bombardier candidate(s) [5] risk instant game loss if executed first.

### [18:45:32] Executed #4 -> Poisoner (EVIL)

#### [18:45:32] Solver Output
Scenarios: 5/37
Definite evil: ['#4']
Definite good: ['#1', '#3', '#5', '#6', '#8']
Evil probabilities: #7=80%, #2=20%

#### [18:45:32] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (80% evil Baa, 20% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 80%, but all reveal branches still lead to a forced win.

### [18:46:05] Executed #7 -> Baa (EVIL)

## [18:46:05] GAME OVER — WIN
Final HP: 5
Notes: Asc54 v5 5HP, wrong exec on Jester #1, Dreamer+Jester+FT abilities, two Bakers


---

# New Game — 2026-04-10 18:47:45
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Scout, Architect, Poet, Medium, Empress
- Outcasts: Doppelganger, Bombardier, Drunk
- Minions: Chancellor
- Demons: Pooka

### [18:48:06] Revealed #1 Empress
Info: {'targets': [2, 3, 5]}

### [18:48:06] Revealed #2 Bombardier
Info: {}

### [18:48:06] Revealed #3 Bombardier
Info: {}

### [18:48:06] Revealed #4 Oracle
Info: {'targets': [3, 8], 'minion_role': 'Chancellor'}

### [18:48:06] Revealed #5 Medium
Info: {'good_position': 4, 'good_role': 'Oracle'}

### [18:48:06] Revealed #6 Poet
Info: {'evil_pairs': 0, 'copied_role': 'Knitter'}

### [18:48:06] Revealed #7 Empress
Info: {'targets': [2, 8, 9]}

### [18:48:06] Revealed #8 Architect
Info: {'side': 'Right'}

### [18:48:06] Revealed #9 Oracle
Info: {'targets': [1, 7], 'minion_role': 'Chancellor'}

#### [18:48:12] Solver Output
Scenarios: 5/2464
Definite good: ['#1', '#5', '#6', '#7', '#9']
Evil probabilities: #2=80%, #4=80%, #3=20%, #8=20%

#### [18:48:12] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (80% evil Pooka, 20% good Oracle).
WARNING: Execution lookahead override -- immediate hit chance is 80%, but all reveal branches still lead to a forced win.

### [18:48:38] Executed #4 -> Pooka (EVIL)

#### [18:48:38] Solver Output
Scenarios: 4/260
Definite evil: ['#2', '#4']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8', '#9']

#### [18:48:38] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Chancellor'})

### [18:49:09] Executed #2 -> Chancellor (EVIL)

## [18:49:09] GAME OVER — WIN
Final HP: 10
Notes: Asc54 v6 perfect 10HP, two apparent Bombardiers (#2 evil disguise, #3 real)


---

# New Game — 2026-04-10 18:50:35
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Jester, Poet, Baker, Empress, Confessor, Medium
- Outcasts: Doppelganger, Bombardier
- Minions: Chancellor
- Demons: Pooka

### [18:52:05] Revealed #2 Empress
Info: {'targets': [1, 4, 6]}

### [18:52:05] Revealed #3 Confessor
Info: {'dizzy': False}

### [18:52:05] Revealed #4 Bombardier
Info: {}

### [18:52:05] Revealed #5 Medium
Info: {'good_position': 6, 'good_role': 'Baker'}

### [18:52:05] Revealed #6 Baker
Info: {'original_role': 'Confessor'}

### [18:52:05] Revealed #7 Scout
Info: {'evil_role': 'Pooka', 'distance': 2}

### [18:52:38] Revealed #1 Jester
Info: {}

### [18:52:38] Revealed #8 Jester
Info: {}

#### [18:52:43] Solver Output
Scenarios: 28/454
Definite good: ['#1', '#3']
Evil probabilities: #7=79%, #4=36%, #6=36%, #2=29%, #5=14%, #8=7%

#### [18:52:43] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#3', '#4', '#7']
Reason: Expected posterior 10.0 scenarios (adjusted 10.0, info gain 1.485 bits) | timing x1.00

### [18:53:12] Revealed #1 Jester
Info: {'targets': [3, 4, 7], 'evil_count': 1}

### [18:53:12] Ability used at #1

#### [18:53:12] Solver Output
Scenarios: 12/454
Definite evil: ['#7']
Definite good: ['#1', '#3', '#4', '#5', '#8']
Evil probabilities: #2=67%, #6=33%

#### [18:53:12] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 12 scenarios (roles: {'Pooka', 'Chancellor'})

### [18:53:41] Executed #7 -> Chancellor (EVIL)

#### [18:53:41] Solver Output
Scenarios: 4/62
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8']

#### [18:53:41] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 4 scenarios (roles: {'Pooka'})

### [18:54:12] Executed #6 -> Pooka (EVIL)

## [18:54:13] GAME OVER — WIN
Final HP: 10
Notes: Asc54 v7 perfect 10HP, ASC54 COMPLETE 7/7, 16-win streak


---

# New Game — 2026-04-11 13:52:24
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Poet, Jester, Oracle, Bard, Fortune_Teller
- Outcasts: Bombardier
- Minions: Poisoner
- Demons: Pooka

### [13:52:58] Revealed #1 Jester
Info: {}

### [13:52:58] Revealed #2 Oracle
Info: {'targets': [1, 2], 'minion_role': 'Poisoner'}

### [13:52:58] Revealed #3 Fortune_Teller
Info: {}

### [13:52:58] Revealed #4 Jester
Info: {}

### [13:52:58] Revealed #5 Bard
Info: {'corruption_distance': -1}

### [13:52:58] Revealed #6 Bombardier
Info: {}

### [13:52:58] Revealed #7 Enlightened
Info: {'direction': 'CW'}

### [13:53:02] Revealed #8 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 6}

#### [13:53:09] Solver Output
Scenarios: 1/86
Definite evil: ['#1', '#4']
Definite good: ['#2', '#3', '#5', '#6', '#7', '#8']

#### [13:53:09] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [13:54:06] Executed #1 -> Poisoner (EVIL)

#### [13:54:09] Solver Output
Scenarios: 1/12
Definite evil: ['#1', '#4']
Definite good: ['#2', '#3', '#5', '#6', '#7', '#8']

#### [13:54:09] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Pooka'})

## [13:54:43] GAME OVER — WIN
Final HP: 10
Notes: Asc55 perfect 10HP, 2 confident exec from 1-scenario solve

### [13:54:46] Executed #4 -> Pooka (EVIL)

## [13:54:48] GAME OVER — WIN
Final HP: 10
Notes: Asc55 perfect 10HP, 2 confident exec from 1-scenario solve


---

# New Game — 2026-04-11 13:56:41
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Witness, Bard, Bishop, Baker, Oracle
- Outcasts: Plague_Doctor, Bombardier
- Minions: Shaman
- Demons: Baa

### [13:57:59] Revealed #1 Bard
Info: {'corruption_distance': 3}

### [13:57:59] Revealed #2 Baker
Info: {'original_role': 'original'}

### [13:57:59] Revealed #3 Bishop
Info: {'targets': [7, 6, 1], 'types': ['Villager', 'Outcast', 'Minion']}

### [13:57:59] Revealed #4 Oracle
Info: {'targets': [1, 8], 'minion_role': 'Shaman'}

### [13:57:59] Revealed #5 Bard
Info: {'corruption_distance': 1}

### [13:57:59] Revealed #6 Judge
Info: {}

### [13:57:59] Revealed #7 Baker
Info: {'original_role': 'Witness'}

### [13:58:20] Revealed #8 Plague_Doctor
Info: {}

#### [13:58:24] Solver Output
Scenarios: 10/266
Definite good: ['#8']
Evil probabilities: #3=90%, #1=20%, #2=20%, #4=20%, #6=20%, #7=20%, #5=10%

#### [13:58:24] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#4']
Reason: Entropy 2.122 (adjusted 2.122) | timing x1.00

#### [13:59:00] Solver Output
Scenarios: 4/266
Definite good: ['#2', '#6', '#7', '#8']
Evil probabilities: #3=75%, #1=50%, #4=50%, #5=25%

#### [13:59:00] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#1']
Reason: Expected posterior 3.3 scenarios (adjusted 4.2, info gain 0.000 bits) | timing x1.00
WARNING: Corruption risk: 50% -- corrupted Judge results are unreliable

### [13:59:25] Revealed #6 Judge
Info: {'target': 1, 'is_lying': True}

### [13:59:29] Ability used at #6

#### [13:59:30] Solver Output
Scenarios: 4/266
Definite good: ['#2', '#6', '#7', '#8']
Evil probabilities: #3=75%, #1=50%, #4=50%, #5=25%

#### [13:59:30] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 25% good Bishop (corrupted), 25% evil Shaman).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [13:59:36] Executed #3 -> Baa (EVIL)

#### [13:59:42] Solver Output
Scenarios: 2/37
Definite evil: ['#3']
Definite good: ['#2', '#5', '#6', '#7', '#8']
Evil probabilities: #1=50%, #4=50%

#### [13:59:42] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% good Bard, 50% evil Shaman).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [13:59:49] Executed #1 -> GOOD (WRONG!)

#### [13:59:56] Solver Output
Scenarios: 1/31
Definite evil: ['#3', '#4']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8']

#### [13:59:56] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [14:00:03] Executed #4 -> Shaman (EVIL)

## [14:00:21] GAME OVER — WIN
Final HP: 5
Notes: Asc55 Shaman+Baa, 5HP, 1 wrong exec on 50/50 lookahead-forced #1, PD probe + Judge corrupted, original Baker #2, Witness-swap Baker #7


---

# New Game — 2026-04-11 14:01:55
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Bishop, Dreamer, Fortune_Teller, Alchemist, Oracle, Confessor, Judge, Knight
- Outcasts: Drunk, Bombardier
- Minions: Poisoner, Shaman
- Demons: Baa

### [14:04:41] Revealed #1 Oracle
Info: {'targets': [1, 2], 'minion_role': 'Poisoner'}

### [14:04:41] Revealed #2 Empress
Info: {'targets': [3, 7, 8]}

### [14:04:41] Revealed #3 Alchemist
Info: {'cured_count': 2}

### [14:04:41] Revealed #4 Dreamer
Info: {}

### [14:04:41] Revealed #5 Confessor
Info: {'dizzy': False}

### [14:04:41] Revealed #6 Confessor
Info: {'dizzy': False}

### [14:04:41] Revealed #7 Fortune_Teller
Info: {}

### [14:04:41] Revealed #8 Bishop
Info: {'targets': [3, 6, 1], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:04:41] Revealed #9 Knight
Info: {}

#### [14:04:49] Solver Output
Scenarios: 22/4564
Definite good: ['#1', '#5', '#6']
Evil probabilities: #9=64%, #4=55%, #8=55%, #3=45%, #7=45%, #2=36%

#### [14:04:49] Recommendation
Action: **EXECUTE** #9
Reason: Knight check: #9 is 64% evil, 9% corruption risk. Expected HP cost: 0.3 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 9% -- corrupted Knight loses immunity + 4 extra damage

### [14:05:32] Executed #9 -> Shaman (EVIL)

#### [14:05:32] Solver Output
Scenarios: 3/508
Definite evil: ['#9']
Definite good: ['#1', '#5', '#6']
Evil probabilities: #4=67%, #2=33%, #3=33%, #7=33%, #8=33%

#### [14:05:32] Recommendation
Action: **USE_ABILITY** #4 (Dreamer) -> targets ['#2']
Reason: Entropy 1.585 (adjusted 1.585) | timing x1.00

### [14:05:59] Revealed #4 Dreamer
Info: {'target': 2, 'evil_role': 'Shaman'}

### [14:06:03] Ability used at #4

#### [14:06:03] Solver Output
Scenarios: 3/508
Definite evil: ['#9']
Definite good: ['#1', '#5', '#6']
Evil probabilities: #4=67%, #2=33%, #3=33%, #7=33%, #8=33%

#### [14:06:03] Recommendation
Action: **USE_ABILITY** #7 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.918 (adjusted 0.918) | timing x1.00

### [14:06:24] Revealed #7 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': True}

### [14:06:28] Ability used at #7

#### [14:06:28] Solver Output
Scenarios: 2/508
Definite evil: ['#9']
Definite good: ['#1', '#5', '#6', '#8']
Evil probabilities: #2=50%, #3=50%, #4=50%, #7=50%

#### [14:06:28] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (50% good Empress (corrupted), 50% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [14:06:35] Executed #2 -> Poisoner (EVIL)

#### [14:06:40] Solver Output
Scenarios: 1/72
Definite evil: ['#2', '#4', '#9']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8']

#### [14:06:40] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Baa'})

### [14:06:47] Executed #4 -> Baa (EVIL)

## [14:07:26] GAME OVER — WIN
Final HP: 10
Notes: Asc55 9-card 3-evil, 10HP perfect, Knight check on Shaman, Drunk-as-Bishop, Dreamer was Baa lying about Shaman, FT confirmed evil in 1-2


---

# New Game — 2026-04-11 14:08:41
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Confessor, Hunter, Gemcrafter, Oracle, Alchemist
- Outcasts: Bombardier
- Minions: Twin_Minion
- Demons: Pooka

### [14:09:27] Revealed #1 Alchemist
Info: {'cured_count': 0}

### [14:09:27] Revealed #2 Jester
Info: {}

### [14:09:28] Revealed #3 Bombardier
Info: {}

### [14:09:28] Revealed #4 Jester
Info: {}

### [14:09:28] Revealed #5 Gemcrafter
Info: {'good_position': 4}

### [14:09:28] Revealed #6 Hunter
Info: {'distance': 2}

### [14:09:28] Revealed #7 Confessor
Info: {'dizzy': False}

### [14:09:28] Revealed #8 Oracle
Info: {'targets': [2, 8], 'minion_role': 'Twin_Minion'}

#### [14:09:33] Solver Output
Scenarios: 1/56
Definite evil: ['#2', '#4']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8']

#### [14:09:33] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [14:09:58] Executed #2 -> Twin_Minion (EVIL)

#### [14:09:58] Solver Output
Scenarios: 1/7
Definite evil: ['#2', '#4']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8']

#### [14:09:58] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Pooka'})

## [14:10:26] GAME OVER — WIN
Final HP: 10
Notes: Asc55 8-card 10HP perfect, manual entry due to memory reader stale-clue bug, 1-scenario solve

### [14:10:29] Executed #4 -> Pooka (EVIL)

## [14:10:29] GAME OVER — WIN
Final HP: 10
Notes: Asc55 8-card 10HP perfect, manual entry due to memory reader stale-clue bug, 1-scenario solve


---

# New Game — 2026-04-11 14:11:50
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Knight, Bard, Knitter, Medium, Bishop
- Outcasts: Bombardier, Plague_Doctor
- Minions: Twin_Minion
- Demons: Pooka

### [14:13:33] Revealed #1 Knitter
Info: {'evil_pairs': 0}

### [14:13:33] Revealed #2 Knight
Info: {}

### [14:13:33] Revealed #3 Bard
Info: {'corruption_distance': -1}

### [14:13:33] Revealed #4 Bombardier
Info: {}

### [14:13:33] Revealed #5 Baker
Info: {'original_role': 'original'}

### [14:13:33] Revealed #7 Baker
Info: {'original_role': 'Bishop'}

### [14:13:33] Revealed #8 Baker
Info: {'original_role': 'Bishop'}

### [14:13:33] Revealed #9 Bombardier
Info: {}

### [14:13:49] Revealed #6 Plague_Doctor
Info: {}

#### [14:13:54] Solver Output
Scenarios: 7/268
Definite good: ['#2', '#5', '#6']
Evil probabilities: #9=71%, #3=29%, #4=29%, #7=29%, #8=29%, #1=14%

#### [14:13:54] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#3']
Reason: Entropy 2.236 (adjusted 2.236) | timing x1.00

#### [14:14:24] Solver Output
Scenarios: 2/268
Definite evil: ['#3', '#9']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7', '#8']

#### [14:14:24] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [14:14:31] Executed #3 -> Pooka (EVIL)

#### [14:14:35] Solver Output
Scenarios: 2/31
Definite evil: ['#3', '#9']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7', '#8']

#### [14:14:35] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 2 scenarios (roles: {'Twin_Minion'})

### [14:14:42] Executed #9 -> Twin Minion (EVIL)

## [14:15:04] GAME OVER — WIN
Final HP: 10
Notes: Asc55 9-card 10HP perfect, Baker chain (3 Bakers from 1), PD probe clean on Pooka, both evils confident exec


---

# New Game — 2026-04-11 14:16:26
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Dreamer, Knitter, Judge, Poet, Alchemist, Lover
- Outcasts: Plague_Doctor, Bombardier
- Minions: Minion, Poisoner
- Demons: Lilis

### [14:19:05] Revealed #1 Poet
Info: {'good_position': 7, 'good_role': 'Judge', 'copied_role': 'Medium'}

### [14:19:05] Revealed #2 Knitter
Info: {'evil_pairs': 2}

### [14:19:05] Revealed #3 Lover
Info: {'evil_adjacent': 2}

### [14:19:05] Revealed #5 Alchemist
Info: {'cured_count': 2}

### [14:19:05] Revealed #6 Scout
Info: {'evil_role': 'Minion', 'distance': 2}

### [14:19:05] Revealed #9 Bombardier
Info: {}

### [14:19:12] Revealed #4 Plague_Doctor
Info: {}

### [14:19:13] Revealed #8 Dreamer
Info: {}

#### [14:19:18] Solver Output
Scenarios: 92/2076
Definite good: ['#1', '#7']
Evil probabilities: #5=77%, #3=63%, #2=59%, #6=45%, #9=30%, #8=16%, #4=10%

#### [14:19:18] Recommendation
Action: **USE_ABILITY** #8 (Dreamer) -> targets ['#5']
Reason: Entropy 2.882 (adjusted 2.663) | timing x1.00
WARNING: Corruption risk: 15%

### [14:19:39] Revealed #8 Dreamer
Info: {'target': 5, 'evil_role': 'Minion'}

### [14:19:40] Ability used at #8

#### [14:19:40] Solver Output
Scenarios: 52/2076
Definite good: ['#1', '#7']
Evil probabilities: #3=67%, #2=62%, #5=60%, #6=54%, #9=31%, #8=19%, #4=8%

#### [14:19:40] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#2']
Reason: Entropy 2.072 (adjusted 2.072) | timing x1.00

#### [14:20:55] Solver Output
Scenarios: 1/2076
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#7', '#8', '#9']

#### [14:20:55] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [14:21:02] Executed #4 -> Poisoner (EVIL)

#### [14:21:08] Solver Output
Scenarios: 1/86
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#7', '#8', '#9']

#### [14:21:08] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Minion'})

### [14:21:15] Executed #5 -> Minion (EVIL)

#### [14:21:19] Solver Output
Scenarios: 1/7
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#7', '#8', '#9']

#### [14:21:19] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [14:21:26] Executed #6 -> Lilis (EVIL)

## [14:21:36] GAME OVER — WIN
Final HP: 6
Notes: Asc55 Lilis 9-card 6HP, night-killed Judge #7, evil PD #4 self-corruption claim, Dreamer ID'd Minion, all 3 evils sequential confident exec

## [14:21:42] GAME OVER — WIN
Final HP: 6
Notes: Asc55 Lilis 9-card 6HP, night-killed Judge #7, evil PD #4 self-corruption claim, Dreamer ID'd Minion, all 3 evils sequential confident exec


---

# New Game — 2026-04-11 14:23:01
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Gemcrafter, Hunter, Alchemist, Knitter, Jester, Scout
- Outcasts: Bombardier, Wretch
- Minions: Shaman
- Demons: Pooka

### [14:24:21] Revealed #2 Bombardier
Info: {}

### [14:24:21] Revealed #3 Hunter
Info: {'distance': 4}

### [14:24:21] Revealed #4 Wretch
Info: {}

### [14:24:21] Revealed #5 Alchemist
Info: {'cured_count': 1}

### [14:24:21] Revealed #6 Scout
Info: {'evil_role': 'Shaman', 'distance': 3}

### [14:24:21] Revealed #7 Knitter
Info: {'evil_pairs': 1}

### [14:24:21] Revealed #8 Knight
Info: {}

### [14:24:21] Revealed #9 Alchemist
Info: {'cured_count': 1}

### [14:24:32] Revealed #1 Jester
Info: {}

#### [14:24:33] Solver Output
Scenarios: 1/72
Definite evil: ['#3', '#6']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#8', '#9']

#### [14:24:33] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [14:24:40] Executed #3 -> Shaman (EVIL)

#### [14:24:46] Solver Output
Scenarios: 1/8
Definite evil: ['#3', '#6']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#8', '#9']

#### [14:24:46] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [14:24:53] Executed #6 -> Pooka (EVIL)

## [14:25:13] GAME OVER — WIN
Final HP: 10
Notes: Asc55 final 7/7 perfect 10HP, ASC55 COMPLETE, 2 Alchemists on board (pool dup), Wretch survived


---

# New Game — 2026-04-11 14:28:00
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Dreamer, Bard, Medium, Slayer, Confessor
- Outcasts: Doppelganger, Plague_Doctor
- Minions: Witch
- Demons: Baa

### [14:28:52] Revealed #1 Medium
Info: {'good_position': 3, 'good_role': 'Slayer'}

### [14:28:52] Revealed #2 Bard
Info: {'corruption_distance': 2}

### [14:28:52] Revealed #4 Confessor
Info: {'dizzy': False}

### [14:28:52] Revealed #6 Knitter
Info: {'evil_pairs': 0}

### [14:28:52] Revealed #7 Confessor
Info: {'dizzy': False}

### [14:28:58] Revealed #3 Slayer
Info: {}

### [14:28:58] Revealed #5 Slayer
Info: {}

#### [14:28:59] Solver Output
Scenarios: 46/1862
Definite good: ['#4', '#7', '#8']
Evil probabilities: #5=74%, #2=48%, #1=30%, #3=26%, #6=22%

#### [14:28:59] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#5']
Reason: Target #5 is 74% evil (adjusted 0.58)
WARNING: Corruption risk: 22% -- Slayer ability disabled if corrupted

### [14:29:33] Ability used at #3

### [14:30:02] Revealed #8 Dreamer
Info: {}

#### [14:30:02] Solver Output
Scenarios: 5/49
Definite evil: ['#2', '#5']
Definite good: ['#1', '#3', '#4', '#6', '#7', '#8']

#### [14:30:02] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 5 scenarios (roles: {'Baa'})

### [14:30:09] Executed #2 -> Baa (EVIL)

## [14:30:29] GAME OVER — WIN
Final HP: 10
Notes: Asc56 v1 8-card 10HP perfect, Slayer #3 killed Witch #5, Baa #2 confident exec, Witch-blocked #8 was Dreamer


---

# New Game — 2026-04-11 14:31:42
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Gemcrafter, Enlightened, Scout, Dreamer, Alchemist, Empress, Poet
- Outcasts: Plague_Doctor, Drunk
- Minions: Shaman
- Demons: Pooka

### [14:33:08] Revealed #1 Alchemist
Info: {'cured_count': 0}

### [14:33:08] Revealed #2 Empress
Info: {'targets': [3, 4, 6]}

### [14:33:08] Revealed #7 Scout
Info: {'evil_role': 'Shaman', 'distance': 2}

### [14:33:08] Revealed #8 Gemcrafter
Info: {'good_position': 9}

### [14:33:08] Revealed #9 Gemcrafter
Info: {'good_position': 8}

### [14:33:22] Revealed #3 Poet
Info: {'evil_adjacent': 0, 'copied_role': 'Lover'}

### [14:33:22] Revealed #4 Poet
Info: {'targets': [3, 5, 7], 'types': ['Minion', 'Outcast', 'Villager'], 'copied_role': 'Bishop'}

### [14:33:22] Revealed #5 Dreamer
Info: {}

### [14:33:22] Revealed #6 Plague_Doctor
Info: {}

#### [14:33:27] Solver Output
Scenarios: 10/2128
Definite good: ['#1']
Evil probabilities: #5=50%, #6=50%, #4=30%, #3=20%, #7=20%, #2=10%, #8=10%, #9=10%

#### [14:33:27] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#7']
Reason: Entropy 2.722 (adjusted 2.722) | timing x1.00

#### [14:33:55] Solver Output
Scenarios: 3/2128
Definite good: ['#1', '#2', '#3', '#4', '#5']
Evil probabilities: #6=67%, #7=67%, #8=33%, #9=33%

#### [14:33:55] Recommendation
Action: **USE_ABILITY** #5 (Dreamer) -> targets ['#1']
Reason: Entropy 0.918 (adjusted 0.612) | timing x1.00
WARNING: Corruption risk: 67%

### [14:34:26] Revealed #5 Dreamer
Info: {'target': 1, 'evil_role': 'Shaman'}

### [14:34:26] Ability used at #5

#### [14:34:27] Solver Output
Scenarios: 3/2128
Definite good: ['#1', '#2', '#3', '#4', '#5']
Evil probabilities: #6=67%, #7=67%, #8=33%, #9=33%

#### [14:34:27] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (67% evil Pooka, 33% good Plague_Doctor).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [14:34:33] Executed #6 -> GOOD (WRONG!)

#### [14:34:41] Solver Output
Scenarios: 1/2016
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#7']

#### [14:34:41] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [14:34:48] Executed #8 -> Pooka (EVIL)

#### [14:34:52] Solver Output
Scenarios: 1/252
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#7']

#### [14:34:52] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [14:34:59] Executed #9 -> Shaman (EVIL)

## [14:35:07] GAME OVER — WIN
Final HP: 5
Notes: Asc56 v2 9-card 5HP, 1 wrong exec on PD #6 (lookahead-forced 67/33), then 2 confident exec, 2 Poets + 2 Gemcrafters from pool dups


---

# New Game — 2026-04-11 14:36:22
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Empress, Fortune_Teller, Alchemist, Bishop, Jester, Poet
- Outcasts: Doppelganger
- Minions: Poisoner
- Demons: Pooka

### [14:37:11] Revealed #1 Alchemist
Info: {'cured_count': 2}

### [14:37:11] Revealed #2 Medium
Info: {'good_position': 4, 'good_role': 'Fortune Teller'}

### [14:37:11] Revealed #5 Bishop
Info: {'targets': [6, 3, 7], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:37:11] Revealed #6 Empress
Info: {'targets': [4, 5, 8]}

### [14:37:11] Revealed #8 Empress
Info: {'targets': [3, 4, 6]}

### [14:37:57] Revealed #3 Jester
Info: {}

### [14:37:57] Revealed #4 Fortune_Teller
Info: {}

### [14:37:57] Revealed #7 Poet
Info: {'good_position': 1, 'copied_role': 'Gemcrafter'}

#### [14:38:01] Solver Output
Scenarios: 17/526
Definite good: ['#3', '#7']
Evil probabilities: #2=59%, #4=47%, #6=29%, #5=24%, #8=24%, #1=18%

#### [14:38:01] Recommendation
Action: **USE_ABILITY** #4 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.998 (adjusted 0.909) | timing x1.00
WARNING: Corruption risk: 18%

### [14:38:30] Revealed #4 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [14:38:30] Ability used at #4

#### [14:38:30] Solver Output
Scenarios: 9/526
Definite good: ['#1', '#3', '#6', '#7']
Evil probabilities: #2=89%, #4=89%, #5=11%, #8=11%

#### [14:38:30] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#2', '#5']
Reason: Expected posterior 5.6 scenarios (adjusted 7.4, info gain 0.277 bits) | timing x1.00
WARNING: Corruption risk: 67%

### [14:38:56] Revealed #3 Jester
Info: {'targets': [1, 2, 5], 'evil_count': 0}

### [14:38:56] Ability used at #3

#### [14:38:56] Solver Output
Scenarios: 6/526
Definite evil: ['#2', '#4']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8']

#### [14:38:56] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 6 scenarios (roles: {'Poisoner', 'Pooka'})

### [14:39:03] Executed #2 -> Poisoner (EVIL)

#### [14:39:09] Solver Output
Scenarios: 3/66
Definite evil: ['#2', '#4']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8']

#### [14:39:09] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 3 scenarios (roles: {'Pooka'})

### [14:39:15] Executed #4 -> Pooka (EVIL)

## [14:39:23] GAME OVER — WIN
Final HP: 10
Notes: Asc56 v3 8-card 10HP perfect, Doppelganger copied Empress, Poet copied Gemcrafter, FT lying (was Pooka), Jester corrupted lying


---

# New Game — 2026-04-11 14:40:40
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Gemcrafter, Confessor, Jester, Architect, Lover, Oracle
- Outcasts: Wretch, Bombardier
- Minions: Shaman, Witch
- Demons: Baa

### [14:41:23] Revealed #1 Lover
Info: {'evil_adjacent': 2}

### [14:41:23] Revealed #2 Confessor
Info: {'dizzy': True}

### [14:41:23] Revealed #3 Bombardier
Info: {}

### [14:41:23] Revealed #5 Bishop
Info: {'targets': [7, 2, 3], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:41:23] Revealed #6 Gemcrafter
Info: {'good_position': 2}

### [14:41:23] Revealed #7 Architect
Info: {'side': 'Equal'}

### [14:41:23] Revealed #8 Bishop
Info: {'targets': [2, 4, 3], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:41:32] Revealed #4 Jester
Info: {}

#### [14:41:33] Solver Output
Scenarios: 4/504
Definite evil: ['#2', '#6', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [14:41:33] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Witch', 'Shaman'})

### [14:41:40] Executed #2 -> Witch (EVIL)

### [14:42:12] Revealed #9 Oracle
Info: {'targets': [3, 4], 'minion_role': 'Witch'}

#### [14:42:13] Solver Output
Scenarios: 2/56
Definite evil: ['#2', '#6', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [14:42:13] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Baa', 'Shaman'})

### [14:42:19] Executed #6 -> Baa (EVIL)

#### [14:42:26] Solver Output
Scenarios: 1/7
Definite evil: ['#2', '#6', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [14:42:26] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [14:42:33] Executed #9 -> Shaman (EVIL)

## [14:42:41] GAME OVER — WIN
Final HP: 10
Notes: Asc56 v4 9-card 10HP perfect, Witch unblocked #9, Oracle (Shaman) lied about dead Witch, Bishop dup, all 3 confident exec


---

# New Game — 2026-04-11 14:43:52
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Scout, Enlightened, Lover, Bard, Hunter, Empress
- Outcasts: Wretch, Bombardier
- Minions: Twin_Minion, Minion
- Demons: Baa

### [14:44:27] Revealed #1 Scout
Info: {'evil_role': 'Twin Minion', 'distance': 3}

### [14:44:27] Revealed #2 Scout
Info: {'evil_role': 'Twin Minion', 'distance': 1}

### [14:44:27] Revealed #3 Lover
Info: {'evil_adjacent': 1}

### [14:44:27] Revealed #4 Wretch
Info: {}

### [14:44:27] Revealed #5 Hunter
Info: {'distance': 1}

### [14:44:27] Revealed #7 Bard
Info: {'corruption_distance': 1}

### [14:44:27] Revealed #8 Empress
Info: {'targets': [2, 5, 6]}

### [14:44:27] Revealed #9 Poet
Info: {'distance': 3, 'copied_role': 'Hunter'}

### [14:44:45] Revealed #9 Poet
Info: {'evil_role': 'Minion', 'distance': 3, 'copied_role': 'Scout'}

### [14:44:46] Revealed #6 Enlightened
Info: {'direction': 'ccw'}

#### [14:44:46] Solver Output
Scenarios: 2/504
Definite evil: ['#1', '#6', '#7']
Definite good: ['#2', '#3', '#4', '#5', '#8', '#9']

#### [14:44:46] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Minion'})

### [14:44:53] Executed #1 -> Minion (EVIL)

#### [14:44:58] Solver Output
Scenarios: 2/56
Definite evil: ['#1', '#6', '#7']
Definite good: ['#2', '#3', '#4', '#5', '#8', '#9']

#### [14:44:58] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Twin_Minion', 'Baa'})

### [14:45:05] Executed #6 -> Twin Minion (EVIL)

#### [14:45:10] Solver Output
Scenarios: 1/7
Definite evil: ['#1', '#6', '#7']
Definite good: ['#2', '#3', '#4', '#5', '#8', '#9']

#### [14:45:10] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Baa'})

### [14:45:17] Executed #7 -> Baa (EVIL)

## [14:45:24] GAME OVER — WIN
Final HP: 10
Notes: Asc56 v5 9-card 10HP perfect, 3-evil all sequential confident exec, Poet copied Scout


---

# New Game — 2026-04-11 14:46:40
Cards: 8, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Medium, Jester, Enlightened, Architect, Judge
- Outcasts: Bombardier
- Minions: Puppeteer
- Demons: Pooka

### [14:47:29] Revealed #1 Architect
Info: {'side': 'Left'}

### [14:47:29] Revealed #3 Medium
Info: {'good_position': 2, 'good_role': 'Jester'}

### [14:47:29] Revealed #4 Enlightened
Info: {'direction': 'CCW'}

### [14:47:29] Revealed #5 Bombardier
Info: {}

### [14:47:29] Revealed #6 Baker
Info: {'original_role': 'original'}

### [14:47:35] Revealed #2 Jester
Info: {}

### [14:47:35] Revealed #7 Judge
Info: {}

### [14:47:35] Revealed #8 Jester
Info: {}

#### [14:47:36] Solver Output
Scenarios: 6/84
Definite evil: ['#2']
Definite good: ['#5', '#6']
Evil probabilities: #8=67%, #3=50%, #1=33%, #7=33%, #4=17%

#### [14:47:36] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 6 scenarios (roles: {'Pooka', 'Puppet'})

### [14:48:03] Executed #2 -> Pooka (EVIL)

#### [14:48:03] Solver Output
Scenarios: 4/10
Definite evil: ['#2']
Definite good: ['#5', '#6']
Evil probabilities: #8=75%, #7=50%, #1=25%, #3=25%, #4=25%

#### [14:48:03] Recommendation
Action: **USE_ABILITY** #8 (Jester) -> targets ['#1', '#3', '#4']
Reason: Expected posterior 1.7 scenarios (adjusted 1.7, info gain 1.263 bits) | timing x1.00

### [14:48:33] Revealed #8 Jester
Info: {'targets': [1, 3, 4], 'evil_count': 2}

### [14:48:33] Ability used at #8

#### [14:48:33] Solver Output
Scenarios: 2/10
Definite evil: ['#2']
Definite good: ['#1', '#5', '#6']
Evil probabilities: #3=50%, #4=50%, #7=50%, #8=50%

#### [14:48:33] Recommendation
Action: **USE_ABILITY** #7 (Judge) -> targets ['#8']
Reason: Expected posterior 1.0 scenarios (adjusted 1.0, info gain 1.000 bits) | timing x1.00

### [14:48:58] Revealed #7 Judge
Info: {'target': 8, 'is_lying': True}

### [14:48:59] Ability used at #7

#### [14:48:59] Solver Output
Scenarios: 1/10
Definite evil: ['#2', '#7', '#8']
Definite good: ['#1', '#3', '#4', '#5', '#6']

#### [14:48:59] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [14:49:06] Executed #7 -> Puppet (EVIL)

#### [14:49:11] Solver Output
Scenarios: 1/2
Definite evil: ['#2', '#7', '#8']
Definite good: ['#1', '#3', '#4', '#5', '#6']

#### [14:49:11] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Puppeteer'})

### [14:49:18] Executed #8 -> Puppeteer (EVIL)

## [14:49:26] GAME OVER — WIN
Final HP: 10
Notes: Asc56 v6 8-card 10HP perfect, Puppeteer game (3 evils from 2-pool minion+demon), Pooka first then Jester+Judge ability sequence


---

# New Game — 2026-04-11 14:50:48
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Confessor, Enlightened, Slayer, Medium, Poet, Judge
- Outcasts: Wretch
- Minions: Shaman, Poisoner
- Demons: Lilis

### [14:52:03] Revealed #2 Wretch
Info: {}

### [14:52:03] Revealed #3 Poet
Info: {'evil_pairs': 1, 'copied_role': 'Knitter'}

### [14:52:03] Revealed #6 Medium
Info: {'good_position': 8, 'good_role': 'Jester'}

### [14:52:13] Revealed #1 Enlightened
Info: {'direction': 'ccw'}

### [14:52:13] Revealed #4 Judge
Info: {}

### [14:52:13] Revealed #5 Slayer
Info: {}

### [14:52:13] Revealed #7 Jester
Info: {}

### [14:52:14] Revealed #8 Jester
Info: {}

#### [14:52:19] Solver Output
Scenarios: 66/704
Definite good: ['#9']
Evil probabilities: #1=94%, #3=39%, #2=33%, #4=33%, #5=33%, #7=33%, #8=18%, #6=15%

#### [14:52:19] Recommendation
Action: **USE_ABILITY** #8 (Jester) -> targets ['#1', '#5', '#7']
Reason: Expected posterior 27.4 scenarios (adjusted 29.1, info gain 1.182 bits) | timing x1.00
WARNING: Corruption risk: 12%

### [14:52:49] Revealed #8 Jester
Info: {'targets': [1, 5, 7], 'evil_count': 2}

### [14:52:49] Ability used at #8

#### [14:52:49] Solver Output
Scenarios: 34/704
Definite evil: ['#1']
Definite good: ['#9']
Evil probabilities: #5=41%, #7=35%, #3=29%, #6=29%, #8=29%, #2=18%, #4=18%

#### [14:52:49] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 34 scenarios (roles: {'Shaman', 'Poisoner', 'Lilis'})

### [14:52:56] Executed #1 -> Lilis (EVIL)

#### [14:53:02] Solver Output
Scenarios: 11/80
Definite evil: ['#1']
Definite good: ['#9']
Evil probabilities: #5=45%, #6=36%, #8=36%, #3=27%, #7=27%, #4=18%, #2=9%

#### [14:53:02] Recommendation
Action: **USE_ABILITY** #7 (Jester) -> targets ['#2', '#3', '#4']
Reason: Expected posterior 5.3 scenarios (adjusted 5.8, info gain 0.932 bits) | timing x1.00
WARNING: Corruption risk: 18%

### [14:53:34] Revealed #7 Jester
Info: {'targets': [2, 3, 4], 'evil_count': 1}

### [14:53:34] Ability used at #7

#### [14:53:34] Solver Output
Scenarios: 5/80
Definite evil: ['#1']
Definite good: ['#9']
Evil probabilities: #6=40%, #7=40%, #8=40%, #2=20%, #3=20%, #4=20%, #5=20%

#### [14:53:34] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#6']
Reason: Target #6 is 40% evil (adjusted 0.24)
WARNING: Corruption risk: 40% -- Slayer ability disabled if corrupted

### [14:54:05] Ability used at #5

#### [14:54:05] Solver Output
Scenarios: 1/9
Definite evil: ['#1', '#6', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#9']

#### [14:54:05] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [14:54:12] Executed #8 -> Poisoner (EVIL)

## [14:54:20] GAME OVER — WIN
Final HP: 6
Notes: Asc56 v7 final 6HP, ASC56 COMPLETE 7/7, Lilis night-killed Jester #9, Slayer killed Shaman #6, Wretch counted as evil in Jester ability


---

# New Game — 2026-04-11 14:56:15
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Architect, Bishop, Alchemist, Oracle, Poet
- Outcasts: Bombardier, Wretch
- Minions: Twin_Minion
- Demons: Lilis

### [14:58:20] Revealed #1 Alchemist
Info: {'cured_count': 0}

### [14:58:20] Revealed #2 Oracle
Info: {'targets': [1, 7], 'minion_role': 'Twin_Minion'}

### [14:58:20] Revealed #3 Bombardier
Info: {}

### [14:58:20] Revealed #4 Wretch
Info: {}

### [14:58:20] Revealed #5 Architect
Info: {'side': 'Right'}

### [14:58:20] Revealed #6 Bishop
Info: {'targets': [4, 3, 8], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:58:20] Revealed #8 Baker
Info: {'original_role': 'original'}

### [14:58:20] Revealed #9 Alchemist
Info: {'cured_count': 1}

#### [14:58:27] Solver Output
Scenarios: 2/72
Definite evil: ['#2', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7', '#8']

#### [14:58:27] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Lilis', 'Twin_Minion'})

### [14:58:34] Executed #2 -> Lilis (EVIL)

#### [14:58:40] Solver Output
Scenarios: 1/8
Definite evil: ['#2', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7', '#8']

#### [14:58:40] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [14:58:47] Executed #9 -> Twin Minion (EVIL)

## [14:58:57] GAME OVER — WIN
Final HP: 6
Notes: Asc57 v1 Lilis 9-card 6HP, night-killed Poet #7 + 4 HP from 2 night phases, both confident exec, #1 needed re-flip


---

# New Game — 2026-04-11 15:00:12
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Fortune_Teller, Dreamer, Medium, Knight, Bard
- Outcasts: Drunk, Doppelganger
- Minions: Witch
- Demons: Baa

### [15:00:51] Revealed #1 Fortune_Teller
Info: {}

### [15:00:51] Revealed #2 Dreamer
Info: {}

### [15:00:51] Revealed #3 Medium
Info: {'good_position': 2, 'good_role': 'Dreamer'}

### [15:00:51] Revealed #4 Knight
Info: {}

### [15:00:51] Revealed #5 Dreamer
Info: {}

### [15:00:51] Revealed #6 Fortune_Teller
Info: {}

#### [15:00:57] Solver Output
Scenarios: 532/1302
Evil probabilities: #5=32%, #1=32%, #6=32%, #3=29%, #4=28%, #7=28%, #2=21%

#### [15:00:57] Recommendation
Action: **EXECUTE** #4
Reason: Knight check: #4 is 28% evil, 12% corruption risk. Expected HP cost: 0.8 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 12% -- corrupted Knight loses immunity + 4 extra damage

#### [15:01:43] Solver Output
Scenarios: 384/930
Definite good: ['#4']
Evil probabilities: #3=35%, #5=35%, #1=35%, #6=35%, #7=33%, #2=26%

#### [15:01:43] Recommendation
Action: **USE_ABILITY** #5 (Dreamer) -> targets ['#1']
Reason: Entropy 2.272 (adjusted 2.154) | timing x1.00
WARNING: Corruption risk: 10%

### [15:02:13] Revealed #5 Dreamer
Info: {'target': 1, 'evil_role': 'Witch'}

### [15:02:14] Ability used at #5

#### [15:02:14] Solver Output
Scenarios: 317/930
Definite good: ['#4']
Evil probabilities: #3=40%, #5=37%, #6=37%, #7=35%, #2=30%, #1=21%

#### [15:02:14] Recommendation
Action: **USE_ABILITY** #2 (Dreamer) -> targets ['#6']
Reason: Entropy 2.189 (adjusted 2.075) | timing x1.00
WARNING: Corruption risk: 10%

### [15:02:40] Revealed #2 Dreamer
Info: {'target': 6, 'evil_role': 'Baa'}

### [15:02:40] Ability used at #2

#### [15:02:41] Solver Output
Scenarios: 263/930
Definite good: ['#4']
Evil probabilities: #3=44%, #5=38%, #7=36%, #2=34%, #1=24%, #6=24%

#### [15:02:41] Recommendation
Action: **USE_ABILITY** #1 (Fortune Teller) -> targets ['#4', '#5']
Reason: Entropy 1.000 (adjusted 0.941) | timing x1.00
WARNING: Corruption risk: 12%

### [15:03:09] Revealed #1 Fortune Teller
Info: {'targets': [4, 5], 'has_evil': False}

### [15:03:09] Ability used at #1

#### [15:03:10] Solver Output
Scenarios: 139/930
Definite good: ['#4']
Evil probabilities: #3=58%, #2=48%, #7=35%, #5=23%, #6=22%, #1=13%

#### [15:03:10] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#4', '#7']
Reason: Entropy 0.998 (adjusted 0.941) | timing x1.00
WARNING: Corruption risk: 12%

### [15:03:43] Revealed #6 Fortune Teller
Info: {'targets': [4, 7], 'has_evil': False}

### [15:03:44] Ability used at #6

#### [15:03:44] Solver Output
Scenarios: 71/930
Definite good: ['#4']
Evil probabilities: #3=62%, #2=56%, #5=23%, #7=21%, #1=20%, #6=18%

#### [15:03:44] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 62% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 62% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card

### [15:04:19] Executed #3 -> Baa (EVIL)

#### [15:04:19] Solver Output
Scenarios: 22/155
Definite evil: ['#3']
Definite good: ['#1', '#4', '#6']
Evil probabilities: #2=91%, #5=5%, #7=5%

#### [15:04:19] Recommendation
Action: **EXECUTE** #2
Reason: No reveals available. #2 is 91% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 91% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card

## [15:04:56] GAME OVER — WIN
Final HP: 10
Notes: Asc57 v2 7-card 10HP perfect, Knight check on #4 worked (free check), 2 dual-FT probes confirmed 4/5/7 good, then 62% Baa exec + 91% Witch exec

### [15:05:04] Executed #2 -> Witch (EVIL)

## [15:05:04] GAME OVER — WIN
Final HP: 10
Notes: Asc57 v2 7-card 10HP perfect, Knight check on #4 worked (free check), 2 dual-FT probes confirmed 4/5/7 good, then 62% Baa exec + 91% Witch exec


---

# New Game — 2026-04-11 15:06:25
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Witness, Bishop, Architect, Hunter, Druid, Knitter, Baker
- Outcasts: Plague_Doctor, Bombardier
- Minions: Minion, Witch
- Demons: Baa

### [15:07:05] Revealed #2 Bishop
Info: {'targets': [6, 7, 8], 'types': ['Villager', 'Outcast', 'Minion']}

### [15:07:05] Revealed #3 Oracle
Info: {'targets': [4, 7], 'minion_role': 'Minion'}

### [15:07:05] Revealed #4 Hunter
Info: {'distance': 3}

### [15:07:05] Revealed #5 Architect
Info: {'side': 'Right'}

### [15:07:05] Revealed #6 Knitter
Info: {'evil_pairs': 1}

### [15:07:05] Revealed #7 Witness
Info: {'affected_position': 5}

### [15:07:05] Revealed #8 Baker
Info: {'original_role': 'original'}

### [15:07:12] Revealed #1 Druid
Info: {}

#### [15:07:12] Solver Output
Scenarios: 14/1848
Definite good: ['#3', '#8', '#9']
Evil probabilities: #4=86%, #2=71%, #7=71%, #1=29%, #5=29%, #6=14%

#### [15:07:12] Recommendation
Action: **USE_ABILITY** #1 (Druid) -> targets ['#2', '#3', '#4']
Reason: Entropy 0.863 (adjusted 0.863) | timing x1.00

### [15:07:50] Revealed #1 Druid
Info: {'targets': [2, 3, 4], 'found_outcast': 'Wretch'}

### [15:07:50] Ability used at #1

#### [15:07:51] Solver Output
Scenarios: 4/1848
Definite evil: ['#1', '#2']
Definite good: ['#3', '#5', '#6', '#8', '#9']
Evil probabilities: #4=50%, #7=50%

#### [15:07:51] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Baa', 'Witch'})

### [15:07:57] Executed #1 -> Witch (EVIL)

### [15:08:27] Revealed #9 Plague_Doctor
Info: {}

#### [15:08:27] Solver Output
Scenarios: 2/224
Definite evil: ['#1', '#2']
Definite good: ['#3', '#5', '#6', '#8', '#9']
Evil probabilities: #4=50%, #7=50%

#### [15:08:27] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Baa'})

### [15:08:34] Executed #2 -> Baa (EVIL)

#### [15:08:41] Solver Output
Scenarios: 2/31
Definite evil: ['#1', '#2']
Definite good: ['#3', '#5', '#6', '#8', '#9']
Evil probabilities: #4=50%, #7=50%

#### [15:08:41] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#4']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

#### [15:09:12] Solver Output
Scenarios: 1/31
Definite evil: ['#1', '#2', '#4']
Definite good: ['#3', '#5', '#6', '#7', '#8', '#9']

#### [15:09:12] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Minion'})

### [15:09:19] Executed #4 -> Minion (EVIL)

## [15:09:27] GAME OVER — WIN
Final HP: 10
Notes: Asc57 v3 9-card 10HP perfect, Druid Wretch lie locked Witch #1, then Baa #2 + Minion #4 sequential confident


---

# New Game — 2026-04-11 15:10:49
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Knitter, Empress, Knight, Poet, Bard
- Outcasts: Plague_Doctor
- Minions: Witch, Minion
- Demons: Pooka

### [15:11:35] Revealed #2 Bard
Info: {'corruption_distance': 2}

### [15:11:35] Revealed #5 Knight
Info: {}

### [15:11:35] Revealed #6 Knitter
Info: {'evil_pairs': 2}

### [15:11:35] Revealed #7 Empress
Info: {'targets': [2, 4, 5]}

### [15:11:48] Revealed #1 Poet
Info: {'targets': [4, 7, 8], 'copied_role': 'Empress'}

### [15:11:48] Revealed #3 Plague_Doctor
Info: {}

### [15:11:48] Revealed #4 Poet
Info: {'targets': [2, 5, 8], 'types': ['Villager', 'Outcast', 'Minion'], 'copied_role': 'Bishop'}

### [15:11:48] Revealed #8 Jester
Info: {}

#### [15:11:54] Solver Output
Scenarios: 44/1848
Evil probabilities: #1=59%, #6=55%, #5=50%, #4=45%, #2=32%, #9=27%, #7=14%, #3=9%, #8=9%

#### [15:11:54] Recommendation
Action: **EXECUTE** #5
Reason: Knight check: #5 is 50% evil, 27% corruption risk. Expected HP cost: 1.2 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 27% -- corrupted Knight loses immunity + 4 extra damage

#### [15:12:56] Solver Output
Scenarios: 22/1176
Definite good: ['#5']
Evil probabilities: #4=82%, #6=64%, #2=55%, #1=27%, #7=27%, #8=18%, #9=18%, #3=9%

#### [15:12:56] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.187 (adjusted 2.187) | timing x1.00

#### [15:13:30] Solver Output
Scenarios: 4/1176
Definite evil: ['#6']
Definite good: ['#2', '#5', '#7', '#8']
Evil probabilities: #1=50%, #3=50%, #4=50%, #9=50%

#### [15:13:30] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 4 scenarios (roles: {'Pooka', 'Witch', 'Minion'})

### [15:14:04] Executed #6 -> Pooka (EVIL)

#### [15:14:04] Solver Output
Scenarios: 2/162
Definite evil: ['#1', '#6', '#9']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#8']

#### [15:14:04] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Minion', 'Witch'})

### [15:14:37] Executed #1 -> Witch (EVIL)

### [15:15:09] Revealed #9 Knitter
Info: {'evil_pairs': 0}

#### [15:15:09] Solver Output
Scenarios: 1/26
Definite evil: ['#1', '#6', '#9']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#8']

#### [15:15:09] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Minion'})

### [15:15:16] Executed #9 -> Minion (EVIL)

## [15:15:28] GAME OVER — WIN
Final HP: 1
Notes: Asc57 v4 9-card 1HP survivor, corrupted Knight #5 wrecked us 10->1 (lookahead failed), but solver locked all 3 evils sequentially after PD probe


---

# New Game — 2026-04-11 15:16:43
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Fortune_Teller, Knight, Gemcrafter, Medium, Dreamer, Druid, Jester
- Outcasts: Plague_Doctor, Bombardier
- Minions: Twin_Minion, Minion
- Demons: Pooka

### [15:17:26] Revealed #1 Bombardier
Info: {}

### [15:17:26] Revealed #2 Knight
Info: {}

### [15:17:26] Revealed #3 Gemcrafter
Info: {'good_position': 4}

### [15:17:26] Revealed #7 Medium
Info: {'good_position': 1, 'good_role': 'Bombardier'}

### [15:17:35] Revealed #4 Druid
Info: {}

### [15:17:35] Revealed #5 Jester
Info: {}

### [15:17:35] Revealed #6 Dreamer
Info: {}

### [15:17:35] Revealed #8 Fortune_Teller
Info: {}

### [15:17:36] Revealed #9 Plague_Doctor
Info: {}

#### [15:17:36] Solver Output
Scenarios: 354/1638
Evil probabilities: #1=53%, #4=48%, #5=36%, #6=35%, #8=35%, #2=34%, #7=29%, #3=20%, #9=10%

#### [15:17:36] Recommendation
Action: **USE_ABILITY** #6 (Dreamer) -> targets ['#1']
Reason: Entropy 2.760 (adjusted 2.394) | timing x1.00
WARNING: Corruption risk: 27%

### [15:18:05] Revealed #6 Dreamer
Info: {'target': 1, 'evil_role': 'Pooka'}

### [15:18:05] Ability used at #6

#### [15:18:05] Solver Output
Scenarios: 294/1638
Evil probabilities: #4=50%, #1=44%, #5=39%, #6=37%, #2=36%, #8=33%, #7=25%, #3=23%, #9=12%

#### [15:18:05] Recommendation
Action: **EXECUTE** #2
Reason: Knight check: #2 is 36% evil, 24% corruption risk. Expected HP cost: 1.4 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 24% -- corrupted Knight loses immunity + 4 extra damage

#### [15:18:47] Solver Output
Scenarios: 188/1056
Definite good: ['#2']
Evil probabilities: #1=52%, #4=47%, #5=46%, #6=45%, #8=40%, #7=32%, #3=27%, #9=12%

#### [15:18:47] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#2']
Reason: Entropy 2.442 (adjusted 2.442) | timing x1.00

#### [15:19:19] Solver Output
Scenarios: 100/1056
Definite good: ['#2']
Evil probabilities: #1=64%, #6=52%, #5=46%, #4=42%, #8=40%, #7=36%, #3=18%, #9=2%

#### [15:19:19] Recommendation
Action: **USE_ABILITY** #4 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 1.583 (adjusted 1.377) | timing x1.00
WARNING: Corruption risk: 26%

### [15:19:56] Revealed #4 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [15:19:56] Ability used at #4

#### [15:19:56] Solver Output
Scenarios: 68/1056
Definite good: ['#2']
Evil probabilities: #6=53%, #4=50%, #1=47%, #5=47%, #8=47%, #3=26%, #7=26%, #9=3%

#### [15:19:56] Recommendation
Action: **USE_ABILITY** #8 (Fortune Teller) -> targets ['#3', '#6']
Reason: Entropy 1.000 (adjusted 0.853) | timing x1.00
WARNING: Corruption risk: 29%

### [15:20:27] Revealed #8 Fortune Teller
Info: {'targets': [3, 6], 'has_evil': False}

### [15:20:27] Ability used at #8

#### [15:20:27] Solver Output
Scenarios: 34/1056
Definite good: ['#2', '#9']
Evil probabilities: #6=76%, #8=53%, #1=47%, #4=47%, #3=29%, #5=29%, #7=18%

#### [15:20:27] Recommendation
Action: **USE_ABILITY** #5 (Jester) -> targets ['#1', '#6', '#8']
Reason: Expected posterior 23.8 scenarios (adjusted 30.8, info gain 0.143 bits) | timing x1.00
WARNING: Corruption risk: 59%

### [15:21:01] Revealed #5 Jester
Info: {'targets': [1, 6, 8], 'evil_count': 3}

### [15:21:02] Ability used at #5

#### [15:21:02] Solver Output
Scenarios: 26/1056
Definite good: ['#2', '#9']
Evil probabilities: #6=69%, #4=62%, #8=46%, #3=38%, #5=38%, #1=31%, #7=15%

#### [15:21:02] Recommendation
Action: **EXECUTE** #6
Reason: No reveals available. #6 is 69% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 69% confident (budget: 2 wrong execs)

### [15:21:38] Executed #6 -> Pooka (EVIL)

#### [15:21:38] Solver Output
Scenarios: 4/142
Definite evil: ['#1', '#6']
Definite good: ['#2', '#3', '#4', '#8', '#9']
Evil probabilities: #5=50%, #7=50%

#### [15:21:38] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Twin_Minion', 'Minion'})

### [15:21:45] Executed #1 -> Twin Minion (EVIL)

#### [15:21:53] Solver Output
Scenarios: 2/26
Definite evil: ['#1', '#6']
Definite good: ['#2', '#3', '#4', '#8', '#9']
Evil probabilities: #5=50%, #7=50%

#### [15:21:53] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (50% good Jester (corrupted), 50% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:21:59] Executed #5 -> Minion (EVIL)

## [15:22:09] GAME OVER — WIN
Final HP: 10
Notes: Asc57 v5 9-card 10HP perfect, Knight check on #2 worked, all 3 evils sequential after probes


---

# New Game — 2026-04-11 15:23:32
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Poet, Fortune_Teller, Druid, Bard
- Outcasts: Bombardier
- Minions: 
- Demons: Pooka

### [15:24:09] Revealed #1 Bard
Info: {'corruption_distance': 3}

### [15:24:09] Revealed #2 Bombardier
Info: {}

### [15:24:09] Revealed #3 Poet
Info: {'good_position': 5, 'good_role': 'Hunter', 'copied_role': 'Medium'}

### [15:24:09] Revealed #4 Druid
Info: {}

### [15:24:09] Revealed #5 Hunter
Info: {'distance': 2}

### [15:24:09] Revealed #6 Fortune_Teller
Info: {}

#### [15:24:10] Solver Output
Scenarios: 2/6
Definite good: ['#2', '#3', '#4', '#5']
Evil probabilities: #1=50%, #6=50%

#### [15:24:10] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 1.000 (adjusted 0.750) | timing x1.00
WARNING: Corruption risk: 50%

### [15:24:41] Revealed #6 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [15:24:41] Ability used at #6

#### [15:24:41] Solver Output
Scenarios: 1/6
Definite evil: ['#1']
Definite good: ['#2', '#3', '#4', '#5', '#6']

#### [15:24:41] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [15:24:48] Executed #1 -> Pooka (EVIL)

## [15:24:57] GAME OVER — WIN
Final HP: 10
Notes: Asc57 v6 6-card 10HP perfect single demon, FT corrupted lying detected, Pooka definite exec


---

# New Game — 2026-04-11 15:26:13
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Alchemist, Knight, Fortune_Teller, Enlightened, Oracle, Hunter
- Outcasts: Wretch
- Minions: Twin_Minion
- Demons: Pooka

### [15:27:01] Revealed #1 Enlightened
Info: {'direction': 'Equidistant'}

### [15:27:01] Revealed #3 Oracle
Info: {'targets': [4, 5], 'minion_role': 'Twin_Minion'}

### [15:27:01] Revealed #4 Hunter
Info: {'distance': 1}

### [15:27:01] Revealed #5 Alchemist
Info: {'cured_count': 1}

### [15:27:01] Revealed #6 Knight
Info: {}

### [15:27:01] Revealed #7 Wretch
Info: {}

### [15:27:01] Revealed #8 Confessor
Info: {'dizzy': False}

### [15:27:09] Revealed #2 Fortune_Teller
Info: {}

#### [15:27:09] Solver Output
Scenarios: 3/56
Definite good: ['#1', '#5', '#7', '#8']
Evil probabilities: #3=67%, #6=67%, #2=33%, #4=33%

#### [15:27:09] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#1', '#3']
Reason: Entropy 0.918 (adjusted 0.765) | timing x1.00
WARNING: Corruption risk: 33%

### [15:27:40] Revealed #2 Fortune Teller
Info: {'targets': [1, 3], 'has_evil': True}

### [15:27:40] Ability used at #2

#### [15:27:41] Solver Output
Scenarios: 2/56
Definite good: ['#1', '#5', '#7', '#8']
Evil probabilities: #2=50%, #3=50%, #4=50%, #6=50%

#### [15:27:41] Recommendation
Action: **EXECUTE** #6
Reason: Knight free check: #6 is 50% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [15:28:17] Executed #6 -> Pooka (EVIL)

#### [15:28:18] Solver Output
Scenarios: 1/7
Definite evil: ['#3', '#6']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#8']

#### [15:28:18] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [15:28:24] Executed #3 -> Twin Minion (EVIL)

## [15:28:33] GAME OVER — WIN
Final HP: 10
Notes: Asc57 v7 final 8-card 10HP perfect, ASC57 COMPLETE 7/7, Knight check on Pooka #6 worked, Twin Minion #3 confident exec


---

# New Game — 2026-04-11 15:30:54
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Scout, Knight, Enlightened, Medium, Hunter, Bard
- Outcasts: Bombardier, Plague_Doctor
- Minions: Poisoner, Minion
- Demons: Lilis

### [15:32:47] Revealed #1 Knight
Info: {}

### [15:32:47] Revealed #2 Hunter
Info: {'distance': 4}

### [15:32:47] Revealed #3 Scout
Info: {'evil_role': 'Lilis', 'distance': 3}

### [15:32:47] Revealed #4 Lover
Info: {'evil_adjacent': 0}

### [15:32:47] Revealed #5 Bard
Info: {'corruption_distance': 1}

### [15:32:47] Revealed #6 Medium
Info: {'good_position': 9, 'good_role': 'Bombardier'}

### [15:32:47] Revealed #9 Bombardier
Info: {}

### [15:32:55] Revealed #7 Plague_Doctor
Info: {}

#### [15:32:56] Solver Output
Scenarios: 14/2586
Definite good: ['#5', '#8']
Evil probabilities: #2=86%, #9=86%, #6=50%, #3=36%, #1=21%, #7=14%, #4=7%

#### [15:32:56] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.353 (adjusted 2.353) | timing x1.00

#### [15:33:34] Solver Output
Scenarios: 9/2586
Definite good: ['#4', '#5', '#8']
Evil probabilities: #2=89%, #6=78%, #9=78%, #3=22%, #7=22%, #1=11%

#### [15:33:34] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (44% evil Minion, 22% evil Lilis, 22% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 89%, but all reveal branches still lead to a forced win.

### [15:33:41] Executed #2 -> Minion (EVIL)

#### [15:33:48] Solver Output
Scenarios: 4/308
Definite evil: ['#2']
Definite good: ['#1', '#4', '#5', '#8']
Evil probabilities: #6=75%, #9=75%, #3=25%, #7=25%

#### [15:33:48] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (50% evil Lilis, 25% good Medium, 25% evil Poisoner).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [15:33:54] Executed #6 -> Lilis (EVIL)

#### [15:34:01] Solver Output
Scenarios: 2/43
Definite evil: ['#2', '#6', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [15:34:01] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 2 scenarios (roles: {'Poisoner'})

### [15:34:08] Executed #9 -> Poisoner (EVIL)

## [15:34:17] GAME OVER — WIN
Final HP: 6
Notes: Asc58 v1 Lilis 9-card 6HP, night-killed Enlightened #8 + 4HP, #1 reflip needed, PD probe locked Lilis #6, all 3 evils sequential


---

# New Game — 2026-04-11 15:35:39
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Enlightened, Gemcrafter, Baker, Architect, Fortune_Teller, Knight, Scout
- Outcasts: Plague_Doctor
- Minions: Minion, Shaman
- Demons: Pooka

### [15:36:53] Revealed #1 Plague_Doctor
Info: {}

### [15:36:53] Revealed #2 Baker
Info: {'original_role': 'Empress'}

### [15:36:53] Revealed #3 Scout
Info: {'evil_role': 'Shaman', 'distance': 2}

### [15:36:53] Revealed #5 Baker
Info: {'original_role': 'Enlightened'}

### [15:36:53] Revealed #6 Gemcrafter
Info: {'good_position': 2}

### [15:36:53] Revealed #7 Empress
Info: {'targets': [1, 4, 5]}

### [15:36:53] Revealed #8 Enlightened
Info: {'direction': 'CW'}

### [15:36:53] Revealed #9 Knight
Info: {}

### [15:36:53] Revealed #10 Architect
Info: {'side': 'Right'}

### [15:37:02] Revealed #4 Enlightened
Info: {'direction': 'cw'}

#### [15:37:02] Solver Output
Scenarios: 44/3240
Definite good: ['#1', '#6']
Evil probabilities: #4=64%, #3=57%, #8=50%, #9=50%, #10=39%, #7=25%, #5=11%, #2=5%

#### [15:37:02] Recommendation
Action: **EXECUTE** #9
Reason: Knight check: #9 is 50% evil, 23% corruption risk. Expected HP cost: 1.0 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 23% -- corrupted Knight loses immunity + 4 extra damage

### [15:37:41] Executed #9 -> Minion (EVIL)

#### [15:37:41] Solver Output
Scenarios: 9/352
Definite evil: ['#9']
Definite good: ['#1', '#2', '#6', '#7', '#8']
Evil probabilities: #4=89%, #10=56%, #3=44%, #5=11%

#### [15:37:41] Recommendation
Action: **USE_ABILITY** #1 (Plague Doctor) -> targets ['#5']
Reason: Entropy 1.891 (adjusted 1.891) | timing x1.00

#### [15:38:16] Solver Output
Scenarios: 3/352
Definite evil: ['#3', '#4', '#9']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8', '#10']

#### [15:38:16] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 3 scenarios (roles: {'Pooka', 'Shaman'})

### [15:38:23] Executed #3 -> Pooka (EVIL)

#### [15:38:33] Solver Output
Scenarios: 1/43
Definite evil: ['#3', '#4', '#9']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8', '#10']

#### [15:38:33] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [15:38:40] Executed #4 -> Shaman (EVIL)

## [15:38:51] GAME OVER — WIN
Final HP: 10
Notes: Asc58 v2 10-card 10HP perfect, Knight check on Minion #9 (no immunity), then PD probe found Pooka #3 + Shaman #4


---

# New Game — 2026-04-11 15:40:29
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Hunter, Architect, Druid, Fortune_Teller, Baker, Confessor
- Outcasts: Drunk
- Minions: Minion
- Demons: Pooka

### [15:40:52] Revealed #1 Hunter
Info: {'distance': 2}

### [15:40:52] Revealed #2 Confessor
Info: {'dizzy': False}

### [15:40:52] Revealed #3 Architect
Info: {'side': 'Left'}

### [15:40:52] Revealed #6 Baker
Info: {'original_role': 'Medium'}

### [15:40:52] Revealed #8 Medium
Info: {'good_position': 7, 'good_role': 'Druid'}

### [15:41:10] Revealed #4 Druid
Info: {}

### [15:41:10] Revealed #5 Fortune_Teller
Info: {}

### [15:41:10] Revealed #7 Druid
Info: {}

#### [15:41:11] Solver Output
Scenarios: 30/336
Definite good: ['#2']
Evil probabilities: #7=57%, #6=37%, #5=33%, #3=30%, #4=23%, #1=10%, #8=10%

#### [15:41:11] Recommendation
Action: **USE_ABILITY** #4 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.997 (adjusted 0.847) | timing x1.00
WARNING: Corruption risk: 30%

### [15:42:38] Revealed #4 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Drunk'}

### [15:42:38] Ability used at #4

#### [15:42:38] Solver Output
Scenarios: 15/336
Definite good: ['#2']
Evil probabilities: #7=53%, #3=33%, #4=33%, #5=33%, #6=27%, #1=13%, #8=7%

#### [15:42:38] Recommendation
Action: **USE_ABILITY** #5 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.997 (adjusted 0.831) | timing x1.00
WARNING: Corruption risk: 33%

### [15:43:01] Revealed #5 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [15:43:01] Ability used at #5

#### [15:43:01] Solver Output
Scenarios: 7/336
Definite good: ['#2', '#5']
Evil probabilities: #7=71%, #4=43%, #1=29%, #6=29%, #3=14%, #8=14%

#### [15:43:01] Recommendation
Action: **USE_ABILITY** #7 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.592 (adjusted 0.549) | timing x1.00
WARNING: Corruption risk: 14%

### [15:43:35] Revealed #7 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Drunk'}

### [15:43:35] Ability used at #7

#### [15:43:35] Solver Output
Scenarios: 4/336
Definite evil: ['#7']
Definite good: ['#1', '#2', '#5', '#8']
Evil probabilities: #4=50%, #3=25%, #6=25%

#### [15:43:35] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 4 scenarios (roles: {'Pooka'})

### [15:43:42] Executed #7 -> Pooka (EVIL)

#### [15:43:50] Solver Output
Scenarios: 4/42
Definite evil: ['#7']
Definite good: ['#1', '#2', '#5', '#8']
Evil probabilities: #4=50%, #3=25%, #6=25%

#### [15:43:50] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (50% good Drunk (corrupted), 50% evil Minion).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:43:57] Executed #4 -> Minion (EVIL)

## [15:44:07] GAME OVER — WIN
Final HP: 10
Notes: Asc58 v3 8-card 10HP perfect, 2 Druids both lying about Drunk in 1,2,3, FT confirmed 1,2 clean, lookahead-forced #4 final


---

# New Game — 2026-04-11 15:45:27
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Slayer, Bishop, Baker, Enlightened, Poet, Druid
- Outcasts: Drunk, Wretch
- Minions: Chancellor, Witch
- Demons: Lilis

### [15:47:06] Revealed #1 Enlightened
Info: {'direction': 'CW'}

### [15:47:06] Revealed #2 Wretch
Info: {}

### [15:47:06] Revealed #3 Bishop
Info: {'targets': [6, 7, 8], 'types': ['Villager', 'Outcast', 'Minion']}

### [15:47:06] Revealed #4 Knitter
Info: {'evil_pairs': 2}

### [15:47:06] Revealed #5 Poet
Info: {'direction': 'CW', 'copied_role': 'Enlightened'}

### [15:47:14] Revealed #6 Slayer
Info: {}

### [15:47:14] Revealed #8 Druid
Info: {}

#### [15:47:15] Solver Output
Scenarios: 142/3412
Definite good: ['#7']
Evil probabilities: #5=56%, #4=51%, #8=50%, #2=42%, #6=37%, #3=31%, #1=16%, #9=16%

#### [15:47:15] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.986 (adjusted 0.951) | timing x1.00
WARNING: Corruption risk: 7%

### [15:47:45] Revealed #8 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [15:47:45] Ability used at #8

#### [15:47:46] Solver Output
Scenarios: 45/3412
Definite good: ['#7']
Evil probabilities: #4=62%, #3=58%, #5=58%, #2=53%, #6=42%, #8=13%, #1=7%, #9=7%

#### [15:47:46] Recommendation
Action: **USE_ABILITY** #6 (Slayer) -> targets ['#4']
Reason: Target #4 is 62% evil (adjusted 0.57)
WARNING: Corruption risk: 9% -- Slayer ability disabled if corrupted

### [15:48:26] Ability used at #6

### [15:49:23] Revealed #9 Baker
Info: {'original_role': 'Poet'}

#### [15:49:40] Solver Output
Scenarios: 0/109

#### [15:49:40] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [15:50:02] Solver Output
Scenarios: 0/0

#### [15:50:02] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [15:50:43] Solver Output
Scenarios: 2/267
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#2', '#6', '#7', '#8', '#9']

#### [15:50:43] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Chancellor', 'Lilis'})

#### [15:50:50] Solver Output
Scenarios: 2/267
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#2', '#6', '#7', '#8', '#9']

#### [15:50:50] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Lilis', 'Chancellor'})

### [15:50:57] Executed #3 -> Chancellor (EVIL)

#### [15:51:05] Solver Output
Scenarios: 1/31
Definite evil: ['#3', '#4', '#5']
Definite good: ['#1', '#2', '#6', '#7', '#8', '#9']

#### [15:51:05] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [15:51:12] Executed #5 -> Lilis (EVIL)

## [15:51:24] GAME OVER — WIN
Final HP: 6
Notes: Asc58 v4 Lilis 9-card 6HP, Slayer killed Witch #4, night_no_kill bug confirmed wrong evil (#9 instead of #5), recovered by manual JSON fix


---

# New Game — 2026-04-11 15:54:26
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Knight, Enlightened, Witness, Scout, Hunter
- Outcasts: Drunk, Bombardier
- Minions: Witch, Twin_Minion
- Demons: Baa

### [15:54:59] Revealed #1 Baker
Info: {'original_role': 'original'}

### [15:54:59] Revealed #2 Bombardier
Info: {}

### [15:54:59] Revealed #3 Scout
Info: {'evil_role': 'Baa', 'distance': 2}

### [15:54:59] Revealed #4 Scout
Info: {'evil_role': 'Twin Minion', 'distance': 1}

### [15:54:59] Revealed #5 Baker
Info: {'original_role': 'Hunter'}

### [15:54:59] Revealed #6 Witness
Info: {'affected_position': 4}

### [15:54:59] Revealed #7 Baker
Info: {'original_role': 'Knight'}

### [15:54:59] Revealed #8 Bombardier
Info: {}

#### [15:55:23] Solver Output
Scenarios: 218/2352
Definite good: ['#1']
Evil probabilities: #2=66%, #6=63%, #8=48%, #3=36%, #4=29%, #9=24%, #7=18%, #5=16%

#### [15:55:23] Recommendation
Action: **EXECUTE** #6
Reason: No reveals available. #6 is 63% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 63% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Low confidence (63% < 63%) -- consider gathering more info

### [15:56:01] Executed #6 -> Baa (EVIL)

#### [15:56:01] Solver Output
Scenarios: 44/252
Definite evil: ['#6']
Definite good: ['#1']
Evil probabilities: #2=64%, #8=55%, #4=27%, #3=18%, #9=18%, #5=9%, #7=9%

#### [15:56:01] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 27% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 27% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Low confidence (27% < 60%) -- consider gathering more info

### [15:56:46] Executed #4 -> Twin_Minion (EVIL)

#### [15:56:46] Solver Output
Scenarios: 6/30
Definite evil: ['#4', '#6']
Definite good: ['#1', '#3', '#5', '#7', '#9']
Evil probabilities: #2=50%, #8=50%

#### [15:56:46] Recommendation
Action: **EXECUTE** #1
Reason: No reveals available. #1 is 0% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 0% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Low confidence (0% < 50%) -- consider gathering more info

#### [15:56:58] Solver Output
Scenarios: 6/30
Definite evil: ['#4', '#6']
Definite good: ['#1', '#3', '#5', '#7', '#9']
Evil probabilities: #2=50%, #8=50%

#### [15:56:58] Recommendation
Action: **EXECUTE** #1
Reason: No reveals available. #1 is 0% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 0% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Low confidence (0% < 50%) -- consider gathering more info

#### [15:58:13] Solver Output
Scenarios: 6/26
Definite evil: ['#4', '#6']
Definite good: ['#1', '#3', '#5', '#7', '#9']
Evil probabilities: #2=50%, #8=50%

#### [15:58:13] Recommendation
Action: **EXECUTE** #1
Reason: No reveals available. #1 is 0% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 0% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Bombardier safety: executing #1 (0%) despite low confidence — Bombardier candidate(s) [2, 8] risk instant game loss if executed first.

### [15:58:26] Executed #1 -> GOOD (WRONG!)

#### [15:58:36] Solver Output
Scenarios: 6/26
Definite evil: ['#4', '#6']
Definite good: ['#1', '#3', '#5', '#7', '#9']
Evil probabilities: #2=50%, #8=50%

#### [15:58:36] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 0% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 0% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Bombardier safety: executing #3 (0%) despite low confidence — Bombardier candidate(s) [2, 8] risk instant game loss if executed first.

#### [15:59:10] Solver Output
Scenarios: 6/26
Definite evil: ['#4', '#6']
Definite good: ['#1', '#3', '#5', '#7', '#9']
Evil probabilities: #2=50%, #8=50%

#### [15:59:10] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 0% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 0% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Bombardier safety: executing #3 (0%) despite low confidence — Bombardier candidate(s) [2, 8] risk instant game loss if executed first.

## [15:59:59] GAME OVER — LOSS
Final HP: 5
Notes: Asc58 v5 LOSS, solver Bombardier safety bug picked 0%-evil targets (#1, #3) leading to guaranteed wrong execs and HP 5->0. Witch was at #8 (Bombardier candidate) but solver refused to execute due to Bombardier risk. Real bug to fix.

## [16:00:11] GAME OVER — LOSS
Final HP: 5
Notes: Asc58 v5 LOSS HP 0. Solver Bombardier safety bug: with Witch at #8 (Bombardier disguise) and #2 also Bombardier candidate, solver picked 0% confident #1 then #3 (both Good), guaranteed wrong execs, HP 10->5->0. Streak broken. Bug: solver should execute #8 even at Bombardier risk if it's the ONLY winning move.

#### [16:22:43] Claude Reasoning


#### [16:22:45] Solver Output
Scenarios: 6/26
Definite evil: ['#4', '#6']
Definite good: ['#1', '#3', '#5', '#7', '#9']
Evil probabilities: #2=50%, #8=50%

#### [16:22:45] Recommendation
Action: **EXECUTE** #2
Reason: No reveals available. #2 is 50% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 50% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Bombardier safety: executing #2 (50%) despite low confidence — Bombardier candidate(s) [2, 8] risk instant game loss if executed first.


---

# New Game — 2026-04-11 16:28:55
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Knitter, Fortune_Teller, Oracle, Hunter, Medium, Scout
- Outcasts: Doppelganger
- Minions: Twin_Minion
- Demons: Pooka

### [16:29:56] Revealed #1 Scout
Info: {'evil_role': 'Twin Minion', 'distance': 3}

### [16:29:56] Revealed #2 Lover
Info: {'evil_adjacent': 0}

### [16:29:56] Revealed #3 Knitter
Info: {'evil_pairs': 1}

### [16:29:56] Revealed #4 Medium
Info: {'good_position': 6, 'good_role': 'Oracle'}

### [16:29:56] Revealed #5 Hunter
Info: {'distance': 2}

### [16:29:56] Revealed #6 Oracle
Info: {'targets': [7, 8], 'minion_role': 'Twin_Minion'}

### [16:29:56] Revealed #7 Medium
Info: {'good_position': 6, 'good_role': 'Oracle'}

### [16:30:11] Revealed #8 Fortune_Teller
Info: {}

#### [16:30:16] Solver Output
Scenarios: 3/336
Definite good: ['#1', '#2', '#5', '#6']
Evil probabilities: #3=67%, #7=67%, #4=33%, #8=33%

#### [16:30:16] Recommendation
Action: **USE_ABILITY** #8 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.918 (adjusted 0.918) | timing x1.00

### [16:30:37] Revealed #8 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': True}

### [16:30:42] Ability used at #8

#### [16:30:42] Solver Output
Scenarios: 1/336
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7']

#### [16:30:42] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [16:30:49] Executed #3 -> Pooka (EVIL)

#### [16:30:58] Solver Output
Scenarios: 1/42
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7']

#### [16:30:58] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [16:31:05] Executed #8 -> Twin Minion (EVIL)

## [16:31:26] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect win, FT targets #1,#2 revealed evil presence -> solver solved to 1 scenario, clean sweep


---

# New Game — 2026-04-11 17:58:25
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Bard, Bishop, Medium, Druid, Judge
- Outcasts: Plague_Doctor
- Minions: 
- Demons: Pooka

### [17:59:37] Revealed #1 Bard
Info: {'corruption_distance': 1}

### [17:59:37] Revealed #2 Bishop
Info: {'targets': [7, 3, 6], 'types': ['Villager', 'Outcast', 'Demon']}

### [17:59:37] Revealed #4 Judge
Info: {}

### [17:59:37] Revealed #5 Medium
Info: {'good_position': 1, 'good_role': 'Bard'}

### [17:59:37] Revealed #7 Architect
Info: {'side': 'Right'}

### [17:59:46] Revealed #3 Plague_Doctor
Info: {}

### [17:59:46] Revealed #6 Druid
Info: {}

#### [17:59:50] Solver Output
Scenarios: 6/31
Definite good: ['#1', '#3', '#4', '#5', '#6']
Evil probabilities: #2=50%, #7=50%

#### [17:59:50] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.459 (adjusted 1.459) | timing x1.00

#### [18:01:09] Solver Output
Scenarios: 3/31
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']

#### [18:01:09] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 3 scenarios (roles: {'Pooka'})

### [18:01:16] Executed #7 -> Pooka (EVIL)

## [18:01:42] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect win, 7-card single demon, PD check on #6 directly revealed #7 as evil


---

# New Game — 2026-04-11 18:10:18
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Hunter, Dreamer, Baker, Gemcrafter, Jester
- Outcasts: Doppelganger, Bombardier
- Minions: Minion
- Demons: Lilis

### [18:11:40] Revealed #1 Hunter
Info: {'distance': 3}

### [18:11:40] Revealed #2 Bard
Info: {'corruption_distance': -1}

### [18:11:40] Revealed #3 Hunter
Info: {'distance': 2}

### [18:11:58] Revealed #4 Jester
Info: {}

### [18:13:50] Revealed #5 Hunter
Info: {'distance': 4}

### [18:13:50] Revealed #6 Bombardier
Info: {}

### [18:13:50] Revealed #9 Baker
Info: {'original_role': 'Jester'}

### [18:13:59] Revealed #8 Dreamer
Info: {}

#### [18:14:04] Solver Output
Scenarios: 12/448
Definite evil: ['#1', '#9']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7', '#8']

#### [18:14:04] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 12 scenarios (roles: {'Minion', 'Lilis'})

### [18:14:11] Executed #1 -> Minion (EVIL)

#### [18:14:21] Solver Output
Scenarios: 6/49
Definite evil: ['#1', '#9']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7', '#8']

#### [18:14:21] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 6 scenarios (roles: {'Lilis'})

### [18:14:28] Executed #9 -> Lilis (EVIL)

## [18:14:50] GAME OVER — WIN
Final HP: 6
Notes: 6HP ASC58 COMPLETE 7/7, Lilis + Bombardier + Doppelganger + Minion, 2 night kills 4HP, solver locked definite #1 Minion then #9 Lilis


---

# New Game — 2026-04-12 11:15:44
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Oracle, Knight, Fortune_Teller, Empress, Gemcrafter
- Outcasts: Bombardier, Wretch, Plague_Doctor
- Minions: Witch, Puppeteer
- Demons: Baa

### [11:17:06] Revealed #1 Empress
Info: {'targets': [4, 7, 9]}

### [11:17:06] Revealed #2 Oracle
Info: {'targets': [1, 9], 'minion_role': 'Puppeteer'}

### [11:17:06] Revealed #3 Wretch
Info: {}

### [11:17:06] Revealed #4 Fortune_Teller
Info: {}

### [11:17:06] Revealed #5 Plague_Doctor
Info: {}

### [11:17:06] Revealed #6 Gemcrafter
Info: {'good_position': 1}

### [11:17:06] Revealed #7 Confessor
Info: {'dizzy': True}

### [11:17:06] Revealed #8 Fortune_Teller
Info: {}

#### [11:18:30] Solver Output
Scenarios: 28/2032
Definite good: ['#1']
Evil probabilities: #2=86%, #3=86%, #7=86%, #8=86%, #4=29%, #5=14%, #6=7%, #9=7%

#### [11:18:30] Recommendation
Action: **USE_ABILITY** #5 (Plague Doctor) -> targets ['#1']
Reason: Entropy 1.698 (adjusted 1.698) | timing x1.00

### [11:19:31] Ability used at #5

#### [11:19:34] Solver Output
Scenarios: 6/2032
Definite evil: ['#7', '#8']
Definite good: ['#1', '#4', '#6']
Evil probabilities: #2=67%, #5=67%, #3=33%, #9=33%

#### [11:19:34] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 6 scenarios (roles: {'Baa', 'Puppet', 'Puppeteer', 'Witch'})

### [11:19:41] Executed #7 -> Puppet (EVIL)

#### [11:19:47] Solver Output
Scenarios: 2/284
Definite evil: ['#2', '#5', '#7', '#8']
Definite good: ['#1', '#3', '#4', '#6', '#9']

#### [11:19:47] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Baa', 'Witch'})

### [11:19:54] Executed #2 -> Baa (EVIL)

#### [11:19:57] Solver Output
Scenarios: 1/44
Definite evil: ['#2', '#5', '#7', '#8']
Definite good: ['#1', '#3', '#4', '#6', '#9']

#### [11:19:57] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Witch'})

### [11:20:04] Executed #5 -> Witch (EVIL)

#### [11:20:09] Solver Output
Scenarios: 1/2
Definite evil: ['#2', '#5', '#7', '#8']
Definite good: ['#1', '#3', '#4', '#6', '#9']

#### [11:20:09] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Puppeteer'})

### [11:21:03] Executed #8 -> Puppeteer (EVIL)

## [11:21:08] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, PD check on #1 revealed corruption, all 4 evils definite


---

# New Game — 2026-04-12 11:23:19
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Lover, Knight, Druid, Jester, Confessor
- Outcasts: Wretch, Doppelganger
- Minions: Puppeteer
- Demons: Lilis

### [11:26:52] Revealed #1 Knight
Info: {}

### [11:26:52] Revealed #2 Confessor
Info: {'dizzy': True}

### [11:26:52] Revealed #3 Confessor
Info: {'dizzy': True}

### [11:26:52] Revealed #4 Lover
Info: {'evil_adjacent': 1}

### [11:26:52] Revealed #8 Bishop
Info: {'targets': [4, 5, 9], 'types': ['Villager', 'Outcast', 'Minion']}

### [11:26:52] Revealed #9 Wretch
Info: {}

### [11:27:15] Revealed #6 Druid
Info: {}

### [11:27:19] Revealed #7 Jester
Info: {}

#### [11:27:23] Solver Output
Scenarios: 74/1036
Definite evil: ['#2', '#3']
Definite good: ['#5']
Evil probabilities: #1=27%, #6=14%, #7=14%, #9=14%, #8=11%, #4=5%

#### [11:27:23] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 74 scenarios (roles: {'Lilis', 'Puppet', 'Puppeteer'})

### [11:27:30] Executed #2 -> Lilis (EVIL)

#### [11:27:36] Solver Output
Scenarios: 10/117
Definite evil: ['#2', '#3']
Definite good: ['#1', '#5', '#6', '#7', '#8', '#9']
Evil probabilities: #4=40%

#### [11:27:36] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 10 scenarios (roles: {'Puppeteer'})

### [11:27:42] Executed #3 -> Puppeteer (EVIL)

## [11:28:02] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, all definite evils


---

# New Game — 2026-04-12 11:29:19
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Confessor, Baker, Medium, Hunter, Slayer
- Outcasts: Bombardier
- Minions: Shaman
- Demons: Lilis

### [11:30:32] Revealed #1 Baker
Info: {'original_role': 'original'}

### [11:30:32] Revealed #2 Jester
Info: {}

### [11:30:32] Revealed #3 Hunter
Info: {'distance': 1}

### [11:30:32] Revealed #4 Slayer
Info: {}

### [11:30:32] Revealed #5 Confessor
Info: {'dizzy': False}

### [11:30:32] Revealed #6 Bombardier
Info: {}

### [11:30:32] Revealed #8 Slayer
Info: {}

#### [11:30:39] Solver Output
Scenarios: 20/56
Definite good: ['#5', '#7']
Evil probabilities: #2=40%, #4=40%, #1=30%, #3=30%, #6=30%, #8=30%

#### [11:30:39] Recommendation
Action: **USE_ABILITY** #2 (Jester) -> targets ['#1', '#4', '#5']
Reason: Expected posterior 9.1 scenarios (adjusted 9.1, info gain 1.134 bits) | timing x1.00

### [11:31:07] Revealed #2 Jester
Info: {'targets': [1, 4, 5], 'evil_count': 1}

### [11:31:07] Ability used at #2

#### [11:31:11] Solver Output
Scenarios: 10/56
Definite good: ['#5', '#7']
Evil probabilities: #2=40%, #4=40%, #6=40%, #8=40%, #1=20%, #3=20%

#### [11:31:11] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#2']
Reason: Target #2 is 40% evil (adjusted 0.40)

### [11:31:59] Ability used at #4

#### [11:32:03] Solver Output
Scenarios: 6/56
Definite good: ['#2', '#5', '#7']
Evil probabilities: #4=67%, #1=33%, #3=33%, #6=33%, #8=33%

#### [11:32:03] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#4']
Reason: Target #4 is 67% evil (adjusted 0.67)

### [11:32:33] Ability used at #8

#### [11:32:38] Solver Output
Scenarios: 4/56
Definite good: ['#2', '#5', '#6', '#7']
Evil probabilities: #1=50%, #3=50%, #4=50%, #8=50%

#### [11:32:38] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% good Baker, 25% evil Lilis, 25% evil Shaman).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [11:32:45] Executed #1 -> GOOD (WRONG!)

### [11:32:52] Executed #1 -> GOOD (WRONG!)

#### [11:32:52] Solver Output
Scenarios: 2/42
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7']

#### [11:32:52] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Shaman', 'Lilis'})

### [11:33:33] Executed #4 -> Shaman (EVIL)

#### [11:33:33] Solver Output
Scenarios: 1/6
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7']

#### [11:33:33] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [11:33:40] Executed #8 -> Lilis (EVIL)

## [11:33:52] GAME OVER — WIN
Final HP: 3
Notes: 3HP, Lilis game, wrong exec #1 but lookahead win


---

# New Game — 2026-04-12 11:35:17
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Bishop, Slayer, Baker, Judge, Knight
- Outcasts: Bombardier, Plague_Doctor
- Minions: Puppeteer
- Demons: Pooka

### [11:35:55] Revealed #1 Baker
Info: {'original_role': 'original'}

### [11:35:55] Revealed #2 Medium
Info: {'good_position': 4, 'good_role': 'Baker'}

### [11:35:55] Revealed #3 Bombardier
Info: {}

### [11:35:55] Revealed #4 Baker
Info: {'original_role': 'Knight'}

### [11:35:55] Revealed #5 Baker
Info: {'original_role': 'Judge'}

### [11:35:55] Revealed #7 Bishop
Info: {'targets': [5, 1, 4], 'types': ['Villager', 'Outcast', 'Minion']}

### [11:35:55] Revealed #9 Judge
Info: {}

### [11:36:22] Revealed #6 Plague_Doctor
Info: {}

### [11:36:22] Revealed #8 Slayer
Info: {}

#### [11:36:27] Solver Output
Scenarios: 51/646
Evil probabilities: #5=59%, #8=49%, #7=47%, #9=33%, #4=24%, #1=14%, #2=12%, #3=10%, #6=10%

#### [11:36:27] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#7']
Reason: Entropy 2.751 (adjusted 2.751) | timing x1.00

### [11:37:01] Ability used at #6

#### [11:37:01] Solver Output
Scenarios: 25/646
Definite good: ['#1']
Evil probabilities: #7=88%, #8=48%, #5=44%, #9=36%, #4=16%, #6=12%, #2=4%, #3=4%

#### [11:37:01] Recommendation
Action: **USE_ABILITY** #9 (Judge) -> targets ['#2']
Reason: Expected posterior 15.0 scenarios (adjusted 16.5, info gain 0.599 bits) | timing x1.00
WARNING: Corruption risk: 20% -- corrupted Judge results are unreliable

### [11:37:31] Revealed #9 Judge
Info: {'target': 2, 'is_lying': True}

### [11:37:31] Ability used at #9

#### [11:37:31] Solver Output
Scenarios: 15/646
Definite good: ['#1', '#2', '#3']
Evil probabilities: #7=87%, #8=47%, #5=40%, #9=40%, #4=27%, #6=13%

#### [11:37:31] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#7']
Reason: Target #7 is 87% evil (adjusted 0.52)
WARNING: Corruption risk: 40% -- Slayer ability disabled if corrupted

### [11:38:00] Ability used at #8

#### [11:38:00] Solver Output
Scenarios: 9/646
Definite good: ['#1', '#2', '#3']
Evil probabilities: #7=78%, #5=44%, #9=44%, #8=33%, #4=22%, #6=22%

#### [11:38:00] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (67% evil Pooka, 22% good Bishop (corrupted), 11% evil Puppeteer).
WARNING: Execution lookahead override -- immediate hit chance is 78%, but all reveal branches still lead to a forced win.

### [11:38:07] Executed #7 -> Puppeteer (EVIL)

#### [11:38:11] Solver Output
Scenarios: 1/70
Definite evil: ['#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6', '#9']

#### [11:38:11] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [11:38:18] Executed #8 -> Pooka (EVIL)

## [11:38:24] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, PD check + Judge + Slayer abilities


---

# New Game — 2026-04-12 11:39:45
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Witness, Oracle, Enlightened, Baker, Lover, Dreamer
- Outcasts: Doppelganger, Drunk
- Minions: Shaman
- Demons: Baa

### [11:40:06] Revealed #1 Baker
Info: {'original_role': 'original'}

### [11:40:06] Revealed #2 Baker
Info: {'original_role': 'original'}

### [11:40:06] Revealed #3 Enlightened
Info: {'direction': 'CW'}

### [11:40:06] Revealed #4 Oracle
Info: {'targets': [2, 3], 'minion_role': 'Shaman'}

### [11:40:06] Revealed #5 Baker
Info: {'original_role': 'original'}

### [11:40:06] Revealed #6 Lover
Info: {'evil_adjacent': 1}

### [11:40:06] Revealed #7 Baker
Info: {'original_role': 'original'}

### [11:40:21] Revealed #7 Baker
Info: {'original_role': 'original'}

#### [11:40:34] Solver Output
Scenarios: 82/1302
Evil probabilities: #4=56%, #2=30%, #7=28%, #3=26%, #5=21%, #6=21%, #1=18%

#### [11:40:34] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 56% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 56% confident (budget: 2 wrong execs)
WARNING: Low confidence (56% < 60%) -- consider gathering more info

### [11:41:21] Executed #4 -> Shaman (EVIL)

#### [11:41:22] Solver Output
Scenarios: 23/186
Definite evil: ['#4']
Definite good: ['#2', '#3']
Evil probabilities: #5=26%, #6=26%, #7=26%, #1=22%

#### [11:41:22] Recommendation
Action: **EXECUTE** #5
Reason: No reveals available. #5 is 26% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 26% confident (budget: 2 wrong execs)
WARNING: Low confidence (26% < 50%) -- consider gathering more info

### [11:42:01] Executed #5 -> GOOD (WRONG!)

#### [11:42:01] Solver Output
Scenarios: 17/155
Definite evil: ['#4']
Definite good: ['#2', '#3', '#5']
Evil probabilities: #6=35%, #7=35%, #1=29%

#### [11:42:01] Recommendation
Action: **ERROR** #7
Reason: #7 is 35% likely evil but budget=1 requires >=85% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 35% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 35% < 85% threshold. Consider manual override if you have extra information.

### [11:42:51] Executed #7 -> GOOD (WRONG!)

## [11:43:02] GAME OVER — LOSS
Final HP: 0
Notes: Loss: 35% probabilistic pick on #7 was wrong, Baa at #6 was the target

## [11:43:13] GAME OVER — LOSS
Final HP: 0
Notes: Loss: 35% probabilistic pick on #7 was wrong, Baa at #6 (Lover) never found


---

# New Game — 2026-04-12 11:52:38
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Bard, Hunter, Fortune_Teller, Baker, Architect, Jester
- Outcasts: Doppelganger
- Minions: Twin_Minion, Witch
- Demons: Lilis

### [11:54:27] Revealed #1 Empress
Info: {'targets': [5, 6, 9]}

### [11:54:27] Revealed #2 Bard
Info: {'corruption_distance': -1}

### [11:54:27] Revealed #3 Empress
Info: {'targets': [1, 6, 9]}

### [11:54:27] Revealed #5 Hunter
Info: {'distance': 1}

### [11:54:27] Revealed #6 Baker
Info: {'original_role': 'Bard'}

### [11:54:27] Revealed #7 Architect
Info: {'side': 'Right'}

### [11:54:38] Revealed #4 Jester
Info: {}

#### [11:54:38] Solver Output
Scenarios: 72/3024
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#9']
Evil probabilities: #4=50%, #8=50%

#### [11:54:38] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 72 scenarios (roles: {'Twin_Minion', 'Witch', 'Lilis'})

### [11:54:45] Executed #6 -> Twin Minion (EVIL)

#### [11:54:50] Solver Output
Scenarios: 24/336
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#9']
Evil probabilities: #4=50%, #8=50%

#### [11:54:50] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 24 scenarios (roles: {'Lilis', 'Witch'})

### [11:54:57] Executed #7 -> Witch (EVIL)

#### [11:55:03] Solver Output
Scenarios: 12/42
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#9']
Evil probabilities: #4=50%, #8=50%

#### [11:55:03] Recommendation
Action: **USE_ABILITY** #4 (Jester) -> targets ['#1', '#2', '#3']
Reason: Expected posterior 6.0 scenarios (adjusted 6.0, info gain 1.000 bits) | timing x1.00

### [11:55:34] Revealed #4 Jester
Info: {'targets': [1, 2, 3], 'evil_count': 0}

### [11:55:34] Ability used at #4

#### [11:55:34] Solver Output
Scenarios: 6/42
Definite evil: ['#6', '#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#9']

#### [11:55:34] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 6 scenarios (roles: {'Lilis'})

### [11:55:41] Executed #8 -> Lilis (EVIL)

## [11:55:48] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis+Witch game, all definite evils


---

# New Game — 2026-04-12 11:57:09
Cards: 8, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Witness, Dreamer, Poet, Baker, Lover, Knight
- Outcasts: Doppelganger, Wretch
- Minions: Puppeteer
- Demons: Baa

### [11:57:31] Revealed #1 Knight
Info: {}

### [11:57:31] Revealed #2 Lover
Info: {'evil_adjacent': 0}

### [11:57:31] Revealed #3 Poet
Info: {'direction': 'Equidistant', 'copied_role': 'Enlightened'}

### [11:57:31] Revealed #4 Wretch
Info: {}

### [11:57:31] Revealed #6 Dreamer
Info: {}

### [11:57:31] Revealed #7 Witness
Info: {'affected_position': 3}

### [11:57:31] Revealed #8 Baker
Info: {'original_role': 'original'}

### [11:57:38] Revealed #5 Enlightened
Info: {'direction': 'cw'}

#### [11:57:38] Solver Output
Scenarios: 5/444
Definite evil: ['#2', '#3', '#5']
Definite good: ['#1', '#4', '#6', '#7', '#8']

#### [11:57:38] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 5 scenarios (roles: {'Puppeteer'})

### [11:57:46] Executed #2 -> Puppeteer (EVIL)

#### [11:57:51] Solver Output
Scenarios: 5/62
Definite evil: ['#2', '#3', '#5']
Definite good: ['#1', '#4', '#6', '#7', '#8']

#### [11:57:51] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 5 scenarios (roles: {'Puppet'})

### [11:57:58] Executed #3 -> Puppet (EVIL)

#### [11:58:04] Solver Output
Scenarios: 5/31
Definite evil: ['#2', '#3', '#5']
Definite good: ['#1', '#4', '#6', '#7', '#8']

#### [11:58:04] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 5 scenarios (roles: {'Baa'})

### [11:58:11] Executed #5 -> Baa (EVIL)

## [11:58:18] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, all definite evils


---

# New Game — 2026-04-12 12:00:12
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Medium, Judge, Empress, Oracle, Gemcrafter, Confessor
- Outcasts: Drunk
- Minions: Twin_Minion, Poisoner
- Demons: Lilis

### [12:01:40] Revealed #1 Medium
Info: {'good_position': 3, 'good_role': 'Gemcrafter'}

### [12:01:40] Revealed #2 Confessor
Info: {'dizzy': True}

### [12:01:40] Revealed #3 Gemcrafter
Info: {'good_position': 10}

### [12:01:40] Revealed #4 Baker
Info: {'original_role': 'original'}

### [12:01:40] Revealed #5 Empress
Info: {'targets': [1, 2, 9]}

### [12:01:40] Revealed #6 Baker
Info: {'original_role': 'Oracle'}

### [12:01:40] Revealed #8 Medium
Info: {'good_position': 1, 'good_role': 'Medium'}

### [12:01:40] Revealed #9 Baker
Info: {'original_role': 'Judge'}

#### [12:01:46] Solver Output
Scenarios: 28/7812
Definite good: ['#1', '#3', '#7', '#8', '#10']
Evil probabilities: #9=93%, #2=86%, #6=57%, #5=36%, #4=29%

#### [12:01:46] Recommendation
Action: **EXECUTE** #9
Reason: No reveals available. #9 is 93% likely evil (HP=6, budget=1 wrong execs)
WARNING: Probabilistic execution -- 93% confident (budget: 1 wrong execs)

### [12:02:44] Executed #9 -> GOOD (WRONG!)

#### [12:02:44] Solver Output
Scenarios: 0/5484

#### [12:02:44] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

## [12:03:30] GAME OVER — LOSS
Final HP: 1
Notes: Loss: 0 scenarios after exec #9 good. Solver bug: 93% confidence on #9 but was good Baker. Investigate Baker chain + 3 Bakers + Drunk.

## [12:03:46] GAME OVER — LOSS
Final HP: 1
Notes: Loss: 0 scenarios after exec #9 good. True evils: #1=Lilis, #3=Poisoner, #10=Twin_Minion. Solver wrongly had 93% on #9. Baker chain + Drunk + 3 Bakers.


---

# New Game — 2026-04-12 12:07:46
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Dreamer, Bard, Medium, Baker, Knight, Enlightened
- Outcasts: Drunk, Plague_Doctor, Bombardier
- Minions: Poisoner, Chancellor
- Demons: Baa

### [12:08:12] Revealed #1 Bombardier
Info: {}

### [12:08:12] Revealed #2 Baker
Info: {'original_role': 'Knight'}

### [12:08:12] Revealed #3 Medium
Info: {'good_position': 5, 'good_role': 'Knight'}

### [12:08:12] Revealed #4 Medium
Info: {'good_position': 1, 'good_role': 'Bombardier'}

### [12:08:12] Revealed #5 Knight
Info: {}

### [12:08:12] Revealed #6 Enlightened
Info: {'direction': 'CCW'}

### [12:08:12] Revealed #7 Bard
Info: {'corruption_distance': 1}

### [12:08:12] Revealed #8 Confessor
Info: {'dizzy': False}

### [12:08:12] Revealed #9 Baker
Info: {'original_role': 'Bard'}

#### [12:08:18] Solver Output
Scenarios: 20/6486
Definite good: ['#8']
Evil probabilities: #9=75%, #2=60%, #6=40%, #1=30%, #4=30%, #5=30%, #7=20%, #3=15%

#### [12:08:18] Recommendation
Action: **EXECUTE** #9
Reason: Execution lookahead: #9 guarantees a win across all reveal branches with current HP budget (45% evil Baa, 30% evil Chancellor, 15% good Baker (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [12:08:25] Executed #9 -> Chancellor (EVIL)

#### [12:08:30] Solver Output
Scenarios: 6/505
Definite evil: ['#9']
Definite good: ['#8']
Evil probabilities: #2=50%, #6=50%, #5=33%, #1=17%, #3=17%, #4=17%, #7=17%

#### [12:08:30] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (33% evil Baa, 33% good Baker (corrupted), 17% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [12:08:37] Executed #2 -> GOOD (WRONG!)

### [12:08:44] Executed #2 -> GOOD (WRONG!)

#### [12:08:44] Solver Output
Scenarios: 3/395
Definite evil: ['#9']
Definite good: ['#2', '#8']
Evil probabilities: #1=33%, #3=33%, #4=33%, #5=33%, #6=33%, #7=33%

#### [12:08:44] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (67% good Drunk (corrupted), 33% evil Baa).
WARNING: Execution lookahead override -- immediate hit chance is 33%, but all reveal branches still lead to a forced win.

### [12:08:51] Executed #7 -> GOOD (WRONG!)

### [12:09:29] Executed #7 -> GOOD (WRONG!)

#### [12:09:29] Solver Output
Scenarios: 2/278
Definite evil: ['#9']
Definite good: ['#2', '#6', '#7', '#8']
Evil probabilities: #1=50%, #3=50%, #4=50%, #5=50%

#### [12:09:29] Recommendation
Action: **EXECUTE** #5
Reason: Knight free check: #5 is 50% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [12:10:08] Executed #5 -> Baa (EVIL)

#### [12:10:08] Solver Output
Scenarios: 1/37
Definite evil: ['#3', '#5', '#9']
Definite good: ['#1', '#2', '#4', '#6', '#7', '#8']

#### [12:10:08] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [12:10:15] Executed #3 -> Poisoner (EVIL)

## [12:10:23] GAME OVER — WIN
Final HP: 3
Notes: 3HP, Knight free check saved the day, 2 wrong execs (corrupted Baker + Drunk)


---

# New Game — 2026-04-12 12:16:19
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Scout, Architect, Knight, Medium
- Outcasts: Drunk, Bombardier
- Minions: Twin_Minion
- Demons: Baa

### [12:16:49] Revealed #1 Scout
Info: {'evil_role': 'Twin Minion', 'distance': 1}

### [12:16:49] Revealed #2 Scout
Info: {'evil_role': 'Twin Minion', 'distance': 2}

### [12:16:49] Revealed #3 Medium
Info: {'good_position': 2, 'good_role': 'Scout'}

### [12:16:49] Revealed #4 Architect
Info: {'side': 'Right'}

### [12:16:49] Revealed #5 Knitter
Info: {'evil_pairs': 1}

### [12:16:49] Revealed #6 Knight
Info: {}

### [12:16:49] Revealed #7 Bombardier
Info: {}

#### [12:16:54] Solver Output
Scenarios: 6/222
Definite evil: ['#3']
Definite good: ['#1', '#5', '#6', '#7']
Evil probabilities: #2=67%, #4=33%

#### [12:16:54] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 6 scenarios (roles: {'Baa', 'Twin_Minion'})

### [12:17:01] Executed #3 -> Baa (EVIL)

#### [12:17:07] Solver Output
Scenarios: 3/31
Definite evil: ['#3']
Definite good: ['#1', '#5', '#6', '#7']
Evil probabilities: #2=67%, #4=33%

#### [12:17:07] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (67% evil Twin_Minion, 33% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [12:17:13] Executed #2 -> Twin Minion (EVIL)

## [12:17:20] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, both evils definite/forced-safe


---

# New Game — 2026-04-12 12:18:27
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Gemcrafter, Confessor, Alchemist, Lover, Knight
- Outcasts: Bombardier
- Minions: Poisoner
- Demons: Pooka

### [12:18:40] Revealed #1 Gemcrafter
Info: {'good_position': 3}

### [12:18:40] Revealed #2 Confessor
Info: {'dizzy': True}

### [12:18:40] Revealed #3 Knight
Info: {}

### [12:18:40] Revealed #4 Hunter
Info: {'distance': 2}

### [12:18:40] Revealed #5 Bombardier
Info: {}

### [12:18:40] Revealed #6 Lover
Info: {'evil_adjacent': 0}

### [12:18:40] Revealed #7 Knight
Info: {}

### [12:18:40] Revealed #8 Alchemist
Info: {'cured_count': 2}

#### [12:18:40] Solver Output
Scenarios: 4/83
Definite good: ['#4', '#5', '#6']
Evil probabilities: #3=75%, #1=50%, #2=25%, #7=25%, #8=25%

#### [12:18:40] Recommendation
Action: **EXECUTE** #3
Reason: Knight check: #3 is 75% evil, 25% corruption risk. Expected HP cost: 0.6 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 25% -- corrupted Knight loses immunity + 4 extra damage

### [12:19:19] Executed #3 -> Poisoner (EVIL)

#### [12:19:20] Solver Output
Scenarios: 1/11
Definite evil: ['#1', '#3']
Definite good: ['#2', '#4', '#5', '#6', '#7', '#8']

#### [12:19:20] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [12:19:26] Executed #1 -> Pooka (EVIL)

## [12:19:34] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect


---

# New Game — 2026-04-12 12:20:36
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Hunter, Bishop, Scout, Knight, Judge
- Outcasts: Wretch
- Minions: Minion, Witch
- Demons: Lilis

### [12:21:20] Revealed #1 Alchemist
Info: {'cured_count': 0}

### [12:21:20] Revealed #2 Bishop
Info: {'targets': [6, 1, 8], 'types': ['Villager', 'Outcast', 'Minion']}

### [12:21:20] Revealed #3 Hunter
Info: {'distance': 1}

### [12:21:20] Revealed #4 Knight
Info: {}

### [12:21:20] Revealed #5 Wretch
Info: {}

### [12:21:20] Revealed #6 Bishop
Info: {'targets': [6, 5], 'types': ['Villager', 'Minion']}

### [12:21:20] Revealed #7 Knight
Info: {}

#### [12:21:26] Solver Output
Scenarios: 62/504
Definite good: ['#1', '#8']
Evil probabilities: #4=65%, #2=61%, #6=58%, #7=55%, #9=29%, #5=23%, #3=10%

#### [12:21:26] Recommendation
Action: **EXECUTE** #4
Reason: Knight free check: #4 is 65% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [12:22:18] Executed #4 -> Minion (EVIL)

#### [12:22:19] Solver Output
Scenarios: 13/56
Definite evil: ['#4']
Definite good: ['#1', '#3', '#8']
Evil probabilities: #6=62%, #2=54%, #7=31%, #9=31%, #5=23%

#### [12:22:19] Recommendation
Action: **EXECUTE** #7
Reason: Knight free check: #7 is 31% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [12:22:58] Executed #7 -> Lilis (EVIL)

#### [12:22:58] Solver Output
Scenarios: 2/7
Definite evil: ['#4', '#7']
Definite good: ['#1', '#3', '#5', '#8', '#9']
Evil probabilities: #2=50%, #6=50%

#### [12:22:58] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (50% good Bishop, 50% evil Witch).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [12:23:05] Executed #2 -> Witch (EVIL)

## [12:23:13] GAME OVER — WIN
Final HP: 8
Notes: 8HP, 2 Knight free checks + forced-safe lookahead


---

# New Game — 2026-04-12 12:24:24
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Fortune_Teller, Dreamer, Empress, Architect, Scout, Poet
- Outcasts: Doppelganger, Wretch
- Minions: Twin_Minion, Puppeteer
- Demons: Baa

### [12:24:38] Revealed #1 Knight
Info: {}

### [12:24:38] Revealed #2 Architect
Info: {'side': 'Left'}

### [12:24:38] Revealed #4 Scout
Info: {'evil_role': 'Puppet', 'distance': 2}

### [12:24:38] Revealed #5 Empress
Info: {'targets': [1, 2, 3]}

### [12:24:38] Revealed #6 Knight
Info: {}

### [12:24:38] Revealed #7 Scout
Info: {'evil_role': 'Puppeteer', 'distance': 3}

### [12:24:38] Revealed #8 Poet
Info: {'direction': 'CCW', 'copied_role': 'Enlightened'}

#### [12:24:38] Solver Output
Scenarios: 66/4536
Definite evil: ['#4', '#7']
Definite good: ['#1', '#3', '#9']
Evil probabilities: #5=67%, #6=55%, #2=48%, #8=30%

#### [12:24:38] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 66 scenarios (roles: {'Puppeteer', 'Twin_Minion', 'Baa'})

### [12:24:45] Executed #4 -> Baa (EVIL)

### [12:24:57] Revealed #3 Fortune_Teller
Info: {}

### [12:24:57] Revealed #9 Dreamer
Info: {}

#### [12:24:57] Solver Output
Scenarios: 28/504
Definite evil: ['#4', '#7']
Definite good: ['#1', '#3', '#9']
Evil probabilities: #6=64%, #5=61%, #2=39%, #8=36%

#### [12:24:57] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 28 scenarios (roles: {'Twin_Minion', 'Puppeteer'})

### [12:25:04] Executed #7 -> Puppeteer (EVIL)

#### [12:25:10] Solver Output
Scenarios: 22/72
Definite evil: ['#4', '#7']
Definite good: ['#1', '#3', '#9']
Evil probabilities: #6=55%, #2=50%, #5=50%, #8=45%

#### [12:25:10] Recommendation
Action: **EXECUTE** #6
Reason: Knight free check: #6 is 55% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

#### [12:26:40] Execution Blocked
#6 Knight immunity — confirmed good, no HP loss

#### [12:26:46] Solver Output
Scenarios: 10/30
Definite evil: ['#4', '#7', '#8']
Definite good: ['#1', '#3', '#6', '#9']
Evil probabilities: #2=50%, #5=50%

#### [12:26:46] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 10 scenarios (roles: {'Puppet'})

### [12:26:53] Executed #8 -> Puppet (EVIL)

#### [12:27:00] Solver Output
Scenarios: 10/30
Definite evil: ['#4', '#7', '#8']
Definite good: ['#1', '#3', '#6', '#9']
Evil probabilities: #2=50%, #5=50%

#### [12:27:00] Recommendation
Action: **USE_ABILITY** #3 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [12:27:32] Revealed #3 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': True}

### [12:27:32] Ability used at #3

#### [12:27:32] Solver Output
Scenarios: 5/30
Definite evil: ['#2', '#4', '#7', '#8']
Definite good: ['#1', '#3', '#5', '#6', '#9']

#### [12:27:32] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 5 scenarios (roles: {'Twin_Minion'})

### [12:27:39] Executed #2 -> Twin Minion (EVIL)

## [12:27:49] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Knight immunity check + FT ability


---

# New Game — 2026-04-12 12:28:46
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Empress, Fortune_Teller, Slayer, Knight, Confessor, Poet
- Outcasts: Doppelganger
- Minions: Minion
- Demons: Pooka

### [12:29:00] Revealed #2 Confessor
Info: {'dizzy': True}

### [12:29:00] Revealed #5 Knight
Info: {}

### [12:29:00] Revealed #6 Confessor
Info: {'dizzy': False}

### [12:29:00] Revealed #7 Knitter
Info: {'evil_pairs': 1}

### [12:29:00] Revealed #8 Empress
Info: {'targets': [4, 5, 6]}

#### [12:29:00] Solver Output
Scenarios: 25/336
Definite good: ['#5', '#6']
Evil probabilities: #1=56%, #2=44%, #8=40%, #3=20%, #4=20%, #7=20%

#### [12:29:00] Recommendation
Action: **REVEAL** #1
Reason: #1: 56% evil, 2.711 bits (7 outcomes)

### [12:29:17] Revealed #1 Poet
Info: {'good_position': 8, 'copied_role': 'Gemcrafter'}

### [12:29:18] Revealed #3 Fortune_Teller
Info: {}

### [12:29:18] Revealed #4 Slayer
Info: {}

#### [12:29:18] Solver Output
Scenarios: 15/336
Definite good: ['#5', '#6']
Evil probabilities: #8=60%, #1=33%, #2=33%, #3=33%, #4=33%, #7=7%

#### [12:29:18] Recommendation
Action: **USE_ABILITY** #3 (Fortune Teller) -> targets ['#4', '#8']
Reason: Entropy 0.971 (adjusted 0.971) | follow-up bonus 0.178 | timing x1.00

### [12:29:47] Revealed #3 Fortune Teller
Info: {'targets': [4, 8], 'has_evil': True}

### [12:29:47] Ability used at #3

#### [12:29:47] Solver Output
Scenarios: 9/336
Definite evil: ['#8']
Definite good: ['#3', '#4', '#5', '#6', '#7']
Evil probabilities: #1=56%, #2=44%

#### [12:29:47] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 9 scenarios (roles: {'Pooka', 'Minion'})

### [12:29:54] Executed #8 -> Minion (EVIL)

#### [12:30:01] Solver Output
Scenarios: 5/42
Definite evil: ['#1', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7']

#### [12:30:01] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 5 scenarios (roles: {'Pooka'})

### [12:30:08] Executed #1 -> Pooka (EVIL)

## [12:30:18] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, FT ability confirmed evils


---

# New Game — 2026-04-12 12:31:14
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Medium, Enlightened, Druid, Jester, Fortune_Teller, Lover, Slayer
- Outcasts: Plague_Doctor, Bombardier
- Minions: Chancellor, Twin_Minion
- Demons: Lilis

### [12:32:07] Revealed #1 Enlightened
Info: {'direction': 'CW'}

### [12:32:07] Revealed #3 Lover
Info: {'evil_adjacent': 1}

### [12:32:07] Revealed #6 Bombardier
Info: {}

### [12:32:07] Revealed #7 Jester
Info: {}

### [12:32:17] Revealed #2 Fortune_Teller
Info: {}

### [12:32:17] Revealed #4 Jester
Info: {}

### [12:32:17] Revealed #5 Slayer
Info: {}

### [12:32:17] Revealed #8 Plague_Doctor
Info: {}

#### [12:32:17] Solver Output
Scenarios: 222/3400
Definite good: ['#9', '#10']
Evil probabilities: #7=77%, #5=44%, #4=43%, #2=42%, #6=37%, #3=36%, #8=11%, #1=10%

#### [12:32:18] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.663 (adjusted 1.663) | timing x1.00

### [12:32:58] Ability used at #8

#### [12:32:58] Solver Output
Scenarios: 154/3400
Definite good: ['#1', '#8', '#9', '#10']
Evil probabilities: #7=84%, #5=53%, #3=47%, #6=42%, #4=38%, #2=36%

#### [12:32:58] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#4', '#5']
Reason: Entropy 0.997 (adjusted 0.887) | follow-up bonus 0.341 | timing x1.00
WARNING: Corruption risk: 22%

### [12:33:33] Revealed #2 Fortune Teller
Info: {'targets': [4, 5], 'has_evil': True}

### [12:33:33] Ability used at #2

#### [12:33:33] Solver Output
Scenarios: 72/3400
Definite good: ['#1', '#8', '#9', '#10']
Evil probabilities: #7=94%, #5=56%, #6=50%, #3=44%, #4=33%, #2=22%

#### [12:33:33] Recommendation
Action: **USE_ABILITY** #4 (Jester) -> targets ['#1', '#2', '#6']
Reason: Expected posterior 38.1 scenarios (adjusted 42.3, info gain 0.766 bits) | timing x1.00
WARNING: Corruption risk: 22%

### [12:34:03] Revealed #4 Jester
Info: {'targets': [1, 2, 6], 'evil_count': 1}

### [12:34:03] Ability used at #4

#### [12:34:04] Solver Output
Scenarios: 36/3400
Definite evil: ['#7']
Definite good: ['#1', '#8', '#9', '#10']
Evil probabilities: #5=56%, #3=44%, #6=44%, #4=33%, #2=22%

#### [12:34:04] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 36 scenarios (roles: {'Chancellor', 'Twin_Minion', 'Lilis'})

### [12:34:12] Ability used at #7

#### [12:34:13] Solver Output
Scenarios: 36/3400
Definite evil: ['#7']
Definite good: ['#1', '#8', '#9', '#10']
Evil probabilities: #5=56%, #3=44%, #6=44%, #4=33%, #2=22%

#### [12:34:13] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 36 scenarios (roles: {'Chancellor', 'Twin_Minion', 'Lilis'})

### [12:35:51] Executed #7 -> Chancellor (EVIL)

#### [12:35:51] Solver Output
Scenarios: 10/286
Definite evil: ['#7']
Definite good: ['#1', '#8', '#9', '#10']
Evil probabilities: #5=60%, #3=40%, #4=40%, #6=40%, #2=20%

#### [12:35:51] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#3']
Reason: Target #3 is 40% evil (adjusted 0.40)

### [12:36:24] Ability used at #5

#### [12:36:24] Solver Output
Scenarios: 8/286
Definite evil: ['#7']
Definite good: ['#1', '#8', '#9', '#10']
Evil probabilities: #5=75%, #4=50%, #2=25%, #3=25%, #6=25%

#### [12:36:24] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (38% evil Lilis, 38% evil Twin_Minion, 25% good Slayer).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [12:36:31] Executed #5 -> Twin Minion (EVIL)

#### [12:36:37] Solver Output
Scenarios: 3/35
Definite evil: ['#5', '#7']
Definite good: ['#1', '#2', '#6', '#8', '#9', '#10']
Evil probabilities: #4=67%, #3=33%

#### [12:36:37] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (67% evil Lilis, 33% good Jester (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [12:36:44] Executed #4 -> Lilis (EVIL)

## [12:36:55] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, PD+FT+Jester+Slayer abilities


---

# New Game — 2026-04-12 12:37:49
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Lover, Dreamer, Medium, Bard, Oracle
- Outcasts: Wretch, Bombardier
- Minions: Shaman
- Demons: Pooka

### [12:38:02] Revealed #1 Bard
Info: {'corruption_distance': -1}

### [12:38:02] Revealed #2 Lover
Info: {'evil_adjacent': 0}

### [12:38:02] Revealed #3 Bombardier
Info: {}

### [12:38:02] Revealed #4 Knight
Info: {}

### [12:38:02] Revealed #5 Wretch
Info: {}

### [12:38:02] Revealed #7 Oracle
Info: {'targets': [1, 3], 'minion_role': 'Shaman'}

### [12:38:02] Revealed #8 Knight
Info: {}

### [12:38:02] Revealed #9 Medium
Info: {'good_position': 2, 'good_role': 'Lover'}

### [12:38:11] Revealed #6 Dreamer
Info: {}

#### [12:38:11] Solver Output
Scenarios: 2/72
Definite good: ['#2', '#4', '#5', '#8', '#9']
Evil probabilities: #1=50%, #3=50%, #6=50%, #7=50%

#### [12:38:11] Recommendation
Action: **USE_ABILITY** #6 (Dreamer) -> targets ['#1']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [12:38:43] Revealed #6 Dreamer
Info: {'target': 1, 'evil_role': 'Shaman'}

### [12:38:44] Ability used at #6

#### [12:38:44] Solver Output
Scenarios: 2/72
Definite good: ['#2', '#4', '#5', '#8', '#9']
Evil probabilities: #1=50%, #3=50%, #6=50%, #7=50%

#### [12:38:44] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% good Bard, 50% evil Shaman).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [12:38:51] Executed #1 -> GOOD (WRONG!)

### [12:39:00] Executed #1 -> GOOD (WRONG!)

#### [12:39:00] Solver Output
Scenarios: 1/56
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8', '#9']

#### [12:39:00] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [12:39:07] Executed #6 -> Pooka (EVIL)

#### [12:39:13] Solver Output
Scenarios: 1/7
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#8', '#9']

#### [12:39:13] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [12:39:20] Executed #7 -> Shaman (EVIL)

## [12:39:32] GAME OVER — WIN
Final HP: 5
Notes: 5HP, Dreamer ability + forced-safe lookahead, ASC60 COMPLETE 7/7


---

# New Game — 2026-04-12 13:43:25
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Confessor, Empress, Scout, Hunter
- Outcasts: Doppelganger, Wretch
- Minions: Witch
- Demons: Baa

### [13:44:30] Revealed #2 Wretch
Info: {}

### [13:44:30] Revealed #3 Hunter
Info: {'distance': 3}

### [13:44:30] Revealed #4 Confessor
Info: {'dizzy': False}

### [13:44:30] Revealed #5 Bard
Info: {'corruption_distance': 1}

### [13:44:30] Revealed #6 Empress
Info: {'targets': [3, 4, 7]}

### [13:44:30] Revealed #7 Bard
Info: {'corruption_distance': -1}

#### [13:44:30] Solver Output
Scenarios: 10/222
Definite evil: ['#3', '#5']
Definite good: ['#1', '#2', '#4', '#6', '#7']

#### [13:44:30] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 10 scenarios (roles: {'Baa', 'Witch'})

### [13:44:37] Executed #3 -> Baa (EVIL)

#### [13:44:47] Solver Output
Scenarios: 5/31
Definite evil: ['#3', '#5']
Definite good: ['#1', '#2', '#4', '#6', '#7']

#### [13:44:47] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 5 scenarios (roles: {'Witch'})

### [13:44:54] Executed #5 -> Witch (EVIL)

## [13:45:04] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, both definite


---

# New Game — 2026-04-12 13:45:58
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Judge, Slayer, Oracle, Confessor, Poet, Bard
- Outcasts: Doppelganger
- Minions: Poisoner
- Demons: Lilis

### [13:46:36] Revealed #2 Bard
Info: {'corruption_distance': 2}

### [13:46:36] Revealed #4 Oracle
Info: {'targets': [2, 8], 'minion_role': 'Poisoner'}

### [13:46:36] Revealed #6 Confessor
Info: {'dizzy': True}

### [13:47:00] Revealed #1 Poet
Info: {'targets': [1, 2, 3], 'types': ['Villager', 'Outcast', 'Minion'], 'copied_role': 'Bishop'}

### [13:47:00] Revealed #3 Slayer
Info: {}

### [13:47:00] Revealed #5 Judge
Info: {}

### [13:47:00] Revealed #7 Slayer
Info: {}

#### [13:47:01] Solver Output
Scenarios: 9/700
Definite evil: ['#6']
Definite good: ['#1', '#4', '#5', '#7', '#8']
Evil probabilities: #2=78%, #3=22%

#### [13:47:01] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 9 scenarios (roles: {'Lilis'})

### [13:47:07] Executed #6 -> Lilis (EVIL)

#### [13:47:14] Solver Output
Scenarios: 9/84
Definite evil: ['#6']
Definite good: ['#1', '#4', '#5', '#7', '#8']
Evil probabilities: #2=78%, #3=22%

#### [13:47:14] Recommendation
Action: **USE_ABILITY** #5 (Judge) -> targets ['#3']
Reason: Expected posterior 5.0 scenarios (adjusted 5.0, info gain 0.848 bits) | timing x1.00

### [13:47:43] Revealed #5 Judge
Info: {'target': 3, 'is_lying': False}

### [13:47:43] Ability used at #5

#### [13:47:43] Solver Output
Scenarios: 6/84
Definite evil: ['#2', '#6']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [13:47:43] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 6 scenarios (roles: {'Poisoner'})

### [13:47:50] Executed #2 -> Poisoner (EVIL)

## [13:48:02] GAME OVER — WIN
Final HP: 8
Notes: 8HP, Lilis game, Judge + Poet-Bishop abilities


---

# New Game — 2026-04-12 13:49:00
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Alchemist, Knight, Hunter, Scout, Slayer, Bard
- Outcasts: Wretch
- Minions: Shaman
- Demons: Pooka

### [13:49:13] Revealed #1 Scout
Info: {'evil_role': 'Shaman', 'distance': 1}

### [13:49:13] Revealed #2 Poet
Info: {'distance': 2, 'copied_role': 'Hunter'}

### [13:49:13] Revealed #4 Wretch
Info: {}

### [13:49:13] Revealed #5 Poet
Info: {'evil_pairs': 1, 'copied_role': 'Knitter'}

### [13:49:13] Revealed #6 Bard
Info: {'corruption_distance': 3}

### [13:49:13] Revealed #7 Hunter
Info: {'distance': 2}

### [13:49:13] Revealed #8 Alchemist
Info: {'cured_count': 2}

### [13:49:20] Revealed #3 Slayer
Info: {}

#### [13:49:20] Solver Output
Scenarios: 2/56
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#4', '#5']
Evil probabilities: #6=50%, #8=50%

#### [13:49:20] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [13:49:27] Executed #7 -> Pooka (EVIL)

#### [13:49:34] Solver Output
Scenarios: 2/7
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#4', '#5']
Evil probabilities: #6=50%, #8=50%

#### [13:49:34] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#6']
Reason: Target #6 is 50% evil (adjusted 0.50)

### [13:50:21] Ability used at #3

## [13:50:21] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Slayer killed Shaman


---

# New Game — 2026-04-12 13:51:23
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Architect, Baker, Confessor, Judge, Empress
- Outcasts: Plague_Doctor, Wretch
- Minions: Puppeteer
- Demons: Pooka

### [13:51:38] Revealed #1 Architect
Info: {'side': 'Left'}

### [13:51:38] Revealed #2 Wretch
Info: {}

### [13:51:38] Revealed #3 Empress
Info: {'targets': [4, 6, 8]}

### [13:51:38] Revealed #5 Baker
Info: {'original_role': 'Architect'}

### [13:51:38] Revealed #7 Confessor
Info: {'dizzy': True}

### [13:51:38] Revealed #8 Baker
Info: {'original_role': 'Slayer'}

### [13:51:46] Revealed #4 Plague_Doctor
Info: {}

### [13:51:46] Revealed #6 Slayer
Info: {}

### [13:51:46] Revealed #9 Judge
Info: {}

#### [13:51:46] Solver Output
Scenarios: 24/374
Definite good: ['#2', '#3', '#4']
Evil probabilities: #8=71%, #7=67%, #5=58%, #6=54%, #9=33%, #1=17%

#### [13:51:46] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#5']
Reason: Entropy 1.749 (adjusted 1.749) | timing x1.00

### [13:52:24] Ability used at #4

#### [13:52:24] Solver Output
Scenarios: 14/374
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4']
Evil probabilities: #7=71%, #6=57%, #8=57%, #9=14%

#### [13:52:24] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 14 scenarios (roles: {'Pooka', 'Puppet', 'Puppeteer'})

### [13:52:31] Executed #5 -> Pooka (EVIL)

#### [13:52:37] Solver Output
Scenarios: 8/49
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4']
Evil probabilities: #7=75%, #8=75%, #6=25%, #9=25%

#### [13:52:37] Recommendation
Action: **USE_ABILITY** #9 (Judge) -> targets ['#6']
Reason: Expected posterior 5.2 scenarios (adjusted 5.9, info gain 0.452 bits) | timing x1.00
WARNING: Corruption risk: 25% -- corrupted Judge results are unreliable

### [13:53:06] Revealed #9 Judge
Info: {'target': 6, 'is_lying': False}

### [13:53:06] Ability used at #9

#### [13:53:07] Solver Output
Scenarios: 4/49
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4']
Evil probabilities: #7=75%, #8=75%, #6=25%, #9=25%

#### [13:53:07] Recommendation
Action: **USE_ABILITY** #6 (Slayer) -> targets ['#7']
Reason: Target #7 is 75% evil (adjusted 0.19)
WARNING: Corruption risk: 75% -- Slayer ability disabled if corrupted

### [13:53:43] Ability used at #6

#### [13:53:44] Solver Output
Scenarios: 3/49
Definite evil: ['#5', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#6']
Evil probabilities: #7=67%, #9=33%

#### [13:53:44] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 3 scenarios (roles: {'Puppeteer', 'Puppet'})

### [13:53:50] Executed #8 -> Puppeteer (EVIL)

#### [13:53:56] Solver Output
Scenarios: 1/8
Definite evil: ['#5', '#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#9']

#### [13:53:56] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [13:54:03] Executed #7 -> Puppet (EVIL)

## [13:54:16] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, PD+Judge+Slayer abilities, all definite


---

# New Game — 2026-04-12 13:55:15
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Bishop, Scout, Empress, Alchemist, Enlightened, Knight, Druid
- Outcasts: Bombardier
- Minions: Poisoner, Twin_Minion
- Demons: Lilis

### [13:56:04] Revealed #1 Empress
Info: {'targets': [5, 6, 7]}

### [13:56:04] Revealed #2 Scout
Info: {'evil_role': 'Lilis', 'distance': 2}

### [13:56:04] Revealed #3 Alchemist
Info: {'cured_count': 1}

### [13:56:04] Revealed #4 Knight
Info: {}

### [13:56:04] Revealed #6 Bishop
Info: {'targets': [4, 10, 9], 'types': ['Villager', 'Outcast', 'Minion']}

### [13:56:04] Revealed #7 Knight
Info: {}

### [13:56:04] Revealed #9 Bombardier
Info: {}

### [13:56:12] Revealed #5 Druid
Info: {}

#### [13:56:12] Solver Output
Scenarios: 49/1096
Definite good: ['#8', '#10']
Evil probabilities: #4=82%, #3=47%, #7=39%, #9=39%, #2=37%, #1=31%, #5=14%, #6=12%

#### [13:56:12] Recommendation
Action: **USE_ABILITY** #5 (Druid) -> targets ['#1', '#2', '#9']
Reason: Entropy 1.503 (adjusted 1.472) | timing x1.00
WARNING: Corruption risk: 4%

### [13:56:50] Revealed #5 Druid
Info: {'targets': [1, 2, 9], 'found_outcast': 'Bombardier'}

### [13:56:50] Ability used at #5

#### [13:56:51] Solver Output
Scenarios: 21/1096
Definite good: ['#5', '#8', '#9', '#10']
Evil probabilities: #4=90%, #2=62%, #3=57%, #1=43%, #7=38%, #6=10%

#### [13:56:51] Recommendation
Action: **EXECUTE** #4
Reason: Knight free check: #4 is 90% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [13:57:31] Executed #4 -> Poisoner (EVIL)

#### [13:57:31] Solver Output
Scenarios: 6/114
Definite evil: ['#2', '#4']
Definite good: ['#3', '#5', '#6', '#8', '#9', '#10']
Evil probabilities: #1=67%, #7=33%

#### [13:57:31] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 6 scenarios (roles: {'Lilis', 'Twin_Minion'})

### [13:57:38] Executed #2 -> Twin Minion (EVIL)

#### [13:57:44] Solver Output
Scenarios: 4/14
Definite evil: ['#2', '#4']
Definite good: ['#3', '#5', '#6', '#8', '#9', '#10']
Evil probabilities: #1=50%, #7=50%

#### [13:57:44] Recommendation
Action: **EXECUTE** #7
Reason: Knight free check: #7 is 50% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

### [13:58:28] Executed #7 -> Lilis (EVIL)

## [13:58:28] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, Knight check + Druid ability


---

# New Game — 2026-04-12 14:00:23
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Enlightened, Dreamer, Oracle, Baker, Jester, Judge, Slayer
- Outcasts: Doppelganger
- Minions: Puppeteer, Twin_Minion
- Demons: Lilis

### [14:01:09] Revealed #5 Oracle
Info: {'targets': [8, 10], 'minion_role': 'Twin_Minion'}

### [14:01:09] Revealed #7 Baker
Info: {'original_role': 'original'}

### [14:01:09] Revealed #8 Baker
Info: {'original_role': 'Lover'}

### [14:01:09] Revealed #9 Baker
Info: {'original_role': 'Judge'}

### [14:01:24] Revealed #1 Enlightened
Info: {'direction': 'cw'}

### [14:01:24] Revealed #2 Dreamer
Info: {}

### [14:01:25] Revealed #3 Slayer
Info: {}

### [14:01:25] Revealed #4 Jester
Info: {}

#### [14:01:25] Solver Output
Scenarios: 216/6720
Definite good: ['#6', '#7', '#10']
Evil probabilities: #2=89%, #5=75%, #3=67%, #4=67%, #9=36%, #1=33%, #8=33%

#### [14:01:25] Recommendation
Action: **USE_ABILITY** #2 (Dreamer) -> targets ['#4']
Reason: Entropy 3.033 (adjusted 3.033) | timing x1.00

### [14:01:55] Revealed #2 Dreamer
Info: {'target': 4, 'evil_role': 'Twin_Minion'}

### [14:01:55] Ability used at #2

#### [14:01:56] Solver Output
Scenarios: 168/6720
Definite evil: ['#2']
Definite good: ['#6', '#7', '#10']
Evil probabilities: #3=71%, #5=71%, #4=57%, #9=39%, #8=32%, #1=29%

#### [14:01:56] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 168 scenarios (roles: {'Puppet', 'Puppeteer', 'Lilis', 'Twin_Minion'})

### [14:02:03] Executed #2 -> Puppet (EVIL)

#### [14:02:11] Solver Output
Scenarios: 24/672
Definite evil: ['#2', '#3']
Definite good: ['#1', '#6', '#7', '#10']
Evil probabilities: #5=75%, #9=75%, #4=25%, #8=25%

#### [14:02:11] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 24 scenarios (roles: {'Puppeteer'})

### [14:02:35] Ability used at #3

### [14:03:09] Executed #3 -> Puppeteer (EVIL)

#### [14:03:09] Solver Output
Scenarios: 24/336
Definite evil: ['#2', '#3']
Definite good: ['#1', '#6', '#7', '#10']
Evil probabilities: #5=75%, #9=75%, #4=25%, #8=25%

#### [14:03:09] Recommendation
Action: **USE_ABILITY** #4 (Jester) -> targets ['#1', '#5', '#6']
Reason: Expected posterior 10.0 scenarios (adjusted 10.0, info gain 1.263 bits) | timing x1.00

### [14:03:47] Revealed #4 Jester
Info: {'targets': [1, 5, 6], 'evil_count': 1}

### [14:03:47] Ability used at #4

#### [14:03:48] Solver Output
Scenarios: 12/336
Definite evil: ['#2', '#3', '#5', '#9']
Definite good: ['#1', '#4', '#6', '#7', '#8', '#10']

#### [14:03:48] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 12 scenarios (roles: {'Lilis', 'Twin_Minion'})

### [14:03:55] Executed #5 -> GOOD (WRONG!)

### [14:04:08] Executed #5 -> GOOD (WRONG!)

#### [14:04:08] Solver Output
Scenarios: 0/252

#### [14:04:08] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [14:05:04] Executed #1 -> Lilis (EVIL)

## [14:05:04] GAME OVER — WIN
Final HP: 1
Notes: 1HP! Solver 0-scenario bug on #5 (Doppelganger as Oracle wrongly marked evil). Investigate.


---

# New Game — 2026-04-12 14:06:01
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Baker, Medium, Lover, Enlightened, Slayer
- Outcasts: Doppelganger
- Minions: Witch
- Demons: Lilis

### [14:06:44] Revealed #1 Lover
Info: {'evil_adjacent': 2}

### [14:06:44] Revealed #4 Enlightened
Info: {'direction': 'CCW'}

### [14:06:44] Revealed #5 Baker
Info: {'original_role': 'original'}

### [14:06:44] Revealed #6 Baker
Info: {'original_role': 'original'}

### [14:06:52] Revealed #2 Enlightened
Info: {'direction': 'equidistant'}

### [14:06:52] Revealed #3 Slayer
Info: {}

#### [14:06:52] Solver Output
Scenarios: 24/336
Definite evil: ['#1']
Definite good: ['#4', '#5', '#6', '#7', '#8']
Evil probabilities: #2=50%, #3=50%

#### [14:06:52] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 24 scenarios (roles: {'Witch', 'Lilis'})

### [14:06:59] Executed #1 -> Lilis (EVIL)

#### [14:07:06] Solver Output
Scenarios: 12/42
Definite evil: ['#1']
Definite good: ['#4', '#5', '#6', '#7', '#8']
Evil probabilities: #2=50%, #3=50%

#### [14:07:06] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#2']
Reason: Target #2 is 50% evil (adjusted 0.50)

### [14:07:49] Ability used at #3

## [14:07:49] GAME OVER — WIN
Final HP: 8
Notes: 8HP, Slayer kills Witch! ASC61 COMPLETE 7/7


---

# New Game — 2026-04-12 14:15:07
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Alchemist, Slayer, Gemcrafter, Hunter, Bard
- Outcasts: Plague_Doctor
- Minions: Shaman
- Demons: Lilis

### [14:17:20] Revealed #2 Alchemist
Info: {'cured_count': 1}

### [14:17:20] Revealed #4 Bard
Info: {'corruption_distance': -1}

### [14:17:20] Revealed #6 Gemcrafter
Info: {'good_position': 8}

### [14:17:20] Revealed #7 Alchemist
Info: {'cured_count': 0}

### [14:17:20] Revealed #8 Hunter
Info: {'distance': 4}

### [14:17:28] Revealed #1 Plague_Doctor
Info: {}

### [14:17:28] Revealed #3 Slayer
Info: {}

#### [14:17:29] Solver Output
Scenarios: 4/230
Definite evil: ['#6', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#7']

#### [14:17:29] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 4 scenarios (roles: {'Shaman', 'Lilis'})

### [14:17:35] Executed #6 -> Shaman (EVIL)

#### [14:17:43] Solver Output
Scenarios: 2/31
Definite evil: ['#6', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#7']

#### [14:17:43] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 2 scenarios (roles: {'Lilis'})

### [14:17:50] Executed #8 -> Lilis (EVIL)

## [14:18:03] GAME OVER — WIN
Final HP: 8
Notes: 8HP, Lilis game, both definite


---

# New Game — 2026-04-12 14:19:04
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Knitter, Scout, Baker, Medium, Dreamer, Empress
- Outcasts: Bombardier, Drunk, Wretch
- Minions: Chancellor, Shaman
- Demons: Lilis

### [14:20:12] Revealed #1 Medium
Info: {'good_position': 2, 'good_role': 'Knitter'}

### [14:20:12] Revealed #2 Knitter
Info: {'evil_pairs': 2}

### [14:20:12] Revealed #3 Medium
Info: {'good_position': 2, 'good_role': 'Knitter'}

### [14:20:12] Revealed #6 Bombardier
Info: {}

### [14:20:12] Revealed #7 Wretch
Info: {}

### [14:20:12] Revealed #8 Baker
Info: {'original_role': 'Architect'}

### [14:20:12] Revealed #9 Empress
Info: {'targets': [2, 4, 5]}

### [14:20:20] Revealed #4 Dreamer
Info: {}

#### [14:20:21] Solver Output
Scenarios: 56/3494
Definite good: ['#5']
Evil probabilities: #7=50%, #8=50%, #1=39%, #3=39%, #4=36%, #9=36%, #6=29%, #2=21%

#### [14:20:21] Recommendation
Action: **USE_ABILITY** #4 (Dreamer) -> targets ['#8']
Reason: Entropy 2.688 (adjusted 2.592) | timing x1.00
WARNING: Corruption risk: 7%

### [14:20:57] Revealed #4 Dreamer
Info: {'target': 8, 'evil_role': 'Shaman'}

### [14:20:57] Ability used at #4

#### [14:20:58] Solver Output
Scenarios: 43/3494
Definite good: ['#5']
Evil probabilities: #1=51%, #3=51%, #7=44%, #4=37%, #8=35%, #9=33%, #2=28%, #6=21%

#### [14:20:58] Recommendation
Action: **EXECUTE** #1
Reason: No reveals available. #1 is 51% likely evil (HP=6, budget=1 wrong execs)
WARNING: Probabilistic execution -- 51% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #1 (51%) despite low confidence — Bombardier candidate(s) [6] risk instant game loss if executed first.

### [14:21:49] Executed #1 -> GOOD (WRONG!)

#### [14:21:50] Solver Output
Scenarios: 15/2346
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#5']
Evil probabilities: #6=60%, #4=53%, #7=47%, #9=40%

#### [14:21:50] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 15 scenarios (roles: {'Chancellor', 'Lilis', 'Shaman'})

### [14:22:14] Executed #8 -> Lilis (EVIL)

#### [14:22:15] Solver Output
Scenarios: 6/293
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#5']
Evil probabilities: #4=67%, #6=67%, #7=33%, #9=33%

#### [14:22:15] Recommendation
Action: **ERROR** #4
Reason: #4 is 67% likely evil but HP too low to risk (HP=1, cost=5, threshold=95%). Need more info.
WARNING: Probabilistic execution -- 67% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [14:23:01] Executed #4 -> Shaman (EVIL)

#### [14:23:01] Solver Output
Scenarios: 2/37
Definite evil: ['#4', '#6', '#8']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#9']

#### [14:23:01] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Chancellor'})

### [14:23:32] Executed #6 -> Chancellor (EVIL)

## [14:23:33] GAME OVER — WIN
Final HP: 1
Notes: 1HP! Lilis game, close call but all definite at the end


---

# New Game — 2026-04-12 14:24:21
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Hunter, Bard, Confessor, Baker, Alchemist, Fortune_Teller
- Outcasts: Wretch, Plague_Doctor
- Minions: Twin_Minion
- Demons: Baa

### [14:24:36] Revealed #1 Hunter
Info: {'distance': 1}

### [14:24:36] Revealed #3 Baker
Info: {'original_role': 'original'}

### [14:24:36] Revealed #4 Baker
Info: {'original_role': 'Poet'}

### [14:24:36] Revealed #5 Confessor
Info: {'dizzy': True}

### [14:24:36] Revealed #7 Bard
Info: {'corruption_distance': -1}

### [14:24:36] Revealed #8 Baker
Info: {'original_role': 'Alchemist'}

### [14:24:51] Revealed #2 Plague_Doctor
Info: {}

### [14:24:51] Revealed #6 Fortune_Teller
Info: {}

#### [14:24:51] Solver Output
Scenarios: 14/266
Definite good: ['#2', '#3', '#4', '#6']
Evil probabilities: #5=71%, #1=43%, #7=43%, #8=43%

#### [14:24:51] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#5']
Reason: Entropy 1.149 (adjusted 1.149) | timing x1.00

### [14:25:49] Ability used at #2

#### [14:25:49] Solver Output
Scenarios: 10/266
Definite evil: ['#5']
Definite good: ['#2', '#3', '#4', '#6']
Evil probabilities: #1=40%, #8=40%, #7=20%

#### [14:25:49] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 10 scenarios (roles: {'Twin_Minion', 'Baa'})

### [14:25:56] Executed #5 -> Baa (EVIL)

#### [14:26:06] Solver Output
Scenarios: 5/37
Definite evil: ['#5']
Definite good: ['#2', '#3', '#4', '#6']
Evil probabilities: #1=40%, #8=40%, #7=20%

#### [14:26:06] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 0.971 (adjusted 0.971) | timing x1.00

### [14:26:53] Revealed #6 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': True}

### [14:26:53] Ability used at #6

#### [14:26:53] Solver Output
Scenarios: 2/37
Definite evil: ['#1', '#5']
Definite good: ['#2', '#3', '#4', '#6', '#7', '#8']

#### [14:26:53] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Twin_Minion'})

### [14:27:00] Executed #1 -> Twin Minion (EVIL)

## [14:27:12] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, PD+FT definite


---

# New Game — 2026-04-12 14:28:02
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Alchemist, Bishop, Bard, Lover, Gemcrafter, Knight
- Outcasts: Drunk, Plague_Doctor
- Minions: Shaman
- Demons: Lilis

### [14:28:45] Revealed #1 Gemcrafter
Info: {'good_position': 9}

### [14:28:45] Revealed #3 Bishop
Info: {'targets': [7, 8, 3], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:28:45] Revealed #6 Gemcrafter
Info: {'good_position': 4}

### [14:28:45] Revealed #7 Alchemist
Info: {'cured_count': 0}

### [14:28:45] Revealed #8 Bishop
Info: {'targets': [2, 8, 6], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:28:45] Revealed #9 Bard
Info: {'corruption_distance': 2}

### [14:28:54] Revealed #2 Plague_Doctor
Info: {}

### [14:28:55] Revealed #4 Enlightened
Info: {'direction': 'ccw'}

#### [14:28:55] Solver Output
Scenarios: 6/3022
Definite good: ['#2', '#5', '#7']
Evil probabilities: #3=50%, #9=50%, #1=33%, #8=33%, #4=17%, #6=17%

#### [14:28:55] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.459 (adjusted 1.459) | timing x1.00

### [14:29:39] Ability used at #2

#### [14:29:39] Solver Output
Scenarios: 1/3022
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8', '#9']

#### [14:29:39] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [14:29:46] Executed #4 -> Lilis (EVIL)

#### [14:29:53] Solver Output
Scenarios: 1/361
Definite evil: ['#4', '#6']
Definite good: ['#1', '#2', '#3', '#5', '#7', '#8', '#9']

#### [14:29:53] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [14:30:00] Executed #6 -> Shaman (EVIL)

## [14:30:10] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, PD check nails both evils


---

# New Game — 2026-04-12 14:30:59
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Oracle, Slayer, Knight, Alchemist, Gemcrafter, Scout, Hunter, Fortune_Teller
- Outcasts: Plague_Doctor
- Minions: Witch, Minion
- Demons: Pooka

### [14:31:18] Revealed #1 Scout
Info: {'evil_role': 'Minion', 'distance': 1}

### [14:31:18] Revealed #2 Scout
Info: {'evil_role': 'Witch', 'distance': 1}

### [14:31:18] Revealed #3 Oracle
Info: {'targets': [7, 8], 'minion_role': 'Minion'}

### [14:31:18] Revealed #4 Gemcrafter
Info: {'good_position': 3}

### [14:31:18] Revealed #6 Knight
Info: {}

### [14:31:18] Revealed #7 Hunter
Info: {'distance': 1}

### [14:31:18] Revealed #9 Alchemist
Info: {'cured_count': 0}

### [14:31:26] Revealed #5 Fortune_Teller
Info: {}

### [14:31:27] Revealed #8 Slayer
Info: {}

#### [14:31:27] Solver Output
Scenarios: 8/3240
Definite evil: ['#3', '#6']
Definite good: ['#4', '#5', '#7', '#8', '#9', '#10']
Evil probabilities: #1=75%, #2=25%

#### [14:31:27] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 8 scenarios (roles: {'Pooka'})

### [14:31:34] Executed #3 -> Pooka (EVIL)

#### [14:31:41] Solver Output
Scenarios: 8/352
Definite evil: ['#3', '#6']
Definite good: ['#4', '#5', '#7', '#8', '#9', '#10']
Evil probabilities: #1=75%, #2=25%

#### [14:31:41] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 8 scenarios (roles: {'Witch', 'Minion'})

### [14:31:48] Executed #6 -> Witch (EVIL)

#### [14:31:58] Solver Output
Scenarios: 5/43
Definite evil: ['#3', '#6']
Definite good: ['#4', '#5', '#7', '#8', '#9', '#10']
Evil probabilities: #1=60%, #2=40%

#### [14:31:58] Recommendation
Action: **USE_ABILITY** #5 (Fortune Teller) -> targets ['#1', '#4']
Reason: Entropy 0.971 (adjusted 0.777) | follow-up bonus 0.800 | timing x1.00
WARNING: Corruption risk: 40%

### [14:32:49] Revealed #5 Fortune Teller
Info: {'targets': [1, 4], 'has_evil': True}

### [14:32:49] Ability used at #5

#### [14:32:49] Solver Output
Scenarios: 3/43
Definite evil: ['#3', '#6']
Definite good: ['#4', '#5', '#7', '#8', '#9', '#10']
Evil probabilities: #1=67%, #2=33%

#### [14:32:49] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#1']
Reason: Target #1 is 67% evil (adjusted 0.67)

### [14:33:59] Ability used at #8

#### [14:33:59] Solver Output
Scenarios: 1/43
Definite evil: ['#2', '#3', '#6']
Definite good: ['#1', '#4', '#5', '#7', '#8', '#9', '#10']

#### [14:33:59] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Minion'})

### [14:34:06] Executed #2 -> Minion (EVIL)

## [14:34:20] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, all definite


---

# New Game — 2026-04-12 14:35:08
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knitter, Bishop, Confessor, Hunter, Fortune_Teller, Dreamer
- Outcasts: Wretch, Plague_Doctor
- Minions: Minion
- Demons: Baa

### [14:35:23] Revealed #1 Bishop
Info: {'targets': [3, 7], 'types': ['Villager', 'Minion']}

### [14:35:23] Revealed #2 Bishop
Info: {'targets': [1, 5, 3], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:35:23] Revealed #4 Hunter
Info: {'distance': 2}

### [14:35:23] Revealed #5 Knitter
Info: {'evil_pairs': 1}

### [14:35:23] Revealed #7 Wretch
Info: {}

### [14:35:23] Revealed #8 Confessor
Info: {'dizzy': False}

### [14:35:30] Revealed #3 Dreamer
Info: {}

### [14:35:31] Revealed #6 Fortune_Teller
Info: {}

#### [14:35:31] Solver Output
Scenarios: 2/56
Definite evil: ['#2', '#6']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [14:35:31] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Baa', 'Minion'})

### [14:35:38] Executed #2 -> Minion (EVIL)

#### [14:35:46] Solver Output
Scenarios: 1/7
Definite evil: ['#2', '#6']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [14:35:46] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Baa'})

### [14:35:56] Ability used at #6

#### [14:35:57] Solver Output
Scenarios: 1/7
Definite evil: ['#2', '#6']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']

#### [14:35:57] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Baa'})

### [14:37:42] Executed #6 -> Baa (EVIL)

## [14:37:42] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, both definite


---

# New Game — 2026-04-12 14:38:33
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Judge, Lover, Hunter, Scout
- Outcasts: Plague_Doctor, Bombardier
- Minions: Shaman
- Demons: Baa

### [14:38:47] Revealed #1 Hunter
Info: {'distance': 1}

### [14:38:47] Revealed #2 Bombardier
Info: {}

### [14:38:47] Revealed #3 Knight
Info: {}

### [14:38:47] Revealed #4 Lover
Info: {'evil_adjacent': 0}

### [14:38:47] Revealed #5 Hunter
Info: {'distance': 1}

#### [14:38:47] Solver Output
Scenarios: 22/194
Evil probabilities: #6=55%, #1=36%, #2=36%, #5=27%, #7=27%, #3=9%, #4=9%

#### [14:38:47] Recommendation
Action: **EXECUTE** #3
Reason: Knight check: #3 is 9% evil, 18% corruption risk. Expected HP cost: 1.5 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 18% -- corrupted Knight loses immunity + 4 extra damage

### [14:38:58] Revealed #6 Judge
Info: {}

### [14:38:58] Revealed #7 Plague_Doctor
Info: {}

#### [14:38:58] Solver Output
Scenarios: 6/142
Definite evil: ['#6']
Definite good: ['#1', '#3', '#4', '#5']
Evil probabilities: #2=67%, #7=33%

#### [14:38:58] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 6 scenarios (roles: {'Baa', 'Shaman'})

### [14:39:24] Ability used at #6

### [14:39:40] Executed #6 -> Baa (EVIL)

#### [14:39:40] Solver Output
Scenarios: 3/22
Definite evil: ['#6']
Definite good: ['#1', '#3', '#4', '#5']
Evil probabilities: #2=67%, #7=33%

#### [14:39:40] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.585 (adjusted 1.585) | timing x1.00

### [14:40:23] Ability used at #7

#### [14:40:23] Solver Output
Scenarios: 1/22
Definite evil: ['#6', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#5']

#### [14:40:23] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [14:40:30] Executed #7 -> Shaman (EVIL)

## [14:40:44] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, PD+Judge abilities, ASC62 COMPLETE 7/7


---

# New Game — 2026-04-12 14:42:11
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Dreamer, Lover, Hunter, Knitter, Fortune_Teller, Knight, Oracle, Architect
- Outcasts: Wretch, Drunk
- Minions: Puppeteer
- Demons: Lilis

### [14:43:30] Revealed #2 Wretch
Info: {}

### [14:43:30] Revealed #3 Oracle
Info: {'targets': [2, 7], 'minion_role': 'Puppeteer'}

### [14:43:30] Revealed #4 Knight
Info: {}

### [14:43:30] Revealed #5 Architect
Info: {'side': 'Right'}

### [14:43:30] Revealed #6 Knitter
Info: {'evil_pairs': 1}

### [14:43:30] Revealed #9 Lover
Info: {'evil_adjacent': 0}

### [14:43:38] Revealed #1 Fortune_Teller
Info: {}

### [14:43:38] Revealed #7 Dreamer
Info: {}

#### [14:43:39] Solver Output
Scenarios: 42/588
Definite good: ['#8']
Evil probabilities: #6=55%, #4=52%, #2=40%, #5=40%, #3=33%, #7=31%, #1=26%, #9=21%

#### [14:43:39] Recommendation
Action: **USE_ABILITY** #7 (Dreamer) -> targets ['#4']
Reason: Entropy 2.734 (adjusted 2.539) | timing x1.00
WARNING: Corruption risk: 14%

### [14:44:14] Revealed #7 Dreamer
Info: {'target': 4, 'evil_role': 'Puppet'}

### [14:44:14] Ability used at #7

#### [14:44:14] Solver Output
Scenarios: 17/588
Definite good: ['#8']
Evil probabilities: #6=59%, #4=53%, #7=47%, #2=41%, #3=41%, #5=29%, #1=18%, #9=12%

#### [14:44:14] Recommendation
Action: **USE_ABILITY** #1 (Fortune Teller) -> targets ['#3', '#4']
Reason: Entropy 0.998 (adjusted 0.909) | timing x1.00
WARNING: Corruption risk: 18%

### [14:44:51] Revealed #1 Fortune Teller
Info: {'targets': [3, 4], 'has_evil': True}

### [14:44:52] Ability used at #1

#### [14:44:52] Solver Output
Scenarios: 9/588
Definite good: ['#1', '#8', '#9']
Evil probabilities: #4=78%, #3=56%, #6=56%, #7=44%, #2=33%, #5=33%

#### [14:44:52] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (44% evil Puppet, 33% good Oracle, 11% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 56%, but all reveal branches still lead to a forced win.

### [14:44:59] Executed #3 -> GOOD (WRONG!)

### [14:45:07] Executed #3 -> GOOD (WRONG!)

#### [14:45:07] Solver Output
Scenarios: 3/408
Definite evil: ['#4']
Definite good: ['#1', '#3', '#8', '#9']
Evil probabilities: #5=67%, #6=67%, #2=33%, #7=33%

#### [14:45:07] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 3 scenarios (roles: {'Lilis', 'Puppeteer', 'Puppet'})

### [14:45:33] Executed #4 -> Puppet (EVIL)

#### [14:45:34] Solver Output
Scenarios: 1/31
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#7', '#8', '#9']

#### [14:45:34] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Puppeteer'})

### [14:45:41] Executed #5 -> Puppeteer (EVIL)

#### [14:45:47] Solver Output
Scenarios: 1/31
Definite evil: ['#4', '#5', '#6']
Definite good: ['#1', '#2', '#3', '#7', '#8', '#9']

#### [14:45:47] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [14:45:54] Executed #6 -> Lilis (EVIL)

## [14:46:05] GAME OVER — WIN
Final HP: 1
Notes: 1HP! Lilis+Puppeteer, forced-safe lookahead


---

# New Game — 2026-04-12 14:46:54
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Judge, Empress, Gemcrafter, Witness, Baker, Slayer
- Outcasts: Drunk, Bombardier
- Minions: Poisoner, Shaman
- Demons: Baa

### [14:47:08] Revealed #1 Knight
Info: {}

### [14:47:08] Revealed #2 Witness
Info: {'affected_position': 9}

### [14:47:08] Revealed #4 Empress
Info: {'targets': [2, 6, 9]}

### [14:47:08] Revealed #6 Gemcrafter
Info: {'good_position': 9}

### [14:47:08] Revealed #7 Bombardier
Info: {}

### [14:47:08] Revealed #8 Baker
Info: {'original_role': 'Knight'}

### [14:47:08] Revealed #9 Empress
Info: {'targets': [3, 5, 6]}

### [14:47:18] Revealed #3 Slayer
Info: {}

### [14:47:19] Revealed #5 Slayer
Info: {}

#### [14:47:19] Solver Output
Scenarios: 178/4542
Definite good: ['#6']
Evil probabilities: #2=58%, #7=55%, #1=45%, #8=44%, #5=33%, #3=28%, #4=25%, #9=12%

#### [14:47:19] Recommendation
Action: **EXECUTE** #1
Reason: Knight check: #1 is 45% evil, 26% corruption risk. Expected HP cost: 1.3 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 26% -- corrupted Knight loses immunity + 4 extra damage

### [14:48:06] Executed #1 -> Baa (EVIL)

#### [14:48:06] Solver Output
Scenarios: 21/478
Definite evil: ['#1']
Definite good: ['#6']
Evil probabilities: #2=48%, #8=43%, #3=33%, #7=33%, #5=24%, #4=14%, #9=5%

#### [14:48:06] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#2']
Reason: Target #2 is 48% evil (adjusted 0.43)
WARNING: Corruption risk: 10% -- Slayer ability disabled if corrupted

### [14:48:53] Ability used at #5

#### [14:48:53] Solver Output
Scenarios: 15/478
Definite evil: ['#1']
Definite good: ['#6']
Evil probabilities: #8=47%, #3=33%, #5=33%, #7=33%, #2=27%, #4=20%, #9=7%

#### [14:48:53] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#8']
Reason: Target #8 is 47% evil (adjusted 0.34)
WARNING: Corruption risk: 27% -- Slayer ability disabled if corrupted

### [14:49:37] Ability used at #3

#### [14:49:37] Solver Output
Scenarios: 1/61
Definite evil: ['#1', '#5', '#8']
Definite good: ['#2', '#3', '#4', '#6', '#7', '#9']

#### [14:49:37] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [14:49:44] Executed #5 -> Poisoner (EVIL)

## [14:49:58] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, Knight check + dual Slayer kills


---

# New Game — 2026-04-12 14:50:53
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Baker, Jester, Slayer, Scout, Bishop, Enlightened
- Outcasts: Drunk, Plague_Doctor, Wretch
- Minions: Chancellor
- Demons: Lilis

### [14:51:41] Revealed #4 Baker
Info: {'original_role': 'Jester'}

### [14:51:41] Revealed #5 Scout
Info: {'evil_role': 'Chancellor', 'distance': 2}

### [14:51:41] Revealed #9 Wretch
Info: {}

### [14:52:00] Revealed #1 Jester
Info: {}

### [14:52:01] Revealed #2 Plague_Doctor
Info: {}

### [14:52:01] Revealed #3 Jester
Info: {}

### [14:52:01] Revealed #6 Slayer
Info: {}

### [14:52:01] Revealed #7 Enlightened
Info: {'direction': 'ccw'}

#### [14:52:01] Solver Output
Scenarios: 169/1830
Definite good: ['#8']
Evil probabilities: #4=52%, #7=37%, #6=33%, #5=24%, #9=20%, #1=17%, #3=12%, #2=5%

#### [14:52:01] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#4']
Reason: Entropy 2.367 (adjusted 2.367) | timing x1.00

### [14:53:12] Ability used at #2

#### [14:53:13] Solver Output
Scenarios: 89/1830
Definite good: ['#8']
Evil probabilities: #4=94%, #7=35%, #6=29%, #5=18%, #1=7%, #9=7%, #2=6%, #3=4%

#### [14:53:13] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#4', '#5', '#8']
Reason: Expected posterior 39.5 scenarios (adjusted 46.2, info gain 0.946 bits) | timing x1.00
WARNING: Corruption risk: 34%

### [14:54:22] Revealed #3 Jester
Info: {'targets': [4, 5, 8], 'evil_count': 1}

### [14:54:22] Ability used at #3

#### [14:54:22] Solver Output
Scenarios: 42/1830
Definite evil: ['#4']
Definite good: ['#2', '#3', '#8']
Evil probabilities: #7=43%, #6=19%, #1=14%, #9=14%, #5=10%

#### [14:54:22] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 42 scenarios (roles: {'Chancellor', 'Lilis'})

### [14:54:29] Executed #4 -> Lilis (EVIL)

#### [14:54:43] Solver Output
Scenarios: 16/187
Definite evil: ['#4']
Definite good: ['#2', '#3', '#8']
Evil probabilities: #7=38%, #6=25%, #1=12%, #5=12%, #9=12%

#### [14:54:43] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#2', '#3', '#7']
Reason: Expected posterior 9.1 scenarios (adjusted 11.4, info gain 0.490 bits) | timing x1.00
WARNING: Corruption risk: 50%

### [14:55:23] Revealed #1 Jester
Info: {'targets': [2, 3, 7], 'evil_count': 2}

### [14:55:23] Ability used at #1

#### [14:55:23] Solver Output
Scenarios: 10/187
Definite evil: ['#4']
Definite good: ['#2', '#3', '#5', '#8', '#9']
Evil probabilities: #6=40%, #7=40%, #1=20%

#### [14:55:23] Recommendation
Action: **USE_ABILITY** #6 (Slayer) -> targets ['#7']
Reason: Target #7 is 40% evil (adjusted 0.32)
WARNING: Corruption risk: 20% -- Slayer ability disabled if corrupted

### [14:56:08] Ability used at #6

#### [14:56:08] Solver Output
Scenarios: 8/187
Definite evil: ['#4']
Definite good: ['#2', '#3', '#5', '#8', '#9']
Evil probabilities: #6=50%, #1=25%, #7=25%

#### [14:56:08] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (50% evil Chancellor, 25% good Slayer, 12% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [14:56:15] Executed #6 -> GOOD (WRONG!)

### [14:56:30] Executed #6 -> GOOD (WRONG!)

#### [14:56:31] Solver Output
Scenarios: 2/147
Definite evil: ['#1', '#4']
Definite good: ['#2', '#3', '#5', '#6', '#7', '#8', '#9']

#### [14:56:31] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Chancellor'})

### [14:57:02] Executed #1 -> Chancellor (EVIL)

## [14:57:02] GAME OVER — WIN
Final HP: 1
Notes: 1HP! Lilis+Chancellor, forced-safe through 2 wrong execs


---

# New Game — 2026-04-12 14:57:56
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Bard, Bishop, Enlightened, Architect
- Outcasts: Plague_Doctor, Bombardier, Doppelganger
- Minions: Chancellor
- Demons: Baa

### [14:58:11] Revealed #1 Bard
Info: {'corruption_distance': -1}

### [14:58:11] Revealed #2 Architect
Info: {'side': 'Left'}

### [14:58:11] Revealed #3 Bishop
Info: {'targets': [5, 7, 6], 'types': ['Villager', 'Outcast', 'Minion']}

### [14:58:11] Revealed #4 Bombardier
Info: {}

### [14:58:11] Revealed #5 Scout
Info: {'evil_role': 'Chancellor', 'distance': 2}

### [14:58:11] Revealed #6 Scout
Info: {'evil_role': 'Chancellor', 'distance': 2}

### [14:58:19] Revealed #7 Plague_Doctor
Info: {}

#### [14:58:19] Solver Output
Scenarios: 11/796
Definite good: ['#4', '#5', '#6']
Evil probabilities: #1=55%, #3=55%, #2=45%, #7=45%

#### [14:58:19] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#1']
Reason: Entropy 0.994 (adjusted 0.994) | timing x1.00

### [14:59:06] Ability used at #7

#### [14:59:07] Solver Output
Scenarios: 5/796
Definite evil: ['#2', '#7']
Definite good: ['#1', '#3', '#4', '#5', '#6']

#### [14:59:07] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 5 scenarios (roles: {'Baa'})

### [14:59:13] Executed #2 -> Baa (EVIL)

#### [14:59:23] Solver Output
Scenarios: 5/124
Definite evil: ['#2', '#7']
Definite good: ['#1', '#3', '#4', '#5', '#6']

#### [14:59:23] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 5 scenarios (roles: {'Chancellor'})

### [14:59:30] Executed #7 -> Chancellor (EVIL)

## [14:59:43] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, both definite, evil PD lie detected


---

# New Game — 2026-04-12 15:00:36
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Hunter, Dreamer, Bishop, Knight, Alchemist, Jester
- Outcasts: Doppelganger, Drunk
- Minions: Twin_Minion
- Demons: Pooka

### [15:00:51] Revealed #2 Alchemist
Info: {'cured_count': 0}

### [15:00:51] Revealed #7 Bishop
Info: {'targets': [8, 7, 2], 'types': ['Villager', 'Outcast', 'Minion']}

### [15:00:51] Revealed #8 Knight
Info: {}

### [15:00:51] Revealed #9 Hunter
Info: {'distance': 4}

### [15:01:06] Revealed #1 Jester
Info: {}

### [15:01:06] Revealed #3 Druid
Info: {}

### [15:01:07] Revealed #4 Jester
Info: {}

### [15:01:07] Revealed #5 Dreamer
Info: {}

### [15:01:07] Revealed #6 Dreamer
Info: {}

#### [15:01:07] Solver Output
Scenarios: 223/3024
Definite good: ['#2']
Evil probabilities: #9=52%, #7=41%, #8=34%, #6=32%, #1=11%, #4=11%, #5=11%, #3=8%

#### [15:01:07] Recommendation
Action: **USE_ABILITY** #5 (Dreamer) -> targets ['#9']
Reason: Entropy 2.237 (adjusted 1.946) | timing x1.00
WARNING: Corruption risk: 26%

### [15:02:01] Revealed #5 Dreamer
Info: {'target': 9, 'evil_role': 'Pooka'}

### [15:02:02] Ability used at #5

#### [15:02:02] Solver Output
Scenarios: 160/3024
Definite good: ['#2']
Evil probabilities: #6=42%, #7=38%, #9=34%, #8=31%, #4=15%, #5=15%, #1=14%, #3=11%

#### [15:02:02] Recommendation
Action: **USE_ABILITY** #6 (Dreamer) -> targets ['#7']
Reason: Entropy 2.152 (adjusted 1.903) | timing x1.00
WARNING: Corruption risk: 23%

### [15:02:36] Revealed #6 Dreamer
Info: {'target': 7, 'evil_role': 'Twin_Minion'}

### [15:02:36] Ability used at #6

#### [15:02:36] Solver Output
Scenarios: 148/3024
Definite good: ['#2']
Evil probabilities: #6=43%, #9=35%, #7=33%, #8=33%, #4=16%, #1=15%, #5=15%, #3=11%

#### [15:02:36] Recommendation
Action: **USE_ABILITY** #4 (Jester) -> targets ['#5', '#7', '#8']
Reason: Expected posterior 55.3 scenarios (adjusted 56.3, info gain 1.395 bits) | timing x1.00
WARNING: Corruption risk: 3%

### [15:03:14] Revealed #4 Jester
Info: {'targets': [5, 7, 8], 'evil_count': 1}

### [15:03:14] Ability used at #4

#### [15:03:14] Solver Output
Scenarios: 68/3024
Definite good: ['#2']
Evil probabilities: #6=44%, #8=41%, #7=38%, #9=31%, #1=16%, #3=15%, #4=7%, #5=7%

#### [15:03:14] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#3', '#7', '#8']
Reason: Expected posterior 26.5 scenarios (adjusted 27.7, info gain 1.295 bits) | timing x1.00
WARNING: Corruption risk: 9%

### [15:03:53] Revealed #1 Jester
Info: {'targets': [3, 7, 8], 'evil_count': 0}

### [15:03:54] Ability used at #1

#### [15:03:54] Solver Output
Scenarios: 31/3024
Definite good: ['#2', '#3']
Evil probabilities: #6=45%, #1=35%, #9=32%, #7=29%, #8=26%, #4=16%, #5=16%

#### [15:03:54] Recommendation
Action: **USE_ABILITY** #3 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 0.000 (adjusted 0.000) | timing x1.00

### [15:04:31] Revealed #3 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': 'Doppelganger'}

### [15:04:31] Ability used at #3

#### [15:04:31] Solver Output
Scenarios: 14/3024
Definite good: ['#2', '#3']
Evil probabilities: #6=50%, #1=36%, #8=29%, #9=29%, #5=21%, #7=21%, #4=14%

#### [15:04:31] Recommendation
Action: **EXECUTE** #8
Reason: Knight check: #8 is 29% evil, 21% corruption risk. Expected HP cost: 1.4 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 21% -- corrupted Knight loses immunity + 4 extra damage

#### [15:05:22] Execution Blocked
#8 Knight immunity — confirmed good, no HP loss

#### [15:05:22] Solver Output
Scenarios: 10/2352
Definite good: ['#2', '#3', '#8']
Evil probabilities: #6=70%, #5=30%, #7=30%, #9=30%, #1=20%, #4=20%

#### [15:05:22] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (70% evil Pooka, 20% good Dreamer (corrupted), 10% good Dreamer).
WARNING: Execution lookahead override -- immediate hit chance is 70%, but all reveal branches still lead to a forced win.

### [15:05:29] Executed #6 -> Pooka (EVIL)

#### [15:05:43] Solver Output
Scenarios: 7/294
Definite evil: ['#6']
Definite good: ['#1', '#2', '#3', '#7', '#8']
Evil probabilities: #5=43%, #4=29%, #9=29%

#### [15:05:43] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (57% good Dreamer (corrupted), 43% evil Twin_Minion).
WARNING: Execution lookahead override -- immediate hit chance is 43%, but all reveal branches still lead to a forced win.

### [15:05:50] Executed #5 -> Twin Minion (EVIL)

## [15:06:08] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, 5 abilities used, Knight check + forced-safe


---

# New Game — 2026-04-12 15:07:20
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Enlightened, Knight, Slayer, Medium, Lover, Oracle
- Outcasts: Bombardier, Drunk
- Minions: Puppeteer
- Demons: Pooka

### [15:07:39] Revealed #1 Medium
Info: {'good_position': 2, 'good_role': 'Enlightened'}

### [15:07:39] Revealed #2 Enlightened
Info: {'direction': 'CW'}

### [15:07:39] Revealed #3 Lover
Info: {'evil_adjacent': 1}

### [15:07:39] Revealed #4 Confessor
Info: {'dizzy': True}

### [15:07:39] Revealed #6 Knight
Info: {}

### [15:07:39] Revealed #7 Oracle
Info: {'targets': [8, 9], 'minion_role': 'Puppeteer'}

### [15:07:39] Revealed #8 Bombardier
Info: {}

### [15:07:39] Revealed #9 Confessor
Info: {'dizzy': True}

### [15:07:51] Revealed #5 Slayer
Info: {}

#### [15:07:51] Solver Output
Scenarios: 9/588
Definite good: ['#2', '#8']
Evil probabilities: #4=67%, #3=56%, #5=44%, #1=33%, #6=33%, #7=33%, #9=33%

#### [15:07:51] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#4']
Reason: Target #4 is 67% evil (adjusted 0.52)
WARNING: Corruption risk: 22% -- Slayer ability disabled if corrupted

### [15:08:38] Ability used at #5

#### [15:08:39] Solver Output
Scenarios: 3/72
Definite evil: ['#4']
Definite good: ['#1', '#2', '#8', '#9']
Evil probabilities: #5=67%, #7=67%, #3=33%, #6=33%

#### [15:08:39] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (67% evil Puppet, 33% good Slayer).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [15:08:45] Executed #5 -> GOOD (WRONG!)

### [15:08:55] Executed #5 -> GOOD (WRONG!)

#### [15:08:55] Solver Output
Scenarios: 1/31
Definite evil: ['#3', '#4', '#7']
Definite good: ['#1', '#2', '#5', '#6', '#8', '#9']

#### [15:08:55] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [15:09:02] Executed #3 -> Puppet (EVIL)

#### [15:09:09] Solver Output
Scenarios: 1/31
Definite evil: ['#3', '#4', '#7']
Definite good: ['#1', '#2', '#5', '#6', '#8', '#9']

#### [15:09:09] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [15:09:16] Executed #7 -> Pooka (EVIL)

## [15:09:30] GAME OVER — WIN
Final HP: 5
Notes: 5HP, Slayer kills Puppeteer, forced-safe wins


---

# New Game — 2026-04-12 15:10:21
Cards: 7, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Gemcrafter, Dreamer, Oracle, Knight
- Outcasts: Wretch, Plague_Doctor, Bombardier
- Minions: Puppeteer
- Demons: Baa

### [15:10:37] Revealed #1 Bombardier
Info: {}

### [15:10:37] Revealed #2 Gemcrafter
Info: {'good_position': 3}

### [15:10:37] Revealed #3 Oracle
Info: {'targets': [2, 4], 'minion_role': 'Puppet'}

### [15:10:37] Revealed #4 Knight
Info: {}

### [15:10:47] Revealed #5 Slayer
Info: {}

### [15:10:47] Revealed #6 Plague_Doctor
Info: {}

### [15:10:47] Revealed #7 Dreamer
Info: {}

#### [15:10:48] Solver Output
Scenarios: 17/150
Evil probabilities: #1=94%, #2=59%, #5=47%, #7=41%, #4=35%, #3=12%, #6=12%

#### [15:10:48] Recommendation
Action: **USE_ABILITY** #7 (Dreamer) -> targets ['#5']
Reason: Entropy 2.419 (adjusted 2.206) | timing x1.00
WARNING: Corruption risk: 18%

### [15:11:26] Revealed #7 Dreamer
Info: {'target': 5, 'evil_role': 'Puppeteer'}

### [15:11:26] Ability used at #7

#### [15:11:26] Solver Output
Scenarios: 13/150
Evil probabilities: #1=92%, #2=62%, #7=46%, #4=38%, #5=31%, #3=15%, #6=15%

#### [15:11:26] Recommendation
Action: **USE_ABILITY** #6 (Plague Doctor) -> targets ['#2']
Reason: Entropy 1.352 (adjusted 1.352) | timing x1.00

### [15:12:13] Ability used at #6

#### [15:12:14] Solver Output
Scenarios: 9/150
Definite evil: ['#1']
Definite good: ['#3', '#6']
Evil probabilities: #2=78%, #4=44%, #7=44%, #5=33%

#### [15:12:14] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 9 scenarios (roles: {'Puppet', 'Baa', 'Puppeteer'})

### [15:12:20] Executed #1 -> Puppeteer (EVIL)

#### [15:12:30] Solver Output
Scenarios: 7/34
Definite evil: ['#1', '#2']
Definite good: ['#3', '#6']
Evil probabilities: #4=43%, #7=43%, #5=14%

#### [15:12:30] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 7 scenarios (roles: {'Puppet'})

### [15:12:36] Executed #2 -> Puppet (EVIL)

#### [15:12:52] Solver Output
Scenarios: 7/17
Definite evil: ['#1', '#2']
Definite good: ['#3', '#6']
Evil probabilities: #4=43%, #7=43%, #5=14%

#### [15:12:52] Recommendation
Action: **EXECUTE** #4
Reason: Knight check: #4 is 43% evil, 14% corruption risk. Expected HP cost: 0.7 (corrupted Knight = 9 HP).
WARNING: Corruption risk: 14% -- corrupted Knight loses immunity + 4 extra damage

#### [15:14:25] Execution Blocked
#4 Knight immunity — confirmed good, no HP loss

#### [15:14:25] Solver Output
Scenarios: 4/13
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#6']
Evil probabilities: #7=75%, #5=25%

#### [15:14:25] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#7']
Reason: Target #7 is 75% evil (adjusted 0.56)
WARNING: Corruption risk: 25% -- Slayer ability disabled if corrupted

### [15:15:36] Ability used at #5

#### [15:15:36] Solver Output
Scenarios: 2/13
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#6']
Evil probabilities: #5=50%, #7=50%

#### [15:15:36] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 50% good Slayer (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:15:43] Executed #5 -> Baa (EVIL)

## [15:16:09] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, PD+Dreamer+Slayer+Knight check, ASC63 COMPLETE 7/7


---

# New Game — 2026-04-12 15:17:38
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Dreamer, Druid, Slayer, Fortune_Teller, Lover
- Outcasts: Plague_Doctor
- Minions: Witch, Minion
- Demons: Pooka

### [15:18:39] Revealed #2 Lover
Info: {'evil_adjacent': 0}

### [15:18:39] Revealed #7 Medium
Info: {'good_position': 4, 'good_role': 'Druid'}

### [15:18:39] Revealed #8 Medium
Info: {'good_position': 9, 'good_role': 'Dreamer'}

### [15:18:54] Revealed #3 Slayer
Info: {}

### [15:18:54] Revealed #4 Druid
Info: {}

### [15:18:55] Revealed #5 Druid
Info: {}

### [15:18:55] Revealed #6 Fortune_Teller
Info: {}

### [15:18:55] Revealed #9 Dreamer
Info: {}

#### [15:18:55] Solver Output
Scenarios: 76/1848
Definite good: ['#1', '#2']
Evil probabilities: #4=84%, #7=68%, #9=55%, #5=32%, #8=32%, #6=16%, #3=13%

#### [15:18:55] Recommendation
Action: **USE_ABILITY** #9 (Dreamer) -> targets ['#4']
Reason: Entropy 2.728 (adjusted 2.548) | timing x1.00
WARNING: Corruption risk: 13%

### [15:19:40] Revealed #9 Dreamer
Info: {'target': 4, 'evil_role': 'Witch'}

### [15:19:40] Ability used at #9

#### [15:19:41] Solver Output
Scenarios: 43/1848
Definite good: ['#1', '#2']
Evil probabilities: #4=72%, #9=67%, #7=56%, #8=44%, #5=40%, #6=12%, #3=9%

#### [15:19:41] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#5', '#8']
Reason: Entropy 0.996 (adjusted 0.788) | follow-up bonus 0.373 | timing x1.00
WARNING: Corruption risk: 42%

### [15:20:27] Revealed #6 Fortune Teller
Info: {'targets': [5, 8], 'has_evil': True}

### [15:20:28] Ability used at #6

#### [15:20:28] Solver Output
Scenarios: 23/1848
Definite good: ['#1', '#2']
Evil probabilities: #4=83%, #9=65%, #7=57%, #8=43%, #5=26%, #6=22%, #3=4%

#### [15:20:28] Recommendation
Action: **USE_ABILITY** #5 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.932 (adjusted 0.750) | timing x1.00
WARNING: Corruption risk: 39%

### [15:21:14] Revealed #5 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Plague_Doctor'}

### [15:21:14] Ability used at #5

#### [15:21:14] Solver Output
Scenarios: 23/1848
Definite good: ['#1', '#2']
Evil probabilities: #4=83%, #9=65%, #7=57%, #8=43%, #5=26%, #6=22%, #3=4%

#### [15:21:14] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#4']
Reason: Target #4 is 83% evil (adjusted 0.43)
WARNING: Corruption risk: 48% -- Slayer ability disabled if corrupted

### [15:21:58] Ability used at #3

#### [15:21:58] Solver Output
Scenarios: 8/224
Definite evil: ['#4']
Definite good: ['#1', '#2', '#3', '#5']
Evil probabilities: #9=88%, #7=62%, #8=38%, #6=12%

#### [15:21:58] Recommendation
Action: **EXECUTE** #9
Reason: Execution lookahead: #9 guarantees a win across all reveal branches with current HP budget (62% evil Witch, 25% evil Pooka, 12% good Dreamer (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 88%, but all reveal branches still lead to a forced win.

### [15:22:05] Executed #9 -> Pooka (EVIL)

#### [15:22:15] Solver Output
Scenarios: 2/31
Definite evil: ['#4', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#6']
Evil probabilities: #7=50%, #8=50%

#### [15:22:15] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% good Medium (corrupted), 50% evil Witch).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:22:22] Executed #7 -> GOOD (WRONG!)

### [15:22:30] Executed #7 -> GOOD (WRONG!)

#### [15:22:30] Solver Output
Scenarios: 1/26
Definite evil: ['#4', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7']

#### [15:22:30] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Witch'})

### [15:22:37] Executed #8 -> Witch (EVIL)

## [15:22:50] GAME OVER — WIN
Final HP: 5
Notes: 5HP, Slayer+Dreamer+FT+Druid, all abilities used


---

# New Game — 2026-04-12 15:23:46
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Enlightened, Slayer, Druid, Lover, Confessor
- Outcasts: Bombardier, Wretch, Drunk
- Minions: Chancellor, Puppeteer
- Demons: Baa

### [15:24:02] Revealed #1 Lover
Info: {'evil_adjacent': 1}

### [15:24:02] Revealed #3 Wretch
Info: {}

### [15:24:02] Revealed #4 Enlightened
Info: {'direction': 'Equidistant'}

### [15:24:13] Revealed #2 Judge
Info: {}

### [15:24:13] Revealed #5 Slayer
Info: {}

### [15:24:13] Revealed #6 Judge
Info: {}

### [15:24:14] Revealed #7 Judge
Info: {}

### [15:24:14] Revealed #8 Judge
Info: {}

### [15:24:14] Revealed #9 Druid
Info: {}

#### [15:24:14] Solver Output
Scenarios: 690/3880
Evil probabilities: #7=65%, #6=56%, #8=55%, #5=49%, #2=44%, #1=39%, #4=37%, #9=29%, #3=27%

#### [15:24:14] Recommendation
Action: **USE_ABILITY** #9 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.966 (adjusted 0.883) | timing x1.00
WARNING: Corruption risk: 17%

### [15:24:56] Revealed #9 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [15:24:56] Ability used at #9

#### [15:24:57] Solver Output
Scenarios: 360/3880
Evil probabilities: #7=60%, #2=60%, #8=53%, #6=52%, #1=42%, #5=42%, #4=38%, #3=29%, #9=24%

#### [15:24:57] Recommendation
Action: **USE_ABILITY** #2 (Judge) -> targets ['#5']
Reason: Expected posterior 201.1 scenarios (adjusted 212.9, info gain 0.758 bits) | timing x1.00
WARNING: Corruption risk: 12% -- corrupted Judge results are unreliable

### [15:25:47] Revealed #2 Judge
Info: {'target': 5, 'is_lying': True}

### [15:25:47] Ability used at #2

#### [15:25:48] Solver Output
Scenarios: 196/3880
Evil probabilities: #7=61%, #2=57%, #6=53%, #8=52%, #5=43%, #1=39%, #4=35%, #9=34%, #3=26%

#### [15:25:48] Recommendation
Action: **USE_ABILITY** #7 (Judge) -> targets ['#9']
Reason: Expected posterior 109.5 scenarios (adjusted 115.9, info gain 0.758 bits) | timing x1.00
WARNING: Corruption risk: 12% -- corrupted Judge results are unreliable

### [15:26:53] Revealed #7 Judge
Info: {'target': 9, 'is_lying': True}

### [15:26:54] Ability used at #7

#### [15:26:54] Solver Output
Scenarios: 110/3880
Evil probabilities: #7=66%, #2=57%, #5=50%, #6=50%, #8=49%, #1=38%, #4=34%, #9=32%, #3=24%

#### [15:26:54] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#3']
Reason: Expected posterior 63.0 scenarios (adjusted 67.6, info gain 0.702 bits) | timing x1.00
WARNING: Corruption risk: 15% -- corrupted Judge results are unreliable

### [15:27:37] Revealed #6 Judge
Info: {'target': 3, 'is_lying': True}

### [15:27:37] Ability used at #6

#### [15:27:37] Solver Output
Scenarios: 64/3880
Evil probabilities: #2=66%, #7=66%, #5=56%, #6=50%, #8=45%, #3=34%, #9=33%, #4=27%, #1=23%

#### [15:27:37] Recommendation
Action: **USE_ABILITY** #8 (Judge) -> targets ['#7']
Reason: Expected posterior 35.3 scenarios (adjusted 36.9, info gain 0.794 bits) | timing x1.00
WARNING: Corruption risk: 9% -- corrupted Judge results are unreliable

### [15:28:43] Revealed #8 Judge
Info: {'target': 7, 'is_lying': False}

### [15:28:43] Ability used at #8

#### [15:28:43] Solver Output
Scenarios: 32/3880
Definite good: ['#1']
Evil probabilities: #5=81%, #6=66%, #9=59%, #8=53%, #2=50%, #7=47%, #3=25%, #4=19%

#### [15:28:43] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#6']
Reason: Target #6 is 66% evil (adjusted 0.66)

### [15:29:34] Ability used at #5

#### [15:29:34] Solver Output
Scenarios: 11/476
Definite evil: ['#6']
Definite good: ['#1', '#3']
Evil probabilities: #2=73%, #5=73%, #9=55%, #7=45%, #4=27%, #8=27%

#### [15:29:34] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (45% evil Baa, 27% evil Chancellor, 27% good Drunk (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 73%, but all reveal branches still lead to a forced win.

### [15:29:41] Executed #2 -> Chancellor (EVIL)

#### [15:29:50] Solver Output
Scenarios: 3/50
Definite evil: ['#2', '#5', '#6']
Definite good: ['#1', '#3', '#4']
Evil probabilities: #7=33%, #8=33%, #9=33%

#### [15:29:50] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 3 scenarios (roles: {'Puppet'})

### [15:29:57] Executed #5 -> Puppet (EVIL)

#### [15:30:05] Solver Output
Scenarios: 3/25
Definite evil: ['#2', '#5', '#6']
Definite good: ['#1', '#3', '#4']
Evil probabilities: #7=33%, #8=33%, #9=33%

#### [15:30:05] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (33% evil Baa, 33% good Drunk (corrupted), 33% good Judge).
WARNING: Execution lookahead override -- immediate hit chance is 33%, but all reveal branches still lead to a forced win.

### [15:30:12] Executed #7 -> Baa (EVIL)

## [15:30:25] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, 4 Judges + Slayer + Druid, massive ability game


---

# New Game — 2026-04-12 15:31:19
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Knight, Architect, Poet, Fortune_Teller, Knitter
- Outcasts: Bombardier
- Minions: Twin_Minion, Minion
- Demons: Lilis

### [15:32:03] Revealed #1 Architect
Info: {'side': 'Left'}

### [15:32:03] Revealed #3 Baker
Info: {'original_role': 'original'}

### [15:32:03] Revealed #4 Baker
Info: {'original_role': 'Knight'}

### [15:32:03] Revealed #6 Bombardier
Info: {}

### [15:32:03] Revealed #7 Knitter
Info: {'evil_pairs': 1}

### [15:32:03] Revealed #9 Bombardier
Info: {}

### [15:32:12] Revealed #2 Fortune_Teller
Info: {}

### [15:32:12] Revealed #8 Poet
Info: {}

#### [15:32:12] Solver Output
Scenarios: 36/504
Definite good: ['#3', '#5']
Evil probabilities: #6=67%, #9=67%, #1=50%, #8=50%, #7=33%, #2=17%, #4=17%

#### [15:32:12] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#3', '#6']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [15:32:55] Revealed #2 Fortune Teller
Info: {'targets': [3, 6], 'has_evil': True}

### [15:32:55] Ability used at #2

#### [15:32:55] Solver Output
Scenarios: 18/504
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5']
Evil probabilities: #8=67%, #9=67%, #1=33%, #7=33%

#### [15:32:55] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 18 scenarios (roles: {'Minion', 'Lilis', 'Twin_Minion'})

### [15:33:02] Executed #6 -> Minion (EVIL)

#### [15:33:11] Solver Output
Scenarios: 6/56
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5']
Evil probabilities: #8=67%, #9=67%, #1=33%, #7=33%

#### [15:33:11] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (33% evil Lilis, 33% good Poet, 33% evil Twin_Minion).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [15:33:18] Executed #8 -> Twin Minion (EVIL)

#### [15:33:27] Solver Output
Scenarios: 2/7
Definite evil: ['#6', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5']
Evil probabilities: #7=50%, #9=50%

#### [15:33:27] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% good Knitter, 50% evil Lilis).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:33:34] Executed #7 -> Lilis (EVIL)

## [15:33:48] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, FT ability + forced-safe


---

# New Game — 2026-04-12 15:34:44
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Lover, Gemcrafter, Empress, Fortune_Teller, Jester
- Outcasts: Bombardier, Doppelganger
- Minions: Witch
- Demons: Baa

### [15:35:06] Revealed #1 Empress
Info: {'targets': [4, 6, 8]}

### [15:35:06] Revealed #4 Gemcrafter
Info: {'good_position': 8}

### [15:35:06] Revealed #5 Medium
Info: {'good_position': 2, 'good_role': 'Fortune Teller'}

### [15:35:06] Revealed #6 Lover
Info: {'evil_adjacent': 0}

### [15:35:19] Revealed #2 Fortune_Teller
Info: {}

### [15:35:19] Revealed #3 Fortune_Teller
Info: {}

### [15:35:20] Revealed #7 Fortune_Teller
Info: {}

#### [15:35:20] Solver Output
Scenarios: 20/392
Definite good: ['#2', '#4', '#5', '#8']
Evil probabilities: #1=50%, #3=50%, #6=50%, #7=50%

#### [15:35:20] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#1', '#3']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [15:36:05] Revealed #2 Fortune Teller
Info: {'targets': [1, 3], 'has_evil': True}

### [15:36:05] Ability used at #2

#### [15:36:06] Solver Output
Scenarios: 12/392
Definite evil: ['#1', '#3']
Definite good: ['#2', '#4', '#5', '#6', '#7', '#8']

#### [15:36:06] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 12 scenarios (roles: {'Baa', 'Witch'})

### [15:36:12] Executed #1 -> Baa (EVIL)

#### [15:36:26] Solver Output
Scenarios: 6/49
Definite evil: ['#1', '#3']
Definite good: ['#2', '#4', '#5', '#6', '#7', '#8']

#### [15:36:26] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 6 scenarios (roles: {'Witch'})

### [15:37:00] Ability used at #3

### [15:37:27] Executed #3 -> Witch (EVIL)

## [15:37:27] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, FT ability nails both


---

# New Game — 2026-04-12 15:38:58
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Confessor, Enlightened, Poet, Druid, Bard, Oracle
- Outcasts: Wretch, Doppelganger
- Minions: Chancellor
- Demons: Pooka

### [15:39:29] Revealed #1 Oracle
Info: {'targets': [6, 8], 'minion_role': 'Chancellor'}

### [15:39:29] Revealed #3 Confessor
Info: {'dizzy': True}

### [15:39:29] Revealed #4 Oracle
Info: {'targets': [5, 8], 'minion_role': 'Chancellor'}

### [15:39:29] Revealed #5 Bard
Info: {'corruption_distance': 3}

### [15:39:29] Revealed #6 Enlightened
Info: {'direction': 'CW'}

### [15:39:29] Revealed #7 Empress
Info: {'targets': [1, 2, 4]}

### [15:39:29] Revealed #8 Wretch
Info: {}

### [15:39:48] Revealed #2 Poet
Info: {'evil_adjacent': 0, 'copied_role': 'Lover'}

#### [15:39:48] Solver Output
Scenarios: 16/454
Definite good: ['#2', '#5', '#8']
Evil probabilities: #3=62%, #1=56%, #4=38%, #6=38%, #7=6%

#### [15:39:48] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% evil Chancellor, 38% good Confessor (corrupted), 12% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 62%, but all reveal branches still lead to a forced win.

### [15:39:55] Executed #3 -> Pooka (EVIL)

#### [15:40:07] Solver Output
Scenarios: 2/52
Definite evil: ['#3']
Definite good: ['#2', '#4', '#5', '#6', '#8']
Evil probabilities: #1=50%, #7=50%

#### [15:40:07] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% evil Chancellor, 50% good Oracle).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [15:40:14] Executed #1 -> GOOD (WRONG!)

### [15:40:26] Executed #1 -> GOOD (WRONG!)

#### [15:40:26] Solver Output
Scenarios: 1/47
Definite evil: ['#3', '#7']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#8']

#### [15:40:26] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 1 scenarios (roles: {'Chancellor'})

### [15:40:33] Executed #7 -> Chancellor (EVIL)

## [15:40:53] GAME OVER — WIN
Final HP: 5
Notes: 5HP, forced-safe through wrong exec


---

# New Game — 2026-04-12 15:42:03
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Medium, Gemcrafter, Scout, Jester, Confessor
- Outcasts: Plague_Doctor
- Minions: Minion
- Demons: Pooka

### [15:42:19] Revealed #1 Gemcrafter
Info: {'good_position': 8}

### [15:42:19] Revealed #2 Confessor
Info: {'dizzy': False}

### [15:42:19] Revealed #4 Scout
Info: {'evil_role': 'Pooka', 'distance': 3}

### [15:42:19] Revealed #7 Medium
Info: {'good_position': 5, 'good_role': 'Jester'}

### [15:42:34] Revealed #3 Plague_Doctor
Info: {}

### [15:42:34] Revealed #5 Jester
Info: {}

### [15:42:35] Revealed #6 Judge
Info: {}

### [15:42:35] Revealed #8 Jester
Info: {}

#### [15:42:35] Solver Output
Scenarios: 12/224
Definite good: ['#1', '#2', '#3']
Evil probabilities: #5=92%, #7=33%, #8=33%, #4=25%, #6=17%

#### [15:42:35] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#7']
Reason: Entropy 2.055 (adjusted 2.055) | timing x1.00

### [15:43:33] Ability used at #3

#### [15:43:33] Solver Output
Scenarios: 7/224
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#7']
Evil probabilities: #8=43%, #4=29%, #6=29%

#### [15:43:33] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 7 scenarios (roles: {'Pooka', 'Minion'})

### [15:44:23] Ability used at #5

### [15:44:43] Executed #5 -> Minion (EVIL)

#### [15:44:43] Solver Output
Scenarios: 5/31
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#7']
Evil probabilities: #8=60%, #4=20%, #6=20%

#### [15:44:43] Recommendation
Action: **USE_ABILITY** #8 (Jester) -> targets ['#1', '#2', '#3']
Reason: Expected posterior 2.8 scenarios (adjusted 2.8, info gain 0.827 bits) | timing x1.00

### [15:45:46] Revealed #8 Jester
Info: {'targets': [1, 2, 3], 'evil_count': 0}

### [15:45:46] Ability used at #8

#### [15:45:46] Solver Output
Scenarios: 2/31
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#7', '#8']
Evil probabilities: #4=50%, #6=50%

#### [15:45:46] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#1']
Reason: Expected posterior 1.0 scenarios (adjusted 1.0, info gain 1.000 bits) | timing x1.00

### [15:46:39] Revealed #6 Judge
Info: {'target': 1, 'is_lying': False}

### [15:46:40] Ability used at #6

#### [15:46:40] Solver Output
Scenarios: 1/31
Definite evil: ['#4', '#5']
Definite good: ['#1', '#2', '#3', '#6', '#7', '#8']

#### [15:46:40] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [15:46:47] Executed #4 -> GOOD (WRONG!)

### [15:47:04] Executed #4 -> GOOD (WRONG!)

#### [15:47:04] Solver Output
Scenarios: 0/26

#### [15:47:04] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [15:47:46] Executed #8 -> Pooka (EVIL)

## [15:47:47] GAME OVER — WIN
Final HP: 5
Notes: 5HP, solver 0-scenario bug on #4 (100% confident wrong). PD+Jester+Judge abilities.


---

# New Game — 2026-04-12 15:51:21
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Dreamer, Empress, Druid, Poet, Medium, Oracle, Scout
- Outcasts: Plague_Doctor
- Minions: Twin_Minion, Minion
- Demons: Pooka

### [15:51:42] Revealed #4 Empress
Info: {'targets': [3, 5, 9]}

### [15:51:42] Revealed #6 Medium
Info: {'good_position': 5, 'good_role': 'Poet'}

### [15:51:42] Revealed #7 Oracle
Info: {'targets': [3, 6], 'minion_role': 'Minion'}

### [15:51:42] Revealed #9 Scout
Info: {'evil_role': 'Pooka', 'distance': 2}

### [15:51:59] Revealed #1 Dreamer
Info: {}

### [15:51:59] Revealed #2 Plague_Doctor
Info: {}

### [15:51:59] Revealed #3 Druid
Info: {}

### [15:52:00] Revealed #5 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 7}

### [15:52:00] Revealed #8 Druid
Info: {}

#### [15:52:00] Solver Output
Scenarios: 36/1848
Definite good: ['#2']
Evil probabilities: #8=64%, #3=44%, #7=44%, #9=44%, #5=39%, #4=31%, #1=19%, #6=14%

#### [15:52:00] Recommendation
Action: **USE_ABILITY** #1 (Dreamer) -> targets ['#8']
Reason: Entropy 2.776 (adjusted 2.313) | timing x1.00
WARNING: Corruption risk: 33%

### [15:52:49] Revealed #1 Dreamer
Info: {'target': 8, 'evil_role': 'Pooka'}

### [15:52:49] Ability used at #1

#### [15:52:49] Solver Output
Scenarios: 24/1848
Definite good: ['#2']
Evil probabilities: #3=58%, #8=46%, #5=42%, #7=42%, #9=42%, #4=38%, #6=21%, #1=12%

#### [15:52:49] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#1']
Reason: Entropy 1.668 (adjusted 1.668) | timing x1.00

### [15:53:38] Ability used at #2

#### [15:53:39] Solver Output
Scenarios: 6/1848
Definite evil: ['#9']
Definite good: ['#1', '#2', '#5', '#6']
Evil probabilities: #7=67%, #8=67%, #3=33%, #4=33%

#### [15:53:39] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 6 scenarios (roles: {'Pooka', 'Twin_Minion'})

### [15:53:46] Executed #9 -> Pooka (EVIL)

#### [15:53:54] Solver Output
Scenarios: 5/224
Definite evil: ['#9']
Definite good: ['#1', '#2', '#5', '#6']
Evil probabilities: #7=80%, #8=80%, #3=20%, #4=20%

#### [15:53:54] Recommendation
Action: **USE_ABILITY** #3 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 0.971 (adjusted 0.777) | timing x1.00
WARNING: Corruption risk: 40%

### [15:54:39] Revealed #3 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': None}

### [15:54:39] Ability used at #3

#### [15:54:39] Solver Output
Scenarios: 3/224
Definite evil: ['#9']
Definite good: ['#1', '#2', '#5', '#6']
Evil probabilities: #7=67%, #8=67%, #3=33%, #4=33%

#### [15:54:39] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.000 (adjusted 0.000) | timing x1.00
WARNING: Corruption risk: 33%

### [15:55:19] Revealed #8 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [15:55:19] Ability used at #8

#### [15:55:19] Solver Output
Scenarios: 3/224
Definite evil: ['#9']
Definite good: ['#1', '#2', '#5', '#6']
Evil probabilities: #7=67%, #8=67%, #3=33%, #4=33%

#### [15:55:19] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (33% evil Minion, 33% good Oracle, 33% evil Twin_Minion).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [15:55:26] Executed #7 -> Minion (EVIL)

#### [15:55:35] Solver Output
Scenarios: 1/31
Definite evil: ['#7', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']

#### [15:55:35] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Twin_Minion'})

### [15:55:42] Executed #8 -> Twin Minion (EVIL)

## [15:55:57] GAME OVER — WIN
Final HP: 10
Notes: 10HP perfect, PD+Dreamer+2xDruid abilities, ASC64 COMPLETE 7/7


---

# New Game — 2026-04-12 15:57:19
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Bard, Dreamer, Enlightened, Slayer, Knight
- Outcasts: Drunk
- Minions: Twin_Minion, Minion
- Demons: Lilis

### [15:58:37] Revealed #2 Bard
Info: {'corruption_distance': 1}

### [15:58:37] Revealed #3 Knight
Info: {}

### [15:58:37] Revealed #7 Knight
Info: {}

### [15:58:37] Revealed #9 Enlightened
Info: {'direction': 'CCW'}

### [15:58:47] Revealed #1 Druid
Info: {}

### [15:58:47] Revealed #4 Slayer
Info: {}

### [15:58:47] Revealed #5 Slayer
Info: {}

### [15:58:47] Revealed #8 Druid
Info: {}

#### [15:58:48] Solver Output
Scenarios: 324/3024
Definite good: ['#6']
Evil probabilities: #8=52%, #4=43%, #5=43%, #7=43%, #3=41%, #2=39%, #1=20%, #9=20%

#### [15:58:48] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.999 (adjusted 0.999) | timing x1.00

### [15:59:30] Revealed #8 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [15:59:30] Ability used at #8

#### [15:59:30] Solver Output
Scenarios: 138/3024
Definite good: ['#6', '#9']
Evil probabilities: #8=65%, #4=48%, #5=48%, #7=48%, #3=39%, #2=35%, #1=17%

#### [15:59:30] Recommendation
Action: **USE_ABILITY** #1 (Druid) -> targets ['#2', '#3', '#4']
Reason: Entropy 0.988 (adjusted 0.859) | timing x1.00
WARNING: Corruption risk: 26%

### [16:00:15] Revealed #1 Druid
Info: {'targets': [2, 3, 4], 'found_outcast': 'Drunk'}

### [16:00:15] Ability used at #1

#### [16:00:16] Solver Output
Scenarios: 114/3024
Definite good: ['#6', '#9']
Evil probabilities: #8=79%, #4=47%, #5=47%, #7=47%, #3=37%, #1=21%, #2=21%

#### [16:00:16] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#8']
Reason: Target #8 is 79% evil (adjusted 0.79)

### [16:01:07] Ability used at #4

#### [16:01:07] Solver Output
Scenarios: 72/3024
Definite good: ['#6', '#9']
Evil probabilities: #4=75%, #8=67%, #1=33%, #2=33%, #5=33%, #7=33%, #3=25%

#### [16:01:07] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#4']
Reason: Target #4 is 75% evil (adjusted 0.75)

### [16:01:55] Ability used at #5

#### [16:01:55] Solver Output
Scenarios: 12/336
Definite evil: ['#4']
Definite good: ['#5', '#6', '#9']
Evil probabilities: #8=83%, #7=50%, #3=33%, #1=17%, #2=17%

#### [16:01:55] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (42% evil Lilis, 42% evil Minion, 17% good Druid).
WARNING: Execution lookahead override -- immediate hit chance is 83%, but all reveal branches still lead to a forced win.

### [16:02:02] Executed #8 -> Lilis (EVIL)

#### [16:02:13] Solver Output
Scenarios: 5/42
Definite evil: ['#4', '#8']
Definite good: ['#1', '#2', '#5', '#6', '#9']
Evil probabilities: #7=60%, #3=40%

#### [16:02:13] Recommendation
Action: **EXECUTE** #7
Reason: Knight free check: #7 is 60% evil. If real Knight, execution blocked (confirms good, 0 HP). If evil disguise, evil dies. No corruption risk.

#### [16:03:07] Execution Blocked
#7 Knight immunity — confirmed good, no HP loss

#### [16:03:07] Solver Output
Scenarios: 2/36
Definite evil: ['#3', '#4', '#8']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#9']

#### [16:03:07] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Minion'})

### [16:03:14] Executed #3 -> Minion (EVIL)

## [16:03:29] GAME OVER — WIN
Final HP: 6
Notes: 6HP, Lilis game, 2xDruid+2xSlayer+Knight check


---

# New Game — 2026-04-12 16:04:27
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Fortune_Teller, Baker, Lover, Empress, Judge, Knitter, Slayer, Knight
- Outcasts: Bombardier, Plague_Doctor
- Minions: Poisoner, Witch
- Demons: Lilis

### [16:05:24] Revealed #1 Bombardier
Info: {}

### [16:05:24] Revealed #4 Knight
Info: {}

### [16:05:24] Revealed #7 Knitter
Info: {'evil_pairs': 0}

### [16:05:24] Revealed #8 Baker
Info: {'original_role': 'original'}

### [16:05:35] Revealed #2 Plague_Doctor
Info: {}

### [16:05:35] Revealed #3 Slayer
Info: {}

### [16:05:36] Revealed #6 Judge
Info: {}

### [16:05:36] Revealed #9 Fortune_Teller
Info: {}

#### [16:05:36] Solver Output
Scenarios: 898/3428
Definite good: ['#5']
Evil probabilities: #8=57%, #6=44%, #7=39%, #4=37%, #10=35%, #3=30%, #9=27%, #1=24%, #2=8%

#### [16:05:36] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#9']
Reason: Entropy 2.199 (adjusted 2.199) | timing x1.00

### [16:06:51] Ability used at #2

#### [16:06:51] Solver Output
Scenarios: 562/3428
Definite good: ['#5']
Evil probabilities: #8=52%, #6=44%, #9=42%, #4=38%, #7=38%, #3=31%, #10=29%, #1=23%, #2=2%

#### [16:06:52] Recommendation
Action: **USE_ABILITY** #9 (Fortune Teller) -> targets ['#5', '#8']
Reason: Entropy 0.996 (adjusted 0.984) | follow-up bonus 0.342 | timing x1.00
WARNING: Corruption risk: 2%

### [16:07:33] Revealed #9 Fortune Teller
Info: {'targets': [5, 8], 'has_evil': True}

### [16:07:33] Ability used at #9

#### [16:07:33] Solver Output
Scenarios: 302/3428
Definite good: ['#5']
Evil probabilities: #8=57%, #6=44%, #9=42%, #4=40%, #7=36%, #3=30%, #10=29%, #1=22%, #2=1%

#### [16:07:33] Recommendation
Action: **USE_ABILITY** #6 (Judge) -> targets ['#10']
Reason: Expected posterior 188.1 scenarios (adjusted 211.1, info gain 0.516 bits) | timing x1.00
WARNING: Corruption risk: 25% -- corrupted Judge results are unreliable

### [16:08:17] Revealed #6 Judge
Info: {'target': 10, 'is_lying': True}

### [16:08:17] Ability used at #6

#### [16:08:17] Solver Output
Scenarios: 192/3428
Definite good: ['#5']
Evil probabilities: #8=62%, #7=41%, #4=39%, #9=36%, #10=35%, #6=34%, #3=27%, #1=24%, #2=1%

#### [16:08:17] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#8']
Reason: Target #8 is 62% evil (adjusted 0.44)
WARNING: Corruption risk: 29% -- Slayer ability disabled if corrupted

### [16:09:10] Ability used at #3

#### [16:09:10] Solver Output
Scenarios: 142/3428
Definite good: ['#5']
Evil probabilities: #8=49%, #9=49%, #4=38%, #3=37%, #6=37%, #10=35%, #7=34%, #1=20%, #2=1%

#### [16:09:10] Recommendation
Action: **EXECUTE** #8
Reason: No reveals available. #8 is 49% likely evil (HP=6, budget=1 wrong execs)
WARNING: Probabilistic execution -- 49% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: Bombardier safety: executing #8 (49%) despite low confidence — Bombardier candidate(s) [1] risk instant game loss if executed first.

### [16:10:05] Executed #8 -> GOOD (WRONG!)

#### [16:10:06] Solver Output
Scenarios: 72/2282
Definite good: ['#5', '#8']
Evil probabilities: #9=97%, #10=44%, #6=42%, #4=36%, #7=31%, #3=25%, #1=22%, #2=3%

#### [16:10:06] Recommendation
Action: **EXECUTE** #9
Reason: No reveals available. #9 is 97% likely evil (HP=1, budget=0 wrong execs)
WARNING: Probabilistic execution -- 97% confident (budget: 0 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [16:10:38] Executed #9 -> Witch (EVIL)

#### [16:10:38] Solver Output
Scenarios: 34/292
Definite evil: ['#9']
Definite good: ['#2', '#5', '#8']
Evil probabilities: #6=44%, #10=44%, #4=32%, #7=32%, #1=24%, #3=24%

#### [16:10:38] Recommendation
Action: **ERROR** #10
Reason: #10 is 44% likely evil but HP too low to risk (HP=1, cost=5, threshold=95%). Need more info.
WARNING: Probabilistic execution -- 44% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [16:11:20] Revealed #10 Empress
Info: {'targets': [3, 6, 8]}

#### [16:11:30] Solver Output
Scenarios: 27/304
Definite evil: ['#9']
Definite good: ['#2', '#5', '#8']
Evil probabilities: #4=48%, #10=48%, #7=41%, #6=33%, #1=26%, #3=4%

#### [16:11:30] Recommendation
Action: **ERROR** #4
Reason: #4 is 48% likely evil but HP too low to risk (HP=1, cost=5, threshold=95%). Need more info.
WARNING: Probabilistic execution -- 48% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

#### [16:12:15] Execution Blocked
#4 Knight immunity — confirmed good, no HP loss

#### [16:12:15] Solver Output
Scenarios: 14/217
Definite evil: ['#9']
Definite good: ['#2', '#4', '#5', '#8']
Evil probabilities: #7=79%, #10=71%, #1=21%, #6=21%, #3=7%

#### [16:12:15] Recommendation
Action: **ERROR** #7
Reason: #7 is 79% likely evil but HP too low to risk (HP=1, cost=5, threshold=95%). Need more info.
WARNING: Probabilistic execution -- 79% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [16:13:34] Executed #7 -> GOOD (WRONG!)

## [16:13:35] GAME OVER — LOSS
Final HP: 0
Notes: Loss: 49% pick on #8 wrong, then 79% on #7 wrong. Lilis+Witch+Poisoner, very hard game.

## [16:13:47] GAME OVER — LOSS
Final HP: 0
Notes: Loss: Lilis+Witch+Poisoner. 49% pick wrong on #8, 79% wrong on #7. True evils: #1=Poisoner,#9=Witch,#10=Lilis


---

# New Game — 2026-04-12 17:30:48
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Knight, Empress, Hunter, Dreamer, Slayer, Scout
- Outcasts: Doppelganger, Drunk, Bombardier
- Minions: Chancellor
- Demons: Pooka

### [17:31:58] Revealed #1 Empress
Info: {'targets': [3, 5, 8]}

### [17:31:58] Revealed #2 Hunter
Info: {'distance': 3}

### [17:31:58] Revealed #3 Bombardier
Info: {}

### [17:31:58] Revealed #4 Scout
Info: {'evil_role': 'Pooka', 'distance': 2}

### [17:31:58] Revealed #5 Empress
Info: {'targets': [3, 4, 8]}

### [17:31:58] Revealed #6 Architect
Info: {'side': 'Right'}

### [17:34:58] Revealed #7 Slayer
Info: {}

### [17:35:01] Revealed #8 Slayer
Info: {}

### [17:37:06] Revealed #9 Dreamer
Info: {}

#### [17:37:09] Solver Output
Scenarios: 243/3756
Evil probabilities: #3=49%, #6=37%, #5=25%, #9=24%, #7=18%, #2=14%, #4=14%, #8=12%, #1=8%

#### [17:37:09] Recommendation
Action: **USE_ABILITY** #9 (Dreamer) -> targets ['#3']
Reason: Entropy 2.103 (adjusted 1.839) | timing x1.00
WARNING: Corruption risk: 25%

### [17:37:40] Revealed #9 Dreamer
Info: {'target': 3, 'evil_role': 'Chancellor'}

### [17:37:43] Ability used at #9

#### [17:37:47] Solver Output
Scenarios: 178/3756
Evil probabilities: #5=34%, #9=33%, #3=31%, #6=30%, #7=20%, #4=16%, #8=16%, #1=11%, #2=10%

#### [17:37:47] Recommendation
Action: **USE_ABILITY** #7 (Slayer) -> targets ['#5']
Reason: Target #5 is 34% evil (adjusted 0.26)
WARNING: Corruption risk: 24% -- Slayer ability disabled if corrupted

#### [17:38:30] Solver Output
Scenarios: 158/3756
Evil probabilities: #3=35%, #9=33%, #5=25%, #6=25%, #7=23%, #4=18%, #8=18%, #1=12%, #2=11%

#### [17:38:30] Recommendation
Action: **USE_ABILITY** #8 (Slayer) -> targets ['#9']
Reason: Target #9 is 33% evil (adjusted 0.24)
WARNING: Corruption risk: 27% -- Slayer ability disabled if corrupted

#### [17:39:10] Solver Output
Scenarios: 120/3756
Evil probabilities: #5=33%, #6=33%, #7=30%, #8=23%, #3=22%, #4=22%, #1=16%, #9=12%, #2=8%

#### [17:39:10] Recommendation
Action: **EXECUTE** #5
Reason: No reveals available. #5 is 33% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 33% confident (budget: 2 wrong execs)
WARNING: Low confidence (33% < 60%) -- consider gathering more info

### [17:39:57] Executed #5 -> GOOD (WRONG!)

#### [17:40:05] Solver Output
Scenarios: 72/2898
Definite good: ['#5']
Evil probabilities: #6=44%, #8=39%, #3=31%, #4=31%, #1=26%, #9=15%, #2=8%, #7=6%

#### [17:40:05] Recommendation
Action: **EXECUTE** #6
Reason: No reveals available. #6 is 44% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 44% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #6 (44%) despite low confidence — Bombardier candidate(s) [3] risk instant game loss if executed first.

### [17:40:42] Executed #6 -> Chancellor (EVIL)

#### [17:40:46] Solver Output
Scenarios: 28/414
Definite evil: ['#6']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#9']
Evil probabilities: #8=71%, #3=29%

#### [17:40:46] Recommendation
Action: **EXECUTE** #8
Reason: No reveals available. #8 is 71% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 71% confident (budget: 1 wrong execs)
WARNING: Bombardier safety: executing #8 (71%) despite low confidence — Bombardier candidate(s) [3] risk instant game loss if executed first.

### [17:41:30] Executed #8 -> Pooka (EVIL)

## [17:41:36] GAME OVER — WIN
Final HP: 5
Notes: Win with 5HP. Wrong exec on #5 Empress (33% pick). Slayer abilities both no-kill. Dreamer random info.


---

# New Game — 2026-04-12 17:43:45
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Alchemist, Bishop, Fortune_Teller, Slayer, Poet
- Outcasts: Bombardier, Doppelganger, Wretch, Plague_Doctor, Drunk
- Minions: Chancellor
- Demons: Baa

### [17:44:39] Revealed #1 Bishop
Info: {'targets': [5, 3, 6], 'types': ['Villager', 'Outcast', 'Minion']}

### [17:44:39] Revealed #4 Bishop
Info: {'targets': [7, 3, 6], 'types': ['Villager', 'Outcast', 'Minion']}

### [17:44:39] Revealed #5 Bombardier
Info: {}

### [17:44:39] Revealed #8 Alchemist
Info: {'cured_count': 2}

### [17:46:04] Revealed #2 Poet
Info: {'evil_pairs': 1, 'copied_role': 'Knitter'}

### [17:46:10] Revealed #3 Plague_Doctor
Info: {}

### [17:46:14] Revealed #6 Slayer
Info: {}

### [17:46:17] Revealed #7 Fortune_Teller
Info: {}

#### [17:46:22] Solver Output
Scenarios: 215/8390
Definite good: ['#6']
Evil probabilities: #8=61%, #1=41%, #4=40%, #2=25%, #5=17%, #7=13%, #3=2%

#### [17:46:22] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#1']
Reason: Entropy 2.144 (adjusted 2.144) | timing x1.00

#### [17:47:21] Solver Output
Scenarios: 5/8390
Definite evil: ['#3']
Definite good: ['#1', '#2', '#5', '#6', '#7']
Evil probabilities: #4=60%, #8=40%

#### [17:47:21] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 5 scenarios (roles: {'Chancellor'})

### [17:47:28] Executed #3 -> Chancellor (EVIL)

#### [17:47:34] Solver Output
Scenarios: 5/396
Definite evil: ['#3']
Definite good: ['#1', '#2', '#5', '#6', '#7']
Evil probabilities: #4=60%, #8=40%

#### [17:47:34] Recommendation
Action: **USE_ABILITY** #7 (Fortune Teller) -> targets ['#2', '#4']
Reason: Entropy 0.971 (adjusted 0.971) | follow-up bonus 0.320 | timing x1.00

### [17:48:01] Revealed #7 Fortune Teller
Info: {'targets': [2, 4], 'has_evil': True}

### [17:48:05] Ability used at #7

#### [17:48:09] Solver Output
Scenarios: 3/396
Definite evil: ['#3', '#4']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8']

#### [17:48:09] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 3 scenarios (roles: {'Baa'})

### [17:48:16] Executed #4 -> GOOD (WRONG!)

#### [17:48:30] Solver Output
Scenarios: 0/365

#### [17:48:30] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

## [17:49:36] GAME OVER — LOSS
Final HP: 5
Notes: SOLVER BUG: 100% confidence wrong exec on #4 (was Good Doppelganger, solver said Baa). 0 scenarios after. Likely FT-as-Drunk not handled - FT #7 was Drunk, gave wrong result, solver trusted it. PD #3 was evil Chancellor, PD check result should be lies.

## [17:49:52] GAME OVER — LOSS
Final HP: 5
Notes: SOLVER BUG: 100% confidence wrong exec on #4 (was Good Doppelganger, solver said Baa). True evils: #3=Chancellor, #8=Baa. FT #7 was Drunk, gave wrong result, solver trusted it. PD #3 was evil Chancellor, PD check result should be lies.

### [18:01:53] Revealed #2 Poet
Info: {'targets': [1, 3], 'copied_role': 'Empress'}

#### [18:01:58] Solver Output
Scenarios: 8/365
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7']

#### [18:01:58] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 8 scenarios (roles: {'Baa'})

#### [18:02:09] Solver Output
Scenarios: 8/365
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7']

#### [18:02:09] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 8 scenarios (roles: {'Baa'})


---

# New Game — 2026-04-12 18:03:46
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Judge, Medium, Fortune_Teller, Empress
- Outcasts: Doppelganger, Plague_Doctor, Wretch, Bombardier
- Minions: Shaman
- Demons: Baa

### [18:04:36] Revealed #1 Empress
Info: {'targets': [2, 6, 7]}

### [18:04:36] Revealed #3 Bombardier
Info: {}

### [18:04:36] Revealed #4 Lover
Info: {'evil_adjacent': 1}

### [18:04:36] Revealed #7 Medium
Info: {'good_position': 3, 'good_role': 'Bombardier'}

### [18:04:36] Revealed #8 Lover
Info: {'evil_adjacent': 1}

### [18:05:05] Revealed #2 Fortune_Teller
Info: {}

### [18:05:05] Revealed #5 Plague_Doctor
Info: {}

### [18:05:05] Revealed #6 Fortune_Teller
Info: {}

#### [18:05:10] Solver Output
Scenarios: 90/1268
Definite good: ['#2', '#6']
Evil probabilities: #1=56%, #3=44%, #4=44%, #7=33%, #5=11%, #8=11%

#### [18:05:10] Recommendation
Action: **USE_ABILITY** #5 (Plague Doctor) -> targets ['#2']
Reason: Entropy 1.669 (adjusted 1.669) | timing x1.00

#### [18:06:07] Solver Output
Scenarios: 20/1268
Definite good: ['#2', '#4', '#6', '#8']
Evil probabilities: #1=50%, #3=50%, #5=50%, #7=50%

#### [18:06:07] Recommendation
Action: **USE_ABILITY** #6 (Fortune Teller) -> targets ['#1', '#2']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [18:06:31] Revealed #6 Fortune Teller
Info: {'targets': [1, 2], 'has_evil': False}

### [18:06:31] Ability used at #6

#### [18:06:35] Solver Output
Scenarios: 12/1268
Definite evil: ['#3', '#7']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#8']

#### [18:06:35] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 12 scenarios (roles: {'Baa', 'Shaman'})

### [18:06:42] Executed #3 -> Shaman (EVIL)

#### [18:06:48] Solver Output
Scenarios: 6/223
Definite evil: ['#3', '#7']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#8']

#### [18:06:48] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 6 scenarios (roles: {'Baa'})

### [18:06:55] Executed #7 -> Baa (EVIL)

## [18:07:02] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. PD revealed #7 evil + #2 corrupted. FT confirmed #1,#2 clean. Both evils 100% confident.


---

# New Game — 2026-04-12 18:09:02
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Empress, Hunter, Bishop, Lover, Scout, Druid
- Outcasts: Plague_Doctor, Drunk
- Minions: Witch
- Demons: Baa

### [18:09:31] Revealed #1 Scout
Info: {'evil_role': 'Witch', 'distance': 1}

### [18:09:31] Revealed #2 Bishop
Info: {'targets': [4, 7, 1], 'types': ['Villager', 'Outcast', 'Minion']}

### [18:09:31] Revealed #3 Empress
Info: {'targets': [5, 6, 8]}

### [18:09:31] Revealed #5 Hunter
Info: {'distance': 4}

### [18:09:31] Revealed #6 Medium
Info: {'good_position': 3, 'good_role': 'Empress'}

### [18:09:31] Revealed #7 Lover
Info: {'evil_adjacent': 2}

### [18:09:39] Revealed #4 Plague_Doctor
Info: {}

#### [18:09:44] Solver Output
Scenarios: 15/1610
Definite good: ['#1', '#3', '#6']
Evil probabilities: #5=67%, #7=67%, #8=33%, #2=27%, #4=7%

#### [18:09:44] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#1']
Reason: Entropy 2.092 (adjusted 2.092) | timing x1.00

#### [18:10:31] Solver Output
Scenarios: 5/1610
Definite evil: ['#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']

#### [18:10:31] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 5 scenarios (roles: {'Witch', 'Baa'})

### [18:10:38] Executed #7 -> Witch (EVIL)

#### [18:10:43] Solver Output
Scenarios: 3/223
Definite evil: ['#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#6']

#### [18:10:43] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 3 scenarios (roles: {'Baa'})

### [18:10:50] Executed #8 -> Baa (EVIL)

## [18:11:00] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. PD confirmed #1 clean. Both evils 100% confident. Witch blocked #8, executed Witch first to unblock.


---

# New Game — 2026-04-12 18:13:03
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Poet, Oracle, Alchemist, Druid, Architect, Slayer
- Outcasts: Bombardier, Plague_Doctor, Wretch
- Minions: Minion, Chancellor
- Demons: Baa

### [18:13:39] Revealed #2 Bombardier
Info: {}

### [18:13:39] Revealed #4 Baker
Info: {'original_role': 'original'}

### [18:13:39] Revealed #5 Architect
Info: {'side': 'Left'}

### [18:13:39] Revealed #6 Wretch
Info: {}

### [18:13:39] Revealed #7 Oracle
Info: {'targets': [1, 4], 'minion_role': 'Chancellor'}

### [18:13:39] Revealed #8 Baker
Info: {'original_role': 'Druid'}

### [18:13:39] Revealed #9 Baker
Info: {'original_role': 'Slayer'}

### [18:14:03] Revealed #1 Druid
Info: {}

### [18:14:03] Revealed #3 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 4}

#### [18:14:08] Solver Output
Scenarios: 12/636
Definite good: ['#2', '#6']
Evil probabilities: #8=83%, #3=67%, #7=50%, #4=33%, #9=33%, #1=17%, #5=17%

#### [18:14:08] Recommendation
Action: **USE_ABILITY** #1 (Druid) -> targets ['#2', '#3', '#4']
Reason: Entropy 0.650 (adjusted 0.650) | timing x1.00

### [18:14:55] Revealed #1 Druid
Info: {'targets': [2, 3, 4], 'found_outcast': None}

### [18:14:55] Ability used at #1

#### [18:15:00] Solver Output
Scenarios: 2/636
Definite evil: ['#1', '#3', '#5']
Definite good: ['#2', '#4', '#6', '#7', '#8', '#9']

#### [18:15:00] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Chancellor'})

### [18:15:07] Executed #1 -> Chancellor (EVIL)

#### [18:15:13] Solver Output
Scenarios: 2/56
Definite evil: ['#1', '#3', '#5']
Definite good: ['#2', '#4', '#6', '#7', '#8', '#9']

#### [18:15:13] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Minion', 'Baa'})

### [18:15:20] Executed #3 -> Minion (EVIL)

#### [18:15:24] Solver Output
Scenarios: 1/7
Definite evil: ['#1', '#3', '#5']
Definite good: ['#2', '#4', '#6', '#7', '#8', '#9']

#### [18:15:24] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Baa'})

### [18:15:31] Executed #5 -> Baa (EVIL)

## [18:15:40] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Triple Baker chain. All 3 evils 100% confident. Only 2 scenarios after reveals+Druid ability.


---

# New Game — 2026-04-12 18:17:45
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Jester, Confessor, Lover, Architect, Alchemist, Poet, Gemcrafter
- Outcasts: Bombardier, Plague_Doctor
- Minions: Minion, Witch
- Demons: Lilis

### [18:20:06] Revealed #1 Alchemist
Info: {'cured_count': 0}

### [18:20:06] Revealed #3 Confessor
Info: {'dizzy': False}

### [18:20:06] Revealed #6 Gemcrafter
Info: {'good_position': 1}

### [18:20:06] Revealed #7 Bombardier
Info: {}

### [18:20:06] Revealed #8 Architect
Info: {'side': 'Left'}

### [18:20:06] Revealed #9 Confessor
Info: {'dizzy': True}

### [18:21:02] Revealed #2 Jester
Info: {}

### [18:21:03] Revealed #4 Poet
Info: {'evil_pairs': 0, 'copied_role': 'Knitter'}

#### [18:21:14] Solver Output
Scenarios: 6/5382
Definite evil: ['#2', '#7', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#8', '#10']

#### [18:21:14] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 6 scenarios (roles: {'Lilis', 'Witch', 'Minion'})

### [18:21:54] Executed #2 -> Lilis (EVIL)

#### [18:21:59] Solver Output
Scenarios: 2/540
Definite evil: ['#2', '#7', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#8', '#10']

#### [18:21:59] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 2 scenarios (roles: {'Minion', 'Witch'})

### [18:22:06] Executed #7 -> Witch (EVIL)

#### [18:22:11] Solver Output
Scenarios: 1/55
Definite evil: ['#2', '#7', '#9']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#8', '#10']

#### [18:22:11] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Minion'})

### [18:22:18] Executed #9 -> Minion (EVIL)

## [18:22:26] GAME OVER — WIN
Final HP: 6
Notes: ASC65 COMPLETE! 6HP. Lilis+Witch game. Night killed #5 Lover and #10 PD. All 3 evils 100% confident with only 6 scenarios.


---

# New Game — 2026-04-13 16:05:57
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Dreamer, Alchemist, Fortune_Teller, Architect, Knight, Empress
- Outcasts: Wretch, Bombardier
- Minions: Minion
- Demons: Pooka

### [16:06:56] Revealed #1 Hunter
Info: {'distance': 3}

### [16:06:56] Revealed #2 Alchemist
Info: {'cured_count': 1}

### [16:06:56] Revealed #3 Fortune_Teller
Info: {}

### [16:06:56] Revealed #4 Knight
Info: {}

### [16:06:56] Revealed #5 Dreamer
Info: {}

### [16:06:56] Revealed #6 Architect
Info: {'side': 'Equal'}

### [16:06:56] Revealed #7 Wretch
Info: {}

### [16:06:56] Revealed #8 Bombardier
Info: {}

### [16:06:56] Revealed #9 Empress
Info: {'targets': [1, 7, 8]}

### [16:11:41] Revealed #3 Fortune Teller
Info: {'targets': [5, 1], 'has_evil': True}

### [16:11:46] Ability used at #3

#### [16:11:54] Solver Output
Scenarios: 3/72
Definite good: ['#3', '#4', '#7', '#8']
Evil probabilities: #6=67%, #1=33%, #2=33%, #5=33%, #9=33%

#### [16:11:54] Recommendation
Action: **USE_ABILITY** #5 (Dreamer) -> targets ['#1']
Reason: Entropy 1.585 (adjusted 1.585) | timing x1.00

### [16:12:28] Revealed #5 Dreamer
Info: {'target': 1, 'evil_role': 'Minion'}

### [16:12:31] Ability used at #5

#### [16:12:35] Solver Output
Scenarios: 2/72
Definite evil: ['#6']
Definite good: ['#1', '#3', '#4', '#7', '#8', '#9']
Evil probabilities: #2=50%, #5=50%

#### [16:12:35] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Minion'})

### [16:12:42] Executed #6 -> Minion (EVIL)

#### [16:12:48] Solver Output
Scenarios: 2/8
Definite evil: ['#6']
Definite good: ['#1', '#3', '#4', '#7', '#8', '#9']
Evil probabilities: #2=50%, #5=50%

#### [16:12:48] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (50% good Alchemist, 50% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [16:12:55] Executed #2 -> GOOD (WRONG!)

#### [16:13:02] Solver Output
Scenarios: 1/7
Definite evil: ['#5', '#6']
Definite good: ['#1', '#2', '#3', '#4', '#7', '#8', '#9']

#### [16:13:02] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [16:13:09] Executed #5 -> Pooka (EVIL)

## [16:13:35] GAME OVER — WIN
Final HP: 5
Notes: FT used on 5,1 True; Dreamer 5 said #1 Minion (lying); 50-50 on #2 lost but lookahead survived


---

# New Game — 2026-04-13 16:16:04
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Baker, Lover, Druid, Confessor, Witness, Architect
- Outcasts: Doppelganger, Wretch
- Minions: Minion
- Demons: Baa

### [16:16:51] Revealed #1 Confessor
Info: {'dizzy': False}

### [16:16:51] Revealed #3 Baker
Info: {'original_role': 'original'}

### [16:16:51] Revealed #4 Wretch
Info: {}

### [16:16:51] Revealed #5 Druid
Info: {}

### [16:16:51] Revealed #6 Baker
Info: {'original_role': 'Lover'}

### [16:16:51] Revealed #7 Architect
Info: {'side': 'Right'}

### [16:17:38] Revealed #2 Witness
Info: {'affected_position': 0}

#### [16:17:42] Solver Output
Scenarios: 30/222
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3']
Evil probabilities: #4=33%, #5=33%, #6=33%

#### [16:17:42] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 30 scenarios (roles: {'Minion', 'Baa'})

### [16:17:48] Executed #7 -> Minion (EVIL)

#### [16:17:54] Solver Output
Scenarios: 15/31
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3']
Evil probabilities: #4=33%, #5=33%, #6=33%

#### [16:17:54] Recommendation
Action: **USE_ABILITY** #5 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.918 (adjusted 0.918) | timing x1.00

### [16:18:26] Revealed #5 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Doppelganger'}

### [16:18:30] Ability used at #5

#### [16:18:33] Solver Output
Scenarios: 8/31
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3']
Evil probabilities: #4=38%, #6=38%, #5=25%

#### [16:18:33] Recommendation
Action: **EXECUTE** #6
Reason: No reveals available. #6 is 38% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 38% confident (budget: 2 wrong execs)
WARNING: Low confidence (38% < 50%) -- consider gathering more info

### [16:19:24] Executed #6 -> GOOD (WRONG!)

#### [16:19:32] Solver Output
Scenarios: 5/26
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#6']
Evil probabilities: #4=60%, #5=40%

#### [16:19:32] Recommendation
Action: **ERROR** #5
Reason: #5 is 40% likely evil but budget=1 requires >=85% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 40% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 40% < 85% threshold. Consider manual override if you have extra information.

#### [16:19:47] Solver Output
Scenarios: 5/26
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#6']
Evil probabilities: #4=60%, #5=40%

#### [16:19:47] Recommendation
Action: **ERROR** #5
Reason: #5 is 40% likely evil but budget=1 requires >=85% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 40% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 40% < 85% threshold. Consider manual override if you have extra information.

#### [16:19:59] Claude Reasoning


### [16:20:52] Executed #5 -> Baa (EVIL)

## [16:20:57] GAME OVER — WIN
Final HP: 5
Notes: Baker chain (3 orig, 6 from Lover); Druid claimed Doppelganger in 1,2,3 (lying); 40% gamble on #5 paid off


---

# New Game — 2026-04-13 16:23:31
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Druid, Gemcrafter, Bishop, Judge, Scout, Architect, Medium
- Outcasts: Bombardier, Drunk
- Minions: Puppeteer, Shaman
- Demons: Lilis

### [16:25:18] Revealed #1 Druid
Info: {}

### [16:25:18] Revealed #2 Gemcrafter
Info: {'good_position': 9}

### [16:25:18] Revealed #3 Bishop
Info: {'targets': [7, 8, 2], 'types': ['Villager', 'Outcast', 'Minion']}

### [16:25:18] Revealed #4 Bishop
Info: {'targets': [10, 7, 2], 'types': ['Villager', 'Outcast', 'Minion']}

### [16:26:48] Revealed #5 Scout
Info: {'evil_role': 'Shaman', 'distance': 1}

### [16:26:48] Revealed #7 Architect
Info: {'side': 'Left'}

### [16:26:48] Revealed #9 Bombardier
Info: {}

### [16:26:55] Revealed #8 Enlightened
Info: {'direction': 'CW'}

#### [16:26:59] Solver Output
Scenarios: 10/2394
Definite evil: ['#10']
Definite good: ['#6', '#7']
Evil probabilities: #3=80%, #1=70%, #5=70%, #4=50%, #2=10%, #8=10%, #9=10%

#### [16:26:59] Recommendation
Action: **EXECUTE** #10
Reason: #10 is evil in ALL 10 scenarios (roles: {'Puppeteer', 'Lilis', 'Shaman', 'Puppet'})

#### [16:27:23] Solver Output
Scenarios: 10/2394
Definite evil: ['#10']
Definite good: ['#6', '#7']
Evil probabilities: #3=80%, #1=70%, #5=70%, #4=50%, #2=10%, #8=10%, #9=10%

#### [16:27:23] Recommendation
Action: **EXECUTE** #10
Reason: #10 is evil in ALL 10 scenarios (roles: {'Puppeteer', 'Lilis', 'Puppet', 'Shaman'})

### [16:28:31] Revealed #1 Druid
Info: {'targets': [2, 3, 9], 'found_outcast': None}

### [16:28:35] Ability used at #1

#### [16:28:39] Solver Output
Scenarios: 5/2394
Definite evil: ['#1', '#10']
Definite good: ['#6', '#7']
Evil probabilities: #3=60%, #4=40%, #5=40%, #2=20%, #8=20%, #9=20%

#### [16:28:39] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 5 scenarios (roles: {'Lilis', 'Puppeteer'})

### [16:28:46] Executed #1 -> Lilis (EVIL)

#### [16:28:53] Solver Output
Scenarios: 2/177
Definite evil: ['#1', '#10']
Definite good: ['#2', '#5', '#6', '#7']
Evil probabilities: #3=50%, #4=50%, #8=50%, #9=50%

#### [16:28:53] Recommendation
Action: **EXECUTE** #10
Reason: #10 is evil in ALL 2 scenarios (roles: {'Shaman'})

### [16:30:59] Executed #8 -> Puppet (EVIL)

#### [16:31:04] Solver Output
Scenarios: 1/13
Definite evil: ['#1', '#8', '#9', '#10']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7']

#### [16:31:04] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Puppeteer'})

### [16:31:11] Executed #9 -> Puppeteer (EVIL)

## [16:31:39] GAME OVER — WIN
Final HP: 6
Notes: Lilis game; 2 nights: #6 Med killed, #10 Shaman killed; Puppet at #8 (from Puppeteer); strategy bug: keeps recommending dead #10

