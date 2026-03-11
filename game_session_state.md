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

