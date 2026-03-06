# Game Session State

---

# New Game — 2026-03-06 16:33:19
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Confessor, Hunter, Lover, Alchemist, Judge, Jester
- Outcasts: Plague_Doctor
- Minions: Poisoner, Minion
- Demons: Pooka

### [16:40:13] Revealed #2 Hunter
Info: {'distance': 2}

### [16:40:13] Revealed #3 Alchemist
Info: {'cured_count': 2}

### [16:40:14] Revealed #4 Confessor
Info: {'dizzy': True}

### [16:40:15] Revealed #5 Judge
Info: {}

### [16:40:16] Revealed #6 Hunter
Info: {'distance': 2}

### [16:40:16] Revealed #7 Lover
Info: {'evil_adjacent': 0}

### [16:40:17] Revealed #8 Medium
Info: {'good_position': 5, 'good_role': 'Judge'}

### [16:40:18] Revealed #9 Plague_Doctor
Info: {}

### [16:41:55] Revealed #1 Jester
Info: {}

#### [16:42:02] Solver Output
Scenarios: 79/2308
Evil probabilities: #2=66%, #3=65%, #6=56%, #4=42%, #1=39%, #7=18%, #5=11%, #8=3%, #9=1%
  Generated 2308 candidate scenarios
  79 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [16:42:02] Recommendation
Action: **USE_ABILITY** #5 (Judge) -> targets ['#6']
Reason: Entropy 0.999 (adjusted 0.702) | timing x1.00
WARNING: Corruption risk: 59%

### [16:42:50] Ability used at #5

### [16:42:50] Revealed #5 Judge
Info: {'target': 6, 'is_lying': True}

#### [16:42:53] Solver Output
Scenarios: 65/2308
Definite good: ['#5', '#8']
Evil probabilities: #3=69%, #2=63%, #6=63%, #1=42%, #4=40%, #7=22%, #9=2%
  Generated 2308 candidate scenarios
  65 scenarios survived validation
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 9]

#### [16:42:53] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#3', '#5', '#6']
Reason: Expected posterior 39.2 scenarios (adjusted 44.6) | timing x1.00
WARNING: Corruption risk: 28%

### [16:43:33] Ability used at #1

### [16:43:33] Revealed #1 Jester
Info: {'targets': [3, 5, 6], 'evil_count': 0}

#### [16:43:36] Solver Output
Scenarios: 35/2308
Definite good: ['#5', '#8', '#9']
Evil probabilities: #3=86%, #6=66%, #2=60%, #1=49%, #4=29%, #7=11%
  Generated 2308 candidate scenarios
  35 scenarios survived validation
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7]

#### [16:43:36] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 86% likely evil (HP=10, can afford 2 wrong execs)
WARNING: Probabilistic execution -- 86% confident

### [16:44:33] Executed #3 -> GOOD (WRONG!)

#### [16:44:37] Solver Output
Scenarios: 5/1476
Definite evil: ['#6']
Definite good: ['#3', '#5', '#8', '#9']
Evil probabilities: #2=60%, #7=60%, #1=40%, #4=40%
  Generated 1476 candidate scenarios
  5 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner', 'Pooka'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 7]

#### [16:44:37] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 5 scenarios (roles: {'Poisoner', 'Pooka'})

### [16:45:10] Executed #6 -> Poisoner (EVIL)

#### [16:45:14] Solver Output
Scenarios: 5/234
Definite evil: ['#6']
Definite good: ['#3', '#5', '#8', '#9']
Evil probabilities: #1=60%, #4=60%, #2=40%, #7=40%
  Generated 234 candidate scenarios
  5 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 7]

#### [16:45:14] Recommendation
Action: **ERROR** #1
Reason: #1 is 60% likely evil but HP too low to risk (HP=5, cost=5). Need more info.
WARNING: Probabilistic execution -- 60% confident
WARNING: CRITICAL: HP=5, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [16:49:56] Executed #2 -> Pooka (EVIL)

#### [16:50:01] Solver Output
Scenarios: 9/58
Definite evil: ['#2', '#6']
Definite good: ['#3', '#5', '#8', '#9']
Evil probabilities: #4=78%, #1=11%, #7=11%
  Generated 58 candidate scenarios
  9 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4, 7]

#### [16:50:01] Recommendation
Action: **ERROR** #4
Reason: #4 is 78% likely evil but HP too low to risk (HP=5, cost=5). Need more info.
WARNING: Probabilistic execution -- 78% confident
WARNING: CRITICAL: HP=5, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

#### [16:51:16] Solver Output
Scenarios: 2/44
Definite evil: ['#2', '#6']
Definite good: ['#3', '#4', '#5', '#8', '#9']
Evil probabilities: #1=50%, #7=50%
  Generated 44 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #6 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [16:51:16] Recommendation
Action: **ERROR** #1
Reason: #1 is 50% likely evil but HP too low to risk (HP=5, cost=5). Need more info.
WARNING: Probabilistic execution -- 50% confident
WARNING: CRITICAL: HP=5, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [16:52:54] Executed #1 -> Minion (EVIL)

## [16:53:04] GAME OVER — WIN
Final HP: 5
Notes: PD ability revealed #2 evil. Manual logic deduced #1 over #7 via Judge corruption analysis.

