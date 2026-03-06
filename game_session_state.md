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


---

# New Game — 2026-03-06 17:01:41
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Scout, Slayer, Bishop, Knitter, Poet, Jester
- Outcasts: Bombardier, Plague_Doctor
- Minions: Minion, Witch
- Demons: Pooka

### [17:03:24] Revealed #1 Bombardier
Info: {}

### [17:03:24] Revealed #2 Plague_Doctor
Info: {}

### [17:03:24] Revealed #3 Jester
Info: {}

### [17:03:24] Revealed #4 Knitter
Info: {'evil_pairs': 0}

### [17:03:25] Revealed #5 Scout
Info: {'evil_role': 'Minion', 'distance': 1}

### [17:03:25] Revealed #6 Slayer
Info: {}

### [17:03:25] Revealed #7 Bishop
Info: {'targets': [4, 7, 9]}

### [17:03:25] Revealed #8 Poet
Info: {'evil_role': 'Pooka', 'distance': 2, 'copied_role': 'Scout'}

#### [17:04:27] Solver Output
Scenarios: 152/1428
Evil probabilities: #3=51%, #8=50%, #9=47%, #4=45%, #7=33%, #1=28%, #5=25%, #6=18%, #2=3%
  Generated 1428 candidate scenarios
  152 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [17:04:27] Recommendation
Action: **USE_ABILITY** #6 (Slayer) -> targets ['#3']
Reason: Target #3 is 51% evil (adjusted 0.37)
WARNING: Corruption risk: 28% -- Slayer ability disabled if corrupted

#### [17:06:17] Solver Output
Scenarios: 110/1428
Evil probabilities: #8=54%, #9=46%, #4=45%, #7=39%, #3=33%, #1=27%, #5=27%, #6=25%, #2=4%
  Generated 1428 candidate scenarios
  110 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [17:06:17] Recommendation
Action: **USE_ABILITY** #3 (Jester) -> targets ['#1', '#2', '#4']
Reason: Expected posterior 53.0 scenarios (adjusted 56.7) | timing x1.00
WARNING: Corruption risk: 14%

### [17:07:00] Ability used at #3

### [17:07:00] Revealed #3 Jester
Info: {'targets': [1, 2, 4], 'evil_count': 3}

#### [17:07:04] Solver Output
Scenarios: 51/1428
Evil probabilities: #3=71%, #4=53%, #8=37%, #9=35%, #7=31%, #5=25%, #6=24%, #1=20%, #2=4%
  Generated 1428 candidate scenarios
  51 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [17:07:04] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 71% likely evil (HP=10, can afford 2 wrong execs)
WARNING: Probabilistic execution -- 71% confident

### [17:19:44] Executed #3 -> Witch (EVIL)

#### [17:19:48] Solver Output
Scenarios: 16/206
Definite evil: ['#3']
Definite good: ['#2']
Evil probabilities: #4=50%, #7=38%, #6=31%, #8=25%, #9=25%, #1=19%, #5=12%
  Generated 206 candidate scenarios
  16 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7, 8, 9]

#### [17:19:48] Recommendation
Action: **REVEAL** #9
Reason: #9: 25% evil, entropy 0.811

### [17:20:37] Revealed #9 Bard
Info: {'corruption_distance': 1}

#### [17:20:41] Solver Output
Scenarios: 11/236
Definite evil: ['#3']
Definite good: ['#2', '#9']
Evil probabilities: #4=64%, #7=55%, #1=27%, #6=27%, #8=18%, #5=9%
  Generated 236 candidate scenarios
  11 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7, 8]

#### [17:20:41] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 64% likely evil (HP=10, can afford 2 wrong execs)
WARNING: Probabilistic execution -- 64% confident

#### [17:21:48] Solver Output
Scenarios: 8/236
Definite evil: ['#3']
Definite good: ['#2', '#5', '#8', '#9']
Evil probabilities: #7=75%, #4=62%, #6=38%, #1=25%
  Generated 236 candidate scenarios
  8 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4, 6, 7]

#### [17:21:48] Recommendation
Action: **EXECUTE** #7
Reason: No reveals available. #7 is 75% likely evil (HP=10, can afford 2 wrong execs)
WARNING: Probabilistic execution -- 75% confident

### [17:22:40] Executed #7 -> GOOD (WRONG!)

#### [17:22:46] Solver Output
Scenarios: 2/172
Definite evil: ['#3', '#6']
Definite good: ['#2', '#5', '#7', '#8', '#9']
Evil probabilities: #1=50%, #4=50%
  Generated 172 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #6 is DEFINITELY EVIL (possible roles: {'Pooka', 'Minion'})
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4]

#### [17:22:46] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Pooka', 'Minion'})

### [17:23:26] Executed #6 -> Pooka (EVIL)

#### [17:23:31] Solver Output
Scenarios: 1/32
Definite evil: ['#1', '#3', '#6']
Definite good: ['#2', '#4', '#5', '#7', '#8', '#9']
  Generated 32 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Minion'})
    #3 is DEFINITELY EVIL (possible roles: {'Witch'})
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [17:23:31] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Minion'})

### [17:24:07] Executed #1 -> Minion (EVIL)

## [17:24:07] GAME OVER — WIN
Final HP: 5
Notes: Witch blocked #9 reveal. PD confirmed #8 corrupted, revealed #3 evil. Wrong exec on #7 (corrupted Bishop) narrowed to 2 scenarios, then solver found #6 and #1.


---

# New Game — 2026-03-06 17:38:54
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: 
- Outcasts: 
- Minions: 
- Demons: 

## Deck
- Villagers: 
- Outcasts: 
- Minions: 
- Demons: 

## Deck
- Villagers: Oracle, Medium, Baker, Empress, Slayer, Judge, Bishop
- Outcasts: Wretch, Doppelganger
- Minions: Plague_Doctor, Chancellor, Puppeteer
- Demons: Lilis

## Deck
- Villagers: Oracle, Medium, Baker, Empress, Slayer, Judge, Bishop
- Outcasts: Wretch, Doppelganger, Plague_Doctor
- Minions: Chancellor, Puppeteer
- Demons: Lilis

### [17:46:36] Revealed #1 Empress
Info: {'targets': [2, 6, 7]}

### [17:46:37] Revealed #2 Medium
Info: {'good_position': 9, 'good_role': 'Oracle'}

### [17:46:37] Revealed #3 Plague Doctor
Info: {}

### [17:46:51] Revealed #4 Judge
Info: {}

### [17:46:51] Revealed #5 Slayer
Info: {}

### [17:46:51] Revealed #6 Baker
Info: {}

### [17:46:51] Revealed #7 Medium
Info: {'good_position': 6, 'good_role': 'Baker'}

### [17:46:51] Revealed #8 Wretch
Info: {}

### [17:46:52] Revealed #9 Oracle
Info: {'targets': [3, 9], 'minion_role': 'Puppeteer'}

#### [17:47:00] Solver Output
Scenarios: 573/16842
Definite good: ['#6']
Evil probabilities: #9=61%, #2=57%, #8=49%, #4=48%, #5=44%, #1=41%, #3=23%, #7=17%
  Generated 16842 candidate scenarios
  573 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [17:47:00] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#9']
Reason: Entropy 2.009 (adjusted 2.009) | timing x1.00

#### [17:52:20] Solver Output
Scenarios: 86/16842
Definite good: ['#6', '#9']
Evil probabilities: #1=63%, #3=58%, #4=58%, #8=53%, #5=47%, #2=42%, #7=21%
  Generated 16842 candidate scenarios
  86 scenarios survived validation
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8]

#### [17:52:20] Recommendation
Action: **USE_ABILITY** #4 (Judge) -> targets ['#7']
Reason: Entropy 0.990 (adjusted 0.990) | timing x1.00

### [17:52:52] Revealed #4 Judge
Info: {'target': 7, 'is_lying': False}

#### [17:52:56] Solver Output
Scenarios: 48/16842
Definite good: ['#6', '#9']
Evil probabilities: #1=79%, #3=71%, #8=62%, #5=50%, #4=42%, #2=29%, #7=12%
  Generated 16842 candidate scenarios
  48 scenarios survived validation
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8]

#### [17:52:56] Recommendation
Action: **USE_ABILITY** #4 (Judge) -> targets ['#5']
Reason: Entropy 0.980 (adjusted 0.980) | timing x1.00

### [17:53:27] Ability used at #4

### [17:53:31] Ability used at #3

#### [17:53:31] Solver Output
Scenarios: 48/16842
Definite good: ['#6', '#9']
Evil probabilities: #1=79%, #3=71%, #8=62%, #5=50%, #4=42%, #2=29%, #7=12%
  Generated 16842 candidate scenarios
  48 scenarios survived validation
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8]

#### [17:53:31] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#1']
Reason: Target #1 is 79% evil (adjusted 0.79)

### [17:54:09] Ability used at #5

#### [17:54:10] Solver Output
Scenarios: 26/16842
Definite good: ['#6', '#9']
Evil probabilities: #5=92%, #1=62%, #3=62%, #2=38%, #4=38%, #8=38%, #7=8%
  Generated 16842 candidate scenarios
  26 scenarios survived validation
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8]

#### [17:54:10] Recommendation
Action: **EXECUTE** #5
Reason: No reveals available. #5 is 92% likely evil (HP=6, can afford 3 wrong execs)
WARNING: Probabilistic execution -- 92% confident

### [17:54:43] Executed #5 -> Lilis (EVIL)

#### [17:54:43] Solver Output
Scenarios: 10/2480
Definite evil: ['#5']
Definite good: ['#6', '#7', '#9']
Evil probabilities: #1=80%, #3=70%, #2=30%, #4=30%, #8=30%
  Generated 2480 candidate scenarios
  10 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 8]

#### [17:54:43] Recommendation
Action: **EXECUTE** #1
Reason: No reveals available. #1 is 80% likely evil (HP=6, can afford 3 wrong execs)
WARNING: Probabilistic execution -- 80% confident

### [17:55:16] Executed #1 -> Chancellor (EVIL)

#### [17:55:16] Solver Output
Scenarios: 8/378
Definite evil: ['#1', '#5']
Definite good: ['#6', '#7', '#8', '#9']
Evil probabilities: #3=88%, #4=38%, #2=12%
  Generated 378 candidate scenarios
  8 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4]

#### [17:55:16] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 88% likely evil (HP=6, can afford 3 wrong execs)
WARNING: Probabilistic execution -- 88% confident

### [17:56:01] Executed #3 -> Puppeteer (EVIL)

#### [17:56:01] Solver Output
Scenarios: 27/102
Definite evil: ['#1', '#3', '#5']
Definite good: ['#2', '#6', '#7', '#8', '#9']
Evil probabilities: #4=37%
  Generated 102 candidate scenarios
  27 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #3 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4]

#### [17:56:01] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 37% likely evil (HP=6, can afford 3 wrong execs)
WARNING: Probabilistic execution -- 37% confident
WARNING: Low confidence (37%) -- consider gathering more info

### [17:57:01] Executed #4 -> Puppet (EVIL)

## [17:57:08] GAME OVER — WIN
Final HP: 6
Notes: Perfect deduction. PD found corruption+evil, Judge confirmed Medium truthful, Slayer fail revealed Lilis. Puppet was 4th evil.

