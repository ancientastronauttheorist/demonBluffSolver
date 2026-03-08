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


---

# New Game — 2026-03-06 18:05:50
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Dreamer, Gemcrafter, Jester, Slayer, Bard, Plague_Doctor
- Outcasts: Enlightened
- Minions: Puppeteer, Minion
- Demons: Pooka


---

# New Game — 2026-03-06 18:14:48
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Alchemist, Dreamer, Gemcrafter, Jester, Slayer, Bard, Enlightened
- Outcasts: Plague_Doctor
- Minions: Puppeteer, Minion
- Demons: Pooka


---

# New Game — 2026-03-06 18:19:08
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Dreamer, Gemcrafter, Jester, Slayer, Bard, Enlightened
- Outcasts: Plague_Doctor
- Minions: Puppeteer, Minion
- Demons: Pooka

### [18:36:14] Revealed #1 Gemcrafter
Info: {'good_position': 4}

### [18:36:43] Revealed #2 Jester
Info: {}

### [18:36:44] Revealed #3 Enlightened
Info: {'direction': 'CW'}

### [18:36:44] Revealed #4 Slayer
Info: {}

### [18:36:45] Revealed #5 Bard
Info: {'corruption_distance': 3}

### [18:36:45] Revealed #6 Bard
Info: {'corruption_distance': 4}

### [18:36:46] Revealed #7 Bard
Info: {'corruption_distance': 0}

### [18:36:47] Revealed #8 Dreamer
Info: {}

### [18:36:47] Revealed #9 Plague_Doctor
Info: {}

### [18:36:48] Revealed #10 Alchemist
Info: {'cured_count': 2}

#### [18:36:58] Solver Output
Scenarios: 67/7048
Definite evil: ['#7']
Evil probabilities: #5=81%, #10=49%, #6=39%, #8=25%, #4=16%, #1=15%, #9=9%, #2=7%, #3=7%
  Generated 7048 candidate scenarios
  67 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Pooka', 'Puppeteer', 'Minion'})
    Uncertain: [1, 2, 3, 4, 5, 6, 8, 9, 10]

#### [18:36:58] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 67 scenarios (roles: {'Pooka', 'Puppeteer', 'Minion'})

### [18:37:55] Executed #7 -> Puppeteer (EVIL)

#### [18:37:58] Solver Output
Scenarios: 23/940
Definite evil: ['#7']
Definite good: ['#2', '#3']
Evil probabilities: #5=65%, #10=52%, #6=48%, #8=48%, #4=17%, #1=13%, #9=9%
  Generated 940 candidate scenarios
  23 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 8, 9, 10]

#### [18:37:59] Recommendation
Action: **USE_ABILITY** #8 (Dreamer) -> targets ['#6']
Reason: Entropy 2.555 (adjusted 2.444) | timing x1.00
WARNING: Corruption risk: 9%

### [18:39:36] Ability used at #8

### [18:39:37] Revealed #8 Dreamer
Info: {'target': 6, 'evil_role': 'Minion'}

#### [18:39:40] Solver Output
Scenarios: 16/940
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#4']
Evil probabilities: #5=94%, #8=62%, #10=56%, #6=25%, #9=6%
  Generated 940 candidate scenarios
  16 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [5, 6, 8, 9, 10]

#### [18:39:40] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.805 (adjusted 1.805) | timing x1.00

#### [18:40:16] Solver Output
Scenarios: 5/940
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#4', '#9', '#10']
Evil probabilities: #5=80%, #6=80%, #8=80%
  Generated 940 candidate scenarios
  5 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [5, 6, 8]

#### [18:40:16] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#5']
Reason: Target #5 is 80% evil (adjusted 0.48)
WARNING: Corruption risk: 40% -- Slayer ability disabled if corrupted

### [18:40:52] Executed #5 -> Minion (EVIL)

#### [18:40:55] Solver Output
Scenarios: 2/143
Definite evil: ['#5', '#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#9', '#10']
Evil probabilities: #6=50%
  Generated 143 candidate scenarios
  2 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #8 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [6]

#### [18:40:55] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [18:41:35] Executed #8 -> Pooka (EVIL)

#### [18:41:39] Solver Output
Scenarios: 2/17
Definite evil: ['#5', '#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#9', '#10']
Evil probabilities: #6=50%
  Generated 17 candidate scenarios
  2 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #8 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [6]

#### [18:41:39] Recommendation
Action: **USE_ABILITY** #2 (Jester) -> targets ['#1', '#3', '#6']
Reason: Expected posterior 1.0 scenarios (adjusted 1.0) | timing x1.00

### [18:42:48] Ability used at #2

### [18:42:48] Revealed #2 Jester
Info: {'targets': [1, 3, 6], 'evil_count': 1}

#### [18:42:52] Solver Output
Scenarios: 1/17
Definite evil: ['#5', '#6', '#7', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#9', '#10']
  Generated 17 candidate scenarios
  1 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #6 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #8 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [18:42:52] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [18:44:05] Executed #6 -> Puppet (EVIL)

## [18:44:12] GAME OVER — WIN
Final HP: 10
Notes: Perfect game 10HP. Slayer killed Minion, Dreamer+PD+Jester confirmed Puppet adjacent to Puppeteer.


---

# New Game — 2026-03-06 18:50:47
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Confessor, Dreamer, Oracle, Enlightened, Alchemist, Gemcrafter
- Outcasts: Plague_Doctor, Bombardier
- Minions: Chancellor, Minion
- Demons: Pooka

### [18:55:21] Revealed #1 Gemcrafter
Info: {'good_position': 2}

### [18:55:21] Revealed #2 Alchemist
Info: {'cured_count': 2}

### [18:55:21] Revealed #3 Bombardier
Info: {}

### [18:55:22] Revealed #4 Dreamer
Info: {}

### [18:55:22] Revealed #5 Oracle
Info: {'targets': [2, 6], 'minion_role': 'Minion'}

### [18:55:22] Revealed #6 Enlightened
Info: {'direction': 'CW'}

### [18:55:22] Revealed #7 Confessor
Info: {'dizzy': True}

### [18:55:22] Revealed #8 Bishop
Info: {'targets': [4, 5, 6], 'types': ['Outcast', 'Villager', 'Minion']}

### [18:55:22] Revealed #9 Gemcrafter
Info: {'good_position': 2}

### [18:55:22] Revealed #10 Plague_Doctor
Info: {}

#### [18:55:28] Solver Output
Scenarios: 7/2822
Definite evil: ['#2']
Definite good: ['#3', '#4', '#5', '#6', '#10']
Evil probabilities: #1=71%, #8=71%, #9=43%, #7=14%
  Generated 2822 candidate scenarios
  7 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Minion', 'Chancellor'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 7, 8, 9]

#### [18:55:28] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 7 scenarios (roles: {'Minion', 'Chancellor'})

### [18:56:06] Executed #2 -> Minion (EVIL)

#### [18:56:06] Solver Output
Scenarios: 7/366
Definite evil: ['#2']
Definite good: ['#3', '#4', '#5', '#6', '#10']
Evil probabilities: #1=71%, #8=71%, #9=43%, #7=14%
  Generated 366 candidate scenarios
  7 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Minion'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 7, 8, 9]

#### [18:56:06] Recommendation
Action: **USE_ABILITY** #4 (Dreamer) -> targets ['#9']
Reason: Entropy 1.842 (adjusted 1.711) | timing x1.00
WARNING: Corruption risk: 14%

### [18:56:43] Ability used at #4

### [18:56:43] Revealed #4 Dreamer
Info: {'target': 9, 'evil_role': 'Minion'}

#### [18:56:43] Solver Output
Scenarios: 4/366
Definite evil: ['#1', '#2', '#8']
Definite good: ['#3', '#4', '#5', '#6', '#7', '#9', '#10']
  Generated 366 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY EVIL (possible roles: {'Minion'})
    #8 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [18:56:43] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Chancellor'})

### [18:58:21] Executed #1 -> Chancellor (EVIL)

### [18:58:21] Executed #8 -> Pooka (EVIL)

## [18:58:28] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Dreamer ability on #9 solved entire board. #9 Gemcrafter, #7 Confessor, #4 Dreamer all corrupted.


---

# New Game — 2026-03-06 18:59:48
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Judge, Scout, Druid, Bishop, Baker, Fortune_Teller
- Outcasts: Plague_Doctor, Wretch
- Minions: Chancellor
- Demons: Pooka

### [19:03:48] Revealed #1 Plague_Doctor
Info: {}

### [19:03:49] Revealed #2 Scout
Info: {'evil_role': 'Pooka', 'distance': 3}

### [19:03:49] Revealed #3 Fortune_Teller
Info: {}

### [19:03:49] Revealed #4 Wretch
Info: {}

### [19:03:49] Revealed #5 Bishop
Info: {'targets': [3, 5, 7], 'types': ['Outcast', 'Minion', 'Villager']}

### [19:03:49] Revealed #6 Baker
Info: {}

### [19:03:50] Revealed #7 Baker
Info: {}

### [19:03:50] Revealed #8 Plague_Doctor
Info: {}

#### [19:03:56] Solver Output
Scenarios: 10/86
Definite good: ['#6', '#7']
Evil probabilities: #1=90%, #4=40%, #2=20%, #3=20%, #5=20%, #8=10%
  Generated 86 candidate scenarios
  10 scenarios survived validation
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 8]

#### [19:03:56] Recommendation
Action: **USE_ABILITY** #1 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.361 (adjusted 1.361) | timing x1.00

#### [19:04:56] Solver Output
Scenarios: 6/86
Definite good: ['#3', '#6', '#7']
Evil probabilities: #1=83%, #4=67%, #2=17%, #5=17%, #8=17%
  Generated 86 candidate scenarios
  6 scenarios survived validation
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 8]

#### [19:04:56] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.252 (adjusted 1.252) | timing x1.00

#### [19:05:36] Solver Output
Scenarios: 4/86
Definite evil: ['#1']
Definite good: ['#3', '#5', '#6', '#7', '#8']
Evil probabilities: #4=75%, #2=25%
  Generated 86 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 4]

#### [19:05:36] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Chancellor'})

### [19:06:36] Executed #1 -> Chancellor (EVIL)

#### [19:06:36] Solver Output
Scenarios: 4/18
Definite evil: ['#1']
Definite good: ['#3', '#5', '#6', '#7', '#8']
Evil probabilities: #4=75%, #2=25%
  Generated 18 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 4]

#### [19:06:36] Recommendation
Action: **USE_ABILITY** #3 (Fortune Teller) -> targets ['#2', '#5']
Reason: Entropy 0.811 (adjusted 0.406) | timing x1.00
WARNING: Corruption risk: 100%

### [19:07:32] Ability used at #3

### [19:07:32] Revealed #3 Fortune Teller
Info: {'targets': [2, 5], 'has_evil': False}

#### [19:07:32] Solver Output
Scenarios: 1/18
Definite evil: ['#1', '#2']
Definite good: ['#3', '#4', '#5', '#6', '#7', '#8']
  Generated 18 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [19:07:32] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [19:08:23] Executed #2 -> Pooka (EVIL)

## [19:08:23] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Baker conversion confirmed #6/#7 good. PD clean checks narrowed field. Corrupted FT inversion solved the 50/50 — #2 Pooka.


---

# New Game — 2026-03-06 19:10:05
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Oracle, Empress, Confessor, Slayer
- Outcasts: Plague_Doctor
- Minions: 
- Demons: Pooka

### [19:11:16] Revealed #1 Oracle
Info: {}

### [19:11:16] Revealed #2 Confessor
Info: {'dizzy': True}

### [19:11:17] Revealed #3 Empress
Info: {'targets': [2, 4, 6]}

### [19:11:17] Revealed #4 Druid
Info: {}

### [19:11:17] Revealed #5 Plague_Doctor
Info: {}

### [19:11:17] Revealed #6 Slayer
Info: {}

#### [19:11:22] Solver Output
Scenarios: 6/21
Definite good: ['#2', '#4', '#5']
Evil probabilities: #3=67%, #1=17%, #6=17%
  Generated 21 candidate scenarios
  6 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [1, 3, 6]

#### [19:11:22] Recommendation
Action: **USE_ABILITY** #5 (Plague Doctor) -> targets ['#1']
Reason: Entropy 1.252 (adjusted 1.252) | timing x1.00

#### [19:12:15] Solver Output
Scenarios: 4/21
Definite good: ['#2', '#4', '#5', '#6']
Evil probabilities: #3=75%, #1=25%
  Generated 21 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 3]

#### [19:12:15] Recommendation
Action: **USE_ABILITY** #4 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.811 (adjusted 0.507) | timing x1.00
WARNING: Corruption risk: 75%

### [19:13:03] Ability used at #4

### [19:13:03] Revealed #4 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Doppelganger'}

#### [19:13:03] Solver Output
Scenarios: 3/21
Definite evil: ['#3']
Definite good: ['#1', '#2', '#4', '#5', '#6']
  Generated 21 candidate scenarios
  3 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [19:13:03] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 3 scenarios (roles: {'Pooka'})

### [19:14:09] Executed #3 -> Pooka (EVIL)

## [19:14:09] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. PD clean on #1 + corrupted Druid claiming Doppelganger (not in deck) solved it. Heavy corruption: #2 #4 #6 all corrupted.


---

# New Game — 2026-03-06 19:16:14
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bard, Judge, Knight, Architect, Scout, Empress, Baker
- Outcasts: Plague_Doctor, Drunk
- Minions: Shaman
- Demons: Baa

### [19:17:40] Revealed #1 Scout
Info: {'evil_role': 'Shaman', 'distance': 1}

### [19:17:40] Revealed #2 Baker
Info: {}

### [19:17:40] Revealed #3 Baker
Info: {}

### [19:17:40] Revealed #4 Plague_Doctor
Info: {}

### [19:17:40] Revealed #5 Empress
Info: {'targets': [3, 7, 8]}

### [19:17:40] Revealed #6 Architect
Info: {'side': 'right'}

### [19:17:41] Revealed #7 Bard
Info: {'corruption_distance': 1}

### [19:17:41] Revealed #8 Baker
Info: {}

#### [19:17:48] Solver Output
Scenarios: 70/416
Definite good: ['#2', '#3', '#8']
Evil probabilities: #5=71%, #6=69%, #1=40%, #7=17%, #4=3%
  Generated 416 candidate scenarios
  70 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7]

#### [19:17:48] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#2']
Reason: Entropy 1.471 (adjusted 1.471) | timing x1.00

#### [19:18:50] Solver Output
Scenarios: 48/416
Definite good: ['#2', '#3', '#4', '#8']
Evil probabilities: #5=75%, #6=67%, #1=50%, #7=8%
  Generated 416 candidate scenarios
  48 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 5, 6, 7]

#### [19:18:50] Recommendation
Action: **EXECUTE** #5
Reason: No reveals available. #5 is 75% likely evil (HP=10, can afford 2 wrong execs)
WARNING: Probabilistic execution -- 75% confident

### [19:19:51] Executed #5 -> Baa (EVIL)

#### [19:19:51] Solver Output
Scenarios: 21/115
Definite evil: ['#5']
Definite good: ['#2', '#3', '#4', '#7', '#8']
Evil probabilities: #6=57%, #1=43%
  Generated 115 candidate scenarios
  21 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 6]

#### [19:19:51] Recommendation
Action: **EXECUTE** #6
Reason: No reveals available. #6 is 57% likely evil (HP=10, can afford 2 wrong execs)
WARNING: Probabilistic execution -- 57% confident

### [19:20:53] Executed #6 -> Shaman (EVIL)

## [19:20:53] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Ascension 9 complete! Baker triple-chain confirmed #2/#3/#8 good. PD clean + Empress/Architect/Scout/Bard constraints narrowed to #5 Baa + #6 Shaman.


---

# New Game — 2026-03-06 19:40:22
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Slayer, Bishop, Confessor, Scout, Architect, Oracle
- Outcasts: Bombardier, Plague_Doctor
- Minions: Minion, Shaman
- Demons: Lilis

### [19:44:07] Revealed #1 Slayer
Info: {}

### [19:44:07] Revealed #2 Plague_Doctor
Info: {}

### [19:44:07] Revealed #3 Bombardier
Info: {}

### [19:44:07] Revealed #4 Scout
Info: {'evil_role': 'Lilis', 'distance': 3}

### [19:44:07] Revealed #5 Oracle
Info: {'targets': [6, 9], 'minion_role': 'Minion'}

### [19:44:07] Revealed #6 Scout
Info: {'evil_role': 'Shaman', 'distance': 4}

### [19:44:07] Revealed #8 Architect
Info: {'side': 'right'}

### [19:44:07] Revealed #9 Confessor
Info: {'dizzy': True}

#### [19:44:19] Solver Output
Scenarios: 32/846
Definite good: ['#2', '#7', '#10']
Evil probabilities: #9=78%, #6=62%, #4=56%, #8=41%, #3=38%, #1=12%, #5=12%
  Generated 846 candidate scenarios
  32 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8, 9]

#### [19:44:19] Recommendation
Action: **USE_ABILITY** #2 (Plague Doctor) -> targets ['#8']
Reason: Entropy 1.505 (adjusted 1.505) | timing x1.00

#### [19:46:01] Solver Output
Scenarios: 21/846
Definite good: ['#2', '#7', '#10']
Evil probabilities: #9=67%, #4=62%, #8=62%, #6=52%, #3=38%, #1=14%, #5=5%
  Generated 846 candidate scenarios
  21 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8, 9]

#### [19:46:01] Recommendation
Action: **USE_ABILITY** #1 (Slayer) -> targets ['#9']
Reason: Target #9 is 67% evil (adjusted 0.63)
WARNING: Corruption risk: 5% -- Slayer ability disabled if corrupted

### [19:46:47] Executed #9 -> Minion (EVIL)

#### [19:46:51] Solver Output
Scenarios: 7/142
Definite evil: ['#9']
Definite good: ['#1', '#2', '#5', '#7', '#10']
Evil probabilities: #8=71%, #4=57%, #3=43%, #6=29%
  Generated 142 candidate scenarios
  7 scenarios survived validation
    #9 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [3, 4, 6, 8]

#### [19:46:51] Recommendation
Action: **EXECUTE** #8
Reason: No reveals available. #8 is 71% likely evil (HP=6, can afford 1 wrong execs)
WARNING: Probabilistic execution -- 71% confident

### [19:47:28] Executed #8 -> Lilis (EVIL)

#### [19:47:31] Solver Output
Scenarios: 4/27
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#5', '#7', '#10']
Evil probabilities: #4=50%, #3=25%, #6=25%
  Generated 27 candidate scenarios
  4 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #9 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [3, 4, 6]

#### [19:47:31] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 50% likely evil (HP=6, can afford 1 wrong execs)
WARNING: Probabilistic execution -- 50% confident

### [19:48:25] Executed #4 -> GOOD (WRONG!)

#### [19:48:25] Solver Output
Scenarios: 2/22
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#4', '#5', '#7', '#10']
Evil probabilities: #3=50%, #6=50%
  Generated 22 candidate scenarios
  2 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #9 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [3, 6]

#### [19:48:25] Recommendation
Action: **ERROR** #3
Reason: #3 is 50% likely evil but HP too low to risk (HP=1, cost=5). Need more info.
WARNING: Probabilistic execution -- 50% confident
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [19:52:05] Executed #3 -> Shaman (EVIL)

## [19:52:05] GAME OVER — WIN
Final HP: 1
Notes: Extra roles mechanic: 12 pool / 10 board. Solver had 50/50 but outcast count deduction (1O on board, both PD and Bombardier visible, PD confirmed good -> Bombardier must be evil disguise) broke the tie. Need to add deck pool vs board count validation to solver.


---

# New Game — 2026-03-06 19:55:25
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Empress, Slayer, Bishop, Confessor, Scout, Architect, Oracle
- Outcasts: Bombardier, Plague_Doctor
- Minions: Minion, Shaman
- Demons: Lilis


---

# New Game — 2026-03-06 20:08:12
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Oracle, Fortune_Teller, Jester, Medium, Confessor, Knitter
- Outcasts: Plague_Doctor
- Minions: Shaman, Minion
- Demons: Pooka

### [20:10:25] Revealed #1 Fortune_Teller
Info: {}

### [20:10:25] Revealed #2 Fortune_Teller
Info: {}

### [20:10:26] Revealed #3 Lover
Info: {'evil_adjacent': 1}

### [20:10:26] Revealed #4 Oracle
Info: {'targets': [1, 8], 'minion_role': 'Minion'}

### [20:10:26] Revealed #5 Confessor
Info: {'dizzy': True}

### [20:10:26] Revealed #6 Jester
Info: {}

### [20:10:26] Revealed #7 Plague_Doctor
Info: {}

### [20:10:26] Revealed #8 Medium
Info: {'good_position': 1, 'good_role': 'Fortune_Teller'}

### [20:10:26] Revealed #9 Plague_Doctor
Info: {}

#### [20:10:31] Solver Output
Scenarios: 68/1638
Definite good: ['#8']
Evil probabilities: #7=82%, #5=57%, #4=43%, #3=37%, #2=34%, #6=24%, #9=18%, #1=6%
  Generated 1638 candidate scenarios
  68 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [20:10:31] Recommendation
Action: **USE_ABILITY** #7 (Plague Doctor) -> targets ['#6']
Reason: Entropy 2.604 (adjusted 2.604) | timing x1.00

#### [20:11:37] Solver Output
Scenarios: 20/1638
Definite good: ['#1', '#8']
Evil probabilities: #7=90%, #4=70%, #2=50%, #6=50%, #5=20%, #3=10%, #9=10%
  Generated 1638 candidate scenarios
  20 scenarios survived validation
    #1 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6, 7, 9]

#### [20:11:37] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#3']
Reason: Entropy 1.722 (adjusted 1.722) | timing x1.00

#### [20:12:23] Solver Output
Scenarios: 12/1638
Definite good: ['#1', '#8']
Evil probabilities: #7=83%, #2=67%, #4=67%, #5=33%, #3=17%, #6=17%, #9=17%
  Generated 1638 candidate scenarios
  12 scenarios survived validation
    #1 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6, 7, 9]

#### [20:12:23] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#1', '#8']
Reason: Entropy 0.918 (adjusted 0.918) | timing x1.00

### [20:13:12] Revealed #2 Fortune Teller
Info: {'targets': [1, 8], 'has_evil': False}

#### [20:13:16] Solver Output
Scenarios: 4/1638
Definite evil: ['#5']
Definite good: ['#1', '#2', '#4', '#8']
Evil probabilities: #3=50%, #6=50%, #7=50%, #9=50%
  Generated 1638 candidate scenarios
  4 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 6, 7, 9]

#### [20:13:16] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 4 scenarios (roles: {'Pooka'})

### [20:14:00] Executed #5 -> Pooka (EVIL)

#### [20:14:00] Solver Output
Scenarios: 4/236
Definite evil: ['#5']
Definite good: ['#1', '#2', '#4', '#8']
Evil probabilities: #3=50%, #6=50%, #7=50%, #9=50%
  Generated 236 candidate scenarios
  4 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 6, 7, 9]

#### [20:14:00] Recommendation
Action: **USE_ABILITY** #1 (Fortune Teller) -> targets ['#2', '#3']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [20:14:47] Revealed #1 Fortune Teller
Info: {'targets': [2, 3], 'has_evil': False}

#### [20:14:47] Solver Output
Scenarios: 0/236
  Generated 236 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #8 Medium: rejected 142/236 (60%)
    #2 Fortune Teller: rejected 136/236 (58%)
    #1 Fortune Teller: rejected 136/236 (58%)
    #3 Lover: rejected 106/236 (45%)
    #4 Oracle: rejected 64/236 (27%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Fortune Teller: 4 scenarios survive  <-- SUSPECT
    WITHOUT #2 Fortune Teller: 3 scenarios survive  <-- SUSPECT
    WITHOUT #3 Lover: 5 scenarios survive  <-- SUSPECT
    WITHOUT #4 Oracle: 2 scenarios survive  <-- SUSPECT
    WITHOUT #5 Confessor: 2 scenarios survive  <-- SUSPECT
    WITHOUT #6 Jester: 2 scenarios survive  <-- SUSPECT
    WITHOUT #8 Medium: 12 scenarios survive  <-- SUSPECT

#### [20:14:47] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [20:18:38] Solver Output
Scenarios: 0/296
  Generated 296 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #8 Medium: rejected 182/296 (61%)
    #2 Fortune Teller: rejected 168/296 (57%)
    #1 Fortune Teller: rejected 168/296 (57%)
    #3 Lover: rejected 134/296 (45%)
    #4 Oracle: rejected 74/296 (25%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Fortune Teller: 8 scenarios survive  <-- SUSPECT
    WITHOUT #2 Fortune Teller: 6 scenarios survive  <-- SUSPECT
    WITHOUT #3 Lover: 10 scenarios survive  <-- SUSPECT
    WITHOUT #4 Oracle: 4 scenarios survive  <-- SUSPECT
    WITHOUT #5 Confessor: 4 scenarios survive  <-- SUSPECT
    WITHOUT #6 Jester: 4 scenarios survive  <-- SUSPECT
    WITHOUT #8 Medium: 20 scenarios survive  <-- SUSPECT

#### [20:18:38] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [20:20:32] Solver Output
Scenarios: 4/296
Definite evil: ['#5']
Definite good: ['#1', '#2', '#4', '#8']
Evil probabilities: #3=50%, #6=50%, #7=50%, #9=50%
  Generated 296 candidate scenarios
  4 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 6, 7, 9]

#### [20:20:32] Recommendation
Action: **USE_ABILITY** #2 (Fortune Teller) -> targets ['#1', '#3']
Reason: Entropy 1.000 (adjusted 1.000) | timing x1.00

### [20:20:39] Ability used at #1

### [20:20:39] Ability used at #2

#### [20:20:39] Solver Output
Scenarios: 4/296
Definite evil: ['#5']
Definite good: ['#1', '#2', '#4', '#8']
Evil probabilities: #3=50%, #6=50%, #7=50%, #9=50%
  Generated 296 candidate scenarios
  4 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 6, 7, 9]

#### [20:20:39] Recommendation
Action: **USE_ABILITY** #6 (Jester) -> targets ['#1', '#2', '#3']
Reason: Expected posterior 3.3 scenarios (adjusted 4.2) | timing x1.00
WARNING: Corruption risk: 50%

### [20:23:58] Executed #9 -> GOOD (WRONG!)

### [20:25:24] Executed #6 -> Minion (EVIL)

### [20:25:24] Executed #7 -> Shaman (EVIL)

## [20:25:29] GAME OVER — WIN
Final HP: 5
Notes: 50/50 on #9 was wrong (was real PD). PD #7 was evil Shaman lying. Multi-PD fix + Medium normalization fix enabled correct solving.


---

# New Game — 2026-03-06 20:27:12
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: 
- Outcasts: 
- Minions: 
- Demons: 

## Deck
- Villagers: Bard, Hunter, Medium, Gemcrafter, Druid, Judge, Lover
- Outcasts: Doppelganger, Drunk
- Minions: Twin_Minion
- Demons: Baa

### [20:29:06] Revealed #1 Medium
Info: {'good_position': 4, 'good_role': 'Lover'}

### [20:29:06] Revealed #2 Hunter
Info: {'distance': 2}

### [20:29:06] Revealed #3 Bard
Info: {'corruption_distance': 0}

### [20:29:06] Revealed #4 Lover
Info: {'evil_adjacent': 1}

### [20:29:07] Revealed #5 Gemcrafter
Info: {'good_position': 8}

### [20:29:07] Revealed #6 Medium
Info: {'good_position': 1, 'good_role': 'Medium'}

### [20:29:07] Revealed #7 Druid
Info: {}

### [20:29:07] Revealed #8 Judge
Info: {}

### [20:30:40] Executed #3 -> GOOD (WRONG!)

### [20:32:57] Executed #5 -> Baa (EVIL)

### [20:32:57] Executed #8 -> Twin_Minion (EVIL)

## [20:33:03] GAME OVER — WIN
Final HP: 5
Notes: Bard 0-corrupted bug: solver used 0 as distance instead of -1 sentinel for 'no corrupted exist'. Wrong exec on #3 (Bard was good). Fixed card_bard to map 0->-1. After fix, solver correctly found #5+#8.


---

# New Game — 2026-03-06 20:34:15
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Druid, Medium, Hunter, Confessor, Lover, Bard, Enlightened
- Outcasts: Judge, Wretch, Plague_Doctor, Drunk, Bombardier
- Minions: Chancellor, Shaman
- Demons: Baa

## Deck
- Villagers: Druid, Medium, Hunter, Confessor, Lover, Bard, Enlightened, Judge
- Outcasts: Wretch, Plague_Doctor, Drunk, Bombardier
- Minions: Chancellor, Shaman
- Demons: Baa

### [20:36:31] Revealed #1 Bard
Info: {'corruption_distance': 1}

### [20:36:32] Revealed #2 Enlightened
Info: {'direction': 'CW'}

### [20:36:32] Revealed #3 Bombardier
Info: {}

### [20:36:32] Revealed #4 Plague_Doctor
Info: {}

### [20:36:32] Revealed #5 Druid
Info: {}

### [20:36:32] Revealed #6 Hunter
Info: {'distance': 2}

### [20:36:32] Revealed #7 Hunter
Info: {'distance': 3}

### [20:36:32] Revealed #8 Medium
Info: {'good_position': 7, 'good_role': 'Hunter'}

### [20:36:32] Revealed #9 Confessor
Info: {'dizzy': False}

#### [20:36:40] Solver Output
Scenarios: 64/8730
Definite good: ['#9']
Evil probabilities: #3=81%, #1=44%, #2=44%, #6=38%, #7=38%, #4=19%, #5=19%, #8=19%
  Generated 8730 candidate scenarios
  64 scenarios survived validation
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [20:36:40] Recommendation
Action: **USE_ABILITY** #4 (Plague Doctor) -> targets ['#6']
Reason: Entropy 1.936 (adjusted 1.936) | timing x1.00

### [20:37:49] Ability used at #4

#### [20:37:51] Solver Output
Scenarios: 24/8730
Definite good: ['#5', '#6', '#9']
Evil probabilities: #1=50%, #2=50%, #3=50%, #4=50%, #7=50%, #8=50%
  Generated 8730 candidate scenarios
  24 scenarios survived validation
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 7, 8]

#### [20:37:51] Recommendation
Action: **USE_ABILITY** #5 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 1.500 (adjusted 1.312) | timing x1.00
WARNING: Corruption risk: 25%

### [20:39:10] Ability used at #5

#### [20:39:10] Solver Output
Scenarios: 24/8730
Definite good: ['#5', '#6', '#9']
Evil probabilities: #1=50%, #2=50%, #3=50%, #4=50%, #7=50%, #8=50%
  Generated 8730 candidate scenarios
  24 scenarios survived validation
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 7, 8]

#### [20:39:10] Recommendation
Action: **EXECUTE** #1
Reason: No reveals available. #1 is 50% likely evil (HP=10, can afford 2 wrong execs)
WARNING: Probabilistic execution -- 50% confident

### [20:42:09] Executed #3 -> GOOD (WRONG!)

## [20:42:09] GAME OVER — LOSS
Final HP: 5
Notes: Bombardier blew up\! Trusted corrupted Drunk's Druid info. #5 was Drunk<Corrupted> not real Druid. Need: 1) Druid validator 2) NEVER execute Bombardier unless confirmed evil 3) Drunk counts as Villager in header board counts


---

# New Game — 2026-03-06 20:46:45
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Confessor, Poet, Jester, Knitter, Gemcrafter, Hunter
- Outcasts: Plague_Doctor
- Minions: Shaman, Witch
- Demons: Lilis

### [20:48:10] Revealed #1 Poet
Info: {'evil_pairs': 0, 'copied_role': 'Knitter'}

#### [20:49:34] Solver Output
Scenarios: 228/504
Evil probabilities: #1=47%, #2=45%, #9=45%, #4=29%, #5=29%, #6=29%, #7=29%, #3=24%, #8=24%
  Generated 504 candidate scenarios
  228 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [20:49:34] Recommendation
Action: **REVEAL** #2
Reason: #2: 45% evil, entropy 1.093

### [20:50:02] Revealed #2 Confessor
Info: {'dizzy': False}

#### [20:50:02] Solver Output
Scenarios: 126/504
Definite good: ['#2']
Evil probabilities: #9=57%, #1=52%, #3=38%, #5=33%, #6=33%, #7=33%, #4=29%, #8=24%
  Generated 504 candidate scenarios
  126 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8, 9]

#### [20:50:02] Recommendation
Action: **REVEAL** #9
Reason: #9: 57% evil, entropy 1.085

### [20:50:33] Revealed #9 Gemcrafter
Info: {'good_position': 8}

#### [20:50:34] Solver Output
Scenarios: 36/504
Definite good: ['#2']
Evil probabilities: #1=83%, #5=50%, #3=33%, #4=33%, #6=33%, #7=33%, #8=17%, #9=17%
  Generated 504 candidate scenarios
  36 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8, 9]

#### [20:50:34] Recommendation
Action: **REVEAL** #5
Reason: #5: 50% evil, entropy 1.100

### [20:51:15] Revealed #5 Knitter
Info: {'evil_pairs': 1}

#### [20:51:16] Solver Output
Scenarios: 12/504
Definite evil: ['#3']
Definite good: ['#2', '#6', '#8', '#9']
Evil probabilities: #1=50%, #4=50%, #5=50%, #7=50%
  Generated 504 candidate scenarios
  12 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis', 'Witch', 'Shaman'})
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 7]

#### [20:51:16] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 12 scenarios (roles: {'Lilis', 'Witch', 'Shaman'})

### [20:52:00] Executed #3 -> GOOD (WRONG!)

### [20:52:30] Revealed #3 Plague_Doctor
Info: {}

### [20:53:21] Executed #8 -> Shaman (EVIL)

#### [20:53:21] Solver Output
Scenarios: 6/120
Definite evil: ['#8']
Definite good: ['#2', '#3', '#5', '#6']
Evil probabilities: #1=67%, #9=67%, #4=33%, #7=33%
  Generated 120 candidate scenarios
  6 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 4, 7, 9]

#### [20:53:21] Recommendation
Action: **REVEAL** #4
Reason: #4: 33% evil, entropy 1.018
WARNING: Witch may be alive -- be cautious about revealing

### [20:54:06] Revealed #4 Plague_Doctor
Info: {}

#### [20:54:06] Solver Output
Scenarios: 2/120
Definite evil: ['#4', '#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#6', '#7']
  Generated 120 candidate scenarios
  2 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Lilis', 'Witch'})
    #8 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #9 is DEFINITELY EVIL (possible roles: {'Lilis', 'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [20:54:06] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Lilis', 'Witch'})

### [20:54:50] Executed #4 -> Witch (EVIL)

### [20:54:50] Executed #9 -> Lilis (EVIL)

## [20:54:51] GAME OVER — WIN
Final HP: 3
Notes: Lilis game, 1-at-a-time flipping. Wrong exec #3 (PD outcast) because PD card wasn't entered — solver couldn't model corruption. #1 Poet was corrupted explaining Knitter contradiction. Recovered by entering PD card post-exec. HP 3/10 survived.


---

# New Game — 2026-03-06 21:12:38
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: FortuneTeller, Baker, Slayer, Confessor, Medium, Bishop, Druid
- Outcasts: Drunk
- Minions: Shaman
- Demons: Pooka

### [21:15:45] Revealed #1 FortuneTeller
Info: {}

### [21:15:45] Revealed #2 Druid
Info: {}

### [21:15:45] Revealed #3 Confessor
Info: {'dizzy': False}

### [21:15:46] Revealed #4 Medium
Info: {'good_position': 5, 'good_role': 'Drunk'}

### [21:15:46] Revealed #5 Bishop
Info: {'targets': [1, 3, 4], 'types': ['Villager', 'Minion', 'Outcast']}

### [21:15:46] Revealed #6 Bishop
Info: {'targets': [1, 4, 8], 'types': ['Outcast', 'Minion', 'Villager']}

### [21:15:46] Revealed #7 Slayer
Info: {}

### [21:15:46] Revealed #8 Druid
Info: {}

#### [21:15:50] Solver Output
Scenarios: 49/392
Definite good: ['#3']
Evil probabilities: #5=78%, #6=35%, #4=27%, #7=18%, #1=14%, #2=14%, #8=14%
  Generated 392 candidate scenarios
  49 scenarios survived validation
    #3 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7, 8]

#### [21:15:50] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.889 (adjusted 0.816) | timing x1.00
WARNING: Corruption risk: 16%

### [21:16:52] Revealed #8 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

### [21:16:55] Ability used at #8

#### [21:16:59] Solver Output
Scenarios: 34/392
Definite good: ['#3', '#8']
Evil probabilities: #5=76%, #6=41%, #4=29%, #1=18%, #2=18%, #7=18%
  Generated 392 candidate scenarios
  34 scenarios survived validation
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7]

#### [21:16:59] Recommendation
Action: **USE_ABILITY** #2 (Druid) -> targets ['#1', '#3', '#4']
Reason: Entropy 0.908 (adjusted 0.841) | timing x1.00
WARNING: Corruption risk: 15%

### [21:18:18] Revealed #2 Druid
Info: {'targets': [1, 3, 4], 'found_outcast': None}

### [21:18:18] Ability used at #2

#### [21:18:19] Solver Output
Scenarios: 23/392
Definite good: ['#2', '#3', '#8']
Evil probabilities: #5=74%, #6=48%, #4=35%, #1=22%, #7=22%
  Generated 392 candidate scenarios
  23 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7]

#### [21:18:19] Recommendation
Action: **USE_ABILITY** #7 (Slayer) -> targets ['#5']
Reason: Target #5 is 74% evil (adjusted 0.45)
WARNING: Corruption risk: 39% -- Slayer ability disabled if corrupted

#### [21:18:59] Solver Output
Scenarios: 14/392
Definite good: ['#2', '#3', '#8']
Evil probabilities: #5=57%, #6=57%, #4=36%, #7=36%, #1=14%
  Generated 392 candidate scenarios
  14 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7]

#### [21:18:59] Recommendation
Action: **EXECUTE** #5
Reason: No reveals available. #5 is 57% likely evil (HP=10, can afford 2 wrong execs)
WARNING: Probabilistic execution -- 57% confident

### [21:19:37] Executed #5 -> GOOD (WRONG!)

#### [21:19:38] Solver Output
Scenarios: 6/294
Definite evil: ['#6']
Definite good: ['#2', '#3', '#5', '#8']
Evil probabilities: #4=67%, #1=17%, #7=17%
  Generated 294 candidate scenarios
  6 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 4, 7]

#### [21:19:38] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 6 scenarios (roles: {'Pooka'})

### [21:20:11] Executed #6 -> Pooka (EVIL)

#### [21:20:11] Solver Output
Scenarios: 6/42
Definite evil: ['#6']
Definite good: ['#2', '#3', '#5', '#8']
Evil probabilities: #4=67%, #1=17%, #7=17%
  Generated 42 candidate scenarios
  6 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 4, 7]

#### [21:20:11] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 67% likely evil (HP=8, can afford 1 wrong execs)
WARNING: Probabilistic execution -- 67% confident

### [21:20:42] Executed #4 -> GOOD (WRONG!)

#### [21:20:42] Solver Output
Scenarios: 2/35
Definite evil: ['#6']
Definite good: ['#2', '#3', '#4', '#5', '#8']
Evil probabilities: #1=50%, #7=50%
  Generated 35 candidate scenarios
  2 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [21:20:42] Recommendation
Action: **ERROR** #1
Reason: #1 is 50% likely evil but HP too low to risk (HP=3, cost=5). Need more info.
WARNING: Probabilistic execution -- 50% confident
WARNING: CRITICAL: HP=3, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [21:25:10] Executed #7 -> Shaman (EVIL)

## [21:27:56] GAME OVER — WIN
Final HP: 3
Notes: FT active ability clutch on 50/50. get_card bug found — CamelCase roles not matching KB.


---

# New Game — 2026-03-06 21:34:30
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Bishop, Judge, Baker, Poet, Medium, Enlightened, Alchemist
- Outcasts: Drunk
- Minions: Witch, Poisoner
- Demons: Lilis

### [21:36:03] Revealed #1 Baker
Info: {}

#### [21:36:34] Solver Output
Scenarios: 3528/3528
Evil probabilities: #1=33%, #2=33%, #3=33%, #4=33%, #5=33%, #6=33%, #7=33%, #8=33%, #9=33%
  Generated 3528 candidate scenarios
  3528 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:36:34] Recommendation
Action: **REVEAL** #2
Reason: #2: 33% evil, entropy 1.018

### [21:37:06] Revealed #2 Bishop
Info: {'targets': [3, 4, 6], 'types': ['Minion', 'Villager', 'Outcast']}

#### [21:37:07] Solver Output
Scenarios: 3216/3528
Evil probabilities: #2=37%, #3=36%, #4=36%, #6=36%, #1=32%, #5=31%, #7=31%, #8=31%, #9=31%
  Generated 3528 candidate scenarios
  3216 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:37:07] Recommendation
Action: **REVEAL** #3
Reason: #3: 36% evil, entropy 1.041

### [21:37:41] Revealed #3 Enlightened
Info: {'direction': 'CCW'}

#### [21:37:42] Solver Output
Scenarios: 1720/3738
Evil probabilities: #2=41%, #4=38%, #3=38%, #5=32%, #6=32%, #7=31%, #8=31%, #9=29%, #1=28%
  Generated 3738 candidate scenarios
  1720 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8, 9]

#### [21:37:42] Recommendation
Action: **REVEAL** #4
Reason: #4: 38% evil, entropy 1.057

### [21:39:32] Revealed #4 Alchemist
Info: {'cured_count': 1}

#### [21:39:37] Solver Output
Scenarios: 996/3948
Definite good: ['#6']
Evil probabilities: #4=59%, #3=46%, #5=43%, #2=39%, #7=30%, #8=30%, #1=27%, #9=27%
  Generated 3948 candidate scenarios
  996 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [21:39:37] Recommendation
Action: **REVEAL** #5
Reason: #5: 43% evil, entropy 1.085

### [21:44:02] Revealed #5 Baker
Info: {}

#### [21:44:07] Solver Output
Scenarios: 1016/4158
Definite good: ['#6']
Evil probabilities: #4=60%, #3=45%, #5=42%, #2=39%, #7=30%, #8=30%, #1=27%, #9=27%
  Generated 4158 candidate scenarios
  1016 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [21:44:07] Recommendation
Action: **REVEAL** #7
Reason: #7: 30% evil, entropy 0.978

### [21:44:52] Revealed #7 Baker
Info: {}

#### [21:44:52] Solver Output
Scenarios: 1016/4368
Definite good: ['#6']
Evil probabilities: #4=60%, #3=45%, #5=42%, #2=39%, #7=30%, #8=30%, #1=27%, #9=27%
  Generated 4368 candidate scenarios
  1016 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [21:44:52] Recommendation
Action: **REVEAL** #8
Reason: #8: 30% evil, entropy 0.978
WARNING: Witch may be alive -- be cautious about revealing

### [21:45:43] Revealed #8 Poet
Info: {'good_position': 5, 'copied_role': 'Gemcrafter'}

#### [21:45:43] Solver Output
Scenarios: 446/4578
Definite good: ['#6']
Evil probabilities: #4=68%, #3=46%, #2=45%, #5=33%, #9=31%, #1=30%, #7=28%, #8=19%
  Generated 4578 candidate scenarios
  446 scenarios survived validation
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 9]

#### [21:45:43] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 68% likely evil (HP=8, can afford 1 wrong execs)
WARNING: Probabilistic execution -- 68% confident

### [21:46:34] Executed #4 -> GOOD (WRONG!)

#### [21:46:35] Solver Output
Scenarios: 142/2982
Definite good: ['#4', '#6']
Evil probabilities: #5=76%, #2=52%, #8=48%, #3=46%, #7=27%, #9=27%, #1=24%
  Generated 2982 candidate scenarios
  142 scenarios survived validation
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 7, 8, 9]

#### [21:46:35] Recommendation
Action: **ERROR** #5
Reason: #5 is 76% likely evil but HP too low to risk (HP=3, cost=5). Need more info.
WARNING: Probabilistic execution -- 76% confident
WARNING: CRITICAL: HP=3, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [21:48:00] Executed #5 -> Poisoner (EVIL)

#### [21:48:00] Solver Output
Scenarios: 38/294
Definite evil: ['#5']
Definite good: ['#4', '#6']
Evil probabilities: #8=68%, #3=53%, #2=47%, #1=11%, #7=11%, #9=11%
  Generated 294 candidate scenarios
  38 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 7, 8, 9]

#### [21:48:00] Recommendation
Action: **ERROR** #8
Reason: #8 is 68% likely evil but HP too low to risk (HP=3, cost=5). Need more info.
WARNING: Probabilistic execution -- 68% confident
WARNING: CRITICAL: HP=3, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [21:49:40] Executed #8 -> GOOD (WRONG!)

#### [21:49:40] Solver Output
Scenarios: 12/210
Definite evil: ['#5']
Definite good: ['#4', '#6', '#8']
Evil probabilities: #2=50%, #3=50%, #1=33%, #7=33%, #9=33%
  Generated 210 candidate scenarios
  12 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 7, 9]

#### [21:49:40] Recommendation
Action: **ERROR** #2
Reason: #2 is 50% likely evil but HP too low to risk (HP=1, cost=5). Need more info.
WARNING: Probabilistic execution -- 50% confident
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [21:50:38] Executed #2 -> Witch (EVIL)

#### [21:50:38] Solver Output
Scenarios: 3/35
Definite evil: ['#2', '#5']
Definite good: ['#3', '#4', '#6', '#8']
Evil probabilities: #1=33%, #7=33%, #9=33%
  Generated 35 candidate scenarios
  3 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Witch'})
    #5 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 7, 9]

#### [21:50:38] Recommendation
Action: **REVEAL** #9
Reason: #9: 33% evil, entropy 1.018

### [21:52:40] Revealed #9 Poet
Info: {}

#### [21:52:40] Solver Output
Scenarios: 3/35
Definite evil: ['#2', '#5']
Definite good: ['#3', '#4', '#6', '#8']
Evil probabilities: #1=33%, #7=33%, #9=33%
  Generated 35 candidate scenarios
  3 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Witch'})
    #5 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 7, 9]

#### [21:52:40] Recommendation
Action: **ERROR** #1
Reason: #1 is 33% likely evil but HP too low to risk (HP=0, cost=5). Need more info.
WARNING: Probabilistic execution -- 33% confident
WARNING: CRITICAL: HP=0, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

## [21:53:33] GAME OVER — LOSS
Final HP: 0
Notes: Lilis game. Wrong exec #4 Alchemist (-5HP), wrong exec #8 Drunk-corrupted (-2HP). Poet corruption fake-out — assumed lying Poet=evil but was corrupted Drunk. Alchemist cured 1 of 2 corruptions. Baker 'I was X' mechanic still unclear.


---

# New Game — 2026-03-06 22:17:15
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Confessor, Alchemist, Dreamer, Enlightened, FortuneTeller, Bard
- Outcasts: Bombardier, PlagueDoctor
- Minions: Witch, Minion
- Demons: Pooka

### [22:19:46] Revealed #1 Bard
Info: {'corruption_distance': -1}

### [22:20:02] Revealed #2 Alchemist
Info: {'cured_count': 2}

### [22:20:02] Revealed #3 FortuneTeller
Info: {}

### [22:20:03] Revealed #4 Enlightened
Info: {'direction': 'cw'}

### [22:20:03] Revealed #5 Dreamer
Info: {}

### [22:20:03] Revealed #6 Bombardier
Info: {}

### [22:20:03] Revealed #7 PlagueDoctor
Info: {}

### [22:20:03] Revealed #8 Confessor
Info: {'dizzy': True}

#### [22:20:14] Solver Output
Scenarios: 2/504
Definite evil: ['#1', '#6', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#7', '#9']
  Generated 504 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #6 is DEFINITELY EVIL (possible roles: {'Minion', 'Witch'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion', 'Witch'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [22:20:14] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [22:21:23] Executed #1 -> wrong (EVIL)

#### [22:21:35] Claude Reasoning


#### [22:22:38] Solver Output
Scenarios: 0/336
  Generated 336 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #2 Alchemist: rejected 180/336 (54%)
    #1 Bard: rejected 162/336 (48%)
    #4 Enlightened: rejected 162/336 (48%)
    #8 Confessor: rejected 150/336 (45%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Bard: 30 scenarios survive  <-- SUSPECT
    WITHOUT #2 Alchemist: 50 scenarios survive  <-- SUSPECT
    WITHOUT #4 Enlightened: 38 scenarios survive  <-- SUSPECT
    WITHOUT #5 Dreamer: 12 scenarios survive  <-- SUSPECT
    WITHOUT #8 Confessor: 30 scenarios survive  <-- SUSPECT

#### [22:22:38] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [22:22:56] Solver Output
Scenarios: 8/336
Definite good: ['#1', '#9']
Evil probabilities: #7=75%, #2=50%, #3=50%, #8=50%, #4=25%, #5=25%, #6=25%
  Generated 336 candidate scenarios
  8 scenarios survived validation
    #1 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6, 7, 8]

#### [22:22:56] Recommendation
Action: **USE_ABILITY** #5 (Dreamer) -> targets ['#7']
Reason: Entropy 2.250 (adjusted 2.250) | timing x1.00

### [22:23:54] Revealed #5 Dreamer
Info: {'target': 7, 'evil_role': 'Minion'}

### [22:23:54] Ability used at #5

#### [22:23:55] Solver Output
Scenarios: 5/336
Definite good: ['#1', '#4', '#9']
Evil probabilities: #3=60%, #7=60%, #8=60%, #2=40%, #5=40%, #6=40%
  Generated 336 candidate scenarios
  5 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 5, 6, 7, 8]

#### [22:24:04] Solver Output
Scenarios: 5/336
Definite good: ['#1', '#4', '#9']
Evil probabilities: #3=60%, #7=60%, #8=60%, #2=40%, #5=40%, #6=40%
  Generated 336 candidate scenarios
  5 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 5, 6, 7, 8]

#### [22:24:04] Recommendation
Action: **ERROR** #8
Reason: #8 is 60% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 60% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: CAUTION: budget=1, confidence 60% < 80% threshold. Consider manual override if you have extra information.

### [22:24:53] Revealed #3 Fortune Teller
Info: {'targets': [7, 8], 'has_evil': True}

### [22:24:53] Ability used at #3

#### [22:24:53] Solver Output
Scenarios: 0/336
  Generated 336 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #2 Alchemist: rejected 180/336 (54%)
    #4 Enlightened: rejected 162/336 (48%)
    #8 Confessor: rejected 150/336 (45%)
    #3 Fortune Teller: rejected 130/336 (39%)
    #5 Dreamer: rejected 72/336 (21%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Bard: still 0
    WITHOUT #2 Alchemist: 2 scenarios survive  <-- SUSPECT
    WITHOUT #3 Fortune Teller: 11 scenarios survive  <-- SUSPECT
    WITHOUT #4 Enlightened: 2 scenarios survive  <-- SUSPECT
    WITHOUT #5 Dreamer: still 0
    WITHOUT #8 Confessor: 26 scenarios survive  <-- SUSPECT

#### [22:24:53] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [22:25:54] Ability used at #7

#### [22:25:54] Solver Output
Scenarios: 0/336
  Generated 336 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #2 Alchemist: rejected 180/336 (54%)
    #4 Enlightened: rejected 162/336 (48%)
    #8 Confessor: rejected 150/336 (45%)
    #3 Fortune Teller: rejected 130/336 (39%)
    #5 Dreamer: rejected 72/336 (21%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Bard: still 0
    WITHOUT #2 Alchemist: 2 scenarios survive  <-- SUSPECT
    WITHOUT #3 Fortune Teller: 11 scenarios survive  <-- SUSPECT
    WITHOUT #4 Enlightened: 2 scenarios survive  <-- SUSPECT
    WITHOUT #5 Dreamer: still 0
    WITHOUT #8 Confessor: 26 scenarios survive  <-- SUSPECT

#### [22:25:54] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [22:29:57] Solver Output
Scenarios: 0/936
  Generated 936 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #4 Enlightened: rejected 428/936 (46%)
    #3 Fortune Teller: rejected 362/936 (39%)
    #8 Confessor: rejected 342/936 (37%)
    #2 Alchemist: rejected 330/936 (35%)
    #5 Dreamer: rejected 72/936 (8%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Bard: 30 scenarios survive  <-- SUSPECT
    WITHOUT #2 Alchemist: 46 scenarios survive  <-- SUSPECT
    WITHOUT #3 Fortune Teller: 75 scenarios survive  <-- SUSPECT
    WITHOUT #4 Enlightened: 52 scenarios survive  <-- SUSPECT
    WITHOUT #5 Dreamer: 30 scenarios survive  <-- SUSPECT
    WITHOUT #8 Confessor: 120 scenarios survive  <-- SUSPECT

#### [22:29:57] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [22:33:54] Solver Output
Scenarios: 0/936
  Generated 936 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #4 Enlightened: rejected 480/936 (51%)
    #3 Fortune Teller: rejected 362/936 (39%)
    #8 Confessor: rejected 342/936 (37%)
    #2 Alchemist: rejected 330/936 (35%)
    #5 Dreamer: rejected 72/936 (8%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Bard: 26 scenarios survive  <-- SUSPECT
    WITHOUT #2 Alchemist: 38 scenarios survive  <-- SUSPECT
    WITHOUT #3 Fortune Teller: 52 scenarios survive  <-- SUSPECT
    WITHOUT #4 Enlightened: 52 scenarios survive  <-- SUSPECT
    WITHOUT #5 Dreamer: 26 scenarios survive  <-- SUSPECT
    WITHOUT #8 Confessor: 74 scenarios survive  <-- SUSPECT

#### [22:33:54] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [22:37:32] Solver Output
Scenarios: 24/936
Definite good: ['#1', '#3', '#4']
Evil probabilities: #6=92%, #8=92%, #5=42%, #9=42%, #2=25%, #7=8%
  Generated 936 candidate scenarios
  24 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [2, 5, 6, 7, 8, 9]

#### [22:37:32] Recommendation
Action: **EXECUTE** #8
Reason: No reveals available. #8 is 92% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 92% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card

### [22:45:12] Executed #8 -> Pooka (EVIL)

#### [22:45:13] Solver Output
Scenarios: 6/152
Definite evil: ['#6', '#8']
Definite good: ['#1', '#3', '#4', '#7']
Evil probabilities: #2=33%, #5=33%, #9=33%
  Generated 152 candidate scenarios
  6 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Minion', 'Witch'})
    #8 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 5, 9]

#### [22:45:13] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 6 scenarios (roles: {'Minion', 'Witch'})

### [22:47:13] Executed #6 -> Minion (EVIL)

#### [22:47:14] Solver Output
Scenarios: 3/27
Definite evil: ['#6', '#8']
Definite good: ['#1', '#3', '#4', '#7']
Evil probabilities: #2=33%, #5=33%, #9=33%
  Generated 27 candidate scenarios
  3 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Minion'})
    #8 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 5, 9]

#### [22:47:14] Recommendation
Action: **ERROR** #2
Reason: #2 is 33% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 33% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: CAUTION: budget=1, confidence 33% < 80% threshold. Consider manual override if you have extra information.

## [22:48:53] GAME OVER — LOSS
Final HP: 0
Notes: 1-in-3 guess at end, solver found 2/3 evils definitively. Fixed PD normalization bug and role_counts normalization bug mid-game.


---

# New Game — 2026-03-07 11:15:53
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Gemcrafter, Baker, Knitter, Architect, Dreamer, Empress
- Outcasts: Doppelganger
- Minions: Minion
- Demons: Pooka

## Deck
- Villagers: Gemcrafter, Baker, Knitter, Architect, Dreamer, Empress
- Outcasts: Doppelganger
- Minions: Minion
- Demons: Pooka

#### [11:16:16] Solver Output
Scenarios: 56/56
Evil probabilities: #1=25%, #2=25%, #3=25%, #4=25%, #5=25%, #6=25%, #7=25%, #8=25%
  Generated 56 candidate scenarios
  56 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [11:16:16] Recommendation
Action: **REVEAL** #1
Reason: #1: 25% evil, entropy 0.911

### [11:20:13] Revealed #1 Gemcrafter
Info: {'good_position': 8}

### [11:20:13] Revealed #3 Empress
Info: {'targets': [2, 6, 8]}

### [11:20:13] Revealed #2 Empress
Info: {'targets': [4, 6, 8]}

### [11:20:13] Revealed #4 Architect
Info: {'side': 'Left'}

#### [11:21:10] Solver Output
Scenarios: 20/392
Definite good: ['#1', '#4']
Evil probabilities: #5=60%, #6=40%, #7=40%, #8=30%, #3=25%, #2=5%
  Generated 392 candidate scenarios
  20 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [2, 3, 5, 6, 7, 8]

#### [11:21:10] Recommendation
Action: **USE_ABILITY** #8 (Dreamer) -> targets ['#5']
Reason: Entropy 2.090 (adjusted 1.777) | timing x1.00
WARNING: Corruption risk: 30%

### [11:22:23] Revealed #8 Dreamer
Info: {'target': 5, 'evil_role': 'Pooka'}

### [11:22:23] Ability used at #8

#### [11:22:31] Solver Output
Scenarios: 15/392
Definite good: ['#1', '#3', '#4']
Evil probabilities: #6=53%, #7=53%, #5=47%, #8=40%, #2=7%
  Generated 392 candidate scenarios
  15 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [2, 5, 6, 7, 8]

#### [11:22:31] Recommendation
Action: **EXECUTE** #6
Reason: No reveals available. #6 is 53% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 53% confident (budget: 2 wrong execs)

### [11:23:15] Executed #6 -> GOOD (WRONG!)

#### [11:23:22] Solver Output
Scenarios: 7/294
Definite evil: ['#5']
Definite good: ['#1', '#3', '#4', '#6', '#7']
Evil probabilities: #8=86%, #2=14%
  Generated 294 candidate scenarios
  7 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka', 'Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 8]

#### [11:23:22] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 7 scenarios (roles: {'Pooka', 'Minion'})

#### [11:23:45] Solver Output
Scenarios: 7/294
Definite evil: ['#5']
Definite good: ['#1', '#3', '#4', '#6', '#7']
Evil probabilities: #8=86%, #2=14%
  Generated 294 candidate scenarios
  7 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion', 'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 8]

#### [11:23:45] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 7 scenarios (roles: {'Minion', 'Pooka'})

### [11:24:26] Executed #5 -> Minion (EVIL)

#### [11:24:31] Solver Output
Scenarios: 6/42
Definite evil: ['#5', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#7']
  Generated 42 candidate scenarios
  6 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #8 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [11:24:31] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 6 scenarios (roles: {'Pooka'})

### [11:25:09] Executed #8 -> Pooka (EVIL)

## [11:27:47] GAME OVER — WIN
Final HP: 5
Notes: Live run. #3=Doppelganger, #1/#7 corrupted. Dreamer #8 -> #5 Pooka. Wrong exec #6, then solved via #5 Minion and #8 Pooka.


---

# New Game — 2026-03-07 13:08:27
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Bard, Judge, Empress, Architect, Hunter, Fortune_Teller, Gemcrafter
- Outcasts: Plague_Doctor
- Minions: Shaman, Poisoner
- Demons: Lilis

### [13:08:28] Revealed #1 Bard
Info: {'corruption_distance': 3}

### [13:08:28] Revealed #2 Hunter
Info: {'distance': 3}

### [13:08:28] Revealed #3 Judge
Info: {'target': 1, 'is_lying': True}

### [13:08:28] Revealed #4 Architect
Info: {'side': 'Left'}

### [13:08:28] Revealed #5 Fortune Teller
Info: {'targets': [1, 3], 'has_evil': False}

### [13:08:28] Ability used at #5

### [13:08:28] Revealed #6 Plague Doctor
Info: {}

### [13:08:29] Revealed #8 Slayer
Info: {}

### [13:08:29] Revealed #9 Empress
Info: {'targets': [1, 3, 5]}

### [13:08:29] Executed #2 -> Lilis (EVIL)

### [13:08:29] Executed #4 -> Shaman (EVIL)

### [13:08:29] Executed #9 -> GOOD (WRONG!)

### [13:08:30] Executed #8 -> Poisoner (EVIL)

## [13:08:30] GAME OVER — WIN
Final HP: 1
Notes: Autonomous live run. #7 was the good Lilis night kill. PD #6 found #1 corrupted and #2 evil. FT #5 on #1/#3 was false. Judge #3 found #1 lying. #1 and #9 were corrupted. Final evils: #2 Lilis, #4 Shaman, #8 Poisoner.

## Deck
- Villagers: Lover, Slayer, Knight, Bard, Confessor, Dreamer, Bishop
- Outcasts: Plague_Doctor
- Minions: Minion, Twin_Minion
- Demons: Pooka


---

# New Game — 2026-03-07 15:48:40
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

### [15:50:56] Revealed #1 Bishop
Info: {'targets': [1, 4, 6], 'types': ['Minion', 'Villager', 'Outcast']}

### [15:50:56] Revealed #5 Confessor
Info: {'dizzy': True}

### [15:50:57] Revealed #3 Dreamer
Info: {}

### [15:50:57] Revealed #2 Bard
Info: {'corruption_distance': 1}

### [15:50:57] Revealed #6 Lover
Info: {'evil_adjacent': 1}

### [15:50:57] Revealed #8 Plague_Doctor
Info: {}

### [15:50:57] Revealed #4 Slayer
Info: {}

### [15:50:59] Revealed #9 Plague_Doctor
Info: {}

## Deck
- Villagers: Lover, Slayer, Knight, Bard, Confessor, Dreamer, Bishop
- Outcasts: Plague_Doctor
- Minions: Minion, Twin_Minion
- Demons: Pooka

### [15:51:19] Revealed #7 Knight
Info: {}

#### [15:51:28] Solver Output
Scenarios: 106/2142
Definite good: ['#2']
Evil probabilities: #9=60%, #5=58%, #1=42%, #8=40%, #4=32%, #6=26%, #3=21%, #7=21%
  Generated 2142 candidate scenarios
  106 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8, 9]

#### [15:51:28] Recommendation
Action: **USE_ABILITY** #8 (Plague Doctor) -> targets ['#1']
Reason: Entropy 2.776 (adjusted 2.776) | timing x1.00

#### [15:52:46] Solver Output
Scenarios: 64/2142
Definite good: ['#2']
Evil probabilities: #9=66%, #5=59%, #1=34%, #4=34%, #8=34%, #3=25%, #6=25%, #7=22%
  Generated 2142 candidate scenarios
  64 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8, 9]

#### [15:52:46] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#9']
Reason: Entropy 2.776 (adjusted 2.429) | timing x1.00
WARNING: Corruption risk: 25%

### [15:53:23] Ability used at #3

### [15:53:23] Revealed #3 Dreamer
Info: {'target': 9, 'evil_role': 'Twin_Minion'}

#### [15:53:23] Solver Output
Scenarios: 44/2142
Definite good: ['#2']
Evil probabilities: #5=59%, #1=50%, #8=50%, #9=50%, #3=32%, #6=25%, #4=18%, #7=16%
  Generated 2142 candidate scenarios
  44 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8, 9]

#### [15:53:23] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.143 (adjusted 2.143) | timing x1.00

#### [15:53:57] Solver Output
Scenarios: 44/2142
Definite good: ['#2']
Evil probabilities: #5=59%, #1=50%, #8=50%, #9=50%, #3=32%, #6=25%, #4=18%, #7=16%
  Generated 2142 candidate scenarios
  44 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8, 9]

#### [15:53:57] Recommendation
Action: **USE_ABILITY** #9 (Plague Doctor) -> targets ['#5']
Reason: Entropy 2.143 (adjusted 2.143) | timing x1.00

#### [15:55:02] Solver Output
Scenarios: 14/2142
Definite evil: ['#5', '#9']
Definite good: ['#1', '#2', '#6', '#8']
Evil probabilities: #3=64%, #4=21%, #7=14%
  Generated 2142 candidate scenarios
  14 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Minion', 'Pooka'})
    #9 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Minion', 'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 7]

#### [15:55:02] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 14 scenarios (roles: {'Twin_Minion', 'Minion', 'Pooka'})

### [15:56:47] Executed #5 -> Minion (EVIL)

#### [15:56:48] Solver Output
Scenarios: 6/242
Definite evil: ['#5', '#9']
Definite good: ['#1', '#2', '#6', '#8']
Evil probabilities: #3=67%, #4=17%, #7=17%
  Generated 242 candidate scenarios
  6 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #9 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 7]

#### [15:56:48] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 6 scenarios (roles: {'Twin_Minion', 'Pooka'})

#### [15:57:18] Solver Output
Scenarios: 6/242
Definite evil: ['#5', '#9']
Definite good: ['#1', '#2', '#6', '#8']
Evil probabilities: #3=67%, #4=17%, #7=17%
  Generated 242 candidate scenarios
  6 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #9 is DEFINITELY EVIL (possible roles: {'Twin_Minion', 'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 7]

#### [15:57:18] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 6 scenarios (roles: {'Twin_Minion', 'Pooka'})

### [15:57:18] Executed #9 -> Twin_Minion (EVIL)

#### [15:57:26] Solver Output
Scenarios: 1/36
Definite evil: ['#5', '#7', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#8']
  Generated 36 candidate scenarios
  1 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #7 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #9 is DEFINITELY EVIL (possible roles: {'Twin_Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [15:57:26] Recommendation
Action: **EXECUTE** #1
Reason: No reveals available. #1 is 0% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 0% confident (budget: 2 wrong execs)
WARNING: Low confidence (0%) -- consider gathering more info

### [15:58:19] Executed #7 -> Pooka (EVIL)

## [15:58:19] GAME OVER — WIN
Final HP: 10
Notes: Autonomous live run. Deck: Lover, Slayer, Knight, Bard, Confessor, Dreamer, Bishop, Plague Doctor, Minion, Twin Minion, Pooka. #1 Bishop and #6 Lover were corrupted. PD #8 found #1 corrupted and #9 evil. Dreamer #3 found #9 could be Twin Minion. PD #9 found #5 corrupted and #2 evil; that branch was treated as corrupted PD misinformation. Final evils: #5 Minion, #7 Pooka, #9 Twin Minion. Finished at 10 HP.

#### [16:23:27] Solver Output
Scenarios: 632/2016
Definite good: ['#7']
Evil probabilities: #5=41%, #8=41%, #3=23%, #1=21%, #4=21%, #9=21%, #6=19%, #2=13%
  Generated 2016 candidate scenarios
  632 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8, 9]

#### [16:23:28] Recommendation
Action: **USE_ABILITY** #4 (Jester) -> targets ['#1', '#5', '#8']
Reason: Expected posterior 264.9 scenarios (adjusted 279.5, info gain 1.177 bits) | timing x1.00
WARNING: Corruption risk: 11%

#### [16:26:00] Solver Output
Scenarios: 300/2016
Definite good: ['#7']
Evil probabilities: #5=65%, #8=31%, #1=29%, #3=21%, #4=19%, #9=19%, #6=10%, #2=7%
  Generated 2016 candidate scenarios
  300 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8, 9]

#### [16:26:00] Recommendation
Action: **USE_ABILITY** #9 (Jester) -> targets ['#2', '#4', '#5']
Reason: Expected posterior 123.1 scenarios (adjusted 130.1, info gain 1.205 bits) | timing x1.00
WARNING: Corruption risk: 11%

#### [16:27:24] Solver Output
Scenarios: 300/2016
Definite good: ['#7']
Evil probabilities: #5=65%, #8=31%, #1=29%, #3=21%, #4=19%, #9=19%, #6=10%, #2=7%
  Generated 2016 candidate scenarios
  300 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8, 9]

#### [16:27:24] Recommendation
Action: **USE_ABILITY** #9 (Jester) -> targets ['#2', '#4', '#5']
Reason: Expected posterior 123.1 scenarios (adjusted 130.1, info gain 1.205 bits) | timing x1.00
WARNING: Corruption risk: 11%

#### [18:38:30] Solver Output
Scenarios: 220/2016
Definite good: ['#7']
Evil probabilities: #5=69%, #1=31%, #8=31%, #3=22%, #9=18%, #4=17%, #6=9%, #2=3%
  Generated 2016 candidate scenarios
  220 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8, 9]

#### [18:38:30] Recommendation
Action: **USE_ABILITY** #9 (Jester) -> targets ['#1', '#2', '#8']
Reason: Expected posterior 93.2 scenarios (adjusted 100.4, info gain 1.132 bits) | timing x1.00
WARNING: Corruption risk: 15%

#### [18:47:28] Solver Output
Scenarios: 96/2016
Definite good: ['#6', '#7']
Evil probabilities: #5=58%, #1=42%, #9=42%, #8=40%, #3=8%, #4=8%, #2=2%
  Generated 2016 candidate scenarios
  96 scenarios survived validation
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 8, 9]

#### [18:47:28] Recommendation
Action: **USE_ABILITY** #8 (Jester) -> targets ['#1', '#2', '#5']
Reason: Expected posterior 47.0 scenarios (adjusted 49.0, info gain 0.971 bits) | timing x1.00
WARNING: Corruption risk: 8%

### [18:54:53] Revealed #8 Jester
Info: {'targets': [1, 2, 5], 'evil_count': 3}

### [18:54:57] Ability used at #8

#### [18:55:05] Solver Output
Scenarios: 46/2016
Definite good: ['#2', '#3', '#4', '#6', '#7']
Evil probabilities: #8=83%, #1=65%, #5=35%, #9=17%
  Generated 2016 candidate scenarios
  46 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 5, 8, 9]

#### [18:55:05] Recommendation
Action: **USE_ABILITY** #5 (Slayer) -> targets ['#8']
Reason: Target #8 is 83% evil (adjusted 0.68)
WARNING: Corruption risk: 17% -- Slayer ability disabled if corrupted

#### [18:58:37] Solver Output
Scenarios: 24/2016
Definite good: ['#2', '#3', '#4', '#6', '#7']
Evil probabilities: #5=67%, #8=67%, #1=33%, #9=33%
  Generated 2016 candidate scenarios
  24 scenarios survived validation
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 5, 8, 9]

#### [18:58:37] Recommendation
Action: **ERROR** #5
Reason: #5 is 67% likely evil but budget=1 requires >=80% confidence (HP=6, cost=5).
WARNING: Probabilistic execution -- 67% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 67% < 80% threshold. Consider manual override if you have extra information.

### [19:02:54] Executed #5 -> GOOD (WRONG!)

#### [19:03:05] Solver Output
Scenarios: 8/1512
Definite evil: ['#1', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7', '#9']
  Generated 1512 candidate scenarios
  8 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Shaman', 'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Shaman', 'Lilis'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [19:03:05] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 8 scenarios (roles: {'Shaman', 'Lilis'})

### [19:04:24] Executed #1 -> Lilis (EVIL)

#### [19:04:28] Solver Output
Scenarios: 4/216
Definite evil: ['#1', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7', '#9']
  Generated 216 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [19:04:28] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 4 scenarios (roles: {'Shaman'})

### [19:05:54] Executed #8 -> Shaman (EVIL)

## [19:07:14] GAME OVER — WIN
Final HP: 4
Notes: Safe endgame line was execute #5 first; #5 revealed Drunk (corrupted) and the final screen showed #9 as Doppelganger. This exposed an execution lookahead gap in strategy.

#### [19:09:22] Solver Output
Scenarios: 4/36
Definite evil: ['#1', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7', '#9']
  Generated 36 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [19:09:22] Recommendation
Action: **WIN**
Reason: All evil characters have been executed!

## Deck
- Villagers: Bishop, Scout, Hunter, Alchemist, Poet, Jester, Witness
- Outcasts: Drunk, Plague_Doctor, Doppelganger
- Minions: Minion, Chancellor
- Demons: Baa


---

# New Game — 2026-03-07 19:33:11
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

### [19:35:41] Revealed #2 Scout
Info: {'evil_role': 'Minion', 'distance': 3}

### [19:35:41] Revealed #6 Bishop
Info: {'targets': [4, 7, 8], 'types': ['Minion', 'Outcast', 'Villager']}

### [19:35:41] Revealed #5 Poet
Info: {}

### [19:35:41] Revealed #4 Scout
Info: {'evil_role': 'Chancellor', 'distance': 1}

### [19:35:41] Revealed #7 Hunter
Info: {'distance': 2}

### [19:35:41] Revealed #9 Jester
Info: {'targets': [1, 2, 3], 'evil_count': 0}

### [19:35:41] Revealed #8 Alchemist
Info: {'cured_count': 0}

### [19:35:41] Revealed #3 Hunter
Info: {'distance': 1}

### [19:35:52] Revealed #9 Jester
Info: {}

#### [19:36:37] Solver Output
Scenarios: 402/18954
Definite good: ['#8']
Evil probabilities: #2=67%, #6=64%, #7=60%, #4=39%, #1=33%, #9=16%, #3=13%, #5=8%
  Generated 18954 candidate scenarios
  402 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [19:36:37] Recommendation
Action: **USE_ABILITY** #9 (Jester) -> targets ['#1', '#4', '#5']
Reason: Expected posterior 155.1 scenarios (adjusted 160.9, info gain 1.321 bits) | timing x1.00
WARNING: Corruption risk: 7%

### [19:38:48] Revealed #9 Jester
Info: {'targets': [1, 4, 5], 'evil_count': 2}

### [19:38:53] Ability used at #9

#### [19:38:59] Solver Output
Scenarios: 164/18954
Definite good: ['#8']
Evil probabilities: #4=60%, #2=59%, #1=51%, #7=40%, #9=39%, #6=32%, #5=13%, #3=6%
  Generated 18954 candidate scenarios
  164 scenarios survived validation
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 9]

#### [19:38:59] Recommendation
Action: **REVEAL** #1
Reason: #1: 51% evil, entropy 1.000

### [19:40:17] Revealed #1 Scout
Info: {'evil_role': 'Baa', 'distance': 2}

#### [19:40:24] Solver Output
Scenarios: 118/20898
Definite good: ['#3', '#8']
Evil probabilities: #4=69%, #1=61%, #2=61%, #9=59%, #7=31%, #5=10%, #6=8%
  Generated 20898 candidate scenarios
  118 scenarios survived validation
    #3 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 6, 7, 9]

#### [19:40:24] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (25% good Scout, 25% evil Baa, 25% evil Chancellor).
WARNING: Execution lookahead override -- immediate hit chance is 69%, but all reveal branches still lead to a forced win.

### [19:41:55] Executed #4 -> Minion (EVIL)

#### [19:42:02] Solver Output
Scenarios: 30/2408
Definite evil: ['#4']
Definite good: ['#3', '#6', '#8']
Evil probabilities: #2=60%, #1=40%, #7=40%, #9=40%, #5=20%
  Generated 2408 candidate scenarios
  30 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 7, 9]

#### [19:42:02] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (40% evil Baa, 33% good Scout, 20% evil Chancellor).
WARNING: Execution lookahead override -- immediate hit chance is 60%, but all reveal branches still lead to a forced win.

### [19:46:10] Executed #2 -> Chancellor (EVIL)

#### [19:46:18] Solver Output
Scenarios: 6/301
Definite evil: ['#2', '#4', '#9']
Definite good: ['#1', '#3', '#5', '#6', '#7', '#8']
  Generated 301 candidate scenarios
  6 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #9 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [19:46:18] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 6 scenarios (roles: {'Baa'})

### [19:47:54] Executed #9 -> GOOD (WRONG!)

#### [19:48:09] Solver Output
Scenarios: 0/258
  Generated 258 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #6 Bishop: rejected 185/258 (72%)
    #9 Jester: rejected 160/258 (62%)
    #7 Hunter: rejected 154/258 (60%)
    #1 Scout: rejected 154/258 (60%)
    #4 Scout: rejected 86/258 (33%)
    #3 Hunter: rejected 73/258 (28%)
    #8 Alchemist: rejected 73/258 (28%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Scout: 6 scenarios survive  <-- SUSPECT
    WITHOUT #2 Scout: still 0
    WITHOUT #3 Hunter: still 0
    WITHOUT #4 Scout: still 0
    WITHOUT #5 Poet: still 0
    WITHOUT #6 Bishop: 12 scenarios survive  <-- SUSPECT
    WITHOUT #7 Hunter: 12 scenarios survive  <-- SUSPECT
    WITHOUT #8 Alchemist: still 0
    WITHOUT #9 Jester: 6 scenarios survive  <-- SUSPECT

#### [19:48:09] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [19:54:35] Solver Output
Scenarios: 0/258
  Generated 258 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #6 Bishop: rejected 185/258 (72%)
    #9 Jester: rejected 160/258 (62%)
    #7 Hunter: rejected 154/258 (60%)
    #1 Scout: rejected 154/258 (60%)
    #4 Scout: rejected 86/258 (33%)
    #3 Hunter: rejected 73/258 (28%)
    #8 Alchemist: rejected 73/258 (28%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Scout: 6 scenarios survive  <-- SUSPECT
    WITHOUT #2 Scout: still 0
    WITHOUT #3 Hunter: still 0
    WITHOUT #4 Scout: still 0
    WITHOUT #5 Poet: still 0
    WITHOUT #6 Bishop: 12 scenarios survive  <-- SUSPECT
    WITHOUT #7 Hunter: 12 scenarios survive  <-- SUSPECT
    WITHOUT #8 Alchemist: still 0
    WITHOUT #9 Jester: 6 scenarios survive  <-- SUSPECT

#### [19:54:35] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [19:59:26] Solver Output
Scenarios: 0/258
  Generated 258 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #6 Bishop: rejected 185/258 (72%)
    #9 Jester: rejected 160/258 (62%)
    #7 Hunter: rejected 154/258 (60%)
    #1 Scout: rejected 154/258 (60%)
    #4 Scout: rejected 86/258 (33%)
    #3 Hunter: rejected 73/258 (28%)
    #8 Alchemist: rejected 73/258 (28%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Scout: 6 scenarios survive  <-- SUSPECT
    WITHOUT #2 Scout: still 0
    WITHOUT #3 Hunter: still 0
    WITHOUT #4 Scout: still 0
    WITHOUT #5 Poet: still 0
    WITHOUT #6 Bishop: 6 scenarios survive  <-- SUSPECT
    WITHOUT #7 Hunter: 6 scenarios survive  <-- SUSPECT
    WITHOUT #8 Alchemist: still 0
    WITHOUT #9 Jester: 6 scenarios survive  <-- SUSPECT

#### [19:59:26] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [20:04:17] Solver Output
Scenarios: 2/258
Definite evil: ['#2', '#4', '#5']
Definite good: ['#1', '#3', '#6', '#7', '#8', '#9']
  Generated 258 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [20:04:17] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 2 scenarios (roles: {'Baa'})

#### [20:05:13] Solver Output
Scenarios: 2/258
Definite evil: ['#2', '#4', '#5']
Definite good: ['#1', '#3', '#6', '#7', '#8', '#9']
  Generated 258 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [20:05:13] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 2 scenarios (roles: {'Baa'})

### [20:07:26] Executed #5 -> Baa (EVIL)

## [20:07:35] GAME OVER — WIN
Final HP: 5
Notes: #1=Drunk(corrupted), #7=Doppelganger. Fixed executed-evil clue handling and hidden-outcast Bishop typing from this run.


---

# New Game — 2026-03-07 20:09:32
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Hunter, Lover, Knitter, Enlightened
- Outcasts: Doppelganger
- Minions: 
- Demons: Pooka

### [20:22:41] Revealed #1 Poet
Info: {'corruption_distance': 1, 'copied_role': 'Bard'}

### [20:22:48] Revealed #2 Poet
Info: {'good_position': 6, 'copied_role': 'Gemcrafter'}

### [20:22:58] Revealed #3 Hunter
Info: {'distance': 2}

### [20:23:05] Revealed #4 Knitter
Info: {'evil_pairs': 1}

### [20:23:11] Revealed #5 Lover
Info: {'evil_adjacent': 1}

### [20:23:15] Revealed #6 Enlightened
Info: {'direction': 'CW'}

#### [20:23:21] Solver Output
Scenarios: 3/30
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4', '#6']
  Generated 30 candidate scenarios
  3 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [20:23:21] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 3 scenarios (roles: {'Pooka'})

### [20:23:52] Executed #5 -> Pooka (EVIL)

## [20:24:11] GAME OVER — WIN
Final HP: 10
Notes: Poet clues came from Bard/Gemcrafter patterns not in play. Final screen: #1=Doppelganger, #4 and #6 corrupted by Pooka.


---

# New Game — 2026-03-07 20:26:24
Cards: 7, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Knight, Confessor, Dreamer, Slayer, Gemcrafter
- Outcasts: Drunk, Plague_Doctor
- Minions: Puppeteer
- Demons: Baa

### [20:29:04] Revealed #1 Gemcrafter
Info: {'good_position': 5}

### [20:29:04] Revealed #2 Slayer
Info: {}

### [20:29:04] Revealed #3 Plague_Doctor
Info: {}

### [20:29:13] Revealed #4 Knight
Info: {}

### [20:29:13] Revealed #5 Gemcrafter
Info: {'good_position': 1}

### [20:29:13] Revealed #7 Confessor
Info: {'dizzy': True}

### [20:29:21] Revealed #6 Dreamer
Info: {}

#### [20:29:26] Solver Output
Scenarios: 103/630
Evil probabilities: #7=63%, #1=57%, #5=49%, #2=37%, #4=37%, #6=37%, #3=20%
  Generated 630 candidate scenarios
  103 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [20:29:26] Recommendation
Action: **USE_ABILITY** #6 (Dreamer) -> targets ['#7']
Reason: Entropy 2.915 (adjusted 2.604) | timing x1.00
WARNING: Corruption risk: 21%

### [20:30:21] Revealed #6 Dreamer
Info: {'target': 7, 'evil_role': 'Baa'}

### [20:30:21] Ability used at #6

#### [20:30:30] Solver Output
Scenarios: 70/630
Evil probabilities: #1=60%, #5=51%, #7=46%, #2=41%, #4=41%, #6=40%, #3=20%
  Generated 630 candidate scenarios
  70 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [20:30:30] Recommendation
Action: **USE_ABILITY** #3 (Plague Doctor) -> targets ['#7']
Reason: Entropy 2.584 (adjusted 2.584) | timing x1.00

#### [20:32:15] Solver Output
Scenarios: 34/630
Evil probabilities: #7=76%, #1=59%, #2=38%, #4=38%, #6=35%, #5=29%, #3=24%
  Generated 630 candidate scenarios
  34 scenarios survived validation
    Uncertain: [1, 2, 3, 4, 5, 6, 7]

#### [20:32:15] Recommendation
Action: **USE_ABILITY** #2 (Slayer) -> targets ['#7']
Reason: Target #7 is 76% evil (adjusted 0.61)
WARNING: Corruption risk: 21% -- Slayer ability disabled if corrupted

### [20:39:44] Executed #7 -> Puppet (EVIL)

#### [20:39:50] Solver Output
Scenarios: 4/106
Definite evil: ['#1', '#5', '#7']
Definite good: ['#2', '#3', '#4', '#6']
  Generated 106 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [20:39:50] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 4 scenarios (roles: {'Puppeteer'})

### [20:40:47] Executed #1 -> Puppeteer (EVIL)

#### [20:40:54] Solver Output
Scenarios: 4/53
Definite evil: ['#1', '#5', '#7']
Definite good: ['#2', '#3', '#4', '#6']
  Generated 53 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [20:40:54] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 4 scenarios (roles: {'Baa'})

### [20:41:53] Executed #5 -> Baa (EVIL)

## [20:41:59] GAME OVER — WIN
Final HP: 10
Notes: Autonomous live run. Slayer #2 killed #7=Puppet after a retarget retry. Final evils: #1 Puppeteer, #5 Baa, #7 Puppet. Final screen showed #6 Dreamer corrupted.


---

# New Game — 2026-03-07 23:42:40
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Baker, Hunter, Medium, Architect, Slayer, Bard
- Outcasts: PlagueDoctor, Doppelganger
- Minions: Shaman, Puppeteer
- Demons: Baa

### [23:47:47] Revealed #1 Medium
Info: {'good_position': 2, 'good_role': 'real'}

### [23:47:47] Revealed #2 Plaguedoctor
Info: {}

### [23:47:48] Revealed #3 Bard
Info: {'corruption_distance': 2}

### [23:47:48] Revealed #4 Hunter
Info: {'distance': 1}

### [23:47:48] Revealed #5 Slayer
Info: {}

### [23:47:48] Revealed #6 Hunter
Info: {'distance': 4}

### [23:47:48] Revealed #7 Baker
Info: {'original_role': 'original'}

### [23:47:49] Revealed #8 Baker
Info: {'original_role': 'hunter'}

### [23:47:49] Revealed #9 Plaguedoctor
Info: {}

### [23:49:22] Executed #1

### [23:49:22] Executed #3

### [23:49:22] Executed #6

## Deck
- Villagers: Baker, Hunter, Medium, Architect, Slayer, Bard
- Outcasts: PlagueDoctor, Doppelganger
- Minions: Shaman, Puppeteer, Puppet
- Demons: Baa

### [00:16:07] Executed #2

## [00:16:08] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Puppeteer+Puppet mechanic = 4 evils from 3-evil deck. Medium lie on #2 revealed Puppeteer. Solver found first 3 instantly (30 scenarios), manual deduction for 4th. #5 Slayer corrupted by Shaman.


---

# New Game — 2026-03-08 00:24:05
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Bard, Bishop, Scout, Confessor, Hunter, Slayer
- Outcasts: Wretch, Drunk
- Minions: Puppeteer
- Demons: Baa

### [00:26:38] Revealed #1 Slayer
Info: {}

### [00:26:38] Revealed #2 Bishop
Info: {'targets': [5, 6, 7], 'types': ['Outcast', 'Villager', 'Minion']}

### [00:26:39] Revealed #3 Wretch
Info: {}

### [00:26:39] Revealed #4 Bard
Info: {'corruption_distance': 3}

### [00:26:39] Revealed #5 Hunter
Info: {'distance': 1}

### [00:26:39] Revealed #6 Scout
Info: {'evil_role': 'Puppet', 'distance': 1}

### [00:26:40] Revealed #7 Confessor
Info: {'dizzy': False}

#### [00:26:52] Solver Output
Scenarios: 22/930
Definite good: ['#7']
Evil probabilities: #4=73%, #2=64%, #3=55%, #5=55%, #1=45%, #6=9%
  Generated 930 candidate scenarios
  22 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6]

#### [00:26:52] Recommendation
Action: **USE_ABILITY** #1 (Slayer) -> targets ['#4']
Reason: Target #4 is 73% evil (adjusted 0.60)
WARNING: Corruption risk: 18% -- Slayer ability disabled if corrupted

### [00:27:58] Executed #4

#### [00:27:59] Solver Output
Scenarios: 0/540
  Generated 540 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #7 Confessor: rejected 324/540 (60%)
    #2 Bishop: rejected 304/540 (56%)
    #4 Bard: rejected 300/540 (56%)
    #6 Scout: rejected 286/540 (53%)
    #5 Hunter: rejected 248/540 (46%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #2 Bishop: 16 scenarios survive  <-- SUSPECT
    WITHOUT #4 Bard: 20 scenarios survive  <-- SUSPECT
    WITHOUT #5 Hunter: 14 scenarios survive  <-- SUSPECT
    WITHOUT #6 Scout: 22 scenarios survive  <-- SUSPECT
    WITHOUT #7 Confessor: 16 scenarios survive  <-- SUSPECT

#### [00:27:59] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [00:33:43] Executed #3

### [00:33:43] Executed #5

## [00:33:43] GAME OVER — LOSS
Final HP: 0
Notes: Ascension 12 first game. Solver 0-scenario on Puppet mechanic. Manual deduction wrong: assumed Scout's Puppet distance was to Baa but Puppet#1 was 1 away from Puppeteer#2. Wrong execs on #5 Hunter and #3 Wretch. Need solver Puppet support and better Scout reasoning.


---

# New Game — 2026-03-08 01:42:45
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Jester, FortuneTeller, Alchemist, Druid, Bard
- Outcasts: Lover, Oracle
- Minions: Poisoner
- Demons: Lilis

## Deck
- Villagers: Jester, FortuneTeller, Alchemist, Druid, Bard, Confessor
- Outcasts: Lover, Oracle, Drunk, Doppelganger
- Minions: Poisoner
- Demons: Lilis

### [01:43:19] Revealed #1 FortuneTeller
Info: {}

### [01:44:22] Revealed #2 Jester
Info: {}

### [01:44:27] Revealed #3 Druid
Info: {}

### [01:44:27] Revealed #4 Bard
Info: {'corruption_distance': 1}

### [01:44:28] Revealed #6 Oracle
Info: {'targets': [7, 9], 'minion_role': 'Poisoner'}

### [01:44:28] Revealed #7 Confessor
Info: {'dizzy': True}

### [01:44:28] Revealed #8 Oracle
Info: {'targets': [5, 7], 'minion_role': 'Poisoner'}

### [01:44:28] Revealed #9 Alchemist
Info: {'cured_count': 1}

#### [01:45:10] Solver Output
Scenarios: 68/5896
Definite good: ['#5', '#6', '#8']
Evil probabilities: #7=94%, #9=34%, #4=21%, #1=19%, #2=19%, #3=13%
  Generated 5896 candidate scenarios
  68 scenarios survived validation
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 7, 9]

#### [01:45:10] Recommendation
Action: **USE_ABILITY** #2 (Jester) -> targets ['#1', '#6', '#7']
Reason: Expected posterior 27.3 scenarios (adjusted 28.1, info gain 1.274 bits) | timing x1.00
WARNING: Corruption risk: 6%

### [01:49:14] Revealed #2 Jester
Info: {'targets': [1, 6, 7], 'evil_count': 1}

#### [01:49:19] Solver Output
Scenarios: 34/5896
Definite evil: ['#7']
Definite good: ['#1', '#2', '#5', '#6', '#8']
Evil probabilities: #9=56%, #3=26%, #4=18%
  Generated 5896 candidate scenarios
  34 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner', 'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 9]

#### [01:49:19] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 34 scenarios (roles: {'Poisoner', 'Lilis'})

### [01:52:56] Executed #7 -> Poisoner (EVIL)

#### [01:52:56] Solver Output
Scenarios: 21/714
Definite evil: ['#7']
Definite good: ['#1', '#2', '#5', '#6', '#8']
Evil probabilities: #3=43%, #4=29%, #9=29%
  Generated 714 candidate scenarios
  21 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 9]

#### [01:52:56] Recommendation
Action: **USE_ABILITY** #2 (Jester) -> targets ['#1', '#3', '#6']
Reason: Expected posterior 10.7 scenarios (adjusted 10.7, info gain 0.971 bits) | timing x1.00

### [01:54:55] Ability used at #2

#### [01:56:20] Solver Output
Scenarios: 21/714
Definite evil: ['#7']
Definite good: ['#1', '#2', '#5', '#6', '#8']
Evil probabilities: #3=43%, #4=29%, #9=29%
  Generated 714 candidate scenarios
  21 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 9]

#### [01:56:20] Recommendation
Action: **USE_ABILITY** #3 (Druid) -> targets ['#1', '#2', '#4']
Reason: Entropy 0.998 (adjusted 0.951) | timing x1.00
WARNING: Corruption risk: 10%

### [01:57:26] Revealed #3 Druid
Info: {'targets': [1, 2, 4], 'found_outcast': None}

### [01:57:26] Ability used at #3

#### [01:57:27] Solver Output
Scenarios: 11/714
Definite evil: ['#7']
Definite good: ['#1', '#2', '#5', '#6', '#8']
Evil probabilities: #3=64%, #4=18%, #9=18%
  Generated 714 candidate scenarios
  11 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 4, 9]

#### [01:57:27] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (64% evil Lilis, 36% good Druid).
WARNING: Execution lookahead override -- immediate hit chance is 64%, but all reveal branches still lead to a forced win.

### [01:58:04] Executed #3 -> Lilis (EVIL)

## [01:58:12] GAME OVER — WIN
Final HP: 6
Notes: Lilis killed #5. Druid+Jester abilities. Clean win 6HP.


---

# New Game — 2026-03-08 03:09:45
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Hunter, Knitter, Empress, Enlightened, Poet, Jester, FortuneTeller
- Outcasts: Baker, Bombardier
- Minions: Puppeteer, Shaman
- Demons: Lilis

### [03:11:07] Revealed #1 Hunter
Info: {'distance': 5}

### [03:11:07] Revealed #2 Enlightened
Info: {'direction': 'equidistant'}

### [03:11:07] Revealed #3 Enlightened
Info: {'direction': 'CW'}

### [03:11:07] Revealed #4 Knitter
Info: {'evil_pairs': 2}

### [03:11:07] Revealed #7 FortuneTeller
Info: {}

### [03:11:07] Revealed #8 Poet
Info: {'copied_role': 'Bounty Hunter', 'evil_position': 7}

### [03:11:08] Revealed #9 Baker
Info: {'original_role': 'original'}

### [03:11:08] Revealed #10 Bombardier
Info: {}

#### [03:11:16] Solver Output
Scenarios: 12/1120
Definite evil: ['#1', '#3']
Definite good: ['#5', '#6', '#9', '#10']
Evil probabilities: #2=67%, #7=50%, #8=50%, #4=33%
  Generated 1120 candidate scenarios
  12 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Shaman', 'Lilis', 'Puppeteer'})
    #3 is DEFINITELY EVIL (possible roles: {'Puppet', 'Puppeteer', 'Lilis', 'Shaman'})
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [2, 4, 7, 8]

#### [03:11:16] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 12 scenarios (roles: {'Shaman', 'Lilis', 'Puppeteer'})

### [03:11:52] Executed #1 -> Puppeteer (EVIL)

#### [03:11:52] Solver Output
Scenarios: 4/112
Definite evil: ['#1', '#2', '#3']
Definite good: ['#4', '#5', '#6', '#9', '#10']
Evil probabilities: #7=50%, #8=50%
  Generated 112 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #2 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #3 is DEFINITELY EVIL (possible roles: {'Lilis', 'Shaman'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [7, 8]

#### [03:11:52] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 4 scenarios (roles: {'Puppet'})

### [03:12:29] Executed #2 -> Puppet (EVIL)

#### [03:12:29] Solver Output
Scenarios: 4/56
Definite evil: ['#1', '#2', '#3']
Definite good: ['#4', '#5', '#6', '#9', '#10']
Evil probabilities: #7=50%, #8=50%
  Generated 56 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #2 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #3 is DEFINITELY EVIL (possible roles: {'Shaman', 'Lilis'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [7, 8]

#### [03:12:29] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 4 scenarios (roles: {'Shaman', 'Lilis'})

### [03:13:03] Executed #3 -> Shaman (EVIL)

#### [03:13:03] Solver Output
Scenarios: 2/7
Definite evil: ['#1', '#2', '#3']
Definite good: ['#4', '#5', '#6', '#9', '#10']
Evil probabilities: #7=50%, #8=50%
  Generated 7 candidate scenarios
  2 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #2 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #3 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [7, 8]

#### [03:13:03] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% good FortuneTeller, 50% evil Lilis).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [03:13:44] Executed #7 -> GOOD (WRONG!)

#### [03:13:44] Solver Output
Scenarios: 1/6
Definite evil: ['#1', '#2', '#3', '#8']
Definite good: ['#4', '#5', '#6', '#7', '#9', '#10']
  Generated 6 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #2 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #3 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [03:13:44] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Lilis'})

### [03:14:25] Executed #8 -> Lilis (EVIL)

## [03:14:26] GAME OVER — WIN
Final HP: 1
Notes: 4 evils, Lilis killed #5 #6. Wrong exec on #7 FortuneTeller. Squeaked by at 1HP.


---

# New Game — 2026-03-08 11:20:49
Cards: 7, Evil: 1, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Dreamer, Bard, Bishop, Hunter, Scout, Alchemist
- Outcasts: PlagueDoctor
- Minions: 
- Demons: Pooka

### [11:21:36] Revealed #1 Hunter
Info: {'distance': 3}

### [11:21:40] Revealed #2 Alchemist
Info: {'cured_count': 1}

### [11:21:44] Revealed #3 Bishop
Info: {'targets': [1, 6, 7], 'types': ['Demon', 'Villager', 'Outcast']}

### [11:21:48] Revealed #4 PlagueDoctor
Info: {}

### [11:21:51] Revealed #5 Bard
Info: {'corruption_distance': 1}

### [11:21:55] Revealed #6 Scout
Info: {'evil_role': 'Pooka', 'distance': 3}

### [11:21:58] Revealed #7 Dreamer
Info: {}

#### [11:22:02] Solver Output
Scenarios: 2/31
Definite evil: ['#2']
Definite good: ['#1', '#3', '#4', '#5', '#6', '#7']
  Generated 31 candidate scenarios
  2 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [11:22:02] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [11:22:31] Executed #2 -> Pooka (EVIL)

## [11:22:36] GAME OVER — WIN
Final HP: 10
Notes: perfect 10HP, 1-shot solve. Corrupted: 1=Hunter,3=Bishop,6=Scout


---

# New Game — 2026-03-08 11:23:41
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Architect, Jester, Knitter, Scout, Confessor, Bishop, Empress
- Outcasts: Slayer, Wretch
- Minions: Chancellor
- Demons: Drunk, Pooka

### [11:24:21] Revealed #1 Scout
Info: {'evil_role': 'Chancellor', 'distance': 2}

### [11:24:25] Revealed #2 Jester
Info: {}

### [11:24:29] Revealed #3 Architect
Info: {'side': 'left'}

### [11:24:32] Revealed #4 Slayer
Info: {}

### [11:24:37] Revealed #5 Empress
Info: {'targets': [2, 4, 7]}

### [11:24:41] Revealed #6 Knitter
Info: {'evil_pairs': 2}

### [11:24:46] Revealed #7 Wretch
Info: {}

### [11:24:51] Revealed #8 Bishop
Info: {'targets': [2, 5, 6], 'types': ['Outcast', 'Minion', 'Villager']}

## Deck
- Villagers: 
- Outcasts: 
- Minions: 
- Demons: 


---

# New Game — 2026-03-08 11:28:10
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Architect, Jester, Knitter, Scout, Confessor, Bishop, Empress
- Outcasts: Slayer, Wretch, Drunk
- Minions: Chancellor
- Demons: Pooka

### [11:28:20] Revealed #1 Scout
Info: {'evil_role': 'Chancellor', 'distance': 2}

### [11:28:20] Revealed #2 Jester
Info: {}

### [11:28:20] Revealed #3 Architect
Info: {'side': 'left'}

### [11:28:20] Revealed #4 Slayer
Info: {}

### [11:28:20] Revealed #5 Empress
Info: {'targets': [2, 4, 7]}

### [11:28:20] Revealed #6 Knitter
Info: {'evil_pairs': 2}

### [11:28:21] Revealed #7 Wretch
Info: {}

### [11:28:21] Revealed #8 Bishop
Info: {'targets': [2, 5, 6], 'types': ['Outcast', 'Minion', 'Villager']}

#### [11:28:25] Solver Output
Scenarios: 11/350
Definite good: ['#5']
Evil probabilities: #8=45%, #1=36%, #6=36%, #7=36%, #3=18%, #4=18%, #2=9%
  Generated 350 candidate scenarios
  11 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8]

#### [11:28:25] Recommendation
Action: **USE_ABILITY** #2 (Jester) -> targets ['#1', '#3', '#8']
Reason: Expected posterior 4.8 scenarios (adjusted 5.4, info gain 1.015 bits) | timing x1.00
WARNING: Corruption risk: 27%

### [11:29:18] Ability used at #2

### [11:29:18] Revealed #2 Jester
Info: {'targets': [1, 3, 8], 'evil_count': 1}

#### [11:29:23] Solver Output
Scenarios: 5/350
Definite good: ['#5', '#6']
Evil probabilities: #7=60%, #8=60%, #1=20%, #2=20%, #3=20%, #4=20%
  Generated 350 candidate scenarios
  5 scenarios survived validation
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 7, 8]

#### [11:29:23] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#7']
Reason: Target #7 is 60% evil (adjusted 0.60)

### [11:30:24] Ability used at #4

#### [11:30:52] Solver Output
Scenarios: 4/350
Definite good: ['#2', '#4', '#5', '#6']
Evil probabilities: #7=75%, #8=75%, #1=25%, #3=25%
  Generated 350 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 3, 7, 8]

#### [11:30:52] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (50% evil Pooka, 25% evil Chancellor, 25% good Wretch).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [11:31:14] Executed #7 -> GOOD (WRONG!)

#### [11:31:18] Solver Output
Scenarios: 1/252
Definite evil: ['#1', '#8']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7']
  Generated 252 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [11:31:18] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [11:31:53] Executed #1 -> Pooka (EVIL)

### [11:32:28] Executed #8 -> Chancellor (EVIL)

## [11:32:36] GAME OVER — WIN
Final HP: 5
Notes: 5HP. Slayer killed Wretch (registers evil)=-5HP. Corrupted: 2=Jester,3=Drunk(disguised as Architect). Drunk was in Outcast pool.


---

# New Game — 2026-03-08 11:49:31
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Hunter, Knight, Alchemist, Oracle, Confessor, FortuneTeller
- Outcasts: Doppelganger, Bombardier
- Minions: Shaman
- Demons: Baa

### [11:50:19] Revealed #1 Knight
Info: {}

### [11:50:24] Revealed #2 Confessor
Info: {'dizzy': True}

### [11:50:30] Revealed #3 Oracle
Info: {'targets': [6, 7], 'minion_role': 'Shaman'}

### [11:50:35] Revealed #4 Alchemist
Info: {'cured_count': 0}

### [11:50:41] Revealed #5 Oracle
Info: {'targets': [5, 6], 'minion_role': 'Shaman'}

### [11:50:47] Revealed #6 Confessor
Info: {'dizzy': True}

### [11:50:52] Revealed #7 Hunter
Info: {'distance': 1}

### [11:50:58] Revealed #8 Alchemist
Info: {'cured_count': 0}

#### [11:51:03] Solver Output
Scenarios: 6/392
Definite evil: ['#2', '#6']
Definite good: ['#1', '#3', '#4', '#5', '#7', '#8']
  Generated 392 candidate scenarios
  6 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Baa'})
    #6 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [11:51:03] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 6 scenarios (roles: {'Baa'})

### [11:51:38] Executed #2 -> Baa (EVIL)

### [11:52:12] Executed #6 -> Shaman (EVIL)

## [11:52:12] GAME OVER — WIN
Final HP: 10
Notes: perfect 10HP, 1-shot solve. Doppelganger#4 disguised as Alchemist.


---

# New Game — 2026-03-08 11:53:40
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Slayer, Architect, FortuneTeller, Confessor, Medium, Alchemist, Knitter, Bishop
- Outcasts: PlagueDoctor
- Minions: Shaman, Minion
- Demons: Pooka

### [11:54:23] Revealed #1 PlagueDoctor
Info: {}

### [11:54:24] Revealed #2 Architect
Info: {'side': 'right'}

### [11:54:24] Revealed #3 Knitter
Info: {'evil_pairs': 0}

### [11:54:24] Revealed #4 Confessor
Info: {'dizzy': True}

### [11:54:24] Revealed #5 Bishop
Info: {'targets': [2, 7, 9], 'types': ['Minion', 'Villager', 'Outcast']}

### [11:54:24] Revealed #6 FortuneTeller
Info: {}

### [11:54:24] Revealed #7 Architect
Info: {'side': 'left'}

### [11:54:24] Revealed #8 Alchemist
Info: {'cured_count': 2}

### [11:54:24] Revealed #9 Slayer
Info: {}

#### [11:54:30] Solver Output
Scenarios: 20/1848
Definite good: ['#1', '#9']
Evil probabilities: #3=60%, #4=60%, #8=60%, #5=50%, #2=30%, #6=20%, #7=20%
  Generated 1848 candidate scenarios
  20 scenarios survived validation
    #1 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6, 7, 8]

#### [11:54:30] Recommendation
Action: **USE_ABILITY** #9 (Slayer) -> targets ['#3']
Reason: Target #3 is 60% evil (adjusted 0.48)
WARNING: Corruption risk: 20% -- Slayer ability disabled if corrupted

#### [11:55:27] Solver Output
Scenarios: 10/1848
Definite evil: ['#8']
Definite good: ['#1', '#6', '#7', '#9']
Evil probabilities: #5=80%, #2=60%, #4=40%, #3=20%
  Generated 1848 candidate scenarios
  10 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Pooka', 'Minion', 'Shaman'})
    #1 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5]

#### [11:55:27] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 10 scenarios (roles: {'Pooka', 'Minion', 'Shaman'})

### [11:56:02] Executed #8 -> Minion (EVIL)

#### [11:56:02] Solver Output
Scenarios: 4/224
Definite evil: ['#5', '#8']
Definite good: ['#1', '#3', '#6', '#7', '#9']
Evil probabilities: #2=75%, #4=25%
  Generated 224 candidate scenarios
  4 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka', 'Shaman'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 4]

#### [11:56:02] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 4 scenarios (roles: {'Pooka', 'Shaman'})

### [11:56:39] Executed #5 -> Shaman (EVIL)

#### [11:56:39] Solver Output
Scenarios: 1/31
Definite evil: ['#4', '#5', '#8']
Definite good: ['#1', '#2', '#3', '#6', '#7', '#9']
  Generated 31 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #5 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [11:56:39] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [11:57:13] Executed #4 -> Pooka (EVIL)

## [11:57:13] GAME OVER — WIN
Final HP: 10
Notes: perfect 10HP. Corrupted: 2=Architect,3=Knitter. Slayer failed on #3 (good but corrupted Knitter).


---

# New Game — 2026-03-08 11:58:09
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Poet, Bishop, Dreamer, Oracle, Baker, Bard, FortuneTeller
- Outcasts: Doppelganger
- Minions: Minion
- Demons: Pooka

### [11:58:53] Revealed #1 Dreamer
Info: {}

### [11:58:53] Revealed #2 FortuneTeller
Info: {}

### [11:58:53] Revealed #3 Oracle
Info: {'targets': [4, 8], 'minion_role': 'Minion'}

### [11:58:54] Revealed #4 Bard
Info: {'corruption_distance': 1}

### [11:58:54] Revealed #5 Baker
Info: {'original_role': 'FortuneTeller'}

### [11:58:54] Revealed #7 Bishop
Info: {'targets': [1, 4, 7], 'types': ['Outcast', 'Minion', 'Villager']}

### [11:58:54] Revealed #8 FortuneTeller
Info: {}

### [11:59:03] Revealed #6 Poet
Info: {}

#### [11:59:09] Solver Output
Scenarios: 10/336
Definite good: ['#1', '#2', '#7']
Evil probabilities: #6=80%, #3=40%, #8=40%, #4=20%, #5=20%
  Generated 336 candidate scenarios
  10 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [3, 4, 5, 6, 8]

#### [11:59:09] Recommendation
Action: **USE_ABILITY** #1 (Dreamer) -> targets ['#3']
Reason: Entropy 0.971 (adjusted 0.971) | timing x1.00

### [12:00:08] Ability used at #1

### [12:00:08] Revealed #1 Dreamer
Info: {'target': 3, 'evil_role': 'Minion'}

#### [12:00:14] Solver Output
Scenarios: 10/336
Definite good: ['#1', '#2', '#7']
Evil probabilities: #6=80%, #3=40%, #8=40%, #4=20%, #5=20%
  Generated 336 candidate scenarios
  10 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [3, 4, 5, 6, 8]

#### [12:00:14] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (80% evil Pooka, 20% good Poet (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 80%, but all reveal branches still lead to a forced win.

### [12:00:55] Executed #6 -> Pooka (EVIL)

#### [12:00:55] Solver Output
Scenarios: 8/42
Definite evil: ['#6']
Definite good: ['#1', '#2', '#4', '#5', '#7']
Evil probabilities: #3=50%, #8=50%
  Generated 42 candidate scenarios
  8 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [3, 8]

#### [12:00:55] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% evil Minion, 38% good Oracle, 12% good Doppelganger).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [12:01:34] Executed #3 -> Minion (EVIL)

## [12:01:34] GAME OVER — WIN
Final HP: 10
Notes: perfect 10HP. Dreamer correctly identified #3 as Minion. Doppelganger#2. Corrupted: 5=Baker,7=Bishop.


---

# New Game — 2026-03-08 12:58:23
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Alchemist, Hunter, Oracle, Scout, Baker, Medium, Poet
- Outcasts: Doppelganger
- Minions: TwinMinion, Poisoner
- Demons: Lilis

### [13:01:22] Revealed #1 Scout
Info: {'evil_role': 'Lilis', 'distance': 3}

### [13:01:25] Revealed #2 Oracle
Info: {'targets': [7, 8], 'minion_role': 'Poisoner'}

### [13:01:28] Revealed #3 Scout
Info: {'evil_role': 'Lilis', 'distance': 1}

### [13:01:31] Revealed #4 Poet
Info: {'evil_pairs': 0, 'copied_role': 'Knitter'}

### [13:01:34] Revealed #6 Hunter
Info: {'distance': 1}

### [13:01:37] Revealed #7 Baker
Info: {'original_role': 'Poet'}

### [13:01:40] Revealed #8 Alchemist
Info: {'cured_count': 1}

### [13:01:45] Revealed #10 Medium
Info: {'good_position': 3, 'good_role': 'Scout'}

#### [13:01:48] Solver Output
Scenarios: 16/5496
Definite evil: ['#3', '#7', '#10']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#8', '#9']
  Generated 5496 candidate scenarios
  16 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis', 'TwinMinion'})
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #10 is DEFINITELY EVIL (possible roles: {'Lilis', 'TwinMinion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [13:01:48] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 16 scenarios (roles: {'Lilis', 'TwinMinion'})

### [13:02:24] Executed #3 -> TwinMinion (EVIL)

#### [13:02:29] Solver Output
Scenarios: 8/530
Definite evil: ['#3', '#7', '#10']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#8', '#9']
  Generated 530 candidate scenarios
  8 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'TwinMinion'})
    #7 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #10 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [13:02:29] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 8 scenarios (roles: {'Poisoner'})

### [13:02:54] Executed #7 -> Poisoner (EVIL)

### [13:03:27] Executed #10 -> Lilis (EVIL)

## [13:03:32] GAME OVER — WIN
Final HP: 6
Notes: Perfect deduction, Lilis game with 2 night kills


---

# New Game — 2026-03-08 13:04:50
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Enlightened, Judge, Oracle, Bishop, Druid, Bard
- Outcasts: Bombardier
- Minions: Poisoner
- Demons: Lilis

### [13:07:02] Revealed #1 Bard
Info: {'corruption_distance': 2}

### [13:07:02] Revealed #2 Oracle
Info: {'targets': [4, 8], 'minion_role': 'Poisoner'}

### [13:07:02] Revealed #3 Bombardier
Info: {}

### [13:07:02] Revealed #4 Judge
Info: {}

### [13:07:02] Revealed #5 Bishop
Info: {'targets': [3, 7, 8], 'types': ['Outcast', 'Minion', 'Villager']}

### [13:07:02] Revealed #6 Druid
Info: {}

### [13:07:03] Revealed #8 Oracle
Info: {'targets': [2, 3], 'minion_role': 'Poisoner'}

#### [13:07:08] Solver Output
Scenarios: 10/84
Definite good: ['#7']
Evil probabilities: #8=80%, #5=30%, #2=20%, #3=20%, #4=20%, #6=20%, #1=10%
  Generated 84 candidate scenarios
  10 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8]

#### [13:07:08] Recommendation
Action: **USE_ABILITY** #6 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 1.371 (adjusted 1.371) | timing x1.00

### [13:07:52] Ability used at #6

### [13:07:52] Revealed #6 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

#### [13:07:57] Solver Output
Scenarios: 4/84
Definite good: ['#1', '#4', '#5', '#7']
Evil probabilities: #8=75%, #3=50%, #6=50%, #2=25%
  Generated 84 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [2, 3, 6, 8]

#### [13:07:57] Recommendation
Action: **USE_ABILITY** #4 (Judge) -> targets ['#3']
Reason: Expected posterior 2.0 scenarios (adjusted 2.0, info gain 1.000 bits) | timing x1.00

### [13:08:41] Ability used at #4

### [13:08:41] Revealed #4 Judge
Info: {'target': 3, 'is_lying': False}

#### [13:08:46] Solver Output
Scenarios: 2/84
Definite evil: ['#6', '#8']
Definite good: ['#1', '#2', '#3', '#4', '#5', '#7']
  Generated 84 candidate scenarios
  2 scenarios survived validation
    #6 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [13:08:46] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 2 scenarios (roles: {'Lilis'})

### [13:09:24] Executed #6 -> Lilis (EVIL)

### [13:10:17] Executed #8 -> Poisoner (EVIL)

## [13:10:17] GAME OVER — WIN
Final HP: 8
Notes: Bard#1 corrupted. Druid lied (was Lilis). 8-card Lilis game, 1 night kill.


---

# New Game — 2026-03-08 13:11:15
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Empress, Alchemist, Knitter, Jester, Medium
- Outcasts: Wretch
- Minions: 
- Demons: Pooka

### [13:12:13] Revealed #1 Knitter
Info: {'evil_pairs': 0}

### [13:12:13] Revealed #2 Empress
Info: {'targets': [1, 4, 6]}

### [13:12:13] Revealed #3 Alchemist
Info: {'cured_count': 1}

### [13:12:13] Revealed #4 Medium
Info: {'good_position': 3, 'good_role': 'Alchemist'}

### [13:12:13] Revealed #5 Wretch
Info: {}

### [13:12:13] Revealed #6 Jester
Info: {}

#### [13:12:19] Solver Output
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

#### [13:12:19] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [13:13:07] Executed #3 -> Pooka (EVIL)

## [13:13:07] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP. Empress#2 and Medium#4 corrupted. 1-scenario solve.


---

# New Game — 2026-03-08 13:14:12
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Druid, Confessor, Slayer, Hunter, Knight, Baker
- Outcasts: Wretch, PlagueDoctor
- Minions: TwinMinion
- Demons: Pooka

### [13:15:38] Revealed #1 Confessor
Info: {'dizzy': False}

### [13:15:38] Revealed #2 Hunter
Info: {'distance': 2}

### [13:15:38] Revealed #3 Slayer
Info: {}

### [13:15:38] Revealed #4 Baker
Info: {'original_role': 'Hunter'}

### [13:15:38] Revealed #5 Baker
Info: {'original_role': 'Confessor'}

### [13:15:38] Revealed #6 Knight
Info: {}

### [13:15:38] Revealed #7 Druid
Info: {}

### [13:15:39] Revealed #8 Wretch
Info: {}

### [13:15:39] Revealed #9 PlagueDoctor
Info: {}

#### [13:15:45] Solver Output
Scenarios: 60/310
Definite good: ['#1', '#8', '#9']
Evil probabilities: #3=40%, #4=38%, #5=32%, #6=32%, #7=32%, #2=27%
  Generated 310 candidate scenarios
  60 scenarios survived validation
    #1 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6, 7]

#### [13:15:45] Recommendation
Action: **USE_ABILITY** #7 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.971 (adjusted 0.833) | timing x1.00
WARNING: Corruption risk: 28%

### [13:16:30] Ability used at #7

### [13:16:30] Revealed #7 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

#### [13:16:37] Solver Output
Scenarios: 24/310
Definite good: ['#1', '#7', '#8', '#9']
Evil probabilities: #3=58%, #5=50%, #4=42%, #2=25%, #6=25%
  Generated 310 candidate scenarios
  24 scenarios survived validation
    #1 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6]

#### [13:16:37] Recommendation
Action: **USE_ABILITY** #3 (Slayer) -> targets ['#5']
Reason: Target #5 is 50% evil (adjusted 0.38)
WARNING: Corruption risk: 25% -- Slayer ability disabled if corrupted

#### [13:17:28] Solver Output
Scenarios: 20/310
Definite good: ['#1', '#7', '#8', '#9']
Evil probabilities: #3=70%, #4=45%, #5=40%, #6=25%, #2=20%
  Generated 310 candidate scenarios
  20 scenarios survived validation
    #1 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6]

#### [13:17:28] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 70% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 70% confident (budget: 2 wrong execs)

### [13:18:15] Executed #3 -> GOOD (WRONG!)

#### [13:18:22] Solver Output
Scenarios: 6/236
Definite good: ['#1', '#3', '#7', '#8', '#9']
Evil probabilities: #4=83%, #5=67%, #6=33%, #2=17%
  Generated 236 candidate scenarios
  6 scenarios survived validation
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 4, 5, 6]

#### [13:18:22] Recommendation
Action: **EXECUTE** #4
Reason: No reveals available. #4 is 83% likely evil (HP=5, budget=1 wrong execs)
WARNING: Probabilistic execution -- 83% confident (budget: 1 wrong execs)

### [13:19:07] Executed #4 -> Pooka (EVIL)

#### [13:19:13] Solver Output
Scenarios: 4/32
Definite evil: ['#4']
Definite good: ['#1', '#2', '#3', '#7', '#8', '#9']
Evil probabilities: #5=50%, #6=50%
  Generated 32 candidate scenarios
  4 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [5, 6]

#### [13:19:13] Recommendation
Action: **ERROR** #5
Reason: #5 is 50% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 50% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 50% < 80% threshold. Consider manual override if you have extra information.

### [13:20:43] Ability used at #9

#### [13:20:50] Solver Output
Scenarios: 2/32
Definite evil: ['#4', '#5']
Definite good: ['#1', '#2', '#3', '#6', '#7', '#8', '#9']
  Generated 32 candidate scenarios
  2 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #5 is DEFINITELY EVIL (possible roles: {'TwinMinion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [13:20:50] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 2 scenarios (roles: {'TwinMinion'})

### [13:21:39] Executed #5 -> TwinMinion (EVIL)

## [13:21:39] GAME OVER — WIN
Final HP: 5
Notes: Slayer#3 and Knight#6 corrupted. PD broke 50/50 tie. 1 wrong exec (corrupted Slayer). HP=5.


---

# New Game — 2026-03-08 13:22:49
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Medium, Gemcrafter, Jester, Scout, Bard, Poet
- Outcasts: Bombardier, Wretch, Doppelganger
- Minions: Chancellor
- Demons: Pooka

### [13:23:56] Revealed #1 Scout
Info: {'evil_role': 'Pooka', 'distance': 3}

### [13:23:56] Revealed #2 Bombardier
Info: {}

### [13:23:56] Revealed #3 Bard
Info: {'corruption_distance': 4}

### [13:23:56] Revealed #4 Scout
Info: {'evil_role': 'Pooka', 'distance': 1}

### [13:23:56] Revealed #5 Wretch
Info: {}

### [13:23:56] Revealed #6 Medium
Info: {'good_position': 7, 'good_role': 'Doppelganger'}

### [13:23:57] Revealed #7 Medium
Info: {'good_position': 8, 'good_role': 'Gemcrafter'}

### [13:23:57] Revealed #8 Gemcrafter
Info: {'good_position': 5}

### [13:23:57] Revealed #9 Poet
Info: {'distance': 3, 'copied_role': 'Hunter'}

#### [13:24:03] Solver Output
Scenarios: 1/464
Definite evil: ['#1', '#9']
Definite good: ['#2', '#3', '#4', '#5', '#6', '#7', '#8']
  Generated 464 candidate scenarios
  1 scenarios survived validation
    #1 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #9 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [13:24:03] Recommendation
Action: **EXECUTE** #1
Reason: #1 is evil in ALL 1 scenarios (roles: {'Chancellor'})

### [13:24:50] Executed #1 -> Chancellor (EVIL)

### [13:25:36] Executed #9 -> Pooka (EVIL)

## [13:25:36] GAME OVER — WIN
Final HP: 10
Notes: Perfect 10HP, 1-scenario solve. Gemcrafter#8 corrupted.


---

# New Game — 2026-03-08 13:27:12
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Bishop, Enlightened, Slayer, Baker, Poet, FortuneTeller, Knitter
- Outcasts: PlagueDoctor, Bombardier
- Minions: Minion, Witch
- Demons: Pooka

### [13:31:04] Revealed #1 Knitter
Info: {'evil_pairs': 1}

### [13:31:04] Revealed #2 Poet
Info: {'direction': 'CW', 'copied_role': 'Enlightened'}

### [13:31:04] Revealed #3 FortuneTeller
Info: {}

### [13:31:04] Revealed #4 Bombardier
Info: {}

### [13:31:04] Revealed #5 PlagueDoctor
Info: {}

### [13:31:04] Revealed #6 Bishop
Info: {'targets': [2, 9, 10], 'types': ['Minion', 'Villager', 'Outcast']}

### [13:31:04] Revealed #7 FortuneTeller
Info: {}

### [13:31:05] Revealed #8 Enlightened
Info: {'direction': 'CW'}

### [13:31:05] Revealed #9 Slayer
Info: {}

### [13:31:22] Revealed #8 Enlightened
Info: {'direction': 'EQUAL'}

#### [13:31:29] Solver Output
Scenarios: 60/2904
Definite good: ['#2', '#10']
Evil probabilities: #4=93%, #7=70%, #3=43%, #8=33%, #6=27%, #9=20%, #5=10%, #1=3%
  Generated 2904 candidate scenarios
  60 scenarios survived validation
    #2 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8, 9]

#### [13:31:29] Recommendation
Action: **USE_ABILITY** #9 (Slayer) -> targets ['#7']
Reason: Target #7 is 70% evil (adjusted 0.58)
WARNING: Corruption risk: 17% -- Slayer ability disabled if corrupted

#### [13:32:35] Solver Output
Scenarios: 0/1974
  Generated 1974 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #1 Knitter: rejected 942/1974 (48%)
    #2 Poet: rejected 926/1974 (47%)
    #8 Enlightened: rejected 860/1974 (44%)
    #6 Bishop: rejected 552/1974 (28%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Knitter: 104 scenarios survive  <-- SUSPECT
    WITHOUT #2 Poet: 152 scenarios survive  <-- SUSPECT
    WITHOUT #6 Bishop: 116 scenarios survive  <-- SUSPECT
    WITHOUT #8 Enlightened: 154 scenarios survive  <-- SUSPECT

#### [13:32:35] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [13:32:57] Executed #7 -> Witch (EVIL)

#### [13:33:05] Solver Output
Scenarios: 4/310
Definite evil: ['#7']
Definite good: ['#1', '#2', '#3', '#9', '#10']
Evil probabilities: #4=75%, #6=50%, #8=50%, #5=25%
  Generated 310 candidate scenarios
  4 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 8]

#### [13:33:05] Recommendation
Action: **REVEAL** #10
Reason: #10: 0% evil, entropy 0.000

### [13:34:01] Ability used at #3

### [13:34:01] Revealed #3 Fortune Teller
Info: {'targets': [4, 6], 'has_evil': True}

#### [13:34:12] Solver Output
Scenarios: 1/310
Definite evil: ['#4', '#6', '#7']
Definite good: ['#1', '#2', '#3', '#5', '#8', '#9', '#10']
  Generated 310 candidate scenarios
  1 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Minion'})
    #6 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #7 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [13:34:12] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 1 scenarios (roles: {'Minion'})

### [13:34:51] Executed #4 -> Minion (EVIL)

## [13:37:08] GAME OVER — WIN
Final HP: 10
Notes: 10 cards, 3 evils. Blocked card at #10 (new Asc13 mechanic). Slayer at #9 killed Witch at #7. Enlightened #8 was corrupted. HP 10/10.


---

# New Game — 2026-03-08 13:38:17
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Lover, Medium, Slayer, Scout, Judge, Dreamer, Bishop
- Outcasts: Doppelganger, Drunk
- Minions: Chancellor
- Demons: Pooka

### [13:39:24] Revealed #1 Dreamer
Info: {}

### [13:39:24] Revealed #2 Judge
Info: {}

### [13:39:28] Revealed #3 Scout
Info: {'evil_role': 'Chancellor', 'distance': 1}

### [13:39:29] Revealed #4 Slayer
Info: {}

### [13:39:29] Revealed #5 Medium
Info: {'good_position': 8, 'good_role': 'Drunk'}

### [13:39:33] Revealed #6 Bishop
Info: {'targets': [1, 3, 4], 'types': ['Villager', 'Minion', 'Outcast']}

### [13:39:33] Revealed #7 Medium
Info: {'good_position': 4, 'good_role': 'Slayer'}

### [13:39:34] Revealed #8 Lover
Info: {'evil_adjacent': 2}

#### [13:39:38] Solver Output
Scenarios: 22/2408
Definite good: ['#4']
Evil probabilities: #6=59%, #8=41%, #3=27%, #2=23%, #5=23%, #7=18%, #1=9%
  Generated 2408 candidate scenarios
  22 scenarios survived validation
    #4 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 6, 7, 8]

#### [13:39:38] Recommendation
Action: **USE_ABILITY** #1 (Dreamer) -> targets ['#6']
Reason: Entropy 2.140 (adjusted 1.946) | timing x1.00
WARNING: Corruption risk: 18%

### [13:40:22] Ability used at #1

### [13:40:26] Revealed #1 Dreamer
Info: {'target': 6, 'evil_role': 'Pooka'}

#### [13:40:30] Solver Output
Scenarios: 14/2408
Definite good: ['#4']
Evil probabilities: #8=64%, #5=36%, #6=36%, #2=29%, #1=14%, #7=14%, #3=7%
  Generated 2408 candidate scenarios
  14 scenarios survived validation
    #4 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 6, 7, 8]

#### [13:40:30] Recommendation
Action: **USE_ABILITY** #2 (Judge) -> targets ['#4']
Reason: Expected posterior 7.1 scenarios (adjusted 7.1, info gain 0.971 bits) | timing x1.00

### [13:41:16] Ability used at #2

### [13:41:20] Revealed #2 Judge
Info: {'target': 4, 'is_lying': True}

#### [13:41:26] Solver Output
Scenarios: 8/2408
Definite good: ['#3', '#4', '#7']
Evil probabilities: #2=50%, #5=50%, #8=50%, #6=38%, #1=12%
  Generated 2408 candidate scenarios
  8 scenarios survived validation
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 6, 8]

#### [13:41:26] Recommendation
Action: **USE_ABILITY** #4 (Slayer) -> targets ['#2']
Reason: Target #2 is 50% evil (adjusted 0.25)
WARNING: Corruption risk: 50% -- Slayer ability disabled if corrupted

### [13:42:19] Ability used at #4

### [13:42:25] Executed #2 -> Pooka (EVIL)

#### [13:42:29] Solver Output
Scenarios: 4/301
Definite evil: ['#2']
Definite good: ['#3', '#4', '#5', '#7', '#8']
Evil probabilities: #6=75%, #1=25%
  Generated 301 candidate scenarios
  4 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 6]

#### [13:42:29] Recommendation
Action: **EXECUTE** #6
Reason: Execution lookahead: #6 guarantees a win across all reveal branches with current HP budget (75% evil Chancellor, 25% good Bishop).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [13:42:58] Executed #6 -> Chancellor (EVIL)

## [13:43:06] GAME OVER — WIN
Final HP: 10
Notes: 8 cards, 2 evils. Slayer #4 killed Pooka #2. Judge #2 was Pooka (lying Judge). Dreamer #1, Scout #3, Drunk #8 corrupted by Pooka. HP 10/10 perfect. Ascension 13 complete — 7/7 wins!


---

# New Game — 2026-03-08 14:31:04
Cards: 10, Evil: 4, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Architect, Bishop, Scout, Confessor, Bard, Knitter, Empress
- Outcasts: Knight, PlageDoctor
- Minions: Puppeteer, Minion
- Demons: Pooka

## Deck
- Villagers: Architect, Bishop, Scout, Confessor, Bard, Knitter, Empress
- Outcasts: Knight, PlagueDoctor
- Minions: Puppeteer, Minion
- Demons: Pooka

### [14:32:38] Revealed #1 Scout
Info: {'evil_role': 'Minion', 'distance': 2}

### [14:32:38] Revealed #2 Architect
Info: {'side': 'CW'}

### [14:32:43] Revealed #3 Confessor
Info: {'dizzy': True}

### [14:32:43] Revealed #4 Knight
Info: {}

### [14:32:43] Revealed #5 Architect
Info: {'side': 'CW'}

### [14:32:48] Revealed #6 Bishop
Info: {'targets': [8, 9, 10], 'types': ['Villager', 'Minion', 'Outcast']}

### [14:32:48] Revealed #7 PlagueDoctor
Info: {}

### [14:32:49] Revealed #8 Bard
Info: {'corruption_distance': 2}

### [14:32:54] Revealed #9 Knitter
Info: {'evil_pairs': 2}

### [14:32:54] Revealed #10 Empress
Info: {'targets': [4, 6, 7]}

#### [14:33:00] Solver Output
Scenarios: 9/3808
Definite evil: ['#4']
Definite good: ['#7']
Evil probabilities: #2=89%, #1=44%, #5=44%, #9=44%, #6=33%, #10=22%, #3=11%, #8=11%
  Generated 3808 candidate scenarios
  9 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Puppet', 'Pooka'})
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 5, 6, 8, 9, 10]

#### [14:33:00] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 9 scenarios (roles: {'Puppet', 'Pooka'})

### [14:34:10] Executed #4 -> GOOD (WRONG!)

#### [14:34:16] Solver Output
Scenarios: 0/2112
  Generated 2112 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #5 Architect: rejected 1038/2112 (49%)
    #1 Scout: rejected 1022/2112 (48%)
    #10 Empress: rejected 1000/2112 (47%)
    #9 Knitter: rejected 968/2112 (46%)
    #6 Bishop: rejected 934/2112 (44%)
    #2 Architect: rejected 886/2112 (42%)
    #8 Bard: rejected 884/2112 (42%)
    #3 Confessor: rejected 864/2112 (41%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Scout: still 0
    WITHOUT #2 Architect: still 0
    WITHOUT #3 Confessor: still 0
    WITHOUT #5 Architect: still 0
    WITHOUT #6 Bishop: still 0
    WITHOUT #8 Bard: still 0
    WITHOUT #9 Knitter: still 0
    WITHOUT #10 Empress: still 0

#### [14:34:16] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [14:36:59] Ability used at #7

### [14:37:55] Executed #2 -> Puppeteer (EVIL)

#### [14:38:00] Solver Output
Scenarios: 0/324
  Generated 324 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #1 Scout: rejected 190/324 (59%)
    #5 Architect: rejected 146/324 (45%)
    #9 Knitter: rejected 136/324 (42%)
    #6 Bishop: rejected 136/324 (42%)
    #10 Empress: rejected 119/324 (37%)
    #8 Bard: rejected 118/324 (36%)
    #3 Confessor: rejected 90/324 (28%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Scout: still 0
    WITHOUT #2 Architect: still 0
    WITHOUT #3 Confessor: still 0
    WITHOUT #5 Architect: still 0
    WITHOUT #6 Bishop: still 0
    WITHOUT #8 Bard: still 0
    WITHOUT #9 Knitter: still 0
    WITHOUT #10 Empress: still 0

#### [14:38:00] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [14:42:14] Executed #6 -> GOOD (WRONG!)

## [14:42:27] GAME OVER — LOSS
Final HP: 0
Notes: 10 cards, 4 evils. SOLVER BUG: all 9 scenarios had #4 as DEFINITELY evil but #4 was good corrupted Knight. Wrong exec cost 9 HP (new Asc14 mechanic?). Corrupted: #4 Knight, #6 Bishop, #10 Empress. Pooka at #5 corrupted adjacent #4 and #6. Heart system: lose 1 heart per village loss.


---

# New Game — 2026-03-08 14:54:29
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Confessor, Knight, Alchemist, Hunter, Medium, Druid
- Outcasts: Doppelganger, Bombardier
- Minions: Shaman
- Demons: Baa

### [14:55:29] Revealed #1 Alchemist
Info: {'cured_count': 0}

### [14:55:30] Revealed #2 Knight
Info: {}

### [14:55:30] Revealed #3 Alchemist
Info: {'cured_count': 0}

### [14:55:36] Revealed #4 Alchemist
Info: {'cured_count': 0}

### [14:55:37] Revealed #5 Medium
Info: {'good_position': 7, 'good_role': 'Confessor'}

### [14:55:37] Revealed #6 Druid
Info: {}

### [14:55:38] Revealed #7 Confessor
Info: {'dizzy': True}

#### [14:55:45] Solver Output
Scenarios: 10/252
Definite evil: ['#5', '#7']
Definite good: ['#1', '#2', '#3', '#4', '#6']
  Generated 252 candidate scenarios
  10 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Shaman', 'Baa'})
    #7 is DEFINITELY EVIL (possible roles: {'Shaman', 'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [14:55:45] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 10 scenarios (roles: {'Shaman', 'Baa'})

### [14:56:32] Executed #5 -> Shaman (EVIL)

### [14:57:14] Executed #7 -> Baa (EVIL)

## [14:57:26] GAME OVER — WIN
Final HP: 10
Notes: 7 cards, 2 evils. 3 Alchemists narrowed to 2 fakes. Solver found both evils at #5 and #7 definitively. HP 10/10 perfect.


---

# New Game — 2026-03-08 14:58:51
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 5

## Deck
- Villagers: Scout, Architect, Alchemist, Lover, Hunter, Enlightened, Confessor, Jester, Gemcrafter
- Outcasts: Drunk
- Minions: Minion, Witch
- Demons: Lilis

### [15:00:46] Revealed #1 Jester
Info: {}

### [15:00:46] Revealed #2 Hunter
Info: {'distance': 1}

### [15:00:46] Revealed #3 Scout
Info: {'evil_role': 'Minion', 'distance': 3}

### [15:00:46] Revealed #4 Enlightened
Info: {'direction': 'CW'}

### [15:00:47] Revealed #6 Gemcrafter
Info: {'good_position': 9}

### [15:00:47] Revealed #8 Lover
Info: {'evil_adjacent': 2}

### [15:00:47] Revealed #9 Confessor
Info: {'dizzy': False}

### [15:00:47] Revealed #10 Architect
Info: {'side': 'CCW'}

#### [15:00:55] Solver Output
Scenarios: 32/5040
Definite good: ['#6', '#9']
Evil probabilities: #4=56%, #8=44%, #1=38%, #3=31%, #2=25%, #10=6%
  Generated 5040 candidate scenarios
  32 scenarios survived validation
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 7, 8, 10]

#### [15:00:55] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#4', '#6', '#9']
Reason: Expected posterior 14.4 scenarios (adjusted 14.4, info gain 1.149 bits) | timing x1.00

### [15:02:05] Ability used at #1

### [15:02:05] Revealed #1 Jester
Info: {'targets': [4, 6, 9], 'evil_count': 0}

#### [15:02:12] Solver Output
Scenarios: 14/5040
Definite good: ['#3', '#6', '#9']
Evil probabilities: #8=57%, #1=43%, #2=43%, #4=43%, #10=14%
  Generated 5040 candidate scenarios
  14 scenarios survived validation
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 5, 7, 8, 10]

#### [15:02:12] Recommendation
Action: **EXECUTE** #8
Reason: Execution lookahead: #8 guarantees a win across all reveal branches with current HP budget (43% good Drunk (corrupted), 21% evil Lilis, 21% evil Witch).
WARNING: Execution lookahead override -- immediate hit chance is 57%, but all reveal branches still lead to a forced win.

### [15:03:01] Executed #8 -> Lilis (EVIL)

#### [15:03:08] Solver Output
Scenarios: 3/504
Definite evil: ['#5', '#8']
Definite good: ['#1', '#3', '#4', '#6', '#7', '#9']
Evil probabilities: #2=67%, #10=33%
  Generated 504 candidate scenarios
  3 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Witch', 'Minion'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 10]

#### [15:03:08] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (33% good Drunk (corrupted), 33% evil Minion, 33% evil Witch).
WARNING: Execution lookahead override -- immediate hit chance is 67%, but all reveal branches still lead to a forced win.

### [15:03:58] Executed #2 -> GOOD (WRONG!)

#### [15:04:05] Solver Output
Scenarios: 1/392
Definite evil: ['#5', '#8', '#10']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#7', '#9']
  Generated 392 candidate scenarios
  1 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #10 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [15:04:05] Recommendation
Action: **EXECUTE** #10
Reason: #10 is evil in ALL 1 scenarios (roles: {'Witch'})

### [15:06:03] Executed #10 -> Witch (EVIL)

## [15:06:08] GAME OVER — WIN
Final HP: 4
Notes: 10 cards, 3 evils. Lilis game with 2 night kills (#5 Minion, #7 unknown). Wrong exec #2 Drunk Corrupted (-2 HP). Final HP 4/10. Corrupted: #2 Drunk.


---

# New Game — 2026-03-08 15:06:29
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Medium, Oracle, Baker, Scout, Architect, Gemcrafter
- Outcasts: Doppelganger, Wretch, PlagueDoctor
- Minions: Chancellor
- Demons: Pooka

### [15:10:21] Revealed #1 Architect
Info: {'side': 'Left'}

### [15:10:21] Revealed #2 Baker
Info: {'original_role': 'Oracle'}

### [15:10:21] Revealed #3 Wretch
Info: {}

### [15:10:22] Revealed #4 Oracle
Info: {'targets': [7, 9], 'minion_role': 'Chancellor'}

### [15:10:22] Revealed #5 PlagueDoctor
Info: {}

### [15:10:22] Revealed #6 Medium
Info: {'good_position': 5, 'good_role': 'PlagueDoctor'}

### [15:10:22] Revealed #7 Architect
Info: {'side': 'Left'}

### [15:10:22] Revealed #8 Medium
Info: {'good_position': 6, 'good_role': 'Medium'}

### [15:10:22] Revealed #9 Gemcrafter
Info: {'good_position': 4}

#### [15:10:26] Solver Output
Scenarios: 16/1962
Definite good: ['#5', '#6', '#9']
Evil probabilities: #1=56%, #7=50%, #2=31%, #3=31%, #4=25%, #8=6%
  Generated 1962 candidate scenarios
  16 scenarios survived validation
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 7, 8]

#### [15:10:26] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (38% good Architect (corrupted), 31% evil Chancellor, 25% evil Pooka).
WARNING: Execution lookahead override -- immediate hit chance is 56%, but all reveal branches still lead to a forced win.

### [15:10:54] Executed #1 -> GOOD (WRONG!)

#### [15:10:58] Solver Output
Scenarios: 7/1504
Definite good: ['#1', '#5', '#6', '#9']
Evil probabilities: #7=86%, #2=71%, #3=14%, #4=14%, #8=14%
  Generated 1504 candidate scenarios
  7 scenarios survived validation
    #1 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 7, 8]

#### [15:10:58] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (86% evil Chancellor, 14% good Architect (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 86%, but all reveal branches still lead to a forced win.

### [15:11:25] Executed #7 -> Chancellor (EVIL)

#### [15:11:25] Solver Output
Scenarios: 6/199
Definite evil: ['#7']
Definite good: ['#1', '#4', '#5', '#6', '#8', '#9']
Evil probabilities: #2=83%, #3=17%
  Generated 199 candidate scenarios
  6 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [2, 3]

#### [15:11:25] Recommendation
Action: **EXECUTE** #2
Reason: Execution lookahead: #2 guarantees a win across all reveal branches with current HP budget (83% evil Pooka, 17% good Baker (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 83%, but all reveal branches still lead to a forced win.

### [15:11:58] Executed #2 -> Pooka (EVIL)

## [15:12:06] GAME OVER — WIN
Final HP: 5
Notes: 9 cards, 2 evils. Wrong exec #1 Architect Corrupted (-5 HP). Corrupted: #1 Architect, #8 Medium. #6 was Doppelganger. Final HP 5/10.


---

# New Game — 2026-03-08 15:12:40
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Bishop, Knitter, Architect, Hunter, Dreamer, Gemcrafter
- Outcasts: PlagueDoctor
- Minions: Poisoner
- Demons: Pooka

### [15:14:11] Revealed #1 Hunter
Info: {'distance': 1}

### [15:14:11] Revealed #2 PlagueDoctor
Info: {}

### [15:14:11] Revealed #3 Knitter
Info: {'evil_pairs': 1}

### [15:14:12] Revealed #4 Architect
Info: {'side': 'Left'}

### [15:14:12] Revealed #5 Gemcrafter
Info: {'good_position': 3}

### [15:14:12] Revealed #6 Bishop
Info: {'targets': [1, 2, 8], 'types': ['Minion', 'Villager', 'Outcast']}

### [15:14:12] Revealed #7 Dreamer
Info: {}

### [15:14:12] Revealed #8 Knitter
Info: {'evil_pairs': 1}

#### [15:14:17] Solver Output
Scenarios: 2/290
Definite evil: ['#3']
Definite good: ['#2', '#4', '#5', '#6', '#7']
Evil probabilities: #1=50%, #8=50%
  Generated 290 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 8]

#### [15:14:17] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 2 scenarios (roles: {'Pooka'})

### [15:15:00] Executed #3 -> Pooka (EVIL)

#### [15:15:00] Solver Output
Scenarios: 2/46
Definite evil: ['#3']
Definite good: ['#2', '#4', '#5', '#6', '#7']
Evil probabilities: #1=50%, #8=50%
  Generated 46 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 8]

#### [15:15:00] Recommendation
Action: **USE_ABILITY** #7 (Dreamer) -> targets ['#1']
Reason: Entropy 1.000 (adjusted 0.750) | timing x1.00
WARNING: Corruption risk: 50%

### [15:15:51] Executed #1 -> GOOD (WRONG!)

#### [15:15:56] Solver Output
Scenarios: 1/41
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#4', '#5', '#6', '#7']
  Generated 41 candidate scenarios
  1 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #8 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD

#### [15:15:56] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 1 scenarios (roles: {'Poisoner'})

### [15:16:37] Executed #8 -> Poisoner (EVIL)

## [15:16:37] GAME OVER — WIN
Final HP: 5
Notes: 8 cards, 2 evils. Wrong exec #1 Hunter (-5 HP). Corrupted: #7 Dreamer, #5 Gemcrafter, #4 Architect. Pooka at #3 corrupted adjacent #4 and #2 area. Final HP 5/10.


---

# New Game — 2026-03-08 15:17:04
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Enlightened, Confessor, Baker, Dreamer, Knitter, Druid, Oracle
- Outcasts: Wretch, Drunk
- Minions: Chancellor, Witch
- Demons: Lilis

### [15:22:47] Revealed #1 Knitter
Info: {'evil_pairs': 3}

### [15:22:48] Revealed #2 Enlightened
Info: {'direction': 'Equidistant'}

### [15:22:48] Revealed #3 Dreamer
Info: {}

### [15:22:48] Revealed #4 Oracle
Info: {'targets': [2, 6], 'minion_role': 'Witch'}

### [15:22:48] Revealed #6 Baker
Info: {'original_role': 'Oracle'}

### [15:22:48] Revealed #7 Wretch
Info: {}

### [15:22:48] Revealed #8 Druid
Info: {}

#### [15:22:53] Solver Output
Scenarios: 227/3078
Definite good: ['#5']
Evil probabilities: #1=76%, #2=53%, #4=46%, #3=32%, #6=28%, #8=22%, #9=22%, #7=20%
  Generated 3078 candidate scenarios
  227 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [15:22:53] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#1']
Reason: Entropy 2.924 (adjusted 2.840) | timing x1.00
WARNING: Corruption risk: 6%

### [15:24:01] Ability used at #3

### [15:24:02] Revealed #3 Dreamer
Info: {'target': 1, 'evil_role': 'Chancellor'}

#### [15:24:10] Solver Output
Scenarios: 136/3078
Definite good: ['#5']
Evil probabilities: #1=60%, #2=51%, #4=47%, #3=38%, #6=30%, #9=26%, #8=26%, #7=22%
  Generated 3078 candidate scenarios
  136 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [15:24:10] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.908 (adjusted 0.878) | timing x1.00
WARNING: Corruption risk: 7%

### [15:25:13] Ability used at #8

### [15:25:13] Revealed #8 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': None}

#### [15:25:22] Solver Output
Scenarios: 71/3078
Definite good: ['#5']
Evil probabilities: #1=69%, #3=45%, #4=45%, #2=42%, #8=37%, #6=27%, #9=20%, #7=15%
  Generated 3078 candidate scenarios
  71 scenarios survived validation
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 7, 8, 9]

#### [15:25:22] Recommendation
Action: **ERROR** #1
Reason: #1 is 69% likely evil but budget=1 requires >=80% confidence (HP=8, cost=5).
WARNING: Probabilistic execution -- 69% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: CAUTION: budget=1, confidence 69% < 80% threshold. Consider manual override if you have extra information.

### [15:26:12] Executed #1 -> GOOD (WRONG!)

#### [15:26:13] Solver Output
Scenarios: 22/2064
Definite evil: ['#8']
Definite good: ['#1', '#5']
Evil probabilities: #2=64%, #4=45%, #9=45%, #6=27%, #3=9%, #7=9%
  Generated 2064 candidate scenarios
  22 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor', 'Witch', 'Lilis'})
    #1 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 6, 7, 9]

#### [15:26:13] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 22 scenarios (roles: {'Chancellor', 'Witch', 'Lilis'})

### [15:26:48] Executed #8 -> Chancellor (EVIL)

#### [15:26:48] Solver Output
Scenarios: 9/264
Definite evil: ['#8']
Definite good: ['#1', '#5']
Evil probabilities: #2=67%, #9=44%, #4=33%, #6=33%, #3=11%, #7=11%
  Generated 264 candidate scenarios
  9 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 6, 7, 9]

#### [15:26:48] Recommendation
Action: **ERROR** #2
Reason: #2 is 67% likely evil but HP too low to risk (HP=3, cost=5). Need more info.
WARNING: Probabilistic execution -- 67% confident (budget: 0 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: CRITICAL: HP=3, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [15:27:49] Executed #2 -> GOOD (WRONG!)

#### [15:27:49] Solver Output
Scenarios: 3/190
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#7']
Evil probabilities: #4=67%, #6=33%
  Generated 190 candidate scenarios
  3 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #9 is DEFINITELY EVIL (possible roles: {'Witch', 'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [15:27:49] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 3 scenarios (roles: {'Witch', 'Lilis'})

### [15:28:29] Executed #9 -> Lilis (EVIL)

#### [15:28:29] Solver Output
Scenarios: 2/31
Definite evil: ['#8', '#9']
Definite good: ['#1', '#2', '#3', '#5', '#7']
Evil probabilities: #4=50%, #6=50%
  Generated 31 candidate scenarios
  2 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #9 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [4, 6]

#### [15:28:29] Recommendation
Action: **ERROR** #4
Reason: #4 is 50% likely evil but HP too low to risk (HP=1, cost=5). Need more info.
WARNING: Probabilistic execution -- 50% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [15:29:49] Executed #6 -> Witch (EVIL)

## [15:29:58] GAME OVER — WIN
Final HP: 1
Notes: 9 cards, 3 evils. Lilis game. Night kill #5 (good). #9 Lilis blocked by Witch (#6) — could not reveal #9. Wrong exec #1 Knitter (-5 HP), wrong exec #2 Drunk Corrupted (-2 HP). Manual deduction: Knitter 3-pair count + Wretch-as-evil identified Witch at #6. Solver had 50/50, Knitter+Wretch interaction not wired. Final HP 1/10. CLOSE CALL.


---

# New Game — 2026-03-08 15:32:04
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Bishop, Dreamer, Medium, FortuneTeller, Druid, Baker
- Outcasts: Wretch
- Minions: Witch, TwinMinion
- Demons: Lilis

### [15:34:08] Revealed #1 Dreamer
Info: {}

### [15:34:08] Revealed #2 Druid
Info: {}

### [15:34:08] Revealed #3 FortuneTeller
Info: {}

### [15:34:09] Revealed #4 Baker
Info: {'original_role': 'original'}

### [15:34:09] Revealed #5 Wretch
Info: {}

### [15:34:09] Revealed #6 Baker
Info: {'original_role': 'Medium'}

### [15:34:09] Revealed #8 FortuneTeller
Info: {}

#### [15:34:15] Solver Output
Scenarios: 216/504
Definite good: ['#7']
Evil probabilities: #3=58%, #8=58%, #1=31%, #2=31%, #4=31%, #5=31%, #6=31%, #9=31%
  Generated 504 candidate scenarios
  216 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8, 9]

#### [15:34:15] Recommendation
Action: **USE_ABILITY** #1 (Dreamer) -> targets ['#3']
Reason: Entropy 2.791 (adjusted 2.791) | timing x1.00

### [15:35:18] Ability used at #1

### [15:35:19] Revealed #1 Dreamer
Info: {'target': 3, 'evil_role': 'Lilis'}

#### [15:35:19] Solver Output
Scenarios: 144/504
Definite good: ['#7']
Evil probabilities: #8=72%, #1=38%, #3=38%, #2=31%, #4=31%, #5=31%, #6=31%, #9=31%
  Generated 504 candidate scenarios
  144 scenarios survived validation
    #7 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 8, 9]

#### [15:35:19] Recommendation
Action: **USE_ABILITY** #2 (Druid) -> targets ['#1', '#3', '#4']
Reason: Entropy 0.888 (adjusted 0.888) | timing x1.00

### [15:36:29] Ability used at #2

### [15:36:29] Revealed #2 Druid
Info: {'targets': [1, 3, 4], 'found_outcast': None}

#### [15:36:29] Solver Output
Scenarios: 100/504
Definite good: ['#2', '#7']
Evil probabilities: #8=72%, #1=44%, #3=40%, #4=36%, #5=36%, #6=36%, #9=36%
  Generated 504 candidate scenarios
  100 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8, 9]

#### [15:36:29] Recommendation
Action: **ERROR** #8
Reason: #8 is 72% likely evil but budget=1 requires >=80% confidence (HP=8, cost=5).
WARNING: Probabilistic execution -- 72% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: CAUTION: budget=1, confidence 72% < 80% threshold. Consider manual override if you have extra information.

### [15:37:49] Ability used at #3

### [15:37:49] Revealed #3 Fortune Teller
Info: {'targets': [6, 8], 'has_evil': False}

#### [15:37:49] Solver Output
Scenarios: 46/504
Definite good: ['#2', '#7']
Evil probabilities: #1=57%, #3=48%, #4=48%, #5=48%, #9=48%, #6=26%, #8=26%
  Generated 504 candidate scenarios
  46 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8, 9]

#### [15:37:49] Recommendation
Action: **ERROR** #1
Reason: #1 is 57% likely evil but budget=1 requires >=80% confidence (HP=8, cost=5).
WARNING: Probabilistic execution -- 57% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: CAUTION: budget=1, confidence 57% < 80% threshold. Consider manual override if you have extra information.

### [15:39:00] Ability used at #8

### [15:39:00] Revealed #8 Fortune Teller
Info: {'targets': [1, 3], 'has_evil': False}

#### [15:39:00] Solver Output
Scenarios: 12/504
Definite evil: ['#3', '#8']
Definite good: ['#2', '#7']
Evil probabilities: #1=33%, #4=17%, #5=17%, #6=17%, #9=17%
  Generated 504 candidate scenarios
  12 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'TwinMinion', 'Lilis', 'Witch'})
    #8 is DEFINITELY EVIL (possible roles: {'TwinMinion', 'Witch', 'Lilis'})
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 9]

#### [15:39:00] Recommendation
Action: **EXECUTE** #3
Reason: #3 is evil in ALL 12 scenarios (roles: {'TwinMinion', 'Lilis', 'Witch'})

### [15:39:43] Executed #3 -> Lilis (EVIL)

#### [15:39:43] Solver Output
Scenarios: 8/56
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#7']
Evil probabilities: #4=25%, #5=25%, #6=25%, #9=25%
  Generated 56 candidate scenarios
  8 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'TwinMinion', 'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 9]

#### [15:39:43] Recommendation
Action: **EXECUTE** #8
Reason: #8 is evil in ALL 8 scenarios (roles: {'TwinMinion', 'Witch'})

### [15:40:28] Executed #8 -> TwinMinion (EVIL)

#### [15:40:28] Solver Output
Scenarios: 4/7
Definite evil: ['#3', '#8']
Definite good: ['#1', '#2', '#7']
Evil probabilities: #4=25%, #5=25%, #6=25%, #9=25%
  Generated 7 candidate scenarios
  4 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #8 is DEFINITELY EVIL (possible roles: {'TwinMinion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 9]

#### [15:40:28] Recommendation
Action: **ERROR** #4
Reason: #4 is 25% likely evil but budget=1 requires >=80% confidence (HP=8, cost=5).
WARNING: Probabilistic execution -- 25% confident (budget: 1 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card
WARNING: CAUTION: budget=1, confidence 25% < 80% threshold. Consider manual override if you have extra information.

### [15:43:24] Executed #9 -> Witch (EVIL)

## [15:43:36] GAME OVER — WIN
Final HP: 8
Notes: 9 cards, 3 evils. Witch at #9 was blocking reveals (facedown). Executed facedown Witch directly - works! No wrong execs. HP 8/10.


---

# New Game — 2026-03-08 15:44:38
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Druid, Confessor, Poet, Baker, Knight, Jester, Medium
- Outcasts: PlagueDoctor, Bombardier
- Minions: Minion
- Demons: Lilis

### [15:46:56] Revealed #1 Jester
Info: {}

### [15:46:56] Revealed #2 PlagueDoctor
Info: {}

### [15:46:56] Revealed #3 Medium
Info: {'good_position': 1, 'good_role': 'Jester'}

### [15:46:57] Revealed #4 Bombardier
Info: {}

### [15:46:57] Revealed #5 Poet
Info: {}

### [15:46:57] Revealed #6 Knight
Info: {}

### [15:46:57] Revealed #7 Baker
Info: {'original_role': 'original'}

### [15:46:57] Revealed #8 Confessor
Info: {'dizzy': True}

#### [15:47:03] Solver Output
Scenarios: 44/298
Definite good: ['#9']
Evil probabilities: #8=68%, #7=32%, #4=27%, #5=27%, #6=27%, #1=9%, #2=5%, #3=5%
  Generated 298 candidate scenarios
  44 scenarios survived validation
    #9 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 5, 6, 7, 8]

#### [15:47:03] Recommendation
Action: **USE_ABILITY** #1 (Jester) -> targets ['#2', '#7', '#8']
Reason: Expected posterior 17.8 scenarios (adjusted 19.4, info gain 1.183 bits) | timing x1.00
WARNING: Corruption risk: 18%

### [15:48:01] Ability used at #1

### [15:48:02] Revealed #1 Jester
Info: {'targets': [2, 7, 8], 'evil_count': 3}

#### [15:48:06] Solver Output
Scenarios: 12/298
Definite good: ['#2', '#9']
Evil probabilities: #8=83%, #1=33%, #3=17%, #4=17%, #5=17%, #6=17%, #7=17%
  Generated 298 candidate scenarios
  12 scenarios survived validation
    #2 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8]

#### [15:48:06] Recommendation
Action: **EXECUTE** #8
Reason: No reveals available. #8 is 83% likely evil (HP=6, budget=1 wrong execs)
WARNING: Probabilistic execution -- 83% confident (budget: 1 wrong execs)

### [15:48:35] Executed #8 -> Minion (EVIL)

#### [15:48:39] Solver Output
Scenarios: 5/36
Definite evil: ['#8']
Definite good: ['#2', '#3', '#9']
Evil probabilities: #1=20%, #4=20%, #5=20%, #6=20%, #7=20%
  Generated 36 candidate scenarios
  5 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7]

#### [15:48:39] Recommendation
Action: **ERROR** #1
Reason: #1 is 20% likely evil but budget=1 requires >=80% confidence (HP=6, cost=5).
WARNING: Probabilistic execution -- 20% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 20% < 80% threshold. Consider manual override if you have extra information.

### [15:50:13] Ability used at #2

#### [15:50:19] Solver Output
Scenarios: 5/36
Definite evil: ['#8']
Definite good: ['#2', '#3', '#9']
Evil probabilities: #1=20%, #4=20%, #5=20%, #6=20%, #7=20%
  Generated 36 candidate scenarios
  5 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 4, 5, 6, 7]

#### [15:50:19] Recommendation
Action: **ERROR** #1
Reason: #1 is 20% likely evil but budget=1 requires >=80% confidence (HP=6, cost=5).
WARNING: Probabilistic execution -- 20% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 20% < 80% threshold. Consider manual override if you have extra information.

### [15:52:57] Executed #1 -> GOOD (WRONG!)

#### [15:53:01] Solver Output
Scenarios: 4/31
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#9']
Evil probabilities: #4=25%, #5=25%, #6=25%, #7=25%
  Generated 31 candidate scenarios
  4 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 7]

#### [15:53:01] Recommendation
Action: **ERROR** #5
Reason: #5 is 25% likely evil but HP too low to risk (HP=1, cost=5). Need more info.
WARNING: Probabilistic execution -- 25% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [15:55:44] Executed #6 -> GOOD (WRONG!)

#### [15:55:54] Solver Output
Scenarios: 3/26
Definite evil: ['#8']
Definite good: ['#1', '#2', '#3', '#6', '#9']
Evil probabilities: #4=33%, #5=33%, #7=33%
  Generated 26 candidate scenarios
  3 scenarios survived validation
    #8 is DEFINITELY EVIL (possible roles: {'Minion'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 5, 7]

#### [15:55:54] Recommendation
Action: **ERROR** #5
Reason: #5 is 33% likely evil but HP too low to risk (HP=1, cost=5). Need more info.
WARNING: Probabilistic execution -- 33% confident (budget: 0 wrong execs)
WARNING: CRITICAL: HP=1, wrong exec costs 5 -- CANNOT afford a mistake! Only execute if certain.

### [15:57:17] Executed #5 -> Lilis (EVIL)

## [15:57:29] GAME OVER — WIN
Final HP: 1
## [15:57:35] GAME OVER — WIN
Final HP: 1
Notes: 9 cards, 2 evils. Lilis game, 2 nights (HP 10 to 6). PD passive corrupted Jester #1. Jester lie (3 evils in 2-evil game) key signal. Wrong exec on corrupted Jester (HP 6 to 1). Knight #6 execution immunity = free check. Poet #5 no corrupted contradicted known corruption = identified as Lilis. 1 HP finish. Corrupted: #1 Jester.


---

# New Game — 2026-03-08 16:03:11
Cards: 9, Evil: 4, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Knight, Gemcrafter, Poet, Hunter, Lover, Bard
- Outcasts: Wretch
- Minions: Witch, Puppeteer
- Demons: Lilis

### [16:06:39] Revealed #1 Bard
Info: {'corruption_distance': -1}

### [16:06:39] Revealed #2 Lover
Info: {'evil_adjacent': 2}

### [16:06:39] Revealed #3 Lover
Info: {'evil_adjacent': 2}

### [16:06:39] Revealed #4 Wretch
Info: {}

### [16:06:40] Revealed #6 Hunter
Info: {'distance': 1}

### [16:06:40] Revealed #7 Bard
Info: {'corruption_distance': 1}

### [16:06:40] Revealed #8 Knight
Info: {}

#### [16:06:47] Solver Output
Scenarios: 32/756
Definite evil: ['#7']
Definite good: ['#5']
Evil probabilities: #2=94%, #9=50%, #8=44%, #1=31%, #3=31%, #4=31%, #6=19%
  Generated 756 candidate scenarios
  32 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Witch', 'Lilis', 'Puppeteer'})
    #5 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 8, 9]

#### [16:06:47] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 32 scenarios (roles: {'Witch', 'Lilis', 'Puppeteer'})

### [16:07:24] Executed #7 -> Puppeteer (EVIL)

#### [16:07:25] Solver Output
Scenarios: 10/84
Definite evil: ['#2', '#7']
Definite good: ['#1', '#3', '#5']
Evil probabilities: #6=60%, #8=60%, #4=40%, #9=40%
  Generated 84 candidate scenarios
  10 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Witch', 'Lilis'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    Uncertain: [4, 6, 8, 9]

#### [16:07:25] Recommendation
Action: **EXECUTE** #2
Reason: #2 is evil in ALL 10 scenarios (roles: {'Witch', 'Lilis'})

### [16:07:54] Executed #2 -> Witch (EVIL)

### [16:08:49] Revealed #9 Gemcrafter
Info: {'good_position': 1}

#### [16:08:49] Solver Output
Scenarios: 3/12
Definite evil: ['#2', '#7']
Definite good: ['#1', '#3', '#5', '#9']
Evil probabilities: #4=67%, #6=67%, #8=67%
  Generated 12 candidate scenarios
  3 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Witch'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [4, 6, 8]

#### [16:08:49] Recommendation
Action: **ERROR** #4
Reason: #4 is 67% likely evil but budget=1 requires >=80% confidence (HP=6, cost=5).
WARNING: Probabilistic execution -- 67% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 67% < 80% threshold. Consider manual override if you have extra information.

### [16:09:58] Executed #8 -> Lilis (EVIL)

#### [16:09:58] Solver Output
Scenarios: 1/1
Definite evil: ['#2', '#6', '#7', '#8']
Definite good: ['#1', '#3', '#4', '#5', '#9']
  Generated 1 candidate scenarios
  1 scenarios survived validation
    #2 is DEFINITELY EVIL (possible roles: {'Witch'})
    #6 is DEFINITELY EVIL (possible roles: {'Puppet'})
    #7 is DEFINITELY EVIL (possible roles: {'Puppeteer'})
    #8 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #1 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD

#### [16:09:58] Recommendation
Action: **EXECUTE** #6
Reason: #6 is evil in ALL 1 scenarios (roles: {'Puppet'})

### [16:10:32] Executed #6 -> Puppet (EVIL)

## [16:10:33] GAME OVER — WIN
Final HP: 6
Notes: 9 cards, 4 evils. Witch blocked #9 until killed. Bard contradiction (#1 no corrupted vs #7 1-away) caught Puppeteer. Knight immunity free-check on #8 revealed Lilis. No wrong execs. HP 6/10.


---

# New Game — 2026-03-08 16:11:44
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Knitter, Knight, Empress, Scout, Medium, Poet, Bard
- Outcasts: Oracle, PlagueDoctor
- Minions: TwinMinion, Minion
- Demons: Pooka

### [16:12:55] Revealed #1 Poet
Info: {}

### [16:12:55] Revealed #2 PlagueDoctor
Info: {}

### [16:12:55] Revealed #3 Empress
Info: {'targets': [4, 7, 8]}

### [16:12:55] Revealed #4 Scout
Info: {'evil_role': 'TwinMinion', 'distance': 3}

### [16:12:55] Revealed #5 Bard
Info: {'corruption_distance': -1}

### [16:12:56] Revealed #6 Knight
Info: {}

### [16:12:56] Revealed #7 Oracle
Info: {'targets': [4, 10], 'minion_role': 'TwinMinion'}

### [16:12:56] Revealed #8 Knitter
Info: {'evil_pairs': 1}

### [16:12:56] Revealed #9 Medium
Info: {'good_position': 2, 'good_role': 'PlagueDoctor'}

### [16:12:56] Revealed #10 Knitter
Info: {'evil_pairs': 2}

### [16:13:10] Revealed #10 Poet
Info: {}

#### [16:13:16] Solver Output
Scenarios: 10/3240
Definite evil: ['#7']
Definite good: ['#2', '#3', '#4', '#8', '#9']
Evil probabilities: #1=70%, #5=60%, #10=40%, #6=30%
  Generated 3240 candidate scenarios
  10 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'TwinMinion', 'Minion', 'Pooka'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 5, 6, 10]

#### [16:13:16] Recommendation
Action: **EXECUTE** #7
Reason: #7 is evil in ALL 10 scenarios (roles: {'TwinMinion', 'Minion', 'Pooka'})

### [16:13:53] Executed #7 -> GOOD (WRONG!)

#### [16:13:59] Solver Output
Scenarios: 0/2184
  Generated 2184 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #9 Medium: rejected 1368/2184 (63%)
    #7 Oracle: rejected 1284/2184 (59%)
    #3 Empress: rejected 1064/2184 (49%)
    #8 Knitter: rejected 986/2184 (45%)
    #4 Scout: rejected 910/2184 (42%)
    #5 Bard: rejected 846/2184 (39%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Poet: still 0
    WITHOUT #3 Empress: still 0
    WITHOUT #4 Scout: still 0
    WITHOUT #5 Bard: still 0
    WITHOUT #7 Oracle: still 0
    WITHOUT #8 Knitter: still 0
    WITHOUT #9 Medium: still 0
    WITHOUT #10 Poet: still 0

#### [16:13:59] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

#### [16:28:47] Solver Output
Scenarios: 30/2184
Definite good: ['#2', '#7', '#9']
Evil probabilities: #10=63%, #5=57%, #8=53%, #4=50%, #1=47%, #6=23%, #3=7%
  Generated 2184 candidate scenarios
  30 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8, 10]

#### [16:28:47] Recommendation
Action: **ERROR** #10
Reason: #10 is 63% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 63% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 63% < 80% threshold. Consider manual override if you have extra information.

#### [16:29:03] Solver Output
Scenarios: 30/2184
Definite good: ['#2', '#7', '#9']
Evil probabilities: #10=63%, #5=57%, #8=53%, #4=50%, #1=47%, #6=23%, #3=7%
  Generated 2184 candidate scenarios
  30 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8, 10]

#### [16:29:03] Recommendation
Action: **ERROR** #10
Reason: #10 is 63% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 63% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 63% < 80% threshold. Consider manual override if you have extra information.

### [16:30:13] Ability used at #2

#### [16:30:13] Solver Output
Scenarios: 26/2184
Definite good: ['#2', '#7', '#9']
Evil probabilities: #10=58%, #1=54%, #5=54%, #8=54%, #4=46%, #6=27%, #3=8%
  Generated 2184 candidate scenarios
  26 scenarios survived validation
    #2 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 8, 10]

#### [16:30:13] Recommendation
Action: **ERROR** #10
Reason: #10 is 58% likely evil but budget=1 requires >=80% confidence (HP=5, cost=5).
WARNING: Probabilistic execution -- 58% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 58% < 80% threshold. Consider manual override if you have extra information.

### [16:33:15] Executed #6 -> GOOD (WRONG!)

## [16:33:21] GAME OVER — LOSS
Final HP: 0
Notes: Loss: Oracle misclassified as Outcast in deck caused wrong exec #7. Knight #6 corrupted by Pooka #5, immunity failed, second wrong exec. Corrupted: #3,#4,#6. Bard #5 was Pooka, lied about no corruption. PD corrupted #3.


---

# New Game — 2026-03-08 16:36:08
Cards: 6, Evil: 1, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Bishop, Knight, Oracle, Alchemist, FortuneTeller, Judge
- Outcasts: Drunk
- Minions: 
- Demons: Pooka

### [16:37:33] Revealed #1 Bishop
Info: {'targets': [2, 3, 5], 'types': ['Villager', 'Demon', 'Outcast']}

### [16:37:33] Revealed #2 FortuneTeller
Info: {}

### [16:37:34] Revealed #3 Oracle
Info: {'targets': [2, 6], 'minion_role': 'Chancellor'}

### [16:37:38] Revealed #4 Alchemist
Info: {'cured_count': 1}

### [16:37:38] Revealed #5 Judge
Info: {}

### [16:37:39] Revealed #6 Knight
Info: {}

#### [16:37:42] Solver Output
Scenarios: 7/30
Definite good: ['#1']
Evil probabilities: #3=43%, #2=14%, #4=14%, #5=14%, #6=14%
  Generated 30 candidate scenarios
  7 scenarios survived validation
    #1 is DEFINITELY GOOD
    Uncertain: [2, 3, 4, 5, 6]

#### [16:37:42] Recommendation
Action: **USE_ABILITY** #5 (Judge) -> targets ['#4']
Reason: Expected posterior 4.6 scenarios (adjusted 5.2, info gain 0.427 bits) | timing x1.00
WARNING: Corruption risk: 29% -- corrupted Judge results are unreliable

### [16:38:36] Ability used at #5

### [16:38:36] Revealed #5 Judge
Info: {'target': 4, 'is_lying': False}

#### [16:38:40] Solver Output
Scenarios: 4/30
Definite good: ['#1', '#2']
Evil probabilities: #3=25%, #4=25%, #5=25%, #6=25%
  Generated 30 candidate scenarios
  4 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    Uncertain: [3, 4, 5, 6]

#### [16:38:40] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 25% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 25% confident (budget: 2 wrong execs)
WARNING: Low confidence (25%) -- consider gathering more info

### [16:39:57] Ability used at #2

### [16:39:57] Revealed #2 Fortune Teller
Info: {'targets': [3, 5], 'has_evil': True}

#### [16:40:01] Solver Output
Scenarios: 1/30
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4', '#6']
  Generated 30 candidate scenarios
  1 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD

#### [16:40:01] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 1 scenarios (roles: {'Pooka'})

### [16:40:33] Executed #5 -> Pooka (EVIL)

## [16:40:40] GAME OVER — WIN
Final HP: 10
Notes: 6 cards, 1 evil. Pooka at #5 disguised as Judge, corrupted #3,#4,#6. Judge lied about #4 truthfulness. Drunk at #3 disguised as Oracle. Perfect HP.


---

# New Game — 2026-03-08 16:41:34
Cards: 9, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Hunter, Baker, Oracle, Confessor, Druid, Empress
- Outcasts: Doppelganger, Wretch
- Minions: Poisoner, Witch
- Demons: Baa

### [16:42:48] Revealed #1 Baker
Info: {'original_role': 'original'}

### [16:42:49] Revealed #2 Druid
Info: {}

### [16:42:49] Revealed #3 Confessor
Info: {'dizzy': True}

### [16:42:50] Revealed #4 Oracle
Info: {'targets': [1, 8], 'minion_role': 'Poisoner'}

### [16:42:54] Revealed #5 Empress
Info: {'targets': [1, 6, 9]}

### [16:42:55] Revealed #6 Hunter
Info: {'distance': 1}

### [16:42:55] Revealed #7 Oracle
Info: {'targets': [1, 6], 'minion_role': 'Poisoner'}

### [16:42:56] Revealed #8 Baker
Info: {'original_role': 'Oracle'}

#### [16:43:13] Solver Output
Scenarios: 100/4542
Definite good: ['#2']
Evil probabilities: #7=80%, #3=70%, #4=44%, #5=38%, #1=20%, #9=20%, #8=16%, #6=12%
  Generated 4542 candidate scenarios
  100 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8, 9]

#### [16:43:13] Recommendation
Action: **USE_ABILITY** #2 (Druid) -> targets ['#1', '#3', '#4']
Reason: Entropy 0.000 (adjusted 0.000) | timing x1.00

### [16:44:06] Ability used at #2

### [16:44:06] Revealed #2 Druid
Info: {'targets': [1, 3, 4], 'found_outcast': None}

#### [16:44:12] Solver Output
Scenarios: 80/4542
Definite good: ['#2']
Evil probabilities: #7=85%, #3=68%, #4=48%, #5=35%, #1=25%, #9=20%, #6=10%, #8=10%
  Generated 4542 candidate scenarios
  80 scenarios survived validation
    #2 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 6, 7, 8, 9]

#### [16:44:12] Recommendation
Action: **EXECUTE** #7
Reason: No reveals available. #7 is 85% likely evil (HP=10, budget=2 wrong execs)
WARNING: Probabilistic execution -- 85% confident (budget: 2 wrong execs)
WARNING: Witch is blocking reveals -- killing Witch would unblock last card

### [16:44:40] Executed #7 -> Witch (EVIL)

### [16:45:07] Revealed #9 Oracle
Info: {'targets': [4, 6], 'minion_role': 'Poisoner'}

#### [16:45:08] Solver Output
Scenarios: 17/602
Definite evil: ['#7']
Definite good: ['#2', '#6', '#8']
Evil probabilities: #4=76%, #3=47%, #1=29%, #5=24%, #9=24%
  Generated 602 candidate scenarios
  17 scenarios survived validation
    #7 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [1, 3, 4, 5, 9]

#### [16:45:08] Recommendation
Action: **EXECUTE** #4
Reason: Execution lookahead: #4 guarantees a win across all reveal branches with current HP budget (76% evil Poisoner, 24% good Oracle (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 76%, but all reveal branches still lead to a forced win.

### [16:45:38] Executed #4 -> Poisoner (EVIL)

#### [16:45:38] Solver Output
Scenarios: 13/84
Definite evil: ['#4', '#7']
Definite good: ['#2', '#6', '#8', '#9']
Evil probabilities: #1=38%, #3=31%, #5=31%
  Generated 84 candidate scenarios
  13 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #7 is DEFINITELY EVIL (possible roles: {'Witch'})
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [1, 3, 5]

#### [16:45:38] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (62% good Baker, 38% evil Baa).
WARNING: Execution lookahead override -- immediate hit chance is 38%, but all reveal branches still lead to a forced win.

### [16:46:12] Executed #1 -> GOOD (WRONG!)

#### [16:46:13] Solver Output
Scenarios: 8/70
Definite evil: ['#4', '#7']
Definite good: ['#1', '#2', '#6', '#8', '#9']
Evil probabilities: #3=50%, #5=50%
  Generated 70 candidate scenarios
  8 scenarios survived validation
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #7 is DEFINITELY EVIL (possible roles: {'Witch'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #9 is DEFINITELY GOOD
    Uncertain: [3, 5]

#### [16:46:13] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 50% good Confessor (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [16:46:59] Executed #3 -> Baa (EVIL)

## [16:46:59] GAME OVER — WIN
Final HP: 5
Notes: 9 cards, 3 evils. Witch blocked #9, killed Witch first. Poisoner at #4 found via Oracle cross-reference. 50/50 Baa at #3 vs #5, got it right. Corrupted: #5 Empress. Wrong exec #1 Baker (HP 10 to 5).


---

# New Game — 2026-03-08 16:47:59
Cards: 9, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Lover, Druid, Jester, Baker, Knight, Knitter, Bard
- Outcasts: Drunk, Doppelganger
- Minions: Shaman
- Demons: Pooka

### [16:49:07] Revealed #1 Bard
Info: {'corruption_distance': 2}

### [16:49:07] Revealed #2 Knitter
Info: {'evil_pairs': 0}

### [16:49:07] Revealed #3 Baker
Info: {'original_role': 'original'}

### [16:49:08] Revealed #4 Knitter
Info: {'evil_pairs': 0}

### [16:49:08] Revealed #5 Druid
Info: {}

### [16:49:08] Revealed #6 Lover
Info: {'evil_adjacent': 2}

### [16:49:08] Revealed #7 Baker
Info: {'original_role': 'Bard'}

### [16:49:08] Revealed #8 Druid
Info: {}

### [16:49:08] Revealed #9 Jester
Info: {}

#### [16:49:14] Solver Output
Scenarios: 165/3024
Definite good: ['#2', '#4']
Evil probabilities: #6=50%, #9=45%, #7=33%, #1=25%, #3=24%, #8=12%, #5=10%
  Generated 3024 candidate scenarios
  165 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 6, 7, 8, 9]

#### [16:49:14] Recommendation
Action: **USE_ABILITY** #5 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.992 (adjusted 0.821) | timing x1.00
WARNING: Corruption risk: 35%

### [16:50:10] Ability used at #5

### [16:50:10] Revealed #5 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Drunk'}

#### [16:50:10] Solver Output
Scenarios: 56/3024
Definite good: ['#2', '#4']
Evil probabilities: #6=59%, #9=39%, #7=27%, #1=25%, #3=21%, #5=16%, #8=12%
  Generated 3024 candidate scenarios
  56 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 6, 7, 8, 9]

#### [16:50:10] Recommendation
Action: **USE_ABILITY** #9 (Jester) -> targets ['#6', '#7', '#8']
Reason: Expected posterior 29.5 scenarios (adjusted 31.9, info gain 0.811 bits) | timing x1.00
WARNING: Corruption risk: 16%

### [16:51:04] Ability used at #9

### [16:51:05] Revealed #9 Jester
Info: {'targets': [6, 7, 8], 'evil_count': 1}

#### [16:51:05] Solver Output
Scenarios: 30/3024
Definite good: ['#2', '#4']
Evil probabilities: #6=53%, #3=37%, #1=30%, #7=30%, #5=20%, #8=17%, #9=13%
  Generated 3024 candidate scenarios
  30 scenarios survived validation
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    Uncertain: [1, 3, 5, 6, 7, 8, 9]

#### [16:51:05] Recommendation
Action: **USE_ABILITY** #8 (Druid) -> targets ['#1', '#2', '#3']
Reason: Entropy 0.971 (adjusted 0.761) | timing x1.00
WARNING: Corruption risk: 43%

### [16:52:14] Ability used at #8

### [16:52:14] Revealed #8 Druid
Info: {'targets': [1, 2, 3], 'found_outcast': 'Doppelganger'}

#### [16:52:14] Solver Output
Scenarios: 9/3024
Definite good: ['#1', '#2', '#4', '#8']
Evil probabilities: #3=56%, #5=44%, #6=33%, #7=33%, #9=33%
  Generated 3024 candidate scenarios
  9 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [3, 5, 6, 7, 9]

#### [16:52:14] Recommendation
Action: **EXECUTE** #3
Reason: Execution lookahead: #3 guarantees a win across all reveal branches with current HP budget (56% evil Shaman, 44% good Baker).
WARNING: Execution lookahead override -- immediate hit chance is 56%, but all reveal branches still lead to a forced win.

### [16:52:51] Executed #3 -> GOOD (WRONG!)

#### [16:52:52] Solver Output
Scenarios: 4/2352
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#8']
Evil probabilities: #9=75%, #7=25%
  Generated 2352 candidate scenarios
  4 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka', 'Shaman'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [7, 9]

#### [16:52:52] Recommendation
Action: **EXECUTE** #5
Reason: #5 is evil in ALL 4 scenarios (roles: {'Pooka', 'Shaman'})

### [16:53:23] Executed #5 -> Pooka (EVIL)

#### [16:53:24] Solver Output
Scenarios: 1/294
Definite evil: ['#5', '#9']
Definite good: ['#1', '#2', '#3', '#4', '#6', '#7', '#8']
  Generated 294 candidate scenarios
  1 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Pooka'})
    #9 is DEFINITELY EVIL (possible roles: {'Shaman'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD

#### [16:53:24] Recommendation
Action: **EXECUTE** #9
Reason: #9 is evil in ALL 1 scenarios (roles: {'Shaman'})

### [16:54:02] Executed #9 -> Shaman (EVIL)

## [16:54:02] GAME OVER — WIN
Final HP: 5
Notes: 9 cards, 2 evils. Pooka at #5 corrupted #6 Lover and #8 Drunk. Doppelganger at #4 immune to corruption. Wrong exec #3 Baker (HP 10 to 5). Druid x2 found both outcasts in 1-2-3 group.


---

# New Game — 2026-03-08 16:55:13
Cards: 7, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Bishop, Baker, Hunter, Oracle, Slayer, Dreamer
- Outcasts: Doppelganger, PlagueDoctor
- Minions: Minion
- Demons: Baa

### [16:56:17] Revealed #1 Hunter
Info: {'distance': 1}

### [16:56:17] Revealed #2 Oracle
Info: {'targets': [3, 5], 'minion_role': 'Minion'}

### [16:56:17] Revealed #3 Dreamer
Info: {}

### [16:56:18] Revealed #4 Oracle
Info: {'targets': [2, 5], 'minion_role': 'Minion'}

### [16:56:18] Revealed #5 Bishop
Info: {'targets': [2, 3, 6], 'types': ['Minion', 'Villager', 'Outcast']}

### [16:56:18] Revealed #6 Slayer
Info: {}

### [16:56:18] Revealed #7 Baker
Info: {'original_role': 'original'}

#### [16:56:25] Solver Output
Scenarios: 15/252
Definite good: ['#3', '#4', '#6']
Evil probabilities: #5=87%, #7=47%, #1=33%, #2=33%
  Generated 252 candidate scenarios
  15 scenarios survived validation
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 7]

#### [16:56:25] Recommendation
Action: **USE_ABILITY** #3 (Dreamer) -> targets ['#5']
Reason: Entropy 1.242 (adjusted 1.242) | timing x1.00

### [16:57:32] Ability used at #3

### [16:57:32] Revealed #3 Dreamer
Info: {'target': 5, 'evil_role': 'Minion'}

#### [16:57:32] Solver Output
Scenarios: 12/252
Definite good: ['#3', '#4', '#6']
Evil probabilities: #5=83%, #7=58%, #1=42%, #2=17%
  Generated 252 candidate scenarios
  12 scenarios survived validation
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 2, 5, 7]

#### [16:57:32] Recommendation
Action: **USE_ABILITY** #6 (Slayer) -> targets ['#5']
Reason: Target #5 is 83% evil (adjusted 0.83)

### [16:58:30] Ability used at #6

#### [16:58:31] Solver Output
Scenarios: 0/180
  Generated 180 candidate scenarios
  0 scenarios survived validation
  NO VALID SCENARIOS — check input data
  
  === ZERO-SCENARIO DIAGNOSTICS ===
  Rejection counts (card -> how many scenarios it rejected):
    #5 Bishop: rejected 162/180 (90%)
    #4 Oracle: rejected 102/180 (57%)
    #2 Oracle: rejected 102/180 (57%)
    #1 Hunter: rejected 60/180 (33%)
  
  Leave-one-out analysis (removing each card's info):
    WITHOUT #1 Hunter: 6 scenarios survive  <-- SUSPECT
    WITHOUT #2 Oracle: 2 scenarios survive  <-- SUSPECT
    WITHOUT #3 Dreamer: 2 scenarios survive  <-- SUSPECT
    WITHOUT #4 Oracle: 8 scenarios survive  <-- SUSPECT
    WITHOUT #5 Bishop: 24 scenarios survive  <-- SUSPECT
    WITHOUT #7 Baker: 2 scenarios survive  <-- SUSPECT

#### [16:58:31] Recommendation
Action: **ERROR**
Reason: No surviving scenarios -- check input data

### [16:59:29] Executed #5 -> Minion (EVIL)

#### [16:59:36] Solver Output
Scenarios: 10/36
Definite evil: ['#5']
Definite good: ['#2', '#3', '#4', '#6']
Evil probabilities: #1=50%, #7=50%
  Generated 36 candidate scenarios
  10 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Minion'})
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    Uncertain: [1, 7]

#### [16:59:36] Recommendation
Action: **EXECUTE** #1
Reason: Execution lookahead: #1 guarantees a win across all reveal branches with current HP budget (50% evil Baa, 40% good Hunter, 10% good Doppelganger).
WARNING: Execution lookahead override -- immediate hit chance is 50%, but all reveal branches still lead to a forced win.

### [17:00:17] Executed #1 -> Baa (EVIL)

## [17:00:17] GAME OVER — WIN
Final HP: 10
Notes: 7 cards, 2 evils. Slayer killed Minion at #5 (free kill). Baa at #1 found via 50/50. Perfect HP. Doppelganger at #4. 0-scenario issue after slayer_result - needed explicit execute to set evil role.


---

# New Game — 2026-03-08 17:01:54
Cards: 8, Evil: 2, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Oracle, Confessor, Hunter, Enlightened, Jester
- Outcasts: Wretch, PlagueDoctor, Doppelganger
- Minions: Poisoner
- Demons: Baa

### [17:03:17] Revealed #1 Confessor
Info: {'dizzy': False}

### [17:03:17] Revealed #2 Enlightened
Info: {'direction': 'CCW'}

### [17:03:17] Revealed #3 PlagueDoctor
Info: {}

### [17:03:17] Revealed #4 Oracle
Info: {'targets': [1, 2], 'minion_role': 'Poisoner'}

### [17:03:17] Revealed #5 Jester
Info: {}

### [17:03:17] Revealed #6 Hunter
Info: {'distance': 4}

### [17:03:17] Revealed #7 Confessor
Info: {'dizzy': True}

### [17:03:18] Revealed #8 Wretch
Info: {}

#### [17:04:47] Solver Output
Scenarios: 47/1396
Definite good: ['#1', '#2', '#3']
Evil probabilities: #7=64%, #6=57%, #5=38%, #8=26%, #4=15%
  Generated 1396 candidate scenarios
  47 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    Uncertain: [4, 5, 6, 7, 8]

#### [17:04:47] Recommendation
Action: **USE_ABILITY** #5 (Jester) -> targets ['#1', '#6', '#7']
Reason: Expected posterior 22.8 scenarios (adjusted 23.7, info gain 0.985 bits) | timing x1.00
WARNING: Corruption risk: 9%

### [17:06:30] Ability used at #5

### [17:06:34] Revealed #5 Jester
Info: {'targets': [1, 6, 7], 'evil_count': 2}

#### [17:06:37] Solver Output
Scenarios: 24/1396
Definite good: ['#1', '#2', '#3', '#4', '#8']
Evil probabilities: #5=75%, #7=75%, #6=50%
  Generated 1396 candidate scenarios
  24 scenarios survived validation
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [5, 6, 7]

#### [17:06:37] Recommendation
Action: **EXECUTE** #5
Reason: Execution lookahead: #5 guarantees a win across all reveal branches with current HP budget (46% evil Poisoner, 29% evil Baa, 17% good Jester).
WARNING: Execution lookahead override -- immediate hit chance is 75%, but all reveal branches still lead to a forced win.

### [17:07:27] Executed #5 -> Baa (EVIL)

#### [17:07:31] Solver Output
Scenarios: 7/172
Definite evil: ['#5']
Definite good: ['#1', '#2', '#3', '#4', '#8']
Evil probabilities: #7=57%, #6=43%
  Generated 172 candidate scenarios
  7 scenarios survived validation
    #5 is DEFINITELY EVIL (possible roles: {'Baa'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #3 is DEFINITELY GOOD
    #4 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    Uncertain: [6, 7]

#### [17:07:31] Recommendation
Action: **EXECUTE** #7
Reason: Execution lookahead: #7 guarantees a win across all reveal branches with current HP budget (57% evil Poisoner, 43% good Confessor (corrupted)).
WARNING: Execution lookahead override -- immediate hit chance is 57%, but all reveal branches still lead to a forced win.

### [17:08:05] Executed #7 -> Poisoner (EVIL)

## [17:08:12] GAME OVER — WIN
Final HP: 10
Notes: 8 cards, 2 evils. Perfect game 10HP. Baa at #5, Poisoner at #7. Corrupted: #4 Oracle, #6 Hunter (both adjacent to Baa #5). Jester found 2 evils in {1,6,7}, solver narrowed to 7 scenarios then guaranteed win path.


---

# New Game — 2026-03-08 17:08:35
Cards: 10, Evil: 3, HP: 10, Wrong exec cost: 2

## Deck
- Villagers: Confessor, Poet, Jester, Empress, Bard, Judge, Medium
- Outcasts: Bishop, PlagueDoctor, Doppelganger
- Minions: Chancellor, Poisoner
- Demons: Lilis

### [17:11:22] Revealed #1 Judge
Info: {}

### [17:11:22] Revealed #2 Poet
Info: {'copied_role': '10'}

### [17:11:22] Revealed #3 Bard
Info: {'corruption_distance': -1}

### [17:11:22] Revealed #4 Empress
Info: {'targets': [1, 2, 7]}

### [17:11:22] Revealed #6 Bishop
Info: {'targets': [1, 4, 7], 'types': ['Minion', 'Outcast', 'Villager']}

### [17:11:22] Revealed #7 PlagueDoctor
Info: {}

### [17:11:22] Revealed #8 Medium
Info: {'good_position': 6, 'good_role': 'Bishop'}

### [17:11:23] Revealed #9 Bard
Info: {'corruption_distance': 3}

#### [17:11:37] Solver Output
Scenarios: 254/16086
Definite good: ['#5', '#7', '#10']
Evil probabilities: #3=72%, #9=63%, #8=42%, #6=40%, #4=33%, #1=32%, #2=18%
  Generated 16086 candidate scenarios
  254 scenarios survived validation
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 8, 9]

#### [17:11:37] Recommendation
Action: **USE_ABILITY** #1 (Judge) -> targets ['#9']
Reason: Expected posterior 139.6 scenarios (adjusted 144.2, info gain 0.816 bits) | timing x1.00
WARNING: Corruption risk: 7% -- corrupted Judge results are unreliable

### [17:12:27] Ability used at #1

### [17:12:27] Revealed #1 Judge
Info: {'target': 9, 'is_lying': False}

#### [17:12:33] Solver Output
Scenarios: 112/16086
Definite good: ['#5', '#7', '#10']
Evil probabilities: #3=84%, #1=64%, #9=64%, #8=39%, #6=24%, #4=18%, #2=6%
  Generated 16086 candidate scenarios
  112 scenarios survived validation
    #5 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 3, 4, 6, 8, 9]

#### [17:12:33] Recommendation
Action: **EXECUTE** #3
Reason: No reveals available. #3 is 84% likely evil (HP=6, budget=1 wrong execs)
WARNING: Probabilistic execution -- 84% confident (budget: 1 wrong execs)

### [17:13:01] Executed #3 -> Lilis (EVIL)

#### [17:13:05] Solver Output
Scenarios: 42/1840
Definite evil: ['#3']
Definite good: ['#5', '#6', '#7', '#10']
Evil probabilities: #9=81%, #1=76%, #4=19%, #8=19%, #2=5%
  Generated 1840 candidate scenarios
  42 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 4, 8, 9]

#### [17:13:05] Recommendation
Action: **EXECUTE** #9
Reason: No reveals available. #9 is 81% likely evil (HP=6, budget=1 wrong execs)
WARNING: Probabilistic execution -- 81% confident (budget: 1 wrong execs)

### [17:13:31] Executed #9 -> Chancellor (EVIL)

#### [17:13:35] Solver Output
Scenarios: 16/228
Definite evil: ['#3', '#9']
Definite good: ['#5', '#6', '#7', '#8', '#10']
Evil probabilities: #1=75%, #2=12%, #4=12%
  Generated 228 candidate scenarios
  16 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD
    Uncertain: [1, 2, 4]

#### [17:13:35] Recommendation
Action: **ERROR** #1
Reason: #1 is 75% likely evil but budget=1 requires >=80% confidence (HP=6, cost=5).
WARNING: Probabilistic execution -- 75% confident (budget: 1 wrong execs)
WARNING: CAUTION: budget=1, confidence 75% < 80% threshold. Consider manual override if you have extra information.

### [17:14:44] Ability used at #7

#### [17:14:49] Solver Output
Scenarios: 2/228
Definite evil: ['#3', '#4', '#9']
Definite good: ['#1', '#2', '#5', '#6', '#7', '#8', '#10']
  Generated 228 candidate scenarios
  2 scenarios survived validation
    #3 is DEFINITELY EVIL (possible roles: {'Lilis'})
    #4 is DEFINITELY EVIL (possible roles: {'Poisoner'})
    #9 is DEFINITELY EVIL (possible roles: {'Chancellor'})
    #1 is DEFINITELY GOOD
    #2 is DEFINITELY GOOD
    #5 is DEFINITELY GOOD
    #6 is DEFINITELY GOOD
    #7 is DEFINITELY GOOD
    #8 is DEFINITELY GOOD
    #10 is DEFINITELY GOOD

#### [17:14:49] Recommendation
Action: **EXECUTE** #4
Reason: #4 is evil in ALL 2 scenarios (roles: {'Poisoner'})

### [17:16:18] Executed #4 -> Poisoner (EVIL)

## [17:16:29] GAME OVER — WIN
Final HP: 6
Notes: 10 cards, 3 evils. Final village of Asc15. Lilis game with 2 night kills (#5, #10, both good). HP 6/10. Corrupted: #1 Judge (by Poisoner #4 adjacent). Two Bards: #3 was Lilis disguised, #9 was Chancellor disguised. Judge ability was corrupted so said #9 truthful (wrong). PD #7 saved the game revealing #4 evil and #1 corrupted. #8 was Doppelganger copying Medium.

