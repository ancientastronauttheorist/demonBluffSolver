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

