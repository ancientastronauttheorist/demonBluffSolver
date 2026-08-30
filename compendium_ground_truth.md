# Demon Bluff Compendium — Ground Truth
# Source: In-game compendium right-click info (March 2026)

## VILLAGERS (Good)

### Alchemist
- **Ability**: Villagers to the left and right of me [Range 2] are cured from Corruption. Learn how many I cured.
- **Hints**: Range 2: 2 Villagers to the left of me and 2 Villagers to the right of me are affected.

### Architect
- **Ability**: Learn which side of the circle is more Evil. Learn 'Equal' if both sides are equally Evil.
- **Hints**: My measurement includes the two characters at the exact midpoints of the circle as well.

### Baker
- **Ability**: Reveal: 1 random Unrevealed Good Villager becomes a Baker. Learn which Villager I was.
- **If Lies**: I say that I was a random Villager role.

### Bard
- **Ability**: Learn how far I am from closest Corrupted character.

### Bishop
- **Ability**: Learn up to 3 characters. Among them are 1 Villager, 1 Outcast and 1 Evil role if possible.
- **If Lies**: All characters in my info are Villagers.

### Confessor
- **Ability**: If I am Evil or Corrupted: "I am dizzy"
- **Hints**: I can not Lie.

### Dreamer
- **Ability**: Pick 1 character: Learn an Evil role. If Evil picked, Learn its info.
- **If NOT Lying**: Wretch looks like a Cabbage to me.
- **Active ability** (pick target)

### Druid
- **Ability**: Pick 3 characters: Learn 1 random Outcast among them (if any).
- **Hints**: Wretch does not appear as an Outcast for her.
- **Active ability** (pick targets)

### Empress
- **Ability**: Learn 3 characters. Only 1 is Evil.
- **If Lies**: All characters in my info are Good.

### Enlightened
- **Ability**: Learn if closest Evil to me is Clockwise or Counter-Clockwise. Learn 'Equidistant' if Evils are at the same distance from me.

### Fortune Teller
- **Ability**: Pick 2 characters: Learn if any of them is Evil.
- **Active ability** (pick targets)

### Gemcrafter
- **Ability**: Learn 1 Good character.

### Hunter
- **Ability**: Learn how far from me is the nearest Evil.

### Jester
- **Ability**: Pick 3 characters: Learn how many of them are Evil.
- **Active ability** (pick targets)

### Judge
- **Ability**: Pick 1 character: Learn if they're Lying.
- **Active ability** (pick target)

### Knight
- **Ability**: I can't die.
- **Hints**: If I am Executed while Corrupted: I deal 4 additional damage to you.

### Knitter
- **Ability**: Learn how many Evils are adjacent to each other.

### Lover
- **Ability**: Learn how many Evil characters I am adjacent to.

### Medium
- **Ability**: Learn a Good character and its role.
- **If Lies**: My info includes Disguised character.

### Oracle
- **Ability**: Learn that 1 out of 2 characters is a specific Minion role.
- **If Lies**: Both characters in my info are Good.

### Poet
- **Ability**: Learn random info.

### Scout
- **Ability**: Learn how far a specific Evil is to another closest Evil.
- **Hints**: Tells you distance from 1 random Evil to its nearest Evil.

### Slayer
- **Ability**: Pick 1 character: If Evil picked, Execute it.
- **If Lies**: I can not kill my target.
- **Active ability** (pick target)

### Witness
- **Ability**: Learn a character that was affected by an Evil ability.
- **Can Learn**: Who is the Puppet, Corrupted character by an Evil, Villager turned into Outcast by Chancellor, Cloned character by Shaman, Who was killed by an Evil.

---

## OUTCASTS (Good)

### Drunk
- **Ability**: I Disguise as a random not in play Villager. I am Corrupted and I Lie. I can not be Cured.
- **Hints**: You receive 2 damage instead of 5 when you Execute me.

### Wretch
- **Ability**: I Register as a random Evil Minion to other characters.
- **Hints**: I can not Register as Good or as Wretch to other characters. I only Register as a random Minion role or as an Evil. Can not be Disguised as.

### Bombardier
- **Ability**: Lose if you Execute me.

### Doppelganger
- **Ability**: Game Start: I Disguise as a Good Villager currently in play.
- **Hints**: Always consider themselves as a Disguised role.

### Plague Doctor
- **Ability**: At truthful Start, uniformly add Corrupted to one eligible apparent Villager that lacks the status and exact resistance; there is no alignment, liveness, or self filter. During Day, pick one character and learn the result below.
- **Native targeting**: The active picker allows any board character, including self and dead cards.
- **Truthful result**: A Corrupted target uniformly reveals one registered/runtime Evil character; Wretch and dead Evil characters remain eligible. A clean target reports `Not Corrupted`.
- **Lying result**: The status answer is inverted. A clean target is called Corrupted and one registered/runtime Good character is uniformly mislabeled Evil; a Corrupted target is called clean.
- **Hints**: A self-check always displays `Not Corrupted`, even if the Plague Doctor is Evil or has the status. Ordinary Drunk is checked by raw active status; asc84_v2's generated Drunk was clean because inherited Alchemist resistance blocked that status.

---

## MINIONS (Evil)

### Chancellor
- **Ability**: Game Start: One Villager becomes an Outcast role. I sit next to it. I Lie and Disguise.

### Witch
- **Ability**: You can not Reveal the last card. I Lie and Disguise.
- **Hints**: You can reveal the last card after I die.

### Minion
- **Ability**: I Lie and Disguise.

### Poisoner
- **Ability**: Game Start: One adjacent Villager is Corrupted (if possible). I Lie and Disguise.

### Twin Minion
- **Ability**: I Lie and Disguise.

### Shaman
- **Ability**: Game Start: There are 2 same Villager roles in current Village. I Lie and Disguise.

### Puppeteer
- **Ability**: Game Start: Random adjacent Villager becomes a Puppet if possible (Its Evil, but can not Lie). I Lie and Disguise.

### Puppet
- **Ability**: I Disguise as a Villager, I don't Lie.
- **Hints**: Can only be created by the Puppeteer.

---

## DEMONS (Evil)

### Baa
- **Ability**: One fake Outcast is added to the Deck view. I Lie and Disguise.

### Pooka
- **Ability**: Game Start: Villagers adjacent to me are Corrupted (if possible). I Lie and Disguise.

### Lilis
- **Ability**: At Night: Kill a random unrevealed character. Deal 2 damage to you. I Lie and Disguise.
- **Hints**: Will prioritize killing Good characters first. Will not kill herself when she is the only card left.

---

## GLOBAL MECHANIC (shown on Drunk "More info" popup)
- **CORRUPTED**: Corrupted characters will Lie and their ability will not work.

## ALIGNMENT TOOLTIPS
- **Good**: Character Alignment. Avoid executing them. Lose 5 Health whenever you execute a Good character.
- **Evil**: (implied opposite — execute to win)
