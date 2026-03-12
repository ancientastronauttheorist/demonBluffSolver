"""Complete knowledge base of all Demon Bluff cards, learned from the game compendium."""

from dataclasses import dataclass, field
from enum import Enum


class Role(Enum):
    VILLAGER = "Villager"
    OUTCAST = "Outcast"
    MINION = "Minion"
    DEMON = "Demon"


class Alignment(Enum):
    GOOD = "Good"
    EVIL = "Evil"


@dataclass
class Card:
    name: str
    role: Role
    alignment: Alignment
    ability: str
    lies: bool  # Does this character lie?
    disguises: bool  # Does this character disguise?
    game_start_ability: bool = False  # Does ability trigger at game start?
    activated_ability: bool = False  # Does the player activate this ability?
    corrupted_note: bool = False  # Does the card mention corruption?
    cant_lie: bool = False  # "I Can't Lie" keyword


# ============================================================
# VILLAGERS (Good) - 24 cards
# ============================================================
VILLAGERS = [
    Card("Alchemist", Role.VILLAGER, Alignment.GOOD,
         "Villagers to the left and right of me [Range 2] are cured from Corruption. Learn how many I cured.",
         lies=False, disguises=False, corrupted_note=True),
    Card("Architect", Role.VILLAGER, Alignment.GOOD,
         "Learn which side of the circle is more Evil. Learn 'Equal' if both sides are equally Evil.",
         lies=False, disguises=False),
    Card("Baker", Role.VILLAGER, Alignment.GOOD,
         "Reveal: 1 random Unrevealed Good Villager becomes a Baker. Learn which Villager I was.",
         lies=False, disguises=False),
    Card("Bard", Role.VILLAGER, Alignment.GOOD,
         "Learn how far I am from closest Corrupted character.",
         lies=False, disguises=False, corrupted_note=True),
    Card("Bishop", Role.VILLAGER, Alignment.GOOD,
         "Learn up to 3 characters. Among them are 1 Villager, 1 Outcast and 1 Evil role if possible.",
         lies=False, disguises=False),
    Card("Confessor", Role.VILLAGER, Alignment.GOOD,
         "If I am Evil or Corrupted: 'I am dizzy'",
         lies=False, disguises=False, cant_lie=True),
    Card("Dreamer", Role.VILLAGER, Alignment.GOOD,
         "Pick 1 character: Learn an Evil role. If Evil picked, Learn its info.",
         lies=False, disguises=False, activated_ability=True),
    Card("Druid", Role.VILLAGER, Alignment.GOOD,
         "Pick 3 characters. Learn 1 random Outcast among them (if any).",
         lies=False, disguises=False, activated_ability=True),
    Card("Empress", Role.VILLAGER, Alignment.GOOD,
         "Learn 3 characters. Only 1 is Evil.",
         lies=False, disguises=False),
    Card("Enlightened", Role.VILLAGER, Alignment.GOOD,
         "Learn if closest Evil to me is Clockwise or Counter-Clockwise. Learn 'Equidistant' if Evils are at the same distance from me.",
         lies=False, disguises=False),
    Card("Fortune Teller", Role.VILLAGER, Alignment.GOOD,
         "Pick 2 characters. Learn if any of them is Evil.",
         lies=False, disguises=False, activated_ability=True),
    Card("Gemcrafter", Role.VILLAGER, Alignment.GOOD,
         "Learn 1 Good character.",
         lies=False, disguises=False),
    Card("Hunter", Role.VILLAGER, Alignment.GOOD,
         "Learn how far from me is the nearest Evil.",
         lies=False, disguises=False),
    Card("Jester", Role.VILLAGER, Alignment.GOOD,
         "Pick 3 characters. Learn how many of them are Evil.",
         lies=False, disguises=False, activated_ability=True),
    Card("Judge", Role.VILLAGER, Alignment.GOOD,
         "Pick 1 character. Learn if they're Lying.",
         lies=False, disguises=False, activated_ability=True),
    Card("Knight", Role.VILLAGER, Alignment.GOOD,
         "I can't die.",
         lies=False, disguises=False),
    Card("Knitter", Role.VILLAGER, Alignment.GOOD,
         "Learn how many Evils are adjacent to each other.",
         lies=False, disguises=False),
    Card("Lover", Role.VILLAGER, Alignment.GOOD,
         "Learn how many Evil characters I am adjacent to.",
         lies=False, disguises=False),
    Card("Medium", Role.VILLAGER, Alignment.GOOD,
         "Learn a Good character and its role.",
         lies=False, disguises=False),
    Card("Oracle", Role.VILLAGER, Alignment.GOOD,
         "Learn that 1 out of 2 characters is a specific Minion role.",
         lies=False, disguises=False),
    Card("Poet", Role.VILLAGER, Alignment.GOOD,
         "Learn random Info.",
         lies=False, disguises=False),
    Card("Scout", Role.VILLAGER, Alignment.GOOD,
         "Learn how far a specific Evil is to another closest Evil.",
         lies=False, disguises=False),
    Card("Slayer", Role.VILLAGER, Alignment.GOOD,
         "Pick 1 character. If Evil picked, Execute it.",
         lies=False, disguises=False, activated_ability=True),
    Card("Witness", Role.VILLAGER, Alignment.GOOD,
         "Learn a character that was affected by an Evil ability.",
         lies=False, disguises=False),
]

# ============================================================
# OUTCASTS (Good) - 5 cards
# ============================================================
OUTCASTS = [
    Card("Drunk", Role.OUTCAST, Alignment.GOOD,
         "I Disguise as a random not in play Villager. I am Corrupted and I Lie. I can not be Cured.",
         lies=True, disguises=True, corrupted_note=True),
    Card("Wretch", Role.OUTCAST, Alignment.GOOD,
         "I Register as a random Evil Minion to other characters.",
         lies=False, disguises=False),
    Card("Bombardier", Role.OUTCAST, Alignment.GOOD,
         "Lose if you Execute me.",
         lies=False, disguises=False),
    Card("Doppelganger", Role.OUTCAST, Alignment.GOOD,
         "Game Start: I Disguise as a Good Villager currently in play.",
         lies=False, disguises=True, game_start_ability=True),
    Card("Plague Doctor", Role.OUTCAST, Alignment.GOOD,
         "Game Start: 1 Good Villager is Corrupted. Pick 1 character. If it's Corrupted, learn an Evil character.",
         lies=False, disguises=False, game_start_ability=True, activated_ability=True, corrupted_note=True),
]

# ============================================================
# MINIONS (Evil) - 8 cards
# ============================================================
MINIONS = [
    Card("Chancellor", Role.MINION, Alignment.EVIL,
         "Game Start: One Villager becomes an Outcast role. I sit next to it. I Lie and Disguise.",
         lies=True, disguises=True, game_start_ability=True),
    Card("Witch", Role.MINION, Alignment.EVIL,
         "You can not Reveal the last card. I Lie and Disguise.",
         lies=True, disguises=True),
    Card("Minion", Role.MINION, Alignment.EVIL,
         "I Lie and Disguise.",
         lies=True, disguises=True),
    Card("Poisoner", Role.MINION, Alignment.EVIL,
         "Game Start: One adjacent Villager is Corrupted (if possible). I Lie and Disguise.",
         lies=True, disguises=True, game_start_ability=True, corrupted_note=True),
    Card("Twin Minion", Role.MINION, Alignment.EVIL,
         "I Lie and Disguise.",
         lies=True, disguises=True),
    Card("Shaman", Role.MINION, Alignment.EVIL,
         "Game Start: There are 2 same Villager roles in current Village. I Lie and Disguise.",
         lies=True, disguises=True, game_start_ability=True),
    Card("Puppeteer", Role.MINION, Alignment.EVIL,
         "Game Start: Random adjacent Villager becomes a Puppet if possible. It's Evil, but can not Lie. I Lie and Disguise.",
         lies=True, disguises=True, game_start_ability=True),
    Card("Puppet", Role.MINION, Alignment.EVIL,
         "I Disguise as a Villager, I don't Lie.",
         lies=False, disguises=True, cant_lie=True),
]

# ============================================================
# DEMONS (Evil) - 3 cards
# ============================================================
DEMONS = [
    Card("Baa", Role.DEMON, Alignment.EVIL,
         "One fake Outcast is added to the Deck view. I Lie and Disguise.",
         lies=True, disguises=True),
    Card("Pooka", Role.DEMON, Alignment.EVIL,
         "Game Start: Villagers adjacent to me are Corrupted (if possible). I Lie and Disguise.",
         lies=True, disguises=True, game_start_ability=True, corrupted_note=True),
    Card("Lilis", Role.DEMON, Alignment.EVIL,
         "At Night: Kill a random unrevealed character. Deal 2 damage to you. I Lie and Disguise.",
         lies=True, disguises=True),
]

ALL_CARDS = VILLAGERS + OUTCASTS + MINIONS + DEMONS

CARDS_BY_NAME = {card.name: card for card in ALL_CARDS}

_NORMALIZED_CARDS = {
    card.name.lower().replace(" ", "").replace("_", ""): card
    for card in ALL_CARDS
}

# Common abbreviations / aliases
_ALIASES = {
    "ft": "fortuneteller",
    "pd": "plaguedoctor",
    "bh": "bountyhunter",
    "tm": "twinminion",
}


def get_card(name: str) -> Card:
    """Look up a card by name (case-insensitive, space-insensitive)."""
    key = name.lower().replace(" ", "").replace("_", "")
    key = _ALIASES.get(key, key)
    return _NORMALIZED_CARDS.get(key)


if __name__ == "__main__":
    print(f"Total cards: {len(ALL_CARDS)}")
    print(f"  Villagers: {len(VILLAGERS)}")
    print(f"  Outcasts: {len(OUTCASTS)}")
    print(f"  Minions: {len(MINIONS)}")
    print(f"  Demons: {len(DEMONS)}")
    print()
    for card in ALL_CARDS:
        flags = []
        if card.lies:
            flags.append("LIES")
        if card.disguises:
            flags.append("DISGUISES")
        if card.cant_lie:
            flags.append("CANT_LIE")
        if card.game_start_ability:
            flags.append("GAME_START")
        if card.activated_ability:
            flags.append("ACTIVATED")
        print(f"  [{card.role.value}] {card.name}: {' '.join(flags)}")
