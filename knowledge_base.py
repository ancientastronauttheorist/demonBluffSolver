"""Complete knowledge base of all Demon Bluff cards, learned from the game compendium."""

from dataclasses import dataclass, field
from enum import Enum


# Wrong-execution HP cost per role. Default is 5 (Asc4+).
# Confirmed empirically:
#   Drunk: 2 (asc78_v6, 2026-04-21 — wrong exec cost -2 not -5)
# Uncertain / not codified:
#   Lilis: historical comment claimed Lilis=2, but Lilis is evil so
#   wrong-executing her doesn't cleanly apply. Not added here.
WRONG_EXEC_COST_OVERRIDES: dict[str, int] = {
    "Drunk": 2,
}
DEFAULT_WRONG_EXEC_COST = 5
KNIGHT_BLUFF_EXTRA_DAMAGE = 4


def wrong_exec_cost_for(role_name: str | None, default: int | None = None) -> int:
    """Return the HP cost for a wrong execution targeting `role_name`.

    If role_name is None or not in the override table, returns `default`
    (or DEFAULT_WRONG_EXEC_COST if default is None).
    """
    fallback = default if default is not None else DEFAULT_WRONG_EXEC_COST
    if role_name is None:
        return fallback
    return WRONG_EXEC_COST_OVERRIDES.get(role_name, fallback)


def execution_cost_for(
    role_name: str | None,
    *,
    apparent_role: str | None = None,
    was_evil: bool = False,
    was_corrupted: bool = False,
    was_killable: bool = False,
    execution_blocked: bool = False,
    default: int | None = None,
) -> int:
    """Return total HP damage for an observed execution result.

    ``wrong_exec_cost_for`` remains the base cost for the target's true role.
    A successfully killed good target showing as Knight fires the Knight bluff's
    separate 4-HP execution effect only when it has the active Corrupted status.
    Execution bookkeeping reports Drunk as clean, so callers resolving damage
    must pass the underlying active-status value separately. Plague Doctor is
    different: its native callback reads active Corrupted directly.
    This distinguishes an ordinary Corrupted Drunk-as-Knight (2 + 4) from a
    Chancellor-generated Drunk whose inherited Alchemist resistance blocked
    that status (2 only).
    Correct evil executions and protected/blocked attempts cost no HP.

    ``was_killable`` is deliberately an observed/post-action input.  Callers
    must not use hidden identity or status to predict whether a target is safe
    to execute.
    """
    if was_evil or execution_blocked:
        return 0

    cost = wrong_exec_cost_for(role_name, default=default)
    apparent_key = (apparent_role or "").strip().replace("_", " ").casefold()
    if (was_killable and apparent_key == "knight"
            and was_corrupted):
        cost += KNIGHT_BLUFF_EXTRA_DAMAGE
    return cost


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
         "I am immune to Corruption. Villagers to the left and right of me [Range 2] are cured from Corruption. "
         "Learn how many Corrupted characters were around me [Range 2] at the start of the Round (before the Cure).",
         lies=False, disguises=False, corrupted_note=False),
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
         "Pick 2 characters. Learn 2 roles; at least one is among them. Wretch yields Cabbage.",
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
# OUTCASTS (Good) - 6 cards
# ============================================================
OUTCASTS = [
    Card("Rambler", Role.OUTCAST, Alignment.GOOD,
         "I tell you something really interesting. Adjacent Truthful characters tell me to shut up "
         "instead of sharing their info.",
         lies=False, disguises=False),
    Card("Drunk", Role.OUTCAST, Alignment.GOOD,
         "I Disguise as a random not in play Villager. I Lie and attempt to add Corrupted to myself at Start; exact resistance can block that status.",
         lies=True, disguises=True, corrupted_note=False),
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
         "Truthful Start: corrupt 1 eligible apparent Villager. Pick 1 character; if it is Corrupted, learn an apparent Evil character.",
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
         "Game Start: Hide one existing Outcast identity in the Deck view. I Lie and Disguise.",
         lies=True, disguises=True, game_start_ability=True),
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
