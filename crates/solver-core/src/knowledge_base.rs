/// Complete knowledge base of all 40 Demon Bluff cards.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Faction {
    Villager,
    Outcast,
    Minion,
    Demon,
}

impl Faction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Faction::Villager => "Villager",
            Faction::Outcast => "Outcast",
            Faction::Minion => "Minion",
            Faction::Demon => "Demon",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Faction> {
        match s.to_lowercase().as_str() {
            "villager" | "v" => Some(Faction::Villager),
            "outcast" | "o" => Some(Faction::Outcast),
            "minion" | "m" => Some(Faction::Minion),
            "demon" | "d" => Some(Faction::Demon),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Alignment {
    Good,
    Evil,
}

#[derive(Debug, Clone)]
pub struct Card {
    pub name: &'static str,
    pub faction: Faction,
    pub alignment: Alignment,
    pub lies: bool,
    pub disguises: bool,
    pub cant_lie: bool,
    pub game_start_ability: bool,
    pub activated_ability: bool,
    pub corrupted_note: bool,
}

/// All 40 cards in the game.
pub static ALL_CARDS: &[Card] = &[
    // ── Villagers (Good) ── 24 cards
    Card { name: "Alchemist",      faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Architect",      faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Baker",          faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Bard",           faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: true },
    Card { name: "Bishop",         faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Confessor",      faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: true,  game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Dreamer",        faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: true,  corrupted_note: false },
    Card { name: "Druid",          faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: true,  corrupted_note: false },
    Card { name: "Empress",        faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Enlightened",    faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Fortune Teller", faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: true,  corrupted_note: false },
    Card { name: "Gemcrafter",     faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Hunter",         faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Jester",         faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: true,  corrupted_note: false },
    Card { name: "Judge",          faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: true,  corrupted_note: false },
    Card { name: "Knight",         faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Knitter",        faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Lover",          faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Medium",         faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Oracle",         faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Poet",           faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Scout",          faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Slayer",         faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: true,  corrupted_note: false },
    Card { name: "Witness",        faction: Faction::Villager, alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },

    // ── Outcasts (Good) ── 6 cards
    Card { name: "Rambler",        faction: Faction::Outcast,  alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Drunk",          faction: Faction::Outcast,  alignment: Alignment::Good, lies: true,  disguises: true,  cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Wretch",         faction: Faction::Outcast,  alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Bombardier",     faction: Faction::Outcast,  alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Doppelganger",   faction: Faction::Outcast,  alignment: Alignment::Good, lies: false, disguises: true,  cant_lie: false, game_start_ability: true,  activated_ability: false, corrupted_note: false },
    Card { name: "Plague Doctor",  faction: Faction::Outcast,  alignment: Alignment::Good, lies: false, disguises: false, cant_lie: false, game_start_ability: true,  activated_ability: true,  corrupted_note: true },

    // ── Minions (Evil) ── 8 cards
    Card { name: "Chancellor",     faction: Faction::Minion,   alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: true,  activated_ability: false, corrupted_note: false },
    Card { name: "Witch",          faction: Faction::Minion,   alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Minion",         faction: Faction::Minion,   alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Poisoner",       faction: Faction::Minion,   alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: true,  activated_ability: false, corrupted_note: true },
    Card { name: "Twin Minion",    faction: Faction::Minion,   alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
    Card { name: "Shaman",         faction: Faction::Minion,   alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: true,  activated_ability: false, corrupted_note: false },
    Card { name: "Puppeteer",      faction: Faction::Minion,   alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: true,  activated_ability: false, corrupted_note: false },
    Card { name: "Puppet",         faction: Faction::Minion,   alignment: Alignment::Evil, lies: false, disguises: true,  cant_lie: true,  game_start_ability: false, activated_ability: false, corrupted_note: false },

    // ── Demons (Evil) ── 3 cards
    Card { name: "Baa",            faction: Faction::Demon,    alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: true,  activated_ability: false, corrupted_note: false },
    Card { name: "Pooka",          faction: Faction::Demon,    alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: true,  activated_ability: false, corrupted_note: true },
    Card { name: "Lilis",          faction: Faction::Demon,    alignment: Alignment::Evil, lies: true,  disguises: true,  cant_lie: false, game_start_ability: false, activated_ability: false, corrupted_note: false },
];

/// Look up a card by name (case-insensitive, space/underscore insensitive).
pub fn get_card(name: &str) -> Option<&'static Card> {
    let key = normalize_role(name);
    // Check aliases first
    let key = match key.as_str() {
        "ft" | "fortuneteller" => "fortuneteller",
        "pd" | "plaguedoctor" => "plaguedoctor",
        "bh" | "bountyhunter" => "bountyhunter",
        "tm" | "twinminion" => "twinminion",
        _ => key.as_str(),
    };
    ALL_CARDS.iter().find(|c| normalize_role(c.name) == key)
}

/// Normalize a role name: lowercase, strip spaces and underscores.
pub fn normalize_role(name: &str) -> String {
    name.to_lowercase()
        .replace(' ', "")
        .replace('_', "")
}

/// Check if a role name is a Villager in the knowledge base.
pub fn is_villager_role(name: &str) -> bool {
    get_card(name).map_or(false, |c| c.faction == Faction::Villager)
}

/// Check if a role name is an Outcast in the knowledge base.
pub fn is_outcast_role(name: &str) -> bool {
    get_card(name).map_or(false, |c| c.faction == Faction::Outcast)
}

/// Check if a role name refers to Plague Doctor (handles space/underscore variants).
pub fn is_plague_doctor(name: &str) -> bool {
    normalize_role(name) == "plaguedoctor"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_count() {
        assert_eq!(ALL_CARDS.len(), 41);
        assert_eq!(ALL_CARDS.iter().filter(|c| c.faction == Faction::Villager).count(), 24);
        assert_eq!(ALL_CARDS.iter().filter(|c| c.faction == Faction::Outcast).count(), 6);
        assert_eq!(ALL_CARDS.iter().filter(|c| c.faction == Faction::Minion).count(), 8);
        assert_eq!(ALL_CARDS.iter().filter(|c| c.faction == Faction::Demon).count(), 3);
    }

    #[test]
    fn test_lookup() {
        assert!(get_card("Enlightened").is_some());
        assert!(get_card("enlightened").is_some());
        assert!(get_card("Fortune Teller").is_some());
        assert!(get_card("Fortune_Teller").is_some());
        assert!(get_card("fortuneteller").is_some());
        assert!(get_card("ft").is_some());
        assert!(get_card("Plague_Doctor").is_some());
        assert!(get_card("pd").is_some());
        assert!(get_card("Twin Minion").is_some());
        assert!(get_card("tm").is_some());
        assert!(get_card("nonsense").is_none());
    }

    #[test]
    fn test_factions() {
        assert!(is_villager_role("Knight"));
        assert!(is_villager_role("Druid"));
        assert!(!is_villager_role("Drunk")); // Outcast, not Villager
        assert!(is_outcast_role("Drunk"));
        assert!(is_outcast_role("Wretch"));
        assert!(is_plague_doctor("Plague_Doctor"));
        assert!(is_plague_doctor("Plague Doctor"));
    }

    #[test]
    fn test_special_flags() {
        let confessor = get_card("Confessor").unwrap();
        assert!(confessor.cant_lie);
        assert!(!confessor.lies);

        let puppet = get_card("Puppet").unwrap();
        assert!(puppet.cant_lie);
        assert!(!puppet.lies);
        assert!(puppet.disguises);
        assert_eq!(puppet.alignment, Alignment::Evil);

        let wretch = get_card("Wretch").unwrap();
        assert_eq!(wretch.alignment, Alignment::Good);
        assert_eq!(wretch.faction, Faction::Outcast);
    }
}
