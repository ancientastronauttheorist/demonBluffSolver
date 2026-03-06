# Solver TODOs

## Bishop validator improvement
The Bishop validator currently only checks "at least one target is Evil if truthful." It could be much stronger by parsing and validating the full type breakdown from the game text (e.g., "1 Villager and 1 Minion"). This would let us check:
- Whether the claimed Villager/Outcast/Minion/Demon counts match the actual types at the target positions
- A lying Bishop's type breakdown must NOT match reality
- "Between #X, #Y" means positions X and Y themselves (NOT the range between them)

## Witness "no affected" auto-detection
Currently requires manual `confirm_good`. Solver should automatically detect when evil roles have no character-affecting abilities (Minion, Baa, Witch) and Witness says "no one affected" — this confirms Witness is Good and not corrupted.

## Gemcrafter reliability
Gemcrafter "Learn 1 Good character" has shown inconsistent behavior with disguises (see MEMORY.md). Need more data to determine exact mechanic before trusting solver's Gemcrafter validator.
