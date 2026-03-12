# Demon Bluff Scoring Formula

Reverse-engineered from IL2CPP dump (`dump.cs`) + live memory reads via `memory_reader.py --score`.

## Formula

```
final_score = total_evil_kills × 50
            + total_unrevealed_cards × 10
            + completed_villages × ascension_level × 50
```

## Score Class (from IL2CPP dump)

```
class Score {
    int killedGoods;          // 0x10 — wrong executions
    int completedStages;      // 0x14 — villages completed
    int tempUnrevealedCards;  // 0x18 — unrevealed cards THIS village
    int unrevealedCards;      // 0x1C — total unrevealed across run
    int killedEvils;          // 0x20 — total evil kills
    int tempKilledEvils;      // 0x24 — evil kills THIS village
    int pointPerKill;         // 0x28 — 50 pts
    int pointsPerUnrevealed;  // 0x2C — 10 pts
    int pointsForCompleting;  // 0x30 — 100 pts (base; actual completion uses ascension × pointPerKill)
    int completedDays;        // 0x34 — completed villages
    int multiplier;           // 0x38 — unused in RoguelikeStandard mode
}
```

## Point Values

| Component | Points | Notes |
|---|---|---|
| Evil kill | +50 | Awarded immediately on execution |
| Unrevealed card | +10 | Per card NOT flipped during the village |
| Village completion | ascension × 50 | At Asc 36 = +1800 per village |

## Verification

| Run | Kills | Unrevealed | Villages | Ascension | Calculated | Displayed | Match |
|---|---|---|---|---|---|---|---|
| Asc 35 full run | 14 | 4 | 7 | 35 | 14×50 + 4×10 + 7×35×50 = 12990 | 12990 | Yes |
| Asc 36 after V1 | 3 | 3 | 1 | 36 | 3×50 + 3×10 + 1×36×50 = 1980 | 1980 | Yes |
| Asc 36 after V2 | 5 | 3 | 2 | 36 | 5×50 + 3×10 + 2×36×50 = 3880 | 3880 | Yes |

## Score Behavior

- `tempKilledEvils` / `tempUnrevealedCards` update mid-village, reset after completion
- `killedEvils` / `unrevealedCards` / `completedStages` accumulate at village completion
- All Score fields reset to 0 at ascension start
- Displayed score updates in real-time (floating "+50" popups on kill)

## Strategy Implications

At high ascensions, the village completion bonus dominates:

- Asc 36: completion = **1800 pts** vs unrevealed card = **10 pts**
- You'd need to skip **180 card reveals** to equal one village win
- A risky blind execution causing a loss costs ~1800+ pts

**Winning consistently >> skipping card reveals.**

## Reading Score from Memory

```
python memory_reader.py --score
```

Reads the Score object from Gameplay's static fields at offset 0x0. Shows point config, running totals, per-village temps, and estimated total.
