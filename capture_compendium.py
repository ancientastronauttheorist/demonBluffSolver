"""Capture all card details from the compendium page by clicking each card."""
import sys
import time
import pyautogui
from screenshot import capture

pyautogui.PAUSE = 0.15
pyautogui.FAILSAFE = True

# Card grid layout for 2560x1440 screen
GRID_LEFT = 180
GRID_RIGHT = 2380
COLS = 9
CARD_WIDTH = (GRID_RIGHT - GRID_LEFT) / COLS
ROW_Y = [300, 580, 860]  # 3 rows

PAGE1_CARDS = [
    # Row 1
    ["alchemist", "architect", "baker", "bard", "bishop", "confessor", "dreamer", "druid", "empress"],
    # Row 2
    ["enlightened", "fortune_teller", "gemcrafter", "hunter", "jester", "judge", "knight", "knitter", "lover"],
    # Row 3
    ["medium", "oracle", "poet", "scout", "slayer", "witness"],
]


def get_card_pos(row, col):
    x = int(GRID_LEFT + CARD_WIDTH * col + CARD_WIDTH / 2)
    y = ROW_Y[row]
    return x, y


def capture_page(cards, page_prefix):
    """Click each card and capture a screenshot."""
    for row_idx, row in enumerate(cards):
        for col_idx, name in enumerate(row):
            x, y = get_card_pos(row_idx, col_idx)
            print(f"Clicking {name} at ({x}, {y})...")
            pyautogui.click(x, y)
            time.sleep(0.8)
            capture(f"{page_prefix}_{name}")
            time.sleep(0.3)
    # Click somewhere neutral to close tooltip
    pyautogui.click(1280, 1200)
    time.sleep(0.3)


if __name__ == "__main__":
    print("Starting capture in 2 seconds...")
    time.sleep(2)
    capture_page(PAGE1_CARDS, "p1")
    print("Done! All page 1 cards captured.")
