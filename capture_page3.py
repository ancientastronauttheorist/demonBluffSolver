"""Capture Outcast cards from page 3."""
import time
import pyautogui
from screenshot import capture

pyautogui.PAUSE = 0.15
pyautogui.FAILSAFE = True

GRID_LEFT = 180
GRID_RIGHT = 2380
COLS = 9
CARD_WIDTH = (GRID_RIGHT - GRID_LEFT) / COLS
ROW_Y = [300, 580, 860]

OUTCASTS = ["drunk", "wretch", "bombardier", "doppelganger", "plague_doctor"]

def get_card_pos(row, col):
    x = int(GRID_LEFT + CARD_WIDTH * col + CARD_WIDTH / 2)
    y = ROW_Y[row]
    return x, y

if __name__ == "__main__":
    print("Starting in 2 seconds...")
    time.sleep(2)
    for i, name in enumerate(OUTCASTS):
        x, y = get_card_pos(0, i)
        print(f"Clicking {name} at ({x}, {y})...")
        pyautogui.click(x, y)
        time.sleep(0.8)
        capture(f"p3_{name}")
        time.sleep(0.3)
    pyautogui.click(1280, 1200)
    print("Done!")
