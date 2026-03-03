"""Capture Minion cards from page 4."""
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

MINIONS = ["chancellor", "witch", "minion", "poisoner", "twin_minion", "shaman", "puppeteer", "puppet"]

def get_card_pos(row, col):
    x = int(GRID_LEFT + CARD_WIDTH * col + CARD_WIDTH / 2)
    y = ROW_Y[row]
    return x, y

if __name__ == "__main__":
    print("Starting in 2 seconds...")
    time.sleep(2)
    for i, name in enumerate(MINIONS):
        row = i // 9
        col = i % 9
        x, y = get_card_pos(row, col)
        print(f"Clicking {name} at ({x}, {y})...")
        pyautogui.click(x, y)
        time.sleep(0.8)
        capture(f"p4_{name}")
        time.sleep(0.3)
    pyautogui.click(1280, 1200)
    print("Done!")
