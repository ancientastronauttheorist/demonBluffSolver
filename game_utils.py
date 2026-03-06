"""Game interaction utilities for Demon Bluff.

Wraps screenshot.py + mouse.py with game-awareness:
reliable focus, card detection, button finding, and common action sequences.
"""

import ctypes
import time
import os
import numpy as np
from PIL import Image

# Fix DPI scaling
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import screenshot
import mouse


# ============================================================
# Window Focus
# ============================================================

def is_game_focused(title_substring: str = "Demon Bluff") -> bool:
    """Check if the game window is currently in the foreground."""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    focused = title_substring.lower() in buf.value.lower()
    if not focused:
        try:
            print(f"[focus] Game NOT focused. Active window: '{buf.value}'")
        except UnicodeEncodeError:
            print(f"[focus] Game NOT focused. Active window: (non-ASCII title)")
    return focused


def ensure_game_focused(title_substring: str = "Demon Bluff") -> bool:
    """Check if game is focused; if not, focus it. Returns True if game is now focused."""
    if is_game_focused(title_substring):
        return True
    return focus_game(title_substring)


def focus_game(title_substring: str = "Demon Bluff") -> bool:
    """Bring the game window to foreground using Win32 API.
    Returns True if window was found and focused."""
    import ctypes.wintypes

    user32 = ctypes.windll.user32
    EnumWindows = user32.EnumWindows
    GetWindowTextW = user32.GetWindowTextW
    SetForegroundWindow = user32.SetForegroundWindow
    IsWindowVisible = user32.IsWindowVisible
    ShowWindow = user32.ShowWindow

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    found_hwnd = [None]

    def enum_callback(hwnd, lparam):
        if IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            GetWindowTextW(hwnd, buf, 256)
            if title_substring.lower() in buf.value.lower():
                found_hwnd[0] = hwnd
                return False  # Stop enumeration
        return True

    EnumWindows(WNDENUMPROC(enum_callback), 0)

    if found_hwnd[0]:
        SW_RESTORE = 9
        ShowWindow(found_hwnd[0], SW_RESTORE)
        SetForegroundWindow(found_hwnd[0])
        time.sleep(0.3)
        print(f"[focus] Focused: {title_substring}")
        return True
    else:
        print(f"[focus] Window not found: {title_substring}")
        return False


# ============================================================
# Card Position Detection
# ============================================================

def detect_card_positions(screenshot_path: str) -> list[tuple[int, int]]:
    """Detect card positions from a screenshot using orange region detection.

    Cards have orange/brown backs. Finds clusters of orange pixels,
    returns centers sorted clockwise from top.

    Returns list of (x, y) tuples in screenshot coordinates.
    """
    from scipy import ndimage

    img = np.array(Image.open(screenshot_path))
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Orange card detection: high red, medium green, low blue
    mask = (r > 150) & (g > 80) & (g < 170) & (b < 100)

    # Label connected components
    labeled, n_features = ndimage.label(mask)
    if n_features == 0:
        print("[cards] No orange regions found")
        return []

    # Get center of mass for each region, filter by size
    raw_centers = []
    for i in range(1, n_features + 1):
        region = (labeled == i)
        size = region.sum()
        if size < 500:  # Skip tiny noise
            continue
        cy, cx = ndimage.center_of_mass(region)
        raw_centers.append((int(cx), int(cy)))

    if not raw_centers:
        print("[cards] No card-sized regions found")
        return []

    # Merge nearby centroids (cards often split into 2 regions)
    MERGE_DIST = 120  # px — card halves are within this distance
    centers = []
    used = [False] * len(raw_centers)
    for i, (x1, y1) in enumerate(raw_centers):
        if used[i]:
            continue
        group_x, group_y, count = x1, y1, 1
        for j, (x2, y2) in enumerate(raw_centers):
            if j <= i or used[j]:
                continue
            if abs(x1 - x2) < MERGE_DIST and abs(y1 - y2) < MERGE_DIST:
                group_x += x2
                group_y += y2
                count += 1
                used[j] = True
        centers.append((group_x // count, group_y // count))
        used[i] = True

    if not centers:
        print("[cards] No card-sized regions found after merge")
        return []

    # Sort clockwise from top: compute angle from centroid
    avg_x = sum(c[0] for c in centers) / len(centers)
    avg_y = sum(c[1] for c in centers) / len(centers)

    import math
    def angle_from_top(point):
        dx = point[0] - avg_x
        dy = point[1] - avg_y
        # Angle from 12 o'clock, clockwise
        return (math.atan2(dx, -dy) + 2 * math.pi) % (2 * math.pi)

    centers.sort(key=angle_from_top)

    print(f"[cards] Found {len(centers)} cards: {centers}")
    return centers


# ============================================================
# Button Detection
# ============================================================

def find_button_in_region(screenshot_path: str,
                          left: int, top: int, right: int, bottom: int,
                          min_brightness: int = 200) -> tuple[int, int] | None:
    """Find a bright text/button cluster in a screen region.

    Looks for clusters of bright pixels (text/button highlights).
    Returns center (x, y) in full-screen coordinates, or None.
    """
    from scipy import ndimage

    img = np.array(Image.open(screenshot_path))
    region = img[top:bottom, left:right]

    # Detect bright pixels (white/yellow text on darker background)
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    brightness = (r.astype(int) + g.astype(int) + b.astype(int)) / 3
    mask = brightness > min_brightness

    labeled, n_features = ndimage.label(mask)
    if n_features == 0:
        return None

    # Find largest bright cluster
    best_size = 0
    best_center = None
    for i in range(1, n_features + 1):
        region_mask = (labeled == i)
        size = region_mask.sum()
        if size > best_size:
            best_size = size
            cy, cx = ndimage.center_of_mass(region_mask)
            best_center = (int(cx) + left, int(cy) + top)

    if best_center and best_size > 50:
        print(f"[button] Found at {best_center} (size={best_size})")
        return best_center

    return None


def find_and_click_button(screenshot_path: str,
                          left: int, top: int, right: int, bottom: int,
                          min_brightness: int = 200) -> bool:
    """Find a button in region and click it. Returns True if found."""
    pos = find_button_in_region(screenshot_path, left, top, right, bottom,
                                min_brightness)
    if pos:
        focus_game()
        mouse.click(pos[0], pos[1])
        return True
    print("[button] No button found in region")
    return False


# ============================================================
# Common Game Actions
# ============================================================

def reveal_card(card_pos: tuple[int, int], wait: float = 1.5):
    """Focus game and click a card to reveal it."""
    focus_game()
    mouse.click(card_pos[0], card_pos[1])
    time.sleep(wait)  # Wait for reveal animation
    print(f"[reveal] Clicked card at {card_pos}")


def click_execute_button():
    """Click the Execute button (red sword, bottom-right corner).
    Takes a screenshot first to find it."""
    path = screenshot.capture("_exec_detect")
    # Execute button is typically in bottom-right quadrant
    pos = find_button_in_region(path, 2200, 1200, 2500, 1400, min_brightness=150)
    if pos:
        focus_game()
        mouse.click(pos[0], pos[1])
        time.sleep(0.5)
        print("[execute] Clicked execute button")
        return True
    # Fallback to known approximate position
    focus_game()
    mouse.click(2380, 1330)
    time.sleep(0.5)
    print("[execute] Clicked execute button (fallback position)")
    return True


def execute_card(card_pos: tuple[int, int], wait: float = 2.0):
    """Execute a card: click execute button, then click the card."""
    click_execute_button()
    time.sleep(0.5)
    mouse.click(card_pos[0], card_pos[1])
    time.sleep(wait)  # Wait for execution animation
    print(f"[execute] Executed card at {card_pos}")


def hold_tab_screenshot(name: str = "deck_view") -> str:
    """Hold Tab to show deck view, capture screenshot, release Tab.
    Returns the screenshot path."""
    import pyautogui

    focus_game()
    time.sleep(0.3)
    pyautogui.keyDown('tab')
    time.sleep(0.5)  # Wait for deck view to appear
    path = screenshot.capture(name)
    pyautogui.keyUp('tab')
    print(f"[deck] Captured deck view: {path}")
    return path


def press_escape():
    """Press Escape (opens/closes pause menu)."""
    import pyautogui
    focus_game()
    pyautogui.press('escape')
    time.sleep(0.3)


def click_next_button(screenshot_path: str = None) -> bool:
    """Find and click the 'Next' button in dialog/victory screens.
    Searches bottom-center area of screen."""
    if screenshot_path is None:
        screenshot_path = screenshot.capture("_next_detect")
    # Next button typically in bottom-center area, y~800-900
    return find_and_click_button(screenshot_path, 1100, 750, 1500, 950)


def take_game_screenshot(name: str = None) -> str:
    """Take a screenshot (doesn't require game focus — mss works regardless)."""
    return screenshot.capture(name)


# ============================================================
# Hover + Verify Pattern
# ============================================================

def hover_verify_click(x: int, y: int, verify_name: str = "_hover_verify"):
    """Move mouse to position, screenshot to verify, then click.
    This is the safe interaction pattern: hover → screenshot → click."""
    focus_game()
    mouse.move(x, y)
    time.sleep(0.3)
    path = screenshot.capture(verify_name)
    mouse.click(x, y)
    return path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: game_utils.py [focus|cards|deck|screenshot]")
        print("  focus     — focus the game window")
        print("  cards     — detect card positions from latest screenshot")
        print("  deck      — capture deck view (hold Tab)")
        print("  screenshot — take a screenshot")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "check":
        is_game_focused()
    elif cmd == "focus":
        focus_game()
    elif cmd == "cards":
        if len(sys.argv) > 2:
            path = sys.argv[2]
        else:
            path = take_game_screenshot("card_detect")
        detect_card_positions(path)
    elif cmd == "deck":
        hold_tab_screenshot()
    elif cmd == "screenshot":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        take_game_screenshot(name)
    else:
        print(f"Unknown command: {cmd}")
