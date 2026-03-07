"""Game interaction utilities for Demon Bluff.

Wraps screenshot.py + mouse.py with game-awareness:
reliable focus, card detection, button finding, and common action sequences.
"""

import ctypes
import time
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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
# Card Position Formula
# ============================================================

def card_position(pos: int, n_cards: int,
                  center: tuple[int, int] = (1280, 690),
                  radius: int = 494) -> tuple[int, int]:
    """Compute the pixel coordinates of card `pos` in an N-card circle.

    Cards are arranged clockwise starting from top-center (12 o'clock).
    pos is 1-indexed (1 = top card, then clockwise).

    Verified to sub-pixel accuracy against template-matched facedown cards
    on a 2560x1440 screen with 9 cards.
    """
    import math
    angle = -math.pi / 2 + (pos - 1) * 2 * math.pi / n_cards
    x = round(center[0] + radius * math.cos(angle))
    y = round(center[1] + radius * math.sin(angle))
    return (x, y)


def all_card_positions(n_cards: int, **kwargs) -> list[tuple[int, int]]:
    """Return list of (x, y) for all card positions 1..n_cards."""
    return [card_position(p, n_cards, **kwargs) for p in range(1, n_cards + 1)]


def game_card_coords(card_num: int, n_cards: int, **kwargs) -> tuple[int, int]:
    """Get pixel coords for a game card number (as labeled in-game).

    The game labels card N at top (12 o'clock), then #1 clockwise from there.
    So game card c maps to circle position (c % N) + 1.
    """
    formula_pos = (card_num % n_cards) + 1
    return card_position(formula_pos, n_cards, **kwargs)


def all_game_card_coords(n_cards: int, **kwargs) -> dict[int, tuple[int, int]]:
    """Return dict mapping game card number -> (x, y) for all cards."""
    return {c: game_card_coords(c, n_cards, **kwargs) for c in range(1, n_cards + 1)}


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


def find_execute_button_center(screenshot_path: str) -> tuple[int, int] | None:
    """Find the large red execute button in the lower-right HUD.

    The generic bright-text detector is too easy to confuse with the nearby
    settings icons. This detector keys off the button's red ring instead.
    """
    from scipy import ndimage

    img = np.array(Image.open(screenshot_path).convert("RGB"))
    height, width = img.shape[:2]

    left = int(width * 0.72)
    top = int(height * 0.68)
    right = int(width * 0.95)
    bottom = int(height * 0.96)
    region = img[top:bottom, left:right]

    r = region[:, :, 0].astype(int)
    g = region[:, :, 1].astype(int)
    b = region[:, :, 2].astype(int)
    max_gb = np.maximum(g, b)

    # The execute button is the only large saturated red control in this area.
    mask = (r > 110) & (g < 95) & (b < 95) & ((r - max_gb) > 35)
    labeled, n_features = ndimage.label(mask)
    if n_features == 0:
        return None

    components = []
    for i in range(1, n_features + 1):
        component = labeled == i
        size = int(component.sum())
        if size < 200:
            continue
        cy, cx = ndimage.center_of_mass(component)
        ys, xs = np.where(component)
        components.append({
            "size": size,
            "cx": int(cx) + left,
            "cy": int(cy) + top,
            "bbox": (
                int(xs.min()) + left,
                int(ys.min()) + top,
                int(xs.max()) + left,
                int(ys.max()) + top,
            ),
        })

    if not components:
        return None

    best = max(components, key=lambda c: c["size"])

    # Merge nearby red clusters from the same circular button while excluding
    # the smaller red settings glyphs further down/right.
    cluster = [
        comp for comp in components
        if abs(comp["cx"] - best["cx"]) <= 120 and abs(comp["cy"] - best["cy"]) <= 120
    ]
    total = sum(comp["size"] for comp in cluster)
    cx = round(sum(comp["cx"] * comp["size"] for comp in cluster) / total)
    cy = round(sum(comp["cy"] * comp["size"] for comp in cluster) / total)
    print(f"[execute] Found execute button at ({cx}, {cy}) from {len(cluster)} red clusters")
    return cx, cy


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
# Deck Capture Helpers
# ============================================================

def _to_pil_image(image_or_path):
    if isinstance(image_or_path, Image.Image):
        return image_or_path
    return Image.open(image_or_path).convert("RGB")


def _deck_roi_bounds(size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return (
        int(width * 0.15),
        int(height * 0.12),
        int(width * 0.85),
        int(height * 0.80),
    )


def _deck_card_mask(image_or_path) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = _to_pil_image(image_or_path)
    img = np.array(image)
    left, top, right, bottom = _deck_roi_bounds(image.size)
    roi = img[top:bottom, left:right]

    rgb_max = roi.max(axis=2)
    rgb_min = roi.min(axis=2)
    spread = rgb_max - rgb_min

    # Deck cards stand out by having strong colored borders and labels.
    mask = (rgb_max > 70) & (spread > 35)
    return mask, (left, top, right, bottom)


def score_deck_frame(image_or_path) -> float:
    """Score how complete/readable a deck screenshot looks."""
    image = _to_pil_image(image_or_path)
    img = np.array(image)
    mask, (left, top, right, bottom) = _deck_card_mask(image)

    roi = img[top:bottom, left:right]
    luminance = roi.mean(axis=2)
    edge_energy = float(np.abs(np.diff(luminance, axis=0)).mean() +
                        np.abs(np.diff(luminance, axis=1)).mean())

    row_threshold = max(20, int(mask.shape[1] * 0.01))
    col_threshold = max(10, int(mask.shape[0] * 0.01))
    active_rows = int((mask.sum(axis=1) > row_threshold).sum())
    active_cols = int((mask.sum(axis=0) > col_threshold).sum())

    return float(mask.sum()) + active_rows * 250 + active_cols * 75 + edge_energy * 400


def deck_overlay_visible(image_or_path) -> bool:
    mask, _ = _deck_card_mask(image_or_path)
    visible_pixels = int(mask.sum())
    minimum_pixels = max(4000, mask.size // 450)
    return visible_pixels >= minimum_pixels


def find_deck_content_bounds(image_or_path, padding: int = 36) -> tuple[int, int, int, int]:
    """Find the deck panel bounds inside a full-screen deck screenshot."""
    image = _to_pil_image(image_or_path)
    mask, (left, top, right, bottom) = _deck_card_mask(image)

    row_threshold = max(20, int(mask.shape[1] * 0.01))
    col_threshold = max(10, int(mask.shape[0] * 0.01))
    rows = np.where(mask.sum(axis=1) > row_threshold)[0]
    cols = np.where(mask.sum(axis=0) > col_threshold)[0]

    if len(rows) == 0 or len(cols) == 0:
        return left, top, right, bottom

    width, height = image.size
    crop_left = max(0, left + int(cols[0]) - padding)
    crop_top = max(0, top + int(rows[0]) - padding)
    crop_right = min(width, left + int(cols[-1]) + padding + 1)
    crop_bottom = min(height, top + int(rows[-1]) + padding + 1)
    return crop_left, crop_top, crop_right, crop_bottom


def crop_deck_view(image_or_path, padding: int = 36) -> Image.Image:
    """Crop a full-screen deck screenshot down to the deck content."""
    image = _to_pil_image(image_or_path)
    bounds = find_deck_content_bounds(image, padding=padding)
    return image.crop(bounds)


def enhance_deck_crop(image_or_path, upscale: float = 1.35) -> Image.Image:
    """Make the deck crop easier to read during manual verification."""
    image = _to_pil_image(image_or_path)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = ImageEnhance.Brightness(image).enhance(1.12)
    image = image.filter(ImageFilter.SHARPEN)

    if upscale > 1.0:
        new_size = (
            int(round(image.size[0] * upscale)),
            int(round(image.size[1] * upscale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


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
    pos = find_execute_button_center(path)
    if pos:
        focus_game()
        mouse.click(pos[0], pos[1])
        time.sleep(0.5)
        print("[execute] Clicked execute button")
        return True
    # Fallback to a verified approximate center on 2560x1440.
    focus_game()
    mouse.click(2280, 1220)
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
    """Capture a robust deck screenshot while holding Tab.

    Writes a full-screen PNG plus an enhanced cropped PNG for easier review.
    Returns the full-screen screenshot path.
    """
    import pyautogui

    focus_game()
    time.sleep(0.3)

    best_image = None
    best_score = float("-inf")
    best_visible = False

    try:
        pyautogui.keyDown('tab')
        time.sleep(0.28)

        for frame_idx in range(5):
            if frame_idx:
                time.sleep(0.12)
            frame = screenshot.grab()
            score = score_deck_frame(frame)
            visible = deck_overlay_visible(frame)
            if (best_image is None or
                    visible and not best_visible or
                    (visible == best_visible and score > best_score)):
                best_image = frame.copy()
                best_score = score
                best_visible = visible
    finally:
        pyautogui.keyUp('tab')

    if best_image is None:
        best_image = screenshot.grab()

    path = screenshot.save_image(best_image, name=name, fmt="PNG")
    crop = enhance_deck_crop(crop_deck_view(best_image))
    crop_path = screenshot.save_image(crop, name=f"{name}_crop", fmt="PNG")
    print(f"[deck] Captured deck view: {path} (score={best_score:.0f}, visible={best_visible})")
    print(f"[deck] Saved enhanced crop: {crop_path}")
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
