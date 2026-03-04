"""UI element detection and interaction for Demon Bluff.

Finds dialog buttons (Close, Next, etc.), menu items, and other clickable
elements by analyzing screenshot pixel data. No OCR dependency — uses
white-text-on-dark-background heuristics tuned to the game's UI style.

Usage:
    python ui.py scan                  — screenshot + list all detected buttons
    python ui.py click close           — find and click the Close button
    python ui.py click next            — find and click the Next button
    python ui.py click <index>         — click button by index from scan
    python ui.py hover close           — hover without clicking (verify aim)
    python ui.py hover <index>         — hover by index
    python ui.py debug                 — save annotated screenshot showing detections
"""

import sys
import os
import time
import numpy as np
from PIL import Image, ImageDraw
from dataclasses import dataclass

import screenshot
import mouse
import game_utils


# ============================================================
# Data Structures
# ============================================================

@dataclass
class TextRegion:
    """A detected horizontal band of text."""
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    pixel_count: int  # number of bright pixels

    @property
    def cx(self) -> int:
        return (self.x_min + self.x_max) // 2

    @property
    def cy(self) -> int:
        return (self.y_min + self.y_max) // 2

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass
class UIButton:
    """A detected clickable button."""
    x: int          # click target x
    y: int          # click target y
    label: str      # best-guess label
    confidence: float
    region: TextRegion


# ============================================================
# Text Line Detection
# ============================================================

def _find_text_lines(arr: np.ndarray,
                     x_min: int, x_max: int,
                     y_min: int, y_max: int,
                     min_brightness: int = 200,
                     min_line_pixels: int = 10,
                     gap_threshold: int = 8) -> list[TextRegion]:
    """Find horizontal text lines (white pixels on dark background) in a region.

    Scans row by row for bright pixels, groups consecutive rows into text lines
    separated by gaps of `gap_threshold` or more dark rows.
    """
    crop = arr[y_min:y_max, x_min:x_max]
    r, g, b = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
    bright = ((r + g + b) / 3 > min_brightness)

    # Per-row: count bright pixels + find x extent
    lines = []
    current_start = None
    current_rows = []

    for row_idx in range(bright.shape[0]):
        row_bright = bright[row_idx]
        count = int(row_bright.sum())

        if count >= min_line_pixels:
            if current_start is None:
                current_start = row_idx
            current_rows.append((row_idx, row_bright))
        else:
            # Gap row — close current line if gap is big enough
            if current_start is not None:
                gap = row_idx - current_rows[-1][0]
                if gap >= gap_threshold:
                    lines.append(_finalize_line(current_rows, x_min, y_min))
                    current_start = None
                    current_rows = []
                # else: small gap within same text line, continue

    # Close last line
    if current_start is not None and current_rows:
        lines.append(_finalize_line(current_rows, x_min, y_min))

    return [l for l in lines if l is not None]


def _finalize_line(rows: list, x_offset: int, y_offset: int) -> TextRegion:
    """Convert accumulated bright rows into a TextRegion."""
    if not rows:
        return None

    all_x_mins = []
    all_x_maxs = []
    total_pixels = 0

    for _, row_bright in rows:
        indices = np.where(row_bright)[0]
        if len(indices) > 0:
            all_x_mins.append(int(indices.min()))
            all_x_maxs.append(int(indices.max()))
            total_pixels += len(indices)

    if not all_x_mins:
        return None

    y_start = rows[0][0]
    y_end = rows[-1][0]

    return TextRegion(
        x_min=min(all_x_mins) + x_offset,
        x_max=max(all_x_maxs) + x_offset,
        y_min=y_start + y_offset,
        y_max=y_end + y_offset,
        pixel_count=total_pixels,
    )


# ============================================================
# Dialog Detection
# ============================================================

def _find_dialog_region(arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find a dialog box (dark rectangle with border) in center of screen.

    Returns (x_min, y_min, x_max, y_max) or None.
    """
    h, w = arr.shape[:2]

    # Dialog is typically centered, occupying ~30-60% of screen width
    # and ~20-40% of screen height
    # Look for a region where the center strip is very dark with bright border
    center_x = w // 2

    # Scan vertically down center column for dark region
    center_col = arr[:, center_x]
    brightness = center_col.mean(axis=1)

    # Find runs of dark pixels (< 40 brightness) in center
    dark = brightness < 40
    in_dark = False
    dark_runs = []
    start = 0

    for i in range(h):
        if dark[i] and not in_dark:
            start = i
            in_dark = True
        elif not dark[i] and in_dark:
            if i - start > 100:  # Minimum dialog height
                dark_runs.append((start, i))
            in_dark = False

    if in_dark and h - start > 100:
        dark_runs.append((start, h))

    if not dark_runs:
        return None

    # Pick the largest dark run in the center area
    best = max(dark_runs, key=lambda r: r[1] - r[0])
    y_min, y_max = best

    # Now find horizontal extent by scanning the middle row
    mid_y = (y_min + y_max) // 2
    mid_row = arr[mid_y].mean(axis=1)
    dark_h = mid_row < 40

    # Find the dark run containing center_x
    left = center_x
    while left > 0 and dark_h[left - 1]:
        left -= 1
    right = center_x
    while right < w - 1 and dark_h[right + 1]:
        right += 1

    if right - left < 200:  # Too narrow for a dialog
        return None

    return (left, y_min, right, y_max)


# ============================================================
# Button Identification
# ============================================================

def find_dialog_buttons(screenshot_path: str = None) -> list[UIButton]:
    """Detect interactive buttons on screen.

    Strategy: find all white text regions in the center band, then classify
    each as body text vs button. Buttons in Demon Bluff are:
    - Narrow (< 400px wide, e.g. "Close", "Next", "OK")
    - Centered horizontally on screen
    - Below or separated from wider body text

    Returns only the detected buttons, sorted top to bottom.
    """
    if screenshot_path is None:
        screenshot_path = screenshot.capture("_ui_detect")

    arr = np.array(Image.open(screenshot_path))
    h, w = arr.shape[:2]
    screen_cx = w // 2

    # Find all text lines in the center 70% of the screen
    margin_x = w // 6
    lines = _find_text_lines(
        arr,
        x_min=margin_x, x_max=w - margin_x,
        y_min=50, y_max=h - 50,
        min_brightness=200, min_line_pixels=8, gap_threshold=6,
    )

    if not lines:
        return []

    # Filter noise: discard tiny lines (artifacts, cursor, stray pixels)
    lines = [l for l in lines if l.height >= 10 and l.pixel_count >= 30]

    if not lines:
        return []

    # Sort by vertical position
    lines.sort(key=lambda l: l.cy)

    # First pass: classify each line by shape
    BUTTON_MAX_WIDTH = 500
    CENTER_TOLERANCE = 300

    classified = []  # (line, is_narrow_centered)
    for line in lines:
        is_narrow = line.width < BUTTON_MAX_WIDTH
        is_centered = abs(line.cx - screen_cx) < CENTER_TOLERANCE
        is_reasonable_height = 15 <= line.height <= 60
        classified.append((line, is_narrow and is_centered and is_reasonable_height))

    # Second pass: a narrow centered line is only a "dialog_button" if it has
    # wider body text above it within 150px (i.e. it's actually in a dialog,
    # not a stray card label or HUD element)
    BODY_PROXIMITY = 200  # max px gap between body text and button

    body_lines = [line for line, narrow in classified if not narrow]

    buttons = []
    for line, is_narrow in classified:
        if is_narrow:
            # Check if there's body text above within proximity
            has_body_above = any(
                0 < (line.cy - bl.cy) < BODY_PROXIMITY
                for bl in body_lines
            )
            if has_body_above:
                label = "dialog_button"
            else:
                label = "element"  # narrow but orphaned — probably HUD/label
        else:
            label = "text"

        buttons.append(UIButton(
            x=line.cx, y=line.cy,
            label=label,
            confidence=0.9 if label == "dialog_button" else 0.3,
            region=line,
        ))

    return buttons


def _find_screen_buttons(arr: np.ndarray, screenshot_path: str) -> list[UIButton]:
    """Find buttons when there's no dialog (e.g., menu screen)."""
    h, w = arr.shape[:2]

    # Look for red-highlighted text (hovered menu items)
    # and white text in center column
    lines = _find_text_lines(
        arr,
        x_min=w // 4, x_max=3 * w // 4,
        y_min=h // 4, y_max=3 * h // 4,
        min_brightness=200, min_line_pixels=15, gap_threshold=10,
    )

    buttons = []
    for line in lines:
        label = _guess_label(line, w)
        buttons.append(_make_button(line, label))

    return buttons


def _make_button(region: TextRegion, label: str) -> UIButton:
    """Create a UIButton from a TextRegion."""
    return UIButton(
        x=region.cx,
        y=region.cy,
        label=label,
        confidence=0.8,
        region=region,
    )


def _guess_label(region: TextRegion, screen_width: int) -> str:
    """Heuristic label based on position and size."""
    # Centered narrow text near bottom of dialog = Close/Next/OK
    centered = abs(region.cx - screen_width // 2) < 200
    narrow = region.width < 300

    if centered and narrow:
        return "dialog_button"
    elif centered:
        return "text"
    else:
        return "element"


# ============================================================
# Actions
# ============================================================

def scan(screenshot_path: str = None) -> list[UIButton]:
    """Take screenshot, find all buttons, print them."""
    if screenshot_path is None:
        screenshot_path = screenshot.capture("_ui_scan")

    buttons = find_dialog_buttons(screenshot_path)

    if not buttons:
        print("No buttons detected.")
        return []

    print(f"\nDetected {len(buttons)} button(s):")
    for i, btn in enumerate(buttons):
        region = btn.region
        print(f"  [{i}] ({btn.x}, {btn.y}) "
              f"label={btn.label} "
              f"bounds=({region.x_min},{region.y_min})-({region.x_max},{region.y_max}) "
              f"size={region.width}x{region.height}")

    return buttons


def hover(target: str, screenshot_path: str = None) -> bool:
    """Find button matching target and hover over it.

    target: "close", "next", "ok", or an integer index.
    """
    buttons = find_dialog_buttons(screenshot_path)
    btn = _resolve_target(target, buttons)
    if btn is None:
        return False

    game_utils.focus_game()
    mouse.move(btn.x, btn.y)
    print(f"Hovering over ({btn.x}, {btn.y}) [{btn.label}]")
    return True


def click(target: str, screenshot_path: str = None) -> bool:
    """Find button matching target and click it.

    target: "close", "next", "ok", or an integer index.
    """
    buttons = find_dialog_buttons(screenshot_path)
    btn = _resolve_target(target, buttons)
    if btn is None:
        return False

    game_utils.focus_game()
    mouse.click(btn.x, btn.y)
    print(f"Clicked ({btn.x}, {btn.y}) [{btn.label}]")
    return True


def _resolve_target(target: str, buttons: list[UIButton]) -> UIButton | None:
    """Find button matching the target string."""
    if not buttons:
        print("No buttons detected to interact with.")
        return None

    # Try numeric index
    try:
        idx = int(target)
        if 0 <= idx < len(buttons):
            return buttons[idx]
        print(f"Index {idx} out of range (0-{len(buttons)-1})")
        return None
    except ValueError:
        pass

    # For named targets, find the best dialog_button match
    # Prefer the bottommost dialog_button (Close/Next are below body text)
    dialog_buttons = [b for b in buttons if b.label == "dialog_button"]

    target_lower = target.lower()
    if target_lower in ("close", "next", "ok", "continue", "back", "button"):
        if dialog_buttons:
            # Return bottommost dialog button
            return max(dialog_buttons, key=lambda b: b.y)
        # Fallback: return last item
        return buttons[-1]

    # Fuzzy: just return last dialog button or last item
    if dialog_buttons:
        return dialog_buttons[-1]
    print(f"Unknown target '{target}', using last detected element")
    return buttons[-1]


def debug(screenshot_path: str = None):
    """Save annotated screenshot showing all detections."""
    if screenshot_path is None:
        screenshot_path = screenshot.capture("_ui_debug")

    arr = np.array(Image.open(screenshot_path))
    img = Image.open(screenshot_path).copy()
    draw = ImageDraw.Draw(img)

    # Draw dialog region
    dialog = _find_dialog_region(arr)
    if dialog:
        dx_min, dy_min, dx_max, dy_max = dialog
        draw.rectangle([dx_min, dy_min, dx_max, dy_max], outline="cyan", width=2)

    # Draw detected buttons
    buttons = find_dialog_buttons(screenshot_path)
    for i, btn in enumerate(buttons):
        r = btn.region
        draw.rectangle([r.x_min, r.y_min, r.x_max, r.y_max], outline="lime", width=2)
        draw.ellipse([btn.x - 5, btn.y - 5, btn.x + 5, btn.y + 5], fill="red")
        draw.text((r.x_max + 5, r.y_min), f"[{i}] {btn.label}", fill="lime")

    out_path = screenshot_path.replace(".png", "_debug.png")
    img.save(out_path)
    print(f"Debug image saved: {out_path}")
    return out_path


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python ui.py <command> [args]")
        print()
        print("Commands:")
        print("  scan              — detect and list all buttons")
        print("  click <target>    — click button (close/next/ok or index)")
        print("  hover <target>    — hover over button")
        print("  debug             — save annotated debug image")
        print()
        print("Examples:")
        print("  python ui.py scan")
        print("  python ui.py click close")
        print("  python ui.py click 0")
        print("  python ui.py hover next")
        return

    cmd = sys.argv[1].lower()

    if cmd == "scan":
        scan()
    elif cmd == "click":
        target = sys.argv[2] if len(sys.argv) > 2 else "button"
        click(target)
    elif cmd == "hover":
        target = sys.argv[2] if len(sys.argv) > 2 else "button"
        hover(target)
    elif cmd == "debug":
        debug()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
