"""Mouse control utility for interacting with the game.

Automatically applies calibration transform if calibration.json exists.
Run `python calibrate.py` to generate the calibration file.
"""
import json
import os
import sys
import time
import ctypes
import pyautogui

# Fix DPI scaling on Windows — without this, coordinates are wrong!
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# Safety: don't move too fast, allow abort by moving mouse to corner
# Below 0.2s causes double-clicks, missed clicks, focus races (see AUTOMATION_PLAN.md)
DEFAULT_PAUSE = 0.3
_MIN_PAUSE = 0.15

# Only set PAUSE if it hasn't been explicitly configured by another module.
# pyautogui's built-in default is 0.1; if it's still that, apply our safe default.
if pyautogui.PAUSE == 0.1:
    pyautogui.PAUSE = DEFAULT_PAUSE

pyautogui.FAILSAFE = True


def set_pause(value):
    """Set the pyautogui inter-action pause duration.

    Args:
        value: Pause in seconds. Must be >= 0.15. Values < 0.2 trigger a warning
               because they risk double-clicks, missed clicks, and focus races.
    """
    global DEFAULT_PAUSE
    if value < _MIN_PAUSE:
        raise ValueError(
            f"PAUSE value {value}s is below minimum {_MIN_PAUSE}s — "
            "risk of double-clicks and missed clicks"
        )
    if value < 0.2:
        import warnings
        warnings.warn(
            f"PAUSE value {value}s is below 0.2s — may cause double-clicks, "
            "missed clicks, or focus races",
            stacklevel=2,
        )
    DEFAULT_PAUSE = value
    pyautogui.PAUSE = value

# Calibration state (lazy-loaded)
_calibration = None
_calibration_loaded = False
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")


def _load_calibration():
    """Lazy-load calibration.json. Only reads file once."""
    global _calibration, _calibration_loaded
    if _calibration_loaded:
        return _calibration
    _calibration_loaded = True
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE) as f:
                _calibration = json.load(f)
            print(f"[calibration] Loaded (max residual: {_calibration['max_residual_px']:.1f} px)")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[calibration] WARNING: Failed to load {CALIBRATION_FILE}: {e}")
            _calibration = None
    return _calibration


def _transform(x, y):
    """Apply inverse calibration transform: screenshot coords → pyautogui coords.

    If no calibration is loaded, returns coordinates unchanged.
    """
    cal = _load_calibration()
    if cal is None:
        return int(round(x)), int(round(y))
    inv = cal["inverse"]
    tx = inv["scale_x"] * x + inv["offset_x"]
    ty = inv["scale_y"] * y + inv["offset_y"]
    return int(round(tx)), int(round(ty))


def _fmt(orig_x, orig_y, trans_x, trans_y):
    """Format coordinate debug string, only showing transform if it changed."""
    if orig_x == trans_x and orig_y == trans_y:
        return f"({orig_x}, {orig_y})"
    return f"({orig_x}, {orig_y})→({trans_x}, {trans_y})"


def click(x, y, button="left"):
    """Click at position (x, y). Coords are in screenshot space."""
    tx, ty = _transform(x, y)
    pyautogui.click(tx, ty, button=button)
    print(f"Clicked {_fmt(x, y, tx, ty)} {button}")


def move(x, y):
    """Move mouse to position (x, y). Coords are in screenshot space."""
    tx, ty = _transform(x, y)
    pyautogui.moveTo(tx, ty)
    print(f"Moved to {_fmt(x, y, tx, ty)}")


def right_click(x, y):
    """Right-click at position (x, y). Coords are in screenshot space."""
    tx, ty = _transform(x, y)
    pyautogui.click(tx, ty, button="right")
    print(f"Right-clicked {_fmt(x, y, tx, ty)}")


def double_click(x, y):
    """Double-click at position (x, y). Coords are in screenshot space."""
    tx, ty = _transform(x, y)
    pyautogui.doubleClick(tx, ty)
    print(f"Double-clicked {_fmt(x, y, tx, ty)}")


def drag(x1, y1, x2, y2, duration=0.5):
    """Drag from (x1, y1) to (x2, y2). Coords are in screenshot space."""
    tx1, ty1 = _transform(x1, y1)
    tx2, ty2 = _transform(x2, y2)
    pyautogui.moveTo(tx1, ty1)
    pyautogui.drag(tx2 - tx1, ty2 - ty1, duration=duration)
    print(f"Dragged {_fmt(x1, y1, tx1, ty1)} -> {_fmt(x2, y2, tx2, ty2)}")


def scroll(amount, x=None, y=None):
    """Scroll up (positive) or down (negative). Coords are in screenshot space."""
    if x is not None and y is not None:
        tx, ty = _transform(x, y)
        pyautogui.moveTo(tx, ty)
    pyautogui.scroll(amount)
    print(f"Scrolled {amount}")


def get_position():
    """Print current mouse position."""
    pos = pyautogui.position()
    print(f"{pos.x},{pos.y}")
    return pos


def get_screen_size():
    """Print screen dimensions."""
    size = pyautogui.size()
    print(f"{size.width}x{size.height}")
    return size


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mouse.py [click|rclick|move|pos|size|scroll] [args...]")
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "click":
        click(int(args[0]), int(args[1]))
    elif cmd == "rclick":
        right_click(int(args[0]), int(args[1]))
    elif cmd == "move":
        move(int(args[0]), int(args[1]))
    elif cmd == "pos":
        get_position()
    elif cmd == "size":
        get_screen_size()
    elif cmd == "scroll":
        amt = int(args[0])
        x = int(args[1]) if len(args) > 2 else None
        y = int(args[2]) if len(args) > 2 else None
        scroll(amt, x, y)
    elif cmd == "dclick":
        double_click(int(args[0]), int(args[1]))
    else:
        print(f"Unknown command: {cmd}")
