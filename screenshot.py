"""Quick screenshot utility. Captures the screen and saves to screenshots/ folder."""
import sys
import os
import time
import ctypes

# Fix DPI scaling on Windows — without this, coordinates are wrong!
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import mss
from PIL import Image

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def capture(name=None, region=None):
    """Capture a screenshot. Returns the file path.

    Args:
        name: Optional filename (without extension). Defaults to timestamp.
        region: Optional dict with top, left, width, height to capture a region.
    """
    if name is None:
        name = f"screen_{int(time.time())}"

    filepath = os.path.join(SCREENSHOT_DIR, f"{name}.jpg")

    with mss.mss() as sct:
        if region:
            monitor = region
        else:
            monitor = sct.monitors[1]  # Primary monitor

        img = sct.grab(monitor)
        Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX").save(filepath, "JPEG", quality=85)

    print(filepath)
    return filepath


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    capture(name)
