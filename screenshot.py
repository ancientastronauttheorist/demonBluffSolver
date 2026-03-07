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


def _normalize_format(fmt):
    fmt = (fmt or "JPEG").upper()
    if fmt == "JPG":
        return "JPEG"
    return fmt


def _extension_for_format(fmt):
    fmt = _normalize_format(fmt)
    if fmt == "PNG":
        return "png"
    return "jpg"


def grab(region=None):
    """Capture a screenshot and return it as a PIL Image."""
    with mss.mss() as sct:
        if region:
            monitor = region
        else:
            monitor = sct.monitors[1]  # Primary monitor

        img = sct.grab(monitor)
        return Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")


def save_image(image, name=None, fmt="JPEG", quality=85):
    """Save a PIL Image into screenshots/ and return the file path."""
    if name is None:
        name = f"screen_{int(time.time())}"

    fmt = _normalize_format(fmt)
    ext = _extension_for_format(fmt)
    filepath = os.path.join(SCREENSHOT_DIR, f"{name}.{ext}")

    save_kwargs = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = quality

    image.save(filepath, fmt, **save_kwargs)
    print(filepath)
    return filepath


def capture(name=None, region=None, fmt="JPEG", quality=85):
    """Capture a screenshot. Returns the file path.

    Args:
        name: Optional filename (without extension). Defaults to timestamp.
        region: Optional dict with top, left, width, height to capture a region.
        fmt: Image format to save. Defaults to JPEG.
        quality: JPEG quality when fmt=JPEG.
    """
    image = grab(region)
    return save_image(image, name=name, fmt=fmt, quality=quality)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    capture(name)
