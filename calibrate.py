"""Calibration script: measures coordinate mapping between pyautogui and mss screenshots.

Moves the cursor to known positions with pyautogui, reads back the actual OS cursor
position with GetCursorPos, and compares against mss monitor coordinates to compute
the correction transform.

Usage:
    python calibrate.py          # Full calibration run
    python calibrate.py --info   # Print screen dimensions only
    python calibrate.py --test   # Run calibration + verify accuracy
"""

import ctypes
import ctypes.wintypes
import json
import os
import sys
import time

import mss
import numpy as np
import pyautogui

# Fix DPI scaling -- must match mouse.py
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

pyautogui.PAUSE = 0.05  # Faster during calibration
pyautogui.FAILSAFE = True

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")
MARGIN = 150  # Pixels from screen edge for test points
SETTLE_TIME = 0.05  # Time to wait after moving cursor


def get_cursor_pos():
    """Get actual OS cursor position via Windows API."""
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def get_screen_info():
    """Return screen dimensions from both pyautogui and mss."""
    pa_size = pyautogui.size()
    with mss.mss() as sct:
        mon = sct.monitors[1]
    return {
        "pyautogui_width": pa_size.width,
        "pyautogui_height": pa_size.height,
        "mss_width": mon["width"],
        "mss_height": mon["height"],
        "mss_left": mon["left"],
        "mss_top": mon["top"],
    }


def get_test_points(screen_w, screen_h):
    """Generate test points in a grid pattern, MARGIN pixels from edges."""
    cx, cy = screen_w // 2, screen_h // 2
    left, right = MARGIN, screen_w - MARGIN
    top, bottom = MARGIN, screen_h - MARGIN

    return [
        (cx, cy),           # Center
        (left, top),        # Top-left
        (right, top),       # Top-right
        (left, bottom),     # Bottom-left
        (right, bottom),    # Bottom-right
        (cx, top),          # Top-center
        (cx, bottom),       # Bottom-center
    ]


def measure_point(target_x, target_y, mss_left, mss_top):
    """Move cursor to target via pyautogui, read back actual position.

    Returns:
        (pyautogui_target, screenshot_actual) tuple, where screenshot_actual
        is the cursor position in mss screenshot coordinates.
    """
    pyautogui.moveTo(target_x, target_y)
    time.sleep(SETTLE_TIME)

    # Read back actual OS position
    os_x, os_y = get_cursor_pos()

    # Convert OS screen coords to mss screenshot coords
    # mss screenshot pixel (0,0) = OS screen (mss_left, mss_top)
    screenshot_x = os_x - mss_left
    screenshot_y = os_y - mss_top

    return (target_x, target_y), (screenshot_x, screenshot_y), (os_x, os_y)


def compute_transform(measurements):
    """Fit linear transform: screenshot_coord = scale * pyautogui_coord + offset.

    Uses numpy least-squares to fit: actual = scale * target + offset
    separately for X and Y.

    Args:
        measurements: list of ((target_x, target_y), (actual_x, actual_y))

    Returns:
        (forward, inverse, residuals) where forward/inverse are dicts
    """
    targets = np.array([m[0] for m in measurements], dtype=np.float64)
    actuals = np.array([m[1] for m in measurements], dtype=np.float64)

    n = len(measurements)

    # Fit X: actual_x = scale_x * target_x + offset_x
    A_x = np.column_stack([targets[:, 0], np.ones(n)])
    result_x = np.linalg.lstsq(A_x, actuals[:, 0], rcond=None)
    scale_x, offset_x = result_x[0]

    # Fit Y: actual_y = scale_y * target_y + offset_y
    A_y = np.column_stack([targets[:, 1], np.ones(n)])
    result_y = np.linalg.lstsq(A_y, actuals[:, 1], rcond=None)
    scale_y, offset_y = result_y[0]

    forward = {
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "offset_x": float(offset_x),
        "offset_y": float(offset_y),
    }

    # Inverse: target = (actual - offset) / scale
    inverse = {
        "scale_x": float(1.0 / scale_x),
        "scale_y": float(1.0 / scale_y),
        "offset_x": float(-offset_x / scale_x),
        "offset_y": float(-offset_y / scale_y),
    }

    # Compute residuals
    predicted_x = scale_x * targets[:, 0] + offset_x
    predicted_y = scale_y * targets[:, 1] + offset_y
    residuals = np.sqrt((predicted_x - actuals[:, 0]) ** 2 + (predicted_y - actuals[:, 1]) ** 2)

    return forward, inverse, residuals


def run_calibration(verify=False):
    """Run the full calibration process."""
    print("=== Demon Bluff Cursor Calibration ===\n")

    screen_info = get_screen_info()
    print(f"pyautogui screen: {screen_info['pyautogui_width']}x{screen_info['pyautogui_height']}")
    print(f"mss screen:       {screen_info['mss_width']}x{screen_info['mss_height']}")
    print(f"mss origin:       ({screen_info['mss_left']}, {screen_info['mss_top']})")
    print()

    screen_w = screen_info["pyautogui_width"]
    screen_h = screen_info["pyautogui_height"]
    mss_left = screen_info["mss_left"]
    mss_top = screen_info["mss_top"]
    test_points = get_test_points(screen_w, screen_h)

    print(f"Testing {len(test_points)} points...\n")
    measurements = []

    for i, (tx, ty) in enumerate(test_points):
        (_, _), (sx, sy), (ox, oy) = measure_point(tx, ty, mss_left, mss_top)
        delta_x, delta_y = sx - tx, sy - ty
        print(f"  Point {i + 1}/{len(test_points)}: "
              f"pyautogui=({tx}, {ty}) -> OS=({ox}, {oy}) -> screenshot=({sx}, {sy}) "
              f"delta=({delta_x:+d}, {delta_y:+d})")
        measurements.append(((tx, ty), (sx, sy)))

    print()

    if len(measurements) < 3:
        print("ERROR: Too few successful measurements (need at least 3). Aborting.")
        sys.exit(1)

    # Fit transform
    forward, inverse, residuals = compute_transform(measurements)

    max_residual = float(residuals.max())
    mean_residual = float(residuals.mean())

    print(f"Forward transform (pyautogui -> screenshot):")
    print(f"  x' = {forward['scale_x']:.6f} * x + {forward['offset_x']:.2f}")
    print(f"  y' = {forward['scale_y']:.6f} * y + {forward['offset_y']:.2f}")
    print(f"Inverse transform (screenshot -> pyautogui):")
    print(f"  x  = {inverse['scale_x']:.6f} * x' + {inverse['offset_x']:.2f}")
    print(f"  y  = {inverse['scale_y']:.6f} * y' + {inverse['offset_y']:.2f}")
    print(f"Max residual: {max_residual:.2f} px")
    print(f"Mean residual: {mean_residual:.2f} px")
    print()

    # Save calibration
    calibration = {
        "forward": forward,
        "inverse": inverse,
        "screen_info": screen_info,
        "max_residual_px": max_residual,
        "mean_residual_px": mean_residual,
        "num_points": len(measurements),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"Saved to {CALIBRATION_FILE}")

    if max_residual > 10:
        print("WARNING: High residual -- calibration may be unreliable.")
    else:
        print("Calibration looks good!")

    # Verification pass
    if verify:
        print("\n=== Verification ===\n")
        verify_calibration(inverse, screen_w, screen_h, mss_left, mss_top)

    return calibration


def verify_calibration(inverse, screen_w, screen_h, mss_left, mss_top):
    """Verify calibration by targeting screenshot coords and checking accuracy."""
    verify_points = [
        (screen_w // 4, screen_h // 4),
        (3 * screen_w // 4, screen_h // 4),
        (screen_w // 2, 3 * screen_h // 4),
    ]

    for vx, vy in verify_points:
        # Apply inverse transform: screenshot -> pyautogui
        px = inverse["scale_x"] * vx + inverse["offset_x"]
        py = inverse["scale_y"] * vy + inverse["offset_y"]

        # Move cursor using transformed coords
        pyautogui.moveTo(int(round(px)), int(round(py)))
        time.sleep(SETTLE_TIME)

        # Read back actual position
        os_x, os_y = get_cursor_pos()
        actual_sx = os_x - mss_left
        actual_sy = os_y - mss_top

        error = ((actual_sx - vx) ** 2 + (actual_sy - vy) ** 2) ** 0.5
        print(f"  Target screenshot ({vx}, {vy}) -> pyautogui ({px:.1f}, {py:.1f}) "
              f"-> actual screenshot ({actual_sx}, {actual_sy})  error={error:.1f} px")


def print_info():
    """Print screen dimension info and exit."""
    screen_info = get_screen_info()
    print("Screen info:")
    for k, v in screen_info.items():
        print(f"  {k}: {v}")

    # Also show current cursor position
    os_x, os_y = get_cursor_pos()
    print(f"\nCurrent cursor (OS): ({os_x}, {os_y})")
    print(f"Current cursor (mss screenshot): ({os_x - screen_info['mss_left']}, {os_y - screen_info['mss_top']})")

    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        print(f"\nExisting calibration ({cal['timestamp']}):")
        fwd = cal["forward"]
        print(f"  Forward: scale=({fwd['scale_x']:.6f}, {fwd['scale_y']:.6f}) "
              f"offset=({fwd['offset_x']:.2f}, {fwd['offset_y']:.2f})")
        print(f"  Max residual: {cal['max_residual_px']:.2f} px")
        print(f"  Points used: {cal['num_points']}")
    else:
        print("\nNo calibration file found. Run: python calibrate.py")


if __name__ == "__main__":
    if "--info" in sys.argv:
        print_info()
    elif "--test" in sys.argv:
        run_calibration(verify=True)
    else:
        run_calibration(verify=False)
