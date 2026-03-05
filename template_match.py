"""Template matching for Demon Bluff UI elements.

Crops UI elements (buttons, icons) from screenshots to create reusable templates,
then finds their exact pixel coordinates in new screenshots using OpenCV template
matching. Eliminates guesswork and misclicks.

Usage:
    python template_match.py crop <screenshot> <x> <y> <w> <h> <name>
        Crop a region from a screenshot and save as a template.

    python template_match.py find <name> [screenshot]
        Find a template in a screenshot (takes one if not provided).
        Prints the center coordinates and confidence.

    python template_match.py find_all <name> [screenshot]
        Find ALL occurrences of a template (e.g., multiple cards).

    python template_match.py list
        List all saved templates.

    python template_match.py show <name>
        Show template info (size, path).

    python template_match.py debug <name> [screenshot]
        Save annotated screenshot showing match location.
"""

import sys
import os
import glob
import cv2
import numpy as np
from PIL import Image

import screenshot

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(TEMPLATE_DIR, exist_ok=True)


def crop_template(screenshot_path: str, x: int, y: int, w: int, h: int, name: str) -> str:
    """Crop a region from a screenshot and save as a named template."""
    img = cv2.imread(screenshot_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {screenshot_path}")

    crop = img[y:y+h, x:x+w]
    out_path = os.path.join(TEMPLATE_DIR, f"{name}.png")
    cv2.imwrite(out_path, crop)
    print(f"Saved template '{name}' ({w}x{h}) -> {out_path}")
    return out_path


def find(name: str, screenshot_path: str = None, threshold: float = 0.7) -> dict | None:
    """Find a template in a screenshot.

    Returns dict with keys: x, y (center), confidence, top_left, bottom_right
    or None if no match above threshold.
    """
    template_path = _resolve_template(name)
    if template_path is None:
        return None

    if screenshot_path is None:
        screenshot_path = screenshot.capture("_tmpl_find")

    img = cv2.imread(screenshot_path)
    tmpl = cv2.imread(template_path)

    if img is None or tmpl is None:
        print(f"Error reading images")
        return None

    # Use grayscale for matching (more robust to slight color shifts)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(img_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    th, tw = tmpl_gray.shape[:2]
    cx = max_loc[0] + tw // 2
    cy = max_loc[1] + th // 2

    if max_val < threshold:
        print(f"No match for '{name}' (best={max_val:.3f}, threshold={threshold:.2f})")
        return None

    match = {
        "x": cx,
        "y": cy,
        "confidence": round(max_val, 4),
        "top_left": max_loc,
        "bottom_right": (max_loc[0] + tw, max_loc[1] + th),
        "template": name,
    }
    print(f"Found '{name}' at ({cx}, {cy}) confidence={max_val:.4f}")
    return match


def find_all(name: str, screenshot_path: str = None,
             threshold: float = 0.7, min_distance: int = 30) -> list[dict]:
    """Find ALL occurrences of a template in a screenshot.

    Uses non-maximum suppression to avoid duplicate detections.
    min_distance: minimum pixel distance between match centers.
    """
    template_path = _resolve_template(name)
    if template_path is None:
        return []

    if screenshot_path is None:
        screenshot_path = screenshot.capture("_tmpl_findall")

    img = cv2.imread(screenshot_path)
    tmpl = cv2.imread(template_path)

    if img is None or tmpl is None:
        print("Error reading images")
        return []

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(img_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    th, tw = tmpl_gray.shape[:2]

    # Find all locations above threshold
    locations = np.where(result >= threshold)
    scores = result[locations]

    if len(scores) == 0:
        print(f"No matches for '{name}' above threshold {threshold}")
        return []

    # Sort by score descending for NMS
    order = np.argsort(-scores)
    locations_y = locations[0][order]
    locations_x = locations[1][order]
    scores = scores[order]

    # Non-maximum suppression
    matches = []
    for i in range(len(scores)):
        x, y, s = int(locations_x[i]), int(locations_y[i]), float(scores[i])
        cx, cy = x + tw // 2, y + th // 2

        # Check distance to already-accepted matches
        too_close = False
        for m in matches:
            dist = ((cx - m["x"])**2 + (cy - m["y"])**2) ** 0.5
            if dist < min_distance:
                too_close = True
                break

        if not too_close:
            matches.append({
                "x": cx,
                "y": cy,
                "confidence": round(s, 4),
                "top_left": (x, y),
                "bottom_right": (x + tw, y + th),
                "template": name,
            })

    print(f"Found {len(matches)} match(es) for '{name}':")
    for m in matches:
        print(f"  ({m['x']}, {m['y']}) conf={m['confidence']:.4f}")

    return matches


def find_best(names: list[str], screenshot_path: str = None, threshold: float = 0.7) -> dict | None:
    """Try multiple template names, return the best match across all."""
    if screenshot_path is None:
        screenshot_path = screenshot.capture("_tmpl_best")

    best = None
    for name in names:
        match = find(name, screenshot_path, threshold)
        if match and (best is None or match["confidence"] > best["confidence"]):
            best = match

    return best


def click(name: str, screenshot_path: str = None, threshold: float = 0.7) -> dict | None:
    """Find a template and click its center. Returns match dict or None."""
    import mouse as mouse_mod
    import game_utils

    if screenshot_path is None:
        screenshot_path = screenshot.capture("_tmpl_click")

    match = find(name, screenshot_path, threshold)
    if match is None:
        return None

    game_utils.focus_game()
    mouse_mod.click(match["x"], match["y"])
    print(f"Clicked '{name}' at ({match['x']}, {match['y']})")
    return match


def hover_template(name: str, screenshot_path: str = None, threshold: float = 0.7) -> dict | None:
    """Find a template and hover over its center. Returns match dict or None."""
    import mouse as mouse_mod
    import game_utils

    if screenshot_path is None:
        screenshot_path = screenshot.capture("_tmpl_hover")

    match = find(name, screenshot_path, threshold)
    if match is None:
        return None

    game_utils.focus_game()
    mouse_mod.move(match["x"], match["y"])
    print(f"Hovering '{name}' at ({match['x']}, {match['y']})")
    return match


def debug(name: str, screenshot_path: str = None) -> str:
    """Save annotated screenshot showing match location."""
    if screenshot_path is None:
        screenshot_path = screenshot.capture("_tmpl_debug")

    match = find(name, screenshot_path, threshold=0.3)  # Low threshold for debug

    img = cv2.imread(screenshot_path)
    if match:
        tl = match["top_left"]
        br = match["bottom_right"]
        cv2.rectangle(img, tl, br, (0, 255, 0), 3)
        cv2.circle(img, (match["x"], match["y"]), 8, (0, 0, 255), -1)
        label = f"{match['template']} ({match['confidence']:.3f})"
        cv2.putText(img, label, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = screenshot_path.replace(".png", f"_debug_{name}.png")
    cv2.imwrite(out_path, img)
    print(f"Debug image: {out_path}")
    return out_path


def list_templates():
    """List all saved templates."""
    files = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "*.png")))
    if not files:
        print("No templates saved yet.")
        return []

    print(f"{len(files)} template(s):")
    names = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f)
        h, w = img.shape[:2] if img is not None else (0, 0)
        print(f"  {name:25s} {w}x{h}")
        names.append(name)
    return names


def _resolve_template(name: str) -> str | None:
    """Resolve template name to file path."""
    path = os.path.join(TEMPLATE_DIR, f"{name}.png")
    if os.path.exists(path):
        return path

    # Try fuzzy match
    candidates = glob.glob(os.path.join(TEMPLATE_DIR, f"*{name}*.png"))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print(f"Ambiguous template '{name}', matches: {[os.path.basename(c) for c in candidates]}")
        return None

    print(f"Template not found: '{name}'")
    return None


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "crop":
        if len(sys.argv) != 8:
            print("Usage: python template_match.py crop <screenshot> <x> <y> <w> <h> <name>")
            return
        crop_template(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                      int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])

    elif cmd == "find":
        name = sys.argv[2]
        ss = sys.argv[3] if len(sys.argv) > 3 else None
        find(name, ss)

    elif cmd == "find_all":
        name = sys.argv[2]
        ss = sys.argv[3] if len(sys.argv) > 3 else None
        find_all(name, ss)

    elif cmd == "click":
        name = sys.argv[2]
        ss = sys.argv[3] if len(sys.argv) > 3 else None
        click(name, ss)

    elif cmd == "hover":
        name = sys.argv[2]
        ss = sys.argv[3] if len(sys.argv) > 3 else None
        hover_template(name, ss)

    elif cmd == "list":
        list_templates()

    elif cmd == "show":
        name = sys.argv[2]
        path = _resolve_template(name)
        if path:
            img = cv2.imread(path)
            h, w = img.shape[:2]
            print(f"Template '{name}': {w}x{h} at {path}")

    elif cmd == "debug":
        name = sys.argv[2]
        ss = sys.argv[3] if len(sys.argv) > 3 else None
        debug(name, ss)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
