"""Template audit: for each template in templates/, report the best template-match
confidence across provided screenshots. Use this before a re-capture pass to learn
WHICH templates are actually stale vs just "not visible in this screenshot."

Usage:
    python audit_templates.py <screenshot.jpg> [screenshot2.jpg ...] [--pattern=glob]

Buckets:
  CONFIRMED (>= 0.85) — matched high in some screenshot; template is fine.
  MARGINAL  (0.50 - 0.85) — matched but weakly; worth a human eye.
  UNTESTED  (< 0.50 in every provided screenshot) — element not in these shots,
                                                    OR template is stale. Add more
                                                    screenshots from relevant game
                                                    states to disambiguate.
"""
import sys
import os
import glob
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import cv2

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
HIGH = 0.85
MARGINAL = 0.50


def match_best(template_path, screenshot_imgs):
    template = cv2.imread(template_path)
    if template is None:
        return None, None
    best = 0.0
    best_ss = None
    th, tw = template.shape[:2]
    for ss_name, ss in screenshot_imgs:
        if ss.shape[0] < th or ss.shape[1] < tw:
            continue
        result = cv2.matchTemplate(ss, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best:
            best = max_val
            best_ss = ss_name
    return best, best_ss


def audit(screenshot_paths, pattern="*"):
    screenshot_imgs = []
    for p in screenshot_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[skip] Cannot read {p}")
            continue
        screenshot_imgs.append((os.path.basename(p), img))
    if not screenshot_imgs:
        print("No readable screenshots.")
        return

    templates = sorted(glob.glob(os.path.join(TEMPLATE_DIR, pattern + ".png")))
    if not templates:
        print(f"No templates matched pattern {pattern!r}.")
        return

    confirmed, marginal, untested = [], [], []
    for tpl in templates:
        name = os.path.splitext(os.path.basename(tpl))[0]
        best, best_ss = match_best(tpl, screenshot_imgs)
        if best is None:
            print(f"[skip] Cannot read template {name}")
            continue
        if best >= HIGH:
            confirmed.append((name, best, best_ss))
        elif best >= MARGINAL:
            marginal.append((name, best, best_ss))
        else:
            untested.append((name, best, best_ss))

    print(f"\n=== TEMPLATE AUDIT ===")
    print(f"Screenshots: {len(screenshot_imgs)}  ({', '.join(s for s, _ in screenshot_imgs)})")
    print(f"Templates:   {len(templates)} (pattern={pattern!r})")
    print(f"\n[CONFIRMED] {len(confirmed)} templates matched >= {HIGH}")
    for name, conf, ss in confirmed:
        print(f"  {name:45s} {conf:.3f}  in {ss}")
    print(f"\n[MARGINAL]  {len(marginal)} templates matched {MARGINAL}-{HIGH}")
    for name, conf, ss in marginal:
        print(f"  {name:45s} {conf:.3f}  in {ss}")
    print(f"\n[UNTESTED]  {len(untested)} templates not found > {MARGINAL} in these shots")
    for name, conf, ss in untested:
        print(f"  {name:45s} best={conf:.3f}  in {ss}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    pattern = "*"
    ss_paths = []
    for a in args:
        if a.startswith("--pattern="):
            pattern = a.split("=", 1)[1]
        else:
            ss_paths.append(a)
    if not ss_paths:
        print("No screenshots provided.")
        sys.exit(1)
    audit(ss_paths, pattern)
