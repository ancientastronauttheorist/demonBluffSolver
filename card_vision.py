"""Card detection and recognition helpers for Demon Bluff screenshots.

The core idea is:
1. detect each card box in a screenshot,
2. normalize each crop to a canonical size, then
3. compare card signatures instead of matching raw whole-screen patches.

This keeps the matching robust to modest scale differences between the
compendium, current-deck overlay, and flipped in-game cards.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from knowledge_base import get_card


CARD_SIZE = (192, 224)
REFERENCE_SCREEN = (2560, 1440)
BOARD_CENTER = (1280, 690)
BOARD_RADIUS = 494
BOARD_SEAT_CROP = (240, 270)
DEAD_TEMPLATE_DIR = Path(__file__).with_name("templates") / "board_states"

COMPENDIUM_PAGE_LAYOUTS: dict[str, list[list[Optional[str]]]] = {
    "page1": [
        ["alchemist", "architect", "baker", "bard", "bishop", "confessor", "dreamer", "druid", "empress"],
        ["enlightened", "fortune_teller", "gemcrafter", "hunter", "jester", "judge", "knight", "knitter", "lover"],
        ["medium", "oracle", "poet", "scout", "slayer", "witness", None, None, None],
    ],
    "page3": [
        ["drunk", "wretch", "bombardier", "doppelganger", "plague_doctor", None, None, None, None],
        [None, None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None, None],
    ],
    "page4": [
        ["chancellor", "witch", "minion", "poisoner", "twin_minion", "shaman", "puppeteer", "puppet", None],
        [None, None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None, None],
    ],
    "page5": [
        ["baa", "pooka", "lilis", None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None, None],
    ],
}


@dataclass(frozen=True)
class CardBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class CardSignature:
    title_gray: np.ndarray
    art_hist: np.ndarray


@dataclass
class CardTemplate:
    name: str
    normalized: np.ndarray
    signature: CardSignature


def _faction(name: Optional[str]) -> Optional[str]:
    """Look up faction (Villager/Outcast/Minion/Demon) from knowledge_base."""
    if name is None:
        return None
    card = get_card(name)
    return card.role.value if card else None


@dataclass
class CardPrediction:
    box: CardBox
    name: Optional[str]
    score: float
    margin: float
    accepted: bool
    alternatives: list[tuple[str, float]]

    @property
    def faction(self) -> Optional[str]:
        return _faction(self.name) if self.accepted else None


def _load_bgr(image_or_path) -> np.ndarray:
    if isinstance(image_or_path, np.ndarray):
        return image_or_path.copy()

    path = str(image_or_path)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def _context_roi(image: np.ndarray, context: str) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    regions = {
        "auto": (0.0, 0.0, 1.0, 1.0),
        "compendium": (0.08, 0.14, 0.93, 0.80),
        "deck": (0.18, 0.26, 0.82, 0.70),
        "board": (0.18, 0.02, 0.82, 0.95),
    }
    left_f, top_f, right_f, bottom_f = regions.get(context, regions["auto"])
    return (
        int(width * left_f),
        int(height * top_f),
        int(width * right_f),
        int(height * bottom_f),
    )


def _mask_colored_cards(roi: np.ndarray, scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = ((saturation > 70) & (value > 60)).astype(np.uint8) * 255

    open_k = max(3, int(round(5 * scale)))
    close_k = max(open_k + 2, int(round(9 * scale)))
    kernel_open = np.ones((open_k, open_k), np.uint8)
    kernel_close = np.ones((close_k, close_k), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def _sort_boxes_by_rows(boxes: list[CardBox], row_tolerance: Optional[float] = None) -> list[CardBox]:
    """Sort boxes by visual rows, then left-to-right inside each row."""
    if not boxes:
        return []

    if row_tolerance is None:
        median_height = sorted(box.h for box in boxes)[len(boxes) // 2]
        row_tolerance = max(18.0, median_height * 0.25)

    rows: list[list[CardBox]] = []
    for box in sorted(boxes, key=lambda item: item.y + item.h / 2):
        cy = box.y + box.h / 2
        placed = False
        for row in rows:
            row_center = sum(item.y + item.h / 2 for item in row) / len(row)
            if abs(cy - row_center) <= row_tolerance:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    rows.sort(key=lambda row: sum(item.y + item.h / 2 for item in row) / len(row))
    ordered: list[CardBox] = []
    for row in rows:
        row.sort(key=lambda item: item.x)
        ordered.extend(row)
    return ordered


def detect_card_boxes(image_or_path, context: str = "auto") -> list[CardBox]:
    """Detect upright card boxes in a screenshot."""
    image = _load_bgr(image_or_path)
    height, width = image.shape[:2]
    left, top, right, bottom = _context_roi(image, context)
    roi = image[top:bottom, left:right]

    scale = min(width / REFERENCE_SCREEN[0], height / REFERENCE_SCREEN[1])
    mask = _mask_colored_cards(roi, scale)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_scale = (width * height) / float(REFERENCE_SCREEN[0] * REFERENCE_SCREEN[1])
    min_area = 12000 * area_scale
    max_area = 80000 * area_scale

    boxes: list[CardBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / h if h else 0.0
        if not (min_area <= area <= max_area):
            continue
        if not (0.55 <= aspect <= 1.0):
            continue
        boxes.append(CardBox(left + x, top + y, w, h))

    if context == "compendium":
        boxes = _sort_boxes_by_rows(boxes)
    else:
        boxes.sort(key=lambda box: (box.y, box.x))
    return boxes


def board_seat_center(
    seat_num: int,
    n_cards: int,
    image_size: tuple[int, int],
) -> tuple[int, int]:
    """Predict the screen center of a numbered board seat."""
    width, height = image_size
    scale = min(width / REFERENCE_SCREEN[0], height / REFERENCE_SCREEN[1])
    center_x = width * (BOARD_CENTER[0] / REFERENCE_SCREEN[0])
    center_y = height * (BOARD_CENTER[1] / REFERENCE_SCREEN[1])
    radius = BOARD_RADIUS * scale

    angle = -math.pi / 2 + (seat_num % n_cards) * 2 * math.pi / n_cards
    x = round(center_x + radius * math.cos(angle))
    y = round(center_y + radius * math.sin(angle))
    return x, y


def detect_board_seat_centers(
    image_or_path,
    n_cards: int,
) -> dict[int, tuple[int, int]]:
    """Map numbered board seats to detected card-box centers.

    Uses the rough circle formula only to assign each detected board card box to
    the nearest seat number, then returns exact box centers for clicking.
    """
    image = _load_bgr(image_or_path)
    height, width = image.shape[:2]
    boxes = detect_card_boxes(image, context="board")
    if not boxes:
        return {}

    predicted = {
        seat: board_seat_center(seat, n_cards, (width, height))
        for seat in range(1, n_cards + 1)
    }

    candidates: list[tuple[float, int, CardBox]] = []
    for seat, (px, py) in predicted.items():
        for box in boxes:
            bx, by = box.center
            distance_sq = float((bx - px) ** 2 + (by - py) ** 2)
            candidates.append((distance_sq, seat, box))

    centers: dict[int, tuple[int, int]] = {}
    used_seats: set[int] = set()
    used_boxes: set[CardBox] = set()

    for _, seat, box in sorted(candidates, key=lambda item: item[0]):
        if seat in used_seats or box in used_boxes:
            continue
        used_seats.add(seat)
        used_boxes.add(box)
        centers[seat] = box.center

    return centers


def resolved_board_seat_center(
    image_or_path,
    seat_num: int,
    n_cards: int,
) -> tuple[int, int]:
    """Return the best click center for a numbered board seat.

    Prefers the detected card-box center from the current screenshot and falls
    back to the circle formula if no box could be assigned.
    """
    image = _load_bgr(image_or_path)
    height, width = image.shape[:2]
    detected = detect_board_seat_centers(image, n_cards)
    return detected.get(seat_num, board_seat_center(seat_num, n_cards, (width, height)))


def extract_board_seat_crop(
    image_or_path,
    seat_num: int,
    n_cards: int,
    crop_size: tuple[int, int] = BOARD_SEAT_CROP,
    normalized_size: tuple[int, int] = CARD_SIZE,
) -> np.ndarray:
    """Extract a normalized crop around one predicted board seat."""
    image = _load_bgr(image_or_path)
    width, height = image.shape[1], image.shape[0]
    scale = min(width / REFERENCE_SCREEN[0], height / REFERENCE_SCREEN[1])
    crop_w = int(round(crop_size[0] * scale))
    crop_h = int(round(crop_size[1] * scale))
    center_x, center_y = resolved_board_seat_center(image, seat_num, n_cards)

    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(width, x1 + crop_w)
    y2 = min(height, y1 + crop_h)
    crop = image[y1:y2, x1:x2]
    interpolation = cv2.INTER_AREA if crop.shape[1] >= normalized_size[0] else cv2.INTER_CUBIC
    return cv2.resize(crop, normalized_size, interpolation=interpolation)


def normalize_card_crop(image_or_path, box: CardBox, size: tuple[int, int] = CARD_SIZE) -> np.ndarray:
    """Crop a detected card box and resize it to a canonical size."""
    image = _load_bgr(image_or_path)
    crop = image[box.y:box.y + box.h, box.x:box.x + box.w]
    interpolation = cv2.INTER_AREA if crop.shape[1] >= size[0] else cv2.INTER_CUBIC
    return cv2.resize(crop, size, interpolation=interpolation)


def extract_card_signature(normalized_bgr: np.ndarray) -> CardSignature:
    """Extract a compact, scale-tolerant signature from a normalized card crop."""
    height, width = normalized_bgr.shape[:2]

    title = normalized_bgr[
        int(height * 0.74):int(height * 0.94),
        int(width * 0.07):int(width * 0.93),
    ]
    title_gray = cv2.cvtColor(title, cv2.COLOR_BGR2GRAY)

    art = normalized_bgr[
        int(height * 0.05):int(height * 0.70),
        int(width * 0.08):int(width * 0.92),
    ]
    art_hsv = cv2.cvtColor(art, cv2.COLOR_BGR2HSV)
    art_hist = cv2.calcHist([art_hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(art_hist, art_hist)

    return CardSignature(title_gray=title_gray, art_hist=art_hist)


def _title_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    score = float(cv2.matchTemplate(left, right, cv2.TM_CCOEFF_NORMED)[0, 0])
    if np.isnan(score):
        return 0.0
    return max(-1.0, min(1.0, score))


def _art_similarity(left: np.ndarray, right: np.ndarray) -> float:
    score = float(cv2.compareHist(left, right, cv2.HISTCMP_CORREL))
    return max(-1.0, min(1.0, score))


def compare_signatures(candidate: CardSignature, template: CardSignature) -> tuple[float, float, float]:
    """Return (combined_score, title_score, art_score)."""
    title_score = _title_similarity(candidate.title_gray, template.title_gray)
    art_score = _art_similarity(candidate.art_hist, template.art_hist)
    combined = 0.45 * title_score + 0.55 * art_score
    return combined, title_score, art_score


def _flatten_page_layout(layout: list[list[Optional[str]]]) -> list[Optional[str]]:
    return [name for row in layout for name in row]


def build_library_from_compendium_page(
    image_or_path,
    page_layout: list[list[Optional[str]]],
) -> list[CardTemplate]:
    """Build a template library from one compendium page screenshot."""
    image = _load_bgr(image_or_path)
    boxes = detect_card_boxes(image, context="compendium")
    names = _flatten_page_layout(page_layout)

    if len(boxes) < len(names):
        raise ValueError(
            f"Detected {len(boxes)} compendium boxes, expected at least {len(names)}"
        )

    templates: list[CardTemplate] = []
    for box, name in zip(boxes, names):
        if not name:
            continue
        normalized = normalize_card_crop(image, box)
        templates.append(
            CardTemplate(
                name=name,
                normalized=normalized,
                signature=extract_card_signature(normalized),
            )
        )
    return templates


def classify_normalized_card(
    normalized_bgr: np.ndarray,
    library: list[CardTemplate],
    accept_score: float = 0.55,
    accept_margin: float = 0.05,
    strong_score: float = 0.85,
) -> CardPrediction:
    """Classify one normalized card crop against a template library."""
    signature = extract_card_signature(normalized_bgr)

    scored: list[tuple[float, str]] = []
    for template in library:
        combined, _, _ = compare_signatures(signature, template.signature)
        scored.append((combined, template.name))
    scored.sort(reverse=True)

    best_score, best_name = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    margin = best_score - second_score
    accepted = best_score >= strong_score or (best_score >= accept_score and margin >= accept_margin)

    return CardPrediction(
        box=CardBox(0, 0, normalized_bgr.shape[1], normalized_bgr.shape[0]),
        name=best_name if accepted else None,
        score=best_score,
        margin=margin,
        accepted=accepted,
        alternatives=[(name, score) for score, name in scored[:5]],
    )


def classify_image(
    image_or_path,
    library: list[CardTemplate],
    context: str,
    accept_score: float = 0.55,
    accept_margin: float = 0.05,
    strong_score: float = 0.85,
) -> list[CardPrediction]:
    """Detect and classify all cards in an image."""
    image = _load_bgr(image_or_path)
    boxes = detect_card_boxes(image, context=context)
    predictions: list[CardPrediction] = []

    for box in boxes:
        normalized = normalize_card_crop(image, box)
        classified = classify_normalized_card(
            normalized,
            library,
            accept_score=accept_score,
            accept_margin=accept_margin,
            strong_score=strong_score,
        )
        predictions.append(
            CardPrediction(
                box=box,
                name=classified.name,
                score=classified.score,
                margin=classified.margin,
                accepted=classified.accepted,
                alternatives=classified.alternatives,
            )
        )
    return predictions


def export_library_images(library: list[CardTemplate], output_dir: str | Path) -> list[str]:
    """Save normalized card templates for visual inspection."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for template in library:
        path = out_dir / f"{template.name}.png"
        cv2.imwrite(str(path), template.normalized)
        saved.append(str(path))
    return saved


def load_library_from_dir(template_dir: str | Path) -> list[CardTemplate]:
    """Load normalized card templates from a directory of PNG files."""
    template_dir = Path(template_dir)
    templates: list[CardTemplate] = []
    for path in sorted(template_dir.glob("*.png")):
        normalized = _load_bgr(path)
        templates.append(
            CardTemplate(
                name=path.stem,
                normalized=normalized,
                signature=extract_card_signature(normalized),
            )
        )
    return templates


def load_library_from_dirs(template_dirs: list[str | Path]) -> list[CardTemplate]:
    """Load and merge template libraries from multiple directories."""
    merged: dict[str, CardTemplate] = {}
    for directory in template_dirs:
        for template in load_library_from_dir(directory):
            merged[template.name] = template
    return [merged[name] for name in sorted(merged)]


def _crop_relative(
    image: np.ndarray,
    left_f: float,
    top_f: float,
    right_f: float,
    bottom_f: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = int(round(width * left_f))
    y1 = int(round(height * top_f))
    x2 = int(round(width * right_f))
    y2 = int(round(height * bottom_f))
    return image[y1:y2, x1:x2]


def _range_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _inverse_range_score(value: float, low: float, high: float) -> float:
    return 1.0 - _range_score(value, low, high)


def _orange_ratio(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 60, 30), (25, 255, 255))
    return float(mask.mean() / 255.0)


def _purple_ratio(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (120, 40, 40), (170, 255, 255))
    return float(mask.mean() / 255.0)


def _edge_ratio(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean() / 255.0)


def _gray_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    score = float(cv2.matchTemplate(left, right, cv2.TM_CCOEFF_NORMED)[0, 0])
    if np.isnan(score):
        return 0.0
    return max(-1.0, min(1.0, score))


def _dead_template_score(candidate_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
    candidate_gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # Compare the stable interior of the dead card instead of the full border.
    candidate_center = _crop_relative(candidate_gray, 0.20, 0.25, 0.80, 0.85)
    template_center = _crop_relative(template_gray, 0.20, 0.25, 0.80, 0.85)
    candidate_bottom = _crop_relative(candidate_gray, 0.15, 0.57, 0.85, 0.98)
    template_bottom = _crop_relative(template_gray, 0.15, 0.57, 0.85, 0.98)

    center_score = _gray_similarity(candidate_center, template_center)
    bottom_score = _gray_similarity(candidate_bottom, template_bottom)
    orange_ratio = _orange_ratio(_crop_relative(candidate_bgr, 0.125, 0.11, 0.875, 0.89))
    purple_ratio = _purple_ratio(_crop_relative(candidate_bgr, 0.10, 0.75, 0.90, 0.97))
    edge_ratio = _edge_ratio(candidate_gray)

    return float(np.clip(
        0.30 * _range_score(center_score, 0.25, 0.45)
        + 0.30 * _range_score(bottom_score, 0.25, 0.45)
        + 0.25 * _range_score(orange_ratio, 0.45, 0.70)
        + 0.10 * _inverse_range_score(purple_ratio, 0.10, 0.35)
        + 0.05 * _inverse_range_score(edge_ratio, 0.08, 0.12),
        0.0,
        1.0,
    ))


def _load_dead_templates(template_dir: str | Path = DEAD_TEMPLATE_DIR) -> list[np.ndarray]:
    template_dir = Path(template_dir)
    templates: list[np.ndarray] = []
    for path in sorted(template_dir.glob("*.png")):
        image = _load_bgr(path)
        if image.shape[1] != CARD_SIZE[0] or image.shape[0] != CARD_SIZE[1]:
            image = cv2.resize(image, CARD_SIZE, interpolation=cv2.INTER_AREA)
        templates.append(image)
    return templates


def detect_dead_board_positions(
    image_or_path,
    n_cards: int,
    threshold: float = 0.85,
    template_dir: str | Path = DEAD_TEMPLATE_DIR,
) -> list[dict]:
    """Detect facedown dead board seats, such as Lilis night kills.

    This checks normalized crops around expected board seats against one or more
    saved dead-card templates. The intended use is the full-board flip screen,
    before executions start, where a dead facedown card should still occupy its
    original seat.
    """
    templates = _load_dead_templates(template_dir)
    if not templates:
        return []

    image = _load_bgr(image_or_path)
    results: list[dict] = []
    for seat_num in range(1, n_cards + 1):
        crop = extract_board_seat_crop(image, seat_num, n_cards)
        score = max(
            _dead_template_score(crop, template)
            for template in templates
        )
        if score >= threshold:
            cx, cy = board_seat_center(seat_num, n_cards, (image.shape[1], image.shape[0]))
            results.append({
                "seat": seat_num,
                "score": score,
                "center": (cx, cy),
            })
    return results


def save_debug_image(image_or_path, predictions: list[CardPrediction], output_path: str | Path) -> str:
    """Render prediction boxes and labels into a debug image."""
    image = _load_bgr(image_or_path)

    for index, prediction in enumerate(predictions, start=1):
        color = (0, 200, 0) if prediction.accepted else (0, 165, 255)
        x, y, w, h = prediction.box.x, prediction.box.y, prediction.box.w, prediction.box.h
        label = prediction.name or "unknown"
        text = f"{index}:{label} {prediction.score:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image,
            text,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    output_path = str(output_path)
    cv2.imwrite(output_path, image)
    return output_path


def _predictions_to_json(predictions: list[CardPrediction]) -> list[dict]:
    return [
        {
            "box": {
                "x": prediction.box.x,
                "y": prediction.box.y,
                "w": prediction.box.w,
                "h": prediction.box.h,
            },
            "name": prediction.name,
            "faction": prediction.faction,
            "score": round(prediction.score, 4),
            "margin": round(prediction.margin, 4),
            "accepted": prediction.accepted,
            "alternatives": [
                {"name": name, "score": round(score, 4)}
                for name, score in prediction.alternatives
            ],
        }
        for prediction in predictions
    ]


def _page_layout(page_name: str) -> list[list[Optional[str]]]:
    if page_name not in COMPENDIUM_PAGE_LAYOUTS:
        known = ", ".join(sorted(COMPENDIUM_PAGE_LAYOUTS))
        raise KeyError(f"Unknown page '{page_name}'. Known page layouts: {known}")
    return COMPENDIUM_PAGE_LAYOUTS[page_name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    detect_parser = subparsers.add_parser("detect", help="Detect card boxes in a screenshot")
    detect_parser.add_argument("image")
    detect_parser.add_argument("--context", default="auto", choices=["auto", "compendium", "deck", "board"])
    detect_parser.add_argument("--debug-out")

    build_parser = subparsers.add_parser("build_page", help="Build normalized templates from one compendium page")
    build_parser.add_argument("image")
    build_parser.add_argument("page", choices=sorted(COMPENDIUM_PAGE_LAYOUTS))
    build_parser.add_argument("--output-dir", required=True)

    classify_parser = subparsers.add_parser("classify", help="Classify cards in an image using one compendium page")
    classify_parser.add_argument("compendium_image")
    classify_parser.add_argument("page", choices=sorted(COMPENDIUM_PAGE_LAYOUTS))
    classify_parser.add_argument("target_image")
    classify_parser.add_argument("--context", default="deck", choices=["auto", "compendium", "deck", "board"])
    classify_parser.add_argument("--debug-out")

    classify_dir_parser = subparsers.add_parser("classify_dirs", help="Classify cards using existing exported template dirs")
    classify_dir_parser.add_argument("target_image")
    classify_dir_parser.add_argument("--context", default="deck", choices=["auto", "compendium", "deck", "board"])
    classify_dir_parser.add_argument("--library-dir", action="append", required=True)
    classify_dir_parser.add_argument("--debug-out")

    dead_parser = subparsers.add_parser("detect_dead", help="Detect facedown dead seats on a board screenshot")
    dead_parser.add_argument("image")
    dead_parser.add_argument("n_cards", type=int)
    dead_parser.add_argument("--threshold", type=float, default=0.85)

    args = parser.parse_args()

    if args.cmd == "detect":
        boxes = detect_card_boxes(args.image, context=args.context)
        payload = [
            {"x": box.x, "y": box.y, "w": box.w, "h": box.h}
            for box in boxes
        ]
        print(json.dumps(payload, indent=2))
        if args.debug_out:
            debug_predictions = [
                CardPrediction(
                    box=box,
                    name=None,
                    score=0.0,
                    margin=0.0,
                    accepted=False,
                    alternatives=[],
                )
                for box in boxes
            ]
            print(save_debug_image(args.image, debug_predictions, args.debug_out))
        return

    if args.cmd == "build_page":
        layout = _page_layout(args.page)
        library = build_library_from_compendium_page(args.image, layout)
        for path in export_library_images(library, args.output_dir):
            print(path)
        return

    if args.cmd == "classify":
        layout = _page_layout(args.page)
        library = build_library_from_compendium_page(args.compendium_image, layout)
        predictions = classify_image(args.target_image, library, context=args.context)
        print(json.dumps(_predictions_to_json(predictions), indent=2))
        if args.debug_out:
            print(save_debug_image(args.target_image, predictions, args.debug_out))
        return

    if args.cmd == "classify_dirs":
        library = load_library_from_dirs(args.library_dir)
        predictions = classify_image(args.target_image, library, context=args.context)
        print(json.dumps(_predictions_to_json(predictions), indent=2))
        if args.debug_out:
            print(save_debug_image(args.target_image, predictions, args.debug_out))
        return

    if args.cmd == "detect_dead":
        print(json.dumps(
            detect_dead_board_positions(args.image, args.n_cards, threshold=args.threshold),
            indent=2,
        ))
        return


if __name__ == "__main__":
    main()
