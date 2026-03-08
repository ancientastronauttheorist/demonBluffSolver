import unittest
from pathlib import Path

import cv2
import numpy as np

from card_vision import (
    CARD_SIZE,
    CardTemplate,
    classify_normalized_card,
    detect_dead_board_positions,
    detect_board_seat_centers,
    detect_card_boxes,
    extract_card_signature,
    normalize_card_crop,
    resolved_board_seat_center,
)


def _make_synthetic_card(
    title: str,
    border_bgr: tuple[int, int, int],
    art_bgr: tuple[int, int, int],
    accent_bgr: tuple[int, int, int],
    size: tuple[int, int] = (213, 251),
) -> np.ndarray:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.rectangle(image, (0, 0), (width - 1, height - 1), border_bgr, thickness=8)
    cv2.rectangle(image, (8, 8), (width - 9, height - 9), (18, 24, 30), thickness=-1)

    cv2.rectangle(image, (22, 24), (width - 22, int(height * 0.66)), art_bgr, thickness=-1)
    cv2.circle(image, (width // 2, int(height * 0.34)), 26, accent_bgr, thickness=-1)
    cv2.line(image, (36, 44), (width - 36, int(height * 0.58)), (240, 240, 240), thickness=4)

    title_top = int(height * 0.74)
    cv2.rectangle(image, (14, title_top), (width - 14, height - 18), (86, 42, 118), thickness=-1)
    cv2.putText(
        image,
        title,
        (18, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return image


class TestCardVision(unittest.TestCase):
    def test_detect_dead_board_positions_finds_after_exec_lilis_kill(self):
        path = Path(__file__).resolve().parent.parent / "screenshots" / "after_exec_2.jpg"

        dead = detect_dead_board_positions(path, 9, threshold=0.85)

        self.assertEqual([entry["seat"] for entry in dead], [7])

    def test_detect_dead_board_positions_finds_clean_lilis_kill(self):
        path = Path(__file__).resolve().parent.parent / "screenshots" / "live_full_flip.png"

        dead = detect_dead_board_positions(path, 9, threshold=0.85)

        self.assertEqual([entry["seat"] for entry in dead], [7])

    def test_detect_dead_board_positions_finds_lilis_kill(self):
        path = Path(__file__).resolve().parent.parent / "screenshots" / "live_full_flip_final_attempt.png"

        dead = detect_dead_board_positions(path, 9, threshold=0.85)

        self.assertEqual([entry["seat"] for entry in dead], [7])

    def test_detect_dead_board_positions_ignores_deck_overlay(self):
        path = Path(__file__).resolve().parent.parent / "screenshots" / "deck_view.jpg"

        dead = detect_dead_board_positions(path, 9, threshold=0.85)

        self.assertEqual(dead, [])

    def test_compendium_detection_sorts_staggered_row_left_to_right(self):
        canvas = np.zeros((1440, 2560, 3), dtype=np.uint8)
        card = _make_synthetic_card("ALPHA", (180, 50, 220), (70, 200, 90), (30, 210, 240), size=(213, 251))
        placements = [(237, 277), (465, 274), (693, 277)]

        for x, y in placements:
            canvas[y:y + card.shape[0], x:x + card.shape[1]] = card

        boxes = detect_card_boxes(canvas, context="compendium")

        self.assertEqual([(box.x, box.y) for box in boxes], placements)

    def test_detect_card_boxes_finds_synthetic_cards(self):
        canvas = np.zeros((1440, 2560, 3), dtype=np.uint8)
        card = _make_synthetic_card("ALPHA", (180, 50, 220), (70, 200, 90), (30, 210, 240), size=(182, 216))
        positions = [(600, 430), (795, 430), (990, 430)]

        for x, y in positions:
            canvas[y:y + card.shape[0], x:x + card.shape[1]] = card

        boxes = detect_card_boxes(canvas, context="auto")

        self.assertEqual(len(boxes), 3)
        centers = [box.center for box in boxes]
        self.assertEqual(centers, [(691, 538), (886, 538), (1081, 538)])

    def test_classify_normalized_card_handles_scaled_candidate(self):
        alpha = _make_synthetic_card("ALPHA", (180, 50, 220), (70, 200, 90), (30, 210, 240))
        beta = _make_synthetic_card("BETA", (180, 50, 220), (200, 80, 80), (240, 220, 30))
        library = [
            CardTemplate("alpha", cv2.resize(alpha, CARD_SIZE), extract_card_signature(cv2.resize(alpha, CARD_SIZE))),
            CardTemplate("beta", cv2.resize(beta, CARD_SIZE), extract_card_signature(cv2.resize(beta, CARD_SIZE))),
        ]

        candidate_canvas = np.zeros((1440, 2560, 3), dtype=np.uint8)
        scaled_alpha = _make_synthetic_card(
            "ALPHA",
            (180, 50, 220),
            (70, 200, 90),
            (30, 210, 240),
            size=(182, 216),
        )
        candidate_canvas[440:440 + 216, 604:604 + 182] = scaled_alpha

        boxes = detect_card_boxes(candidate_canvas, context="auto")
        self.assertEqual(len(boxes), 1)

        normalized = normalize_card_crop(candidate_canvas, boxes[0])
        prediction = classify_normalized_card(normalized, library)

        self.assertTrue(prediction.accepted)
        self.assertEqual(prediction.name, "alpha")

    def test_detect_board_seat_centers_prefers_detected_boxes_on_live_board(self):
        path = Path(__file__).resolve().parent.parent / "screenshots" / "full_flip7_clean.jpg"

        centers = detect_board_seat_centers(path, 7)

        self.assertEqual(centers[1], (1666, 406))
        self.assertEqual(centers[2], (1761, 823))
        self.assertEqual(centers[7], (1280, 220))

    def test_resolved_board_seat_center_falls_back_to_detected_live_center(self):
        path = Path(__file__).resolve().parent.parent / "screenshots" / "full_flip7_clean.jpg"

        self.assertEqual(resolved_board_seat_center(path, 7, 7), (1280, 220))

    def test_unknown_card_remains_unaccepted_against_small_library(self):
        alpha = _make_synthetic_card("ALPHA", (180, 50, 220), (70, 200, 90), (30, 210, 240))
        beta = _make_synthetic_card("BETA", (180, 50, 220), (200, 80, 80), (240, 220, 30))
        gamma = _make_synthetic_card("GAMMA", (180, 50, 220), (80, 120, 220), (40, 240, 120))

        library = [
            CardTemplate("alpha", cv2.resize(alpha, CARD_SIZE), extract_card_signature(cv2.resize(alpha, CARD_SIZE))),
            CardTemplate("beta", cv2.resize(beta, CARD_SIZE), extract_card_signature(cv2.resize(beta, CARD_SIZE))),
        ]
        prediction = classify_normalized_card(cv2.resize(gamma, CARD_SIZE), library)

        self.assertFalse(prediction.accepted)
        self.assertIsNone(prediction.name)


if __name__ == "__main__":
    unittest.main()
