"""Reproduce a pinned, partial archived UI review, not player-history admission.

The semantic observations below were authored after visual inspection of both
exact captures. Hash/dimension checks bind that review; this script performs no
OCR and cannot discover or approve different captures. Proprietary image pixels
remain in the private screenshot workspace.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


CAPTURES = (
    (
        "asc84_v1_start.jpg",
        "951893116f0cbc2a261a6530d44582699f8e54cda0fe37efcc3a4739a66bc1c4",
        "all_board_cards_face_down",
    ),
    (
        "asc84_v1_after_flip.jpg",
        "69b12fef258586204ad4f51aebbfb6aac94d8ecce13718e98eecc88f54343ba8",
        "all_board_cards_face_up",
    ),
)


def audit(capture_root: Path):
    captures = []
    for filename, expected_sha256, board_visibility in CAPTURES:
        source = capture_root / filename
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"capture no longer matches reviewed pixels: {filename}")
        with Image.open(source) as original:
            dimensions = list(original.size)
            if dimensions != [2560, 1440] or original.format != "JPEG":
                raise ValueError(f"unexpected original capture format: {filename}")
        captures.append(
            {
                "private_capture_basename": filename,
                "sha256": digest,
                "dimensions": dimensions,
                "format": "JPEG",
                "review_method": "tool_assisted_visual_inspection",
                "public_pixels": {
                    "n_cards": 10,
                    "find_and_execute_evil_count": 3,
                    "objective_subtitle": "2 Minions and 1 Demon",
                    "evils_killed": [0, 3],
                    "village": [1, 7],
                    "ascension": 84,
                    "score": 0,
                    "hp": 10,
                    "hud_counts_left_to_right": [6, 1, 2, 1],
                    "board_visibility": board_visibility,
                    "clockwise_display_ids_from_top": [10, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "visible_deck_strip_labels": [
                        "DOPPELGANGER", "CHANCELLOR", "MINION", "POOKA", "PLAGUE DOCTOR"
                    ],
                    "full_deck_occurrence_multiset": None,
                    "wrong_execution_cost": None,
                    "phase": None,
                },
            }
        )
    captures[1]["public_pixels"]["hunter_capture"] = {
        "display_id": 3,
        "apparent_role_label": "HUNTER",
        "visible_words": "I am 1 card away from closest Evil",
        "visible_line_layout": [
            "I am 1 card", "away from", "closest Evil"
        ],
        "literal_managed_string_newlines": None,
        "explicit_numbered_speech_targets": [],
        "public_geometric_adjacent_ids": [2, 4],
        "native_acted_reference_order": None,
    }
    return {
        "schema": "archived_hunter_ui_review_v1",
        "information_lane": "public_pixel_review_development",
        "review_scope": "two_exact_archived_captures_only",
        "build_binding": {
            "status": "unresolved",
            "candidate_build_id": "f530404b0f3f_807de4a83df4",
            "capture_time_binary_or_asset_fingerprints": None,
        },
        "catalog_order_is_action_chronology": False,
        "captures": captures,
        "player_history_admitted": False,
        "conditional_hunter_baa_domain_admitted": False,
        "exclusions": [
            "No intermediate single-card captures or verified action/reveal order.",
            "No capture-time binary/asset identity; file labels and times are not proof.",
            "Full public role occurrence multiset and wrong-execution cost unavailable.",
            "Ten-card mixed-role board is outside the four/five-card Hunter/Baa domain.",
            "Visual line wrapping is not proof of managed-string newline encoding.",
            "Native target order, hidden truth, statuses, queue and RNG are not observations.",
            "No live control, solver action, phase-legality, held-out or policy claim.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.capture_root)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "captures_reviewed": len(result["captures"]),
        "build_binding": result["build_binding"]["status"],
        "player_history_admitted": result["player_history_admitted"],
    }))
