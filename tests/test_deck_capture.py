import unittest

from PIL import Image, ImageDraw

from game_utils import (
    crop_deck_view,
    deck_overlay_visible,
    enhance_deck_crop,
    find_deck_content_bounds,
    score_deck_frame,
)


class TestDeckCaptureHelpers(unittest.TestCase):
    def _make_frame(self, include_bottom_row: bool = True):
        image = Image.new("RGB", (1600, 900), (8, 8, 14))
        draw = ImageDraw.Draw(image)

        def draw_card(x, y, color):
            draw.rounded_rectangle((x, y, x + 120, y + 170), radius=14,
                                   fill=(28, 26, 36), outline=color, width=10)
            draw.rectangle((x + 18, y + 128, x + 102, y + 150), fill=(235, 235, 235))

        top_row = [360, 500, 640, 780, 920, 1060]
        bottom_row = [500, 640, 780, 920]
        colors = [
            (190, 90, 220),
            (190, 90, 220),
            (190, 90, 220),
            (190, 90, 220),
            (190, 90, 220),
            (190, 90, 220),
            (225, 210, 70),
            (220, 70, 70),
            (220, 70, 70),
            (220, 70, 70),
        ]

        for idx, x in enumerate(top_row):
            draw_card(x, 260, colors[idx])

        if include_bottom_row:
            for idx, x in enumerate(bottom_row, start=len(top_row)):
                draw_card(x, 470, colors[idx])

        return image

    def test_blank_frame_is_not_visible(self):
        blank = Image.new("RGB", (1600, 900), (8, 8, 14))
        self.assertFalse(deck_overlay_visible(blank))

    def test_full_frame_scores_higher_than_partial(self):
        partial = self._make_frame(include_bottom_row=False)
        full = self._make_frame(include_bottom_row=True)

        self.assertGreater(score_deck_frame(full), score_deck_frame(partial))
        self.assertTrue(deck_overlay_visible(full))

    def test_crop_finds_card_cluster(self):
        frame = self._make_frame(include_bottom_row=True)

        left, top, right, bottom = find_deck_content_bounds(frame, padding=24)

        self.assertGreater(left, 250)
        self.assertGreater(top, 180)
        self.assertLess(right, 1250)
        self.assertLess(bottom, 700)

        cropped = crop_deck_view(frame, padding=24)
        self.assertLess(cropped.size[0], frame.size[0])
        self.assertLess(cropped.size[1], frame.size[1])

    def test_enhanced_crop_upscales_image(self):
        frame = self._make_frame(include_bottom_row=True)
        cropped = crop_deck_view(frame)
        enhanced = enhance_deck_crop(cropped)

        self.assertGreater(enhanced.size[0], cropped.size[0])
        self.assertGreater(enhanced.size[1], cropped.size[1])


if __name__ == "__main__":
    unittest.main()
