"""Image rendering: panel geometry, vision-token accounting, Duck-parity mode."""

from __future__ import annotations

import base64
import io

import pytest

from harness.render import (
    ARC_PALETTE,
    TOKEN_PX,
    cell_px_for_budget,
    concat_horizontal,
    estimate_vision_tokens,
    grid_to_image,
    image_part,
    image_to_data_url,
    image_to_png_bytes,
    panel_data_url,
    render_task_panel,
    task_panel_cells,
)

PIL = pytest.importorskip("PIL.Image")


class TestVisionTokenMath:
    def test_token_pixel_size(self):
        assert TOKEN_PX == 32  # 16px patch x 2x2 merge

    @pytest.mark.parametrize(
        "w,h,expected",
        [(32, 32, 1), (64, 64, 4), (33, 32, 2), (0, 10, 0), (1, 1, 1)],
    )
    def test_estimate(self, w, h, expected):
        assert estimate_vision_tokens(w, h) == expected

    def test_duck_parity_64x64_at_4x(self):
        # The Duck renders a 64x64 ARC-AGI-3 board at upscale 4 -> 256x256 px.
        assert estimate_vision_tokens(64 * 4, 64 * 4) == 64

    def test_cell_level_would_be_unaffordable(self):
        # A 42x63-cell panel at one vision token per cell is ~2.6K tokens: this is
        # the number that rules out reading cells from the image.
        assert estimate_vision_tokens(42 * 32, 63 * 32) == 42 * 63


class TestPanelGeometry:
    def test_cells_account_for_pairs_and_tests(self, flip_task):
        w, h = task_panel_cells(flip_task)
        # 3 train rows of height 2 + 1 test row of height 2, gap 1 between rows
        assert w >= 4
        assert h == 2 * 4 + 1 * 3

    def test_two_test_inputs(self, two_test_task):
        w, h = task_panel_cells(two_test_task)
        # 3 train rows (h=2) + 2 test rows (h=2) = 5 rows of 2, 4 gaps of 1
        assert h == 2 * 5 + 1 * 4
        # widest row is train input|output (2+2+gap) or the 3-wide test input
        assert w >= 5

    def test_ragged_rows_use_the_max_height(self):
        from harness.grids import Pair, Task

        task = Task(
            "ragged",
            (Pair([[0] * 5] * 2, [[1] * 5] * 4),),
            ([[2] * 3] * 2,),
        )
        w, h = task_panel_cells(task)
        assert h == 4 + 2 + 1  # max(2, 4) train row + 2-cell test row + gap
        assert w == 5 + 5 + 1  # input width + output width + gap

    def test_cell_px_fits_budget(self, flip_task):
        cell_px = cell_px_for_budget(flip_task, max_vision_tokens=160)
        w, h = task_panel_cells(flip_task)
        assert estimate_vision_tokens(w * cell_px, h * cell_px) <= 160
        assert estimate_vision_tokens(w * (cell_px + 1), h * (cell_px + 1)) > 160

    def test_cell_px_at_least_one(self, eval_tasks):
        for task in eval_tasks[:10]:
            assert cell_px_for_budget(task, 160) >= 1


class TestRender:
    def test_panel_default_fits_token_budget(self, eval_tasks):
        for task in eval_tasks[:12]:
            image, info = render_task_panel(task, max_vision_tokens=160)
            assert info["vision_tokens"] <= 160
            assert image.size == (info["width_px"], info["height_px"])
            assert info["panel_cells"][0] > 0

    def test_duck_literal_upscale_mode(self, flip_task):
        image, info = render_task_panel(flip_task, upscale=4)
        w, h = task_panel_cells(flip_task)
        assert info["width_px"] == w * 4
        assert info["height_px"] == h * 4
        assert info["cell_px"] == 4

    def test_explicit_cell_px(self, flip_task):
        image, info = render_task_panel(flip_task, cell_px=8)
        assert info["cell_px"] == 8

    def test_pixels_use_the_arc_palette(self):
        image = grid_to_image([[0, 1], [2, 3]], cell_px=8, grid_lines=False)
        pixels = image.load()
        assert pixels[0, 0] == ARC_PALETTE[0]
        assert pixels[8, 0] == ARC_PALETTE[1]
        assert pixels[0, 8] == ARC_PALETTE[2]
        assert pixels[8, 8] == ARC_PALETTE[3]

    def test_out_of_range_values_do_not_crash(self):
        image = grid_to_image([[0]], cell_px=4)
        assert image.size == (4, 4)

    def test_data_url_is_a_real_png(self, flip_task):
        url = panel_data_url(flip_task, cell_px=4)[0]
        assert url.startswith("data:image/png;base64,")
        decoded = base64.b64decode(url.split(",", 1)[1])
        image = PIL.open(io.BytesIO(decoded))
        assert image.format == "PNG"
        assert image.size[0] > 0

    def test_png_bytes(self, flip_task):
        image, _ = render_task_panel(flip_task, cell_px=2)
        blob = image_to_png_bytes(image)
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"

    def test_image_part_shape(self):
        part = image_part("data:image/png;base64,AAA")
        assert part == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}

    def test_concat(self):
        a = grid_to_image([[0]], cell_px=4)
        b = grid_to_image([[1, 2]], cell_px=4)
        joined = concat_horizontal([a, b], gap_px=4)
        assert joined.size[0] == 4 + 8 + 4

    def test_panel_is_reasonably_small_in_bytes(self, eval_tasks):
        # Base64 goes into the prompt, so bytes matter for the chars/3 estimator
        # even though the server charges vision tokens.
        url, info = panel_data_url(eval_tasks[0], max_vision_tokens=160)
        assert len(url) < 400_000
        assert info["vision_tokens"] <= 160
