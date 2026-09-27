"""PNG rendering of ARC grids for the multimodal channel.

The Duck injects a 4x-upscaled image of the board into *every* turn, because its
vision encoder works in 16x16 pixel patches and an ARC-AGI-3 board is a fixed
64x64 -- so 4x puts one board-side at 8 vision tokens.

Two things are different here, and they drive this module:

1. **Our data is static.**  In ARC-AGI-3 the board changes every action, so a
   fresh image every turn carries new information.  An ARC-AGI-2 task is a fixed
   bundle of train pairs plus test inputs; re-sending the same pixels on turn 7
   has exactly zero information gain and a strictly positive token cost.  So the
   panel is injected **once**, on the first turn.

2. **Our grids are small and the panel is tall.**  A mean eval-split task is
   ~3 train pairs of ~19x19 grids, i.e. a panel of roughly 42x63 *cells*.  At
   32px per vision token (16px patch x 2x2 merge) one-cell-per-token would be
   ~2600 vision tokens -- unaffordable, and pointless, because a competitor's
   A/B on Qwen found it "cannot accurately infer coordinates based on image".
   The image is therefore sized to a **vision-token budget** (default 160),
   which is gestalt resolution: enough to see symmetry, tiling, object counts
   and colour layout; not enough to read a cell.  Reading cells is the ASCII
   text channel's and the REPL's job.

``cell_px`` is still exposed so the Duck's literal fixed-upscale behaviour can be
reproduced for an A/B.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Sequence

try:  # PIL is present on Kaggle and in the dev venv; keep the import soft for tests
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only where PIL is absent
    Image = None  # type: ignore[assignment]

from .grids import Grid, Task
from .oracle import shape_of

# Standard ARC palette, indices 0-9.
ARC_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}

GRID_LINE = (255, 255, 255)
BACKGROUND = (24, 24, 28)
GAP_CELLS = 1

PATCH_PX = 16
MERGE = 2
TOKEN_PX = PATCH_PX * MERGE  # 32 px per vision token on Qwen3-VL


def estimate_vision_tokens(width_px: int, height_px: int) -> int:
    """Vision tokens for an image of this pixel size (16px patch, 2x2 merge)."""
    if width_px <= 0 or height_px <= 0:
        return 0
    return max(1, math.ceil(width_px / TOKEN_PX) * math.ceil(height_px / TOKEN_PX))


def _require_pil() -> None:
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for image rendering")


def _draw_grid(image, grid: Grid, origin: tuple[int, int], cell_px: int, grid_lines: bool) -> None:
    ox, oy = origin
    pixels = image.load()
    rows, cols = shape_of(grid)
    for r in range(rows):
        for c in range(cols):
            color = ARC_PALETTE.get(int(grid[r][c]), ARC_PALETTE[0])
            x0, y0 = ox + c * cell_px, oy + r * cell_px
            for y in range(y0, y0 + cell_px):
                for x in range(x0, x0 + cell_px):
                    pixels[x, y] = color
    if grid_lines and cell_px >= 6:
        for r in range(rows + 1):
            y = oy + r * cell_px
            if y >= image.size[1]:
                continue
            for x in range(ox, min(ox + cols * cell_px, image.size[0])):
                pixels[x, y] = GRID_LINE
        for c in range(cols + 1):
            x = ox + c * cell_px
            if x >= image.size[0]:
                continue
            for y in range(oy, min(oy + rows * cell_px, image.size[1])):
                pixels[x, y] = GRID_LINE


def grid_to_image(grid: Grid, cell_px: int = 8, grid_lines: bool = True):
    """One grid as a PIL image, ``cell_px`` pixels per cell."""
    _require_pil()
    rows, cols = shape_of(grid)
    cell_px = max(1, int(cell_px))
    image = Image.new("RGB", (max(1, cols * cell_px), max(1, rows * cell_px)), BACKGROUND)
    _draw_grid(image, grid, (0, 0), cell_px, grid_lines)
    return image


def task_panel_cells(task: Task) -> tuple[int, int]:
    """Panel size in *cells* for a task: one row per train pair (input | output),
    then one row per test input."""
    gap = GAP_CELLS
    widths, heights = [], []
    for pair in task.train:
        ir, ic = shape_of(pair.input)
        orr, oc = shape_of(pair.output)
        widths.append(ic + oc + gap)
        heights.append(max(ir, orr))
    for grid in task.test_inputs:
        r, c = shape_of(grid)
        widths.append(c)
        heights.append(r)
    if not widths:
        return 0, 0
    return max(widths), sum(heights) + gap * (len(heights) - 1)


def cell_px_for_budget(task: Task, max_vision_tokens: int = 160) -> int:
    """Largest whole ``cell_px`` whose panel fits the vision-token budget."""
    panel_w, panel_h = task_panel_cells(task)
    if panel_w <= 0 or panel_h <= 0:
        return 1
    best = 1
    for cell_px in range(1, 65):
        if estimate_vision_tokens(panel_w * cell_px, panel_h * cell_px) <= max_vision_tokens:
            best = cell_px
        else:
            break
    return best


def render_task_panel(
    task: Task,
    *,
    cell_px: int | None = None,
    max_vision_tokens: int = 160,
    upscale: int | None = None,
):
    """Composite panel: every train pair as ``input | output``, then each test input.

    ``upscale`` reproduces the Duck's fixed-multiplier behaviour (render at 1px
    per cell then NEAREST-resize); ``cell_px``/``max_vision_tokens`` is the
    token-budgeted path used by default.  Returns ``(image, info)``.
    """
    _require_pil()
    gap = GAP_CELLS
    panel_w_cells, panel_h_cells = task_panel_cells(task)
    if panel_w_cells <= 0:
        raise ValueError("cannot render an empty task panel")

    if upscale is not None:
        base_cell = 1
        scale = max(1, int(upscale))
    else:
        base_cell = int(cell_px) if cell_px else cell_px_for_budget(task, max_vision_tokens)
        scale = 1

    width = panel_w_cells * base_cell
    height = panel_h_cells * base_cell
    image = Image.new("RGB", (width, height), BACKGROUND)

    y = 0
    for pair in task.train:
        ir, ic = shape_of(pair.input)
        orr, oc = shape_of(pair.output)
        _draw_grid(image, pair.input, (0, y), base_cell, base_cell >= 6)
        _draw_grid(image, pair.output, ((ic + gap) * base_cell, y), base_cell, base_cell >= 6)
        y += (max(ir, orr) + gap) * base_cell
    for grid in task.test_inputs:
        r, c = shape_of(grid)
        _draw_grid(image, grid, (0, y), base_cell, base_cell >= 6)
        y += (r + gap) * base_cell

    if scale > 1:
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)

    final_w, final_h = image.size
    info = {
        "width_px": final_w,
        "height_px": final_h,
        "cell_px": base_cell * scale,
        "panel_cells": [panel_h_cells, panel_w_cells],
        "vision_tokens": estimate_vision_tokens(final_w, final_h),
    }
    return image, info


def image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def image_to_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_part(data_url: str) -> dict:
    """OpenAI-compatible ``image_url`` content part."""
    return {"type": "image_url", "image_url": {"url": data_url}}


def panel_data_url(task: Task, **kwargs) -> tuple[str, dict]:
    image, info = render_task_panel(task, **kwargs)
    return image_to_data_url(image), info


def concat_horizontal(images: Sequence, gap_px: int = 8, background=BACKGROUND):
    _require_pil()
    height = max(im.size[1] for im in images)
    width = sum(im.size[0] for im in images) + gap_px * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0] + gap_px
    return canvas
