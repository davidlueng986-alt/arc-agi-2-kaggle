"""Grid representation, CSV rendering, and tolerant parsing.

CSV is the grid encoding Land benchmarked best out of 9 (`0,1,2\\n3,4,5`); a bad
encoding costs roughly 10% of score. Everything here is pure stdlib so it runs
identically on a Windows dev box and inside a Kaggle rerun container.
"""

from __future__ import annotations

import re
from collections.abc import Sequence as _Seq
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Grid = List[List[int]]
GridKey = Tuple[Tuple[int, ...], ...]

MIN_CELL = 0
MAX_CELL = 9
MAX_PROSE_GAP = 1


def is_grid(obj: object) -> bool:
    if isinstance(obj, (str, bytes)) or not isinstance(obj, _Seq):
        return False
    if len(obj) == 0:
        return False
    width: Optional[int] = None
    for row in obj:
        if isinstance(row, (str, bytes)) or not isinstance(row, _Seq) or len(row) == 0:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int):
                return False
            if cell < MIN_CELL or cell > MAX_CELL:
                return False
    return True


def to_key(grid: Grid) -> GridKey:
    return tuple(tuple(int(c) for c in row) for row in grid)


def from_key(key: GridKey) -> Grid:
    return [list(row) for row in key]


def shape(grid: Grid) -> Tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def to_csv(grid: Grid) -> str:
    return "\n".join(",".join(str(int(c)) for c in row) for row in grid)


def to_csv_indented(grid: Grid, padding: str = "    ") -> str:
    return "\n".join(padding + ",".join(str(int(c)) for c in row) for row in grid)


def parse_csv_strict(text: str) -> Optional[Grid]:
    lines = [ln for ln in text.strip().splitlines() if ln.strip() != ""]
    if not lines:
        return None
    grid: Grid = []
    width: Optional[int] = None
    for line in lines:
        tokens = line.strip().split(",")
        if not tokens or not all(t.strip().isdigit() for t in tokens):
            return None
        row = [int(t.strip()) for t in tokens]
        if width is None:
            width = len(row)
        elif len(row) != width:
            return None
        if any(c < MIN_CELL or c > MAX_CELL for c in row):
            return None
        grid.append(row)
    return grid or None


_CSV_ROW_RE = re.compile(r"^[\s`]*((?:\d{1,2}\s*,)+\s*\d{1,2})[\s`]*$")
_LEADING_ENUM_RE = re.compile(r"^\s*(?:\d+\s*[\.\)]|[-*+]|\**Row\s*\d+\**?\s*:?)\s*", re.IGNORECASE)


def _candidate_row(line: str) -> Optional[List[int]]:
    raw = line.strip().strip("`")
    if not raw:
        return None
    m = _CSV_ROW_RE.match(raw)
    if not m:
        return None
    body = m.group(1).replace("`", "")
    tokens = [t.strip() for t in body.split(",")]
    if not tokens or not all(t.isdigit() for t in tokens):
        return None
    row = [int(t) for t in tokens]
    if any(c < MIN_CELL or c > MAX_CELL for c in row):
        return None
    return row


def _row_after_label(line: str) -> Optional[List[int]]:
    stripped = _LEADING_ENUM_RE.sub("", line.strip())
    stripped = re.sub(r"^\**\s*(?:Output|Answer|Result|Grid)\s*\d*\**\s*:\s*", "", stripped, flags=re.IGNORECASE)
    if ":" in stripped:
        stripped = stripped.split(":", 1)[1].strip()
    stripped = stripped.strip().strip("`").strip()
    stripped = stripped.lstrip("[").rstrip("]").strip()
    if not stripped:
        return None
    tokens = [t.strip() for t in stripped.split(",")]
    if len(tokens) < 2 or not all(t.isdigit() for t in tokens):
        return None
    row = [int(t) for t in tokens]
    if any(c < MIN_CELL or c > MAX_CELL for c in row):
        return None
    return row


def _mode_int(values: Iterable[int]) -> int:
    counts: Dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts.items()), key=lambda kv: kv[1])[0]


def _mode_shape(values: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    counts: Dict[Tuple[int, int], int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts.items()), key=lambda kv: kv[1])[0]


def regularize(rows: Sequence[List[int]]) -> Optional[Grid]:
    """Repair a mildly ragged row set into a rectangular grid.

    Models occasionally drop a trailing comma or let a prose line leak in. We keep
    the modal width and discard rows that do not match it, accepting only if the
    surviving rows are a strict majority. Short rows are never padded: inventing
    cells would fabricate an answer.
    """
    if not rows:
        return None
    widths = [len(r) for r in rows]
    width = _mode_int(widths)
    kept = [list(r) for r in rows if len(r) == width]
    if not kept:
        return None
    if len(kept) * 2 <= len(rows):
        return None
    return kept


_SINGLE_CELL_RE = re.compile(r"^\d$")


def parse_grid_blocks(text: str, allow_single_cell: bool = False) -> List[Grid]:
    """Return every plausible CSV grid block in `text`, in order of appearance.

    Blocks break on markdown fences, blank lines, and large width jumps; a single
    interleaved prose line is tolerated. `allow_single_cell` additionally accepts a
    lone digit as a 1x1 grid, which is only safe in a slice already delimited by an
    explicit answer marker.
    """
    if not text:
        return []
    lines = text.splitlines()
    rows_at: List[Tuple[int, List[int]]] = []
    fence_lines: List[int] = []

    for idx, line in enumerate(lines):
        if line.strip().startswith("```"):
            fence_lines.append(idx)
            continue
        row = _candidate_row(line)
        if row is None:
            row = _row_after_label(line)
        if row is None and allow_single_cell:
            m = _SINGLE_CELL_RE.match(line.strip())
            if m:
                row = [int(m.group(0))]
        if row is not None:
            rows_at.append((idx, row))

    if not rows_at:
        return []

    def split_between(a: int, b: int) -> bool:
        """True when lines a and b belong to different grid blocks."""
        gap = b - a - 1
        if any(a < f < b for f in fence_lines):
            return True
        if gap <= 0:
            return False
        intervening = lines[a + 1 : b]
        if all(ln.strip() == "" for ln in intervening):
            return True
        return gap > MAX_PROSE_GAP

    blocks: List[List[Tuple[int, List[int]]]] = []
    current: List[Tuple[int, List[int]]] = [rows_at[0]]
    for prev, cur in zip(rows_at, rows_at[1:]):
        width_jump = abs(len(cur[1]) - len(current[0][1])) > 2
        if width_jump or split_between(prev[0], cur[0]):
            blocks.append(current)
            current = [cur]
        else:
            current.append(cur)
    blocks.append(current)

    out: List[Grid] = []
    for block in blocks:
        grid = regularize([row for _, row in block])
        if grid is not None:
            out.append(grid)
    return out


def parse_last_grid(text: str) -> Optional[Grid]:
    blocks = parse_grid_blocks(text)
    return blocks[-1] if blocks else None


def fallback_grid(test_input: Grid, train: Sequence[Tuple[Grid, Grid]]) -> Grid:
    """Deterministic last-resort answer for when nothing parsed.

    Uses the modal training-output shape filled with the modal non-background
    training-output colour; degrades to the identity transform. Any deterministic
    guess beats an empty grid, which can never match.
    """
    if train:
        target = _mode_shape([shape(o) for _, o in train])
        colors = [c for _, o in train for row in o for c in row if c != 0]
        fill = _mode_int(colors) if colors else 0
        if 1 <= target[0] <= 30 and 1 <= target[1] <= 30:
            return [[fill] * target[1] for _ in range(target[0])]
    return [list(row) for row in test_input]


def grids_equal(a: Grid, b: Grid) -> bool:
    return to_key(a) == to_key(b)
