"""Grid utilities: formatting, tolerant CSV parsing (ported from Land's src/grid.py
and src/judges.py::extract_all_grids), shape-prior inference, distances."""

from __future__ import annotations

import re
from typing import Any, Optional

from harness.types import Grid

MAX_GAP = 2  # Land: allow a small gap of noise lines within a grid block
MAX_WIDTH_DIFF = 5  # Land: tolerate ragged rows up to 5 columns apart


def normalize_grid(obj: Any) -> Optional[Grid]:
    """Coerce nested sequences / numpy arrays to a rectangular list[list[int]] or None."""
    if obj is None:
        return None
    if hasattr(obj, "tolist"):
        try:
            obj = obj.tolist()
        except Exception:
            return None
    if not isinstance(obj, list) or len(obj) == 0:
        return None
    rows: Grid = []
    width: Optional[int] = None
    for row in obj:
        if hasattr(row, "tolist"):
            try:
                row = row.tolist()
            except Exception:
                return None
        if not isinstance(row, list) or len(row) == 0:
            return None
        if width is None:
            width = len(row)
        elif len(row) != width:
            return None
        try:
            rows.append([int(c) for c in row])
        except (TypeError, ValueError):
            return None
    return rows


def shape_of(grid: Optional[Grid]) -> Optional[tuple[int, int]]:
    if not grid or not grid[0]:
        return None
    return (len(grid), len(grid[0]))


def grid_key(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(c) for c in row) for row in grid)


def key_to_grid(key: tuple[tuple[int, ...], ...]) -> Grid:
    return [list(row) for row in key]


def grid_to_csv(grid: Optional[Grid]) -> str:
    """Land's winning format: CSV rows (best of 9 tested; a bad format costs ~10%)."""
    if not grid:
        return ""
    return "\n".join(",".join(str(c) for c in row) for row in grid)


def grid_to_diagram(grid: Grid) -> str:
    """Poetiq <Diagram> style: space-separated cells."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def grids_equal(a: Optional[Grid], b: Optional[Grid]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return grid_key(a) == grid_key(b)


def normalized_distance(a: Optional[Grid], b: Optional[Grid]) -> float:
    """0 = identical, 1 = maximally different (shape mismatch counts as 1.0)."""
    if a is None or b is None:
        return 1.0
    if shape_of(a) != shape_of(b):
        return 1.0
    total = len(a) * len(a[0])
    if total == 0:
        return 0.0
    diff = sum(1 for ra, rb in zip(a, b) for ca, cb in zip(ra, rb) if ca != cb)
    return diff / total


def _parse_row(stripped: str) -> Optional[list[int]]:
    """Parse one candidate CSV row (Land's tolerant row parser)."""
    if not stripped:
        return None
    if re.match(r"^Row\s+\d+:?$", stripped, re.IGNORECASE):
        return None
    if stripped.startswith(("-", "*", "+")):
        return None
    try:
        clean = stripped.replace("`", " ").replace("[", " ").replace("]", " ").strip()
        numbered = re.match(r"^\d+[.)]\s+", clean)
        if numbered:
            clean = clean[numbered.end():]
        tokens = clean.split(",")
        if tokens and all(t.strip().isdigit() for t in tokens):
            return [int(t.strip()) for t in tokens]
        # Fallback: "Row 1: 8,8,8" / "Output: 1,2,3" — slice first-digit..last-digit
        if ":" in clean:
            clean = clean.split(":")[-1].strip()
        first = re.search(r"\d", clean)
        if first:
            last_digit = max((i for i, ch in enumerate(clean) if ch.isdigit()), default=-1)
            if last_digit >= first.start():
                sub = clean[first.start(): last_digit + 1]
                sub_tokens = sub.split(",")
                if len(sub_tokens) > 1 and all(t.strip().isdigit() for t in sub_tokens):
                    remainder = clean[last_digit + 1:].strip()
                    if not any(c.isalpha() for c in remainder):
                        return [int(t.strip()) for t in sub_tokens]
    except ValueError:
        pass
    return None


def _scan_blocks(text: str) -> list[list[list[int]]]:
    """Group candidate rows into blocks; code fences are hard separators."""
    lines = text.splitlines()
    candidate_rows: list[Optional[list[int]]] = []
    hard_separators: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            candidate_rows.append(None)
            hard_separators.append(i)
            continue
        candidate_rows.append(_parse_row(stripped))

    blocks: list[list[list[int]]] = []
    current: list[list[int]] = []
    last_row = -1
    for i, row in enumerate(candidate_rows):
        if row is None:
            continue
        if not current:
            current = [row]
            last_row = i
            continue
        has_hard_sep = any(last_row < sep < i for sep in hard_separators)
        gap = i - last_row - 1
        width_ok = abs(len(row) - len(current[0])) <= MAX_WIDTH_DIFF
        if has_hard_sep or gap > MAX_GAP or not width_ok:
            blocks.append(current)
            current = [row]
        else:
            current.append(row)
        last_row = i
    if current:
        blocks.append(current)
    return blocks


def extract_all_grids(text: str) -> list[Grid]:
    """All CSV-like grid blocks in a text, in order (Land's judges.extract_all_grids)."""
    if not text:
        return []
    return _scan_blocks(text.strip())


def parse_grid_from_text(text: str) -> Optional[Grid]:
    """The LAST valid grid block in a text (Land: the final answer). None if absent."""
    blocks = extract_all_grids(text)
    return blocks[-1] if blocks else None


def distinct_grids(grids: list[Grid]) -> list[Grid]:
    seen: set[tuple] = set()
    out: list[Grid] = []
    for g in grids:
        k = grid_key(g)
        if k not in seen:
            seen.add(k)
            out.append(g)
    return out


def modal_color(train_out: list[Grid]) -> int:
    """Most frequent cell value across train outputs (0 excluded when possible)."""
    counts: dict[int, int] = {}
    for g in train_out:
        for row in g or []:
            for c in row:
                counts[c] = counts.get(c, 0) + 1
    if not counts:
        return 0
    nonzero = {c: n for c, n in counts.items() if c != 0}
    pool = nonzero or counts
    return max(sorted(pool), key=lambda c: pool[c])


# ---------------------------------------------------------------------------
# Shape prior: the true output shape is a near-deterministic function of the
# train pairs (measured on the eval split: relation consistent in 98.3% of
# tasks; identity-train -> identity-test 117/117). Used by the pass@2
# shape-repair hedge.
# ---------------------------------------------------------------------------

def _axis_candidates(hs_in: list[int], hs_out: list[int]) -> list[tuple[int, int]]:
    """All (a, b) with h_out = a*h_in + b fitting every train pair, a in 0..4."""
    fits: list[tuple[int, int]] = []
    for a in range(0, 5):
        bs = {ho - a * hi for hi, ho in zip(hs_in, hs_out)}
        if len(bs) == 1:
            b = bs.pop()
            if -12 <= b <= 12:
                fits.append((a, b))
    return fits


def suggest_output_shape(
    train_in: list[Grid], train_out: list[Grid], test_input: Grid
) -> Optional[tuple[int, int]]:
    """Predict the test output shape from the train pairs, or None if ambiguous.

    Rule precedence matches the measured eval-split statistics:
      1. identity (all train pairs shape-preserving) -> test input shape
         (measured: 117/117 test outputs identity when train is all-identity);
      2. constant output dims across train -> those dims;
      3. per-axis linear fit h' = a*h + b, a in 0..4, b in [-12, 12], required to
         fit every train pair AND agree on a single test prediction (else None).
    """
    if not train_in or not train_out or len(train_in) != len(train_out):
        return None
    ti_shape = shape_of(test_input)
    if ti_shape is None:
        return None
    in_shapes = [shape_of(g) for g in train_in]
    out_shapes = [shape_of(g) for g in train_out]
    if any(s is None for s in in_shapes) or any(s is None for s in out_shapes):
        return None

    if all(i == o for i, o in zip(in_shapes, out_shapes)):
        return ti_shape
    # Constant-output rule needs >= 2 train pairs: with a single pair it is
    # indistinguishable from input-dependent (e.g. scaling) output dims.
    if len(set(out_shapes)) == 1 and len(train_in) >= 2:
        return out_shapes[0]

    pred: list[int] = []
    for axis in (0, 1):
        ins = [s[axis] for s in in_shapes]
        outs = [s[axis] for s in out_shapes]
        fits = _axis_candidates(ins, outs)
        if not fits:
            return None
        preds = {a * ti_shape[axis] + b for a, b in fits}
        if len(preds) != 1:
            return None  # fits disagree on the test input -> ambiguous
        p = preds.pop()
        if not (1 <= p <= 64):
            return None
        pred.append(p)
    return (pred[0], pred[1])
