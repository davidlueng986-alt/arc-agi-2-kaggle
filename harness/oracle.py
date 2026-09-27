"""Exact-replay oracle for ARC-AGI-2 candidates.

This module is intentionally self-contained -- standard library only, no project
imports, no ``from __future__`` -- so its source can be spliced verbatim into the
REPL sandbox bootstrap.  That mirrors how the Duck splices
``inference/utils/segmentation.py`` into ``python_tool_sandbox.py``, and it means
there is exactly one implementation of "did this program reproduce the train
outputs", shared by the sandbox, the host-side candidate pool, and the tests.

The oracle is the ARC-AGI-2 analogue of the Duck's ``action(actions)`` host
round-trip: in ARC-AGI-3 the model acts on a game and observes the new frame;
here the model acts on a *free, exact, deterministic oracle* (the ground-truth
train outputs) and observes exact pass/fail plus diff diagnostics.
"""

import contextlib
import signal
import time

COLOR_CHARS = ".brgysmocd"
COLOR_LEGEND = (
    ".=black, b=blue, r=red, g=green, y=yellow, "
    "s=grey, m=magenta, o=orange, c=sky, d=maroon"
)

MAX_GRID_SIDE = 64
MAX_SAMPLES = 8
CROP_RADIUS = 2


class OracleTimeout(Exception):
    """Raised when a single candidate execution exceeds its time slice."""


@contextlib.contextmanager
def time_slice(seconds):
    """Best-effort wall-clock guard for one candidate execution.

    Uses ``SIGALRM`` where available (Linux / Kaggle).  On platforms without it
    (Windows local dev) this is a no-op and the *host* deadline on the whole
    sandbox process is the only guard, which is still hard.
    """
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise OracleTimeout()

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def normalize_grid(raw):
    """Coerce anything grid-like into ``list[list[int]]`` or ``None``."""
    tolist = getattr(raw, "tolist", None)
    if tolist is not None and not isinstance(raw, (str, bytes)):
        try:
            raw = tolist()
        except Exception:
            return None
    if not isinstance(raw, (list, tuple)):
        return None
    rows = []
    for row in raw:
        tolist = getattr(row, "tolist", None)
        if tolist is not None:
            try:
                row = tolist()
            except Exception:
                return None
        if not isinstance(row, (list, tuple)):
            return None
        cells = []
        for cell in row:
            try:
                value = int(cell)
            except (TypeError, ValueError):
                try:
                    value = int(round(float(cell)))
                except (TypeError, ValueError):
                    return None
            if value < 0 or value > 9:
                return None
            cells.append(value)
        rows.append(cells)
    if not rows:
        return None
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return None
    if len(rows) > MAX_GRID_SIDE or width > MAX_GRID_SIDE:
        return None
    return rows


def shape_of(grid):
    if not grid:
        return [0, 0]
    return [len(grid), len(grid[0])]


def grid_equal(a, b):
    return bool(a) == bool(b) and a == b


def cell_accuracy(expected, got):
    """Fraction of matching cells; only meaningful when shapes agree."""
    if shape_of(expected) != shape_of(got):
        return 0.0
    total = len(expected) * len(expected[0])
    if total == 0:
        return 0.0
    same = sum(
        1
        for r, row in enumerate(expected)
        for c, value in enumerate(row)
        if got[r][c] == value
    )
    return same / total


def ascii_grid(grid, color_chars=COLOR_CHARS):
    if not grid:
        return "(empty)"
    n = len(color_chars)
    return "\n".join(
        "".join(color_chars[max(0, min(n - 1, int(v)))] for v in row) for row in grid
    )


def crop_around(grid, row, col, radius=CROP_RADIUS, color_chars=COLOR_CHARS):
    """Tiny ASCII window centred on one cell -- cheap, exact, coordinate-free."""
    if not grid:
        return ""
    rows, cols = len(grid), len(grid[0])
    r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
    c0, c1 = max(0, col - radius), min(cols, col + radius + 1)
    lines = []
    for r in range(r0, r1):
        marks = []
        for c in range(c0, c1):
            char = color_chars[max(0, min(len(color_chars) - 1, int(grid[r][c])))]
            marks.append(char.upper() if (r == row and c == col) else char)
        lines.append("".join(marks))
    return "\n".join(lines)


def grid_diff(expected, got, max_samples=MAX_SAMPLES):
    """Compact, decision-oriented diff between two grids.

    Returns a dict with ``kind`` of ``shape`` or ``cells``.  The payload is
    deliberately small: bounding box, sample ``[row, col, expected, got]``
    quadruples, and an ASCII crop of both grids around the first mismatch.
    Never dumps a whole grid -- that was one of the failure modes the Duck built
    its segmentation tool to avoid.
    """
    exp_shape, got_shape = shape_of(expected), shape_of(got)
    if exp_shape != got_shape:
        return {
            "kind": "shape",
            "expected_shape": exp_shape,
            "got_shape": got_shape,
            "delta_rows": got_shape[0] - exp_shape[0],
            "delta_cols": got_shape[1] - exp_shape[1],
        }

    rows, cols = exp_shape
    mismatches = []
    for r in range(rows):
        exp_row, got_row = expected[r], got[r]
        for c in range(cols):
            if exp_row[c] != got_row[c]:
                mismatches.append((r, c))
    if not mismatches:
        return {"kind": "exact", "n_diff": 0}

    total = rows * cols
    first_r, first_c = mismatches[0]
    last_r, last_c = mismatches[-1]
    samples = [[r, c, expected[r][c], got[r][c]] for r, c in mismatches[:max_samples]]
    exp_hist, got_hist = {}, {}
    for r, c in mismatches:
        exp_hist[expected[r][c]] = exp_hist.get(expected[r][c], 0) + 1
        got_hist[got[r][c]] = got_hist.get(got[r][c], 0) + 1
    return {
        "kind": "cells",
        "n_diff": len(mismatches),
        "n_cells": total,
        "diff_frac": round(len(mismatches) / total, 4),
        "accuracy": round(1.0 - len(mismatches) / total, 4),
        "bbox": [first_r, first_c, last_r, last_c],
        "samples": samples,
        "samples_truncated": len(mismatches) > max_samples,
        "expected_hist_at_diffs": {str(k): v for k, v in sorted(exp_hist.items())},
        "got_hist_at_diffs": {str(k): v for k, v in sorted(got_hist.items())},
        "expected_crop": crop_around(expected, first_r, first_c),
        "got_crop": crop_around(got, first_r, first_c),
    }


def pair_credit(expected, got):
    """Partial credit for one pair, in [0, 1]. 1.0 iff exact."""
    if got is None:
        return 0.0
    if grid_equal(expected, got):
        return 1.0
    if shape_of(expected) == shape_of(got):
        return round(0.9 * cell_accuracy(expected, got), 4)
    exp_cells = len(expected) * len(expected[0])
    got_cells = len(got) * len(got[0])
    ratio = min(exp_cells, got_cells) / max(exp_cells, got_cells)
    return round(0.3 * ratio, 4)


def augment(grid, name):
    """Dihedral transforms used as a cheap overfitting probe."""
    if name == "identity":
        return [list(row) for row in grid]
    if name == "transpose":
        rows, cols = len(grid), len(grid[0])
        return [[grid[r][c] for r in range(rows)] for c in range(cols)]
    if name == "rot180":
        return [list(reversed(row)) for row in reversed(grid)]
    if name == "flipud":
        return [list(row) for row in reversed(grid)]
    if name == "fliplr":
        return [list(reversed(row)) for row in grid]
    raise ValueError("unknown augmentation: %r" % (name,))


AUGMENTATIONS = ("transpose", "rot180", "flipud")


def _run_once(fn, grid, seconds):
    """Execute ``fn`` on one grid under a time slice; return (grid_or_None, error)."""
    started = time.perf_counter()
    try:
        with time_slice(seconds):
            raw = fn([list(row) for row in grid])
    except OracleTimeout:
        return None, "timeout(%.1fs)" % seconds, time.perf_counter() - started
    except Exception as exc:
        text = "%s: %s" % (type(exc).__name__, exc)
        return None, text[:300], time.perf_counter() - started
    normalized = normalize_grid(raw)
    if normalized is None:
        return None, "returned a non-grid value (%s)" % type(raw).__name__, time.perf_counter() - started
    return normalized, None, time.perf_counter() - started


def verify_candidate(
    train_pairs,
    test_inputs,
    fn,
    *,
    per_call_seconds=1.5,
    total_seconds=12.0,
    use_augmentations=True,
    probe_label_dependence=True,
    max_samples=MAX_SAMPLES,
):
    """Run ``fn`` against every train pair and return compact exact diagnostics.

    ``train_pairs`` is a list of ``(input_grid, output_grid)``.  The report also
    carries the candidate's predictions for ``test_inputs`` so the host-side pool
    can keep them without re-executing the program later.
    """
    deadline = time.perf_counter() + max(0.5, float(total_seconds))
    report = {
        "pairs": len(train_pairs),
        "passed": 0,
        "pass_rate": 0.0,
        "exact": False,
        "soft_score": 0.0,
        "failures": [],
        "errors": [],
        "test": [],
    }

    credits = []
    for index, pair in enumerate(train_pairs):
        expected = pair[1]
        got, error, _elapsed = _run_once(fn, pair[0], per_call_seconds)
        if error is not None:
            report["errors"].append({"pair": index, "error": error})
            credits.append(0.0)
            continue
        if grid_equal(expected, got):
            report["passed"] += 1
            credits.append(1.0)
            continue
        credits.append(pair_credit(expected, got))
        entry = {"pair": index}
        entry.update(grid_diff(expected, got, max_samples=max_samples))
        report["failures"].append(entry)
        if time.perf_counter() > deadline:
            report["budget_exhausted"] = True
            break

    n = max(1, len(train_pairs))
    report["pass_rate"] = round(report["passed"] / n, 4)
    report["exact"] = report["passed"] == len(train_pairs) and len(train_pairs) > 0
    report["soft_score"] = round(sum(credits) / n, 4)

    if use_augmentations and train_pairs and time.perf_counter() < deadline:
        aug = {}
        for name in AUGMENTATIONS:
            passed = 0
            for pair in train_pairs:
                if time.perf_counter() > deadline:
                    break
                try:
                    exp = augment(pair[1], name)
                    src = augment(pair[0], name)
                except Exception:
                    continue
                got, error, _ = _run_once(fn, src, per_call_seconds)
                if error is None and grid_equal(exp, got):
                    passed += 1
            aug[name] = round(passed / max(1, len(train_pairs)), 3)
        report["aug"] = aug

    if (
        probe_label_dependence
        and report["exact"]
        and train_pairs
        and time.perf_counter() < deadline
    ):
        # A legitimate rule does not need the ground-truth labels to be present
        # as data.  Blank them out and re-run: if the candidate still passes it
        # never read them; if it fails it does, which is only legitimate for
        # deriving a mapping (e.g. a palette) and never for table lookup.
        blanked = [(pair[0], [[0] * len(pair[1][0]) for _ in pair[1]]) for pair in train_pairs]
        passed = 0
        for pair in blanked:
            got, error, _ = _run_once(fn, pair[0], per_call_seconds)
            if error is None and got is not None:
                passed += 1
        report["runs_without_labels"] = round(passed / max(1, len(blanked)), 3)

    for index, grid in enumerate(test_inputs or []):
        remaining = max(0.05, deadline - time.perf_counter())
        got, error, _ = _run_once(fn, grid, min(per_call_seconds, remaining))
        entry = {"index": index}
        if error is not None:
            entry["error"] = error
        else:
            entry["shape"] = shape_of(got)
            entry["first_row"] = got[0][:12]
            entry["last_row"] = got[-1][:12]
            entry["_grid"] = got
        report["test"].append(entry)

    report["summary"] = summarize(report)
    return report


def summarize(report):
    """One short line for the ledger -- survives context eviction."""
    if report.get("exact"):
        text = "VERIFIED %d/%d train pairs exact" % (report["passed"], report["pairs"])
        aug = report.get("aug") or {}
        if aug and min(aug.values()) < 1.0:
            text += " (aug %s -- not symmetry-equivariant, usually fine)" % (
                ",".join("%s=%.2f" % (k, v) for k, v in sorted(aug.items())),
            )
        return text
    if report.get("errors"):
        return "ERROR %s" % report["errors"][0]["error"][:120]
    shapes = sum(1 for f in report.get("failures", []) if f.get("kind") == "shape")
    return "FAIL %d/%d exact, soft=%.2f, %d shape mismatches" % (
        report["passed"],
        report["pairs"],
        report.get("soft_score", 0.0),
        shapes,
    )


def strip_private(report):
    """Drop the raw predicted grids before the report is shown to the model."""
    out = {}
    for key, value in report.items():
        if key == "test":
            out[key] = [
                {k: v for k, v in entry.items() if not k.startswith("_")}
                for entry in value
            ]
        else:
            out[key] = value
    return out
