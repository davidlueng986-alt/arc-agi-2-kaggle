"""Train-pair oracle: soft scoring, diff diagnostics, feedback construction.

Port of Poetiq's _soft_score / _array_diff / _build_feedback / _eval_on_train_and_test
with local-model hardening (grid normalisation, per-run error capture).

Verification semantics (Land, measured): precision ~97%, recall ~43% — passing all
train pairs is a strong positive signal, failing is a weak negative one, so failed
candidates are never discarded, only down-ranked.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from harness.grids import normalize_grid
from harness.sandbox import DEFAULT_TIMEOUT_S, run_code
from harness.types import Grid, RunResult


def soft_score(pred: Optional[Grid], truth: Grid) -> float:
    """Mean per-cell accuracy; 0.0 on shape mismatch, 1.0 on exact match."""
    if pred is None:
        return 0.0
    if len(pred) != len(truth) or any(len(p) != len(t) for p, t in zip(pred, truth)):
        return 0.0
    total = sum(len(t) for t in truth)
    if total == 0:
        return 1.0
    match = sum(1 for p, t in zip(pred, truth) for cp, ct in zip(p, t) if cp == ct)
    return match / total


def array_diff(pred: Grid, truth: Grid) -> str:
    """Poetiq diff visualisation: correct cells as-is, wrong cells 'pred/correct'."""
    out = []
    for pr, tr in zip(pred, truth):
        row = []
        for cp, ct in zip(pr, tr):
            row.append(str(int(cp)) if cp == ct else f"{int(cp)}/{int(ct)}")
        out.append(" ".join(row))
    return "\n".join(out)


async def evaluate_code(
    code: str,
    train_in: list[Grid],
    train_out: list[Grid],
    test_in: list[Grid],
    timeout_s: float = DEFAULT_TIMEOUT_S,
    entry: str = "transform",
) -> tuple[list[RunResult], list[RunResult]]:
    """Execute `code` on every train pair (scored) and test input (unscored)."""
    train_results: list[RunResult] = []
    for iin, oout in zip(train_in, train_out):
        ok, out_str = await run_code(code, iin, timeout_s=timeout_s, entry=entry)
        rr = RunResult(raw_output=out_str)
        if not ok:
            rr.error = out_str or "Execution failed."
        else:
            try:
                parsed = normalize_grid(json.loads(out_str))
            except Exception:
                parsed = None
            if parsed is None:
                rr.error = f"Failed to parse output as a rectangular 2D int grid. Output was:\n{out_str[:500]}"
            else:
                rr.grid = parsed
                rr.soft_score = soft_score(parsed, oout)
                rr.success = parsed == oout
        train_results.append(rr)

    test_results: list[RunResult] = []
    for iin in test_in:
        ok, out_str = await run_code(code, iin, timeout_s=timeout_s, entry=entry)
        rr = RunResult(raw_output=out_str)
        if not ok:
            rr.error = out_str or "Execution failed."
        else:
            try:
                parsed = normalize_grid(json.loads(out_str))
            except Exception:
                parsed = None
            if parsed is None:
                rr.error = "Failed to parse output as a rectangular 2D int grid."
            else:
                rr.grid = parsed
        test_results.append(rr)
    return train_results, test_results


def evaluate_code_sync(*args, **kwargs) -> tuple[list[RunResult], list[RunResult]]:
    return asyncio.run(evaluate_code(*args, **kwargs))


def build_feedback(
    train_results: list[RunResult], train_out: list[Grid]
) -> tuple[str, float]:
    """Poetiq feedback: per-example diagnosis (shape mismatch / diff grid / error)
    plus a mean soft score in [0, 1]."""
    parts: list[str] = []
    scores: list[float] = []

    for i, rr in enumerate(train_results):
        if rr.success:
            parts.append(f"Solves Example #{i + 1} correctly. ")
            scores.append(1.0)
            continue
        lines = [f"Solves Example #{i + 1} incorrectly. "]
        if rr.grid is None:
            scores.append(0.0)
            lines.append("\nThe output has to be a rectangular grid of numbers.\n")
        else:
            truth = train_out[i]
            if len(rr.grid) != len(truth) or any(
                len(p) != len(t) for p, t in zip(rr.grid, truth)
            ):
                scores.append(0.0)
                pred_shape = (len(rr.grid), len(rr.grid[0]) if rr.grid else 0)
                true_shape = (len(truth), len(truth[0]) if truth else 0)
                lines.append(
                    f"\n\nShape mismatch: your prediction's shape was {pred_shape}, "
                    f"while the correct shape was {true_shape}."
                )
            else:
                lines.append(
                    "\nYour code's output does not match the expected output."
                    "\n\nBelow is a visualization of the 2D array your code produced as well as the expected output.\n"
                    "Correctly predicted values are shown as-is while the incorrectly predicted values are shown "
                    "in the format 'prediction/correct':\n"
                )
                lines.append(f"\n```\n{array_diff(rr.grid, truth)}\n```\n")
                s = soft_score(rr.grid, truth)
                scores.append(s)
                lines.append(f"Output accuracy: {s:.2f} (0 is worst, 1 is best).\n")
        if rr.error:
            lines.append(f"\n\nYour code produced the following error:\n{rr.error[:800]}\n")
        parts.append("".join(lines))

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return "\n\n".join(parts), mean_score
