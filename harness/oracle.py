"""Train-pair oracle: exact verification, soft scoring and diff diagnostics.

The train pairs are a free, exact, deterministic oracle for synthesized
programs. Land's held-out measurement (precision ~97%, recall ~43%) means
verification is a STRONG POSITIVE and a WEAK NEGATIVE signal, so nothing is
ever discarded for failing it: soft scores keep partial programs ranked
(see pool.py / disambiguate.py fallback path).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .sandbox import Sandbox
from .types import ExecResult, Grid, PairScore, ProgramEval


def score_output(pred: Optional[Grid], truth: Grid) -> PairScore:
    """Score one predicted grid against ground truth."""
    if pred is None:
        return PairScore(exact=False, soft=0.0, shape_ok=False, error="no output", diff=None)
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    shape_ok = pred.shape == truth.shape
    if not shape_ok:
        return PairScore(
            exact=False,
            soft=0.0,
            shape_ok=False,
            error=f"shape mismatch: got {pred.shape}, expected {truth.shape}",
            diff=None,
        )
    eq = pred == truth
    exact = bool(eq.all())
    soft = float(np.nan_to_num(np.mean(eq)))
    return PairScore(exact=exact, soft=soft, shape_ok=True, error=None, diff=array_diff(pred, truth))


def array_diff(pred: Grid, truth: Grid) -> str:
    """Poetiq-style diagnostic: matching cells as-is, mismatches as 'pred/truth'."""
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    lines = []
    for i in range(truth.shape[0]):
        row = []
        for j in range(truth.shape[1]):
            if pred[i, j] == truth[i, j]:
                row.append(str(int(truth[i, j])))
            else:
                row.append(f"{int(pred[i, j])}/{int(truth[i, j])}")
        lines.append(" ".join(row))
    return "\n".join(lines)


def evaluate_program(
    sandbox: Sandbox,
    code: str,
    pairs: Sequence[tuple[Grid, Grid]],
) -> ProgramEval:
    """Run a program over (input, output) pairs; exact verification + soft scores."""
    jobs = [(code, x) for x, _ in pairs]
    results: list[ExecResult] = sandbox.run_batch(jobs)
    per_pair: list[PairScore] = []
    n_err = 0
    for res, (_, truth) in zip(results, pairs):
        if not res.ok:
            n_err += 1
            per_pair.append(
                PairScore(exact=False, soft=0.0, shape_ok=False, error=res.error or "exec failed", diff=None)
            )
        else:
            per_pair.append(score_output(res.output, truth))
    verified = bool(per_pair) and all(ps.exact for ps in per_pair)
    soft_mean = float(np.mean([ps.soft for ps in per_pair])) if per_pair else 0.0
    return ProgramEval(per_pair=per_pair, verified=verified, soft_mean=soft_mean, n_exec_errors=n_err)


def run_on_inputs(
    sandbox: Sandbox, code: str, inputs: Sequence[Grid]
) -> list[Optional[Grid]]:
    """Run a program on raw inputs (no truth); failed runs yield None."""
    results = sandbox.run_batch([(code, x) for x in inputs])
    return [r.output if r.ok else None for r in results]


def build_feedback(eval_result: ProgramEval, pairs: Sequence[tuple[Grid, Grid]]) -> tuple[str, float]:
    """Textual feedback for the refinement loop + overall soft score."""
    parts: list[str] = []
    for i, ps in enumerate(eval_result.per_pair):
        if ps.exact:
            parts.append(f"Example #{i + 1}: CORRECT (exact match).")
            continue
        head = f"Example #{i + 1}: WRONG."
        if ps.error:
            parts.append(f"{head} Error: {ps.error}")
            continue
        block = [head]
        if not ps.shape_ok:
            block.append(ps.error or "shape mismatch")
        else:
            block.append(
                "Cells shown as `predicted/expected` where wrong, plain where right:"
            )
            block.append("```\n" + (ps.diff or "") + "\n```")
            block.append(f"Cell accuracy: {ps.soft:.2f}")
        parts.append("\n".join(block))
    return "\n\n".join(parts), eval_result.soft_mean
