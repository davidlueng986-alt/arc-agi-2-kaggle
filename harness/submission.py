"""Candidate pool, 2-attempt selection, submission schema and scoring.

The pool is what makes the second attempt free.  Every ``verify`` call already
runs the candidate on the test inputs and reports exact train behaviour, so the
harness ends up holding a ranked list of (program, train score, test outputs)
without spending a single extra token on it.  ``attempt_1`` is the best verified
program's output; ``attempt_2`` is the highest-ranked program whose test output
actually *differs*, which under binary exact-match scoring is strictly better
than repeating ``attempt_1``.

Land's held-out verification experiment is the reason ``attempt_2`` is not just a
copy: pass rate 88%, verified rate 38%, verified-but-wrong 1 -- precision ~97%,
recall ~43%.  Verification is a strong positive signal and a weak negative one,
so a verified program leads, but an unverified high-``soft_score`` program is a
legitimate second attempt rather than a discard.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from .grids import Grid, Task, shape_of
from .oracle import cell_accuracy, grid_equal, normalize_grid

ATTEMPT_KEYS = ("attempt_1", "attempt_2")


@dataclass
class Candidate:
    """One registered program plus its exact oracle verdict."""

    rank_key: tuple = ()
    source: str | None = None
    exact: bool = False
    soft_score: float = 0.0
    pass_rate: float = 0.0
    n_pairs: int = 0
    summary: str = ""
    test_outputs: list[Any] = field(default_factory=list)
    test_errors: list[str] = field(default_factory=list)
    turn_index: int = 0
    source_lines: int = 0
    distinct_id: int = 0

    def sort_key(self) -> tuple:
        """Higher is better. Verified first, then soft score, then simpler, then earlier."""
        return (
            1 if self.exact else 0,
            round(self.soft_score, 6),
            -self.source_lines,
            -self.turn_index,
        )


class CandidatePool:
    """Thread-confined per task; capped so a chatty agent cannot grow it forever."""

    def __init__(self, max_size: int = 24) -> None:
        self.max_size = max(1, int(max_size))
        self.candidates: list[Candidate] = []
        self._counter = 0
        self.rejected = 0

    def add(
        self,
        *,
        source: str | None,
        exact: bool,
        soft_score: float,
        pass_rate: float,
        n_pairs: int,
        summary: str,
        test_outputs: Sequence[Any],
        test_errors: Sequence[str] = (),
        turn_index: int = 0,
    ) -> tuple[Candidate | None, dict[str, Any]]:
        cleaned: list[Any] = []
        for grid in test_outputs:
            normalized = normalize_grid(grid)
            if normalized is not None:
                cleaned.append(normalized)
        if not cleaned:
            self.rejected += 1
            return None, {
                "registered": False,
                "reason": "candidate produced no valid test output grid (values must be ints 0-9, rectangular, <=64 per side)",
            }
        if len(cleaned) < len(test_outputs):
            # Keep going: a candidate that answers 1 of 2 test inputs is still
            # worth registering, and `_pad` fills the gap from the input grid.
            pass
        if len(self.candidates) >= self.max_size:
            # Keep the pool bounded: drop the current worst if this one is better.
            worst_index = min(range(len(self.candidates)), key=lambda i: self.candidates[i].sort_key())
            if self.candidates[worst_index].sort_key() >= self._key(
                exact, soft_score, source, turn_index
            ):
                self.rejected += 1
                return None, {"registered": False, "reason": "candidate pool full of better candidates"}
            self.candidates.pop(worst_index)
        self._counter += 1
        candidate = Candidate(
            source=source,
            exact=bool(exact),
            soft_score=float(soft_score),
            pass_rate=float(pass_rate),
            n_pairs=int(n_pairs),
            summary=str(summary),
            test_outputs=cleaned,
            test_errors=[str(e) for e in test_errors],
            turn_index=int(turn_index),
            source_lines=len((source or "").strip().splitlines()),
            distinct_id=self._counter,
        )
        candidate.rank_key = candidate.sort_key()
        self.candidates.append(candidate)
        return candidate, {"registered": True}

    @staticmethod
    def _key(exact: bool, soft_score: float, source: str | None, turn_index: int) -> tuple:
        return (1 if exact else 0, round(float(soft_score), 6), -len((source or "").splitlines()), -turn_index)

    def ranked(self) -> list[Candidate]:
        return sorted(self.candidates, key=lambda c: c.sort_key(), reverse=True)

    def verified(self) -> list[Candidate]:
        return [c for c in self.ranked() if c.exact]

    def distinct_verified_outputs(self) -> int:
        seen: list[list[Any]] = []
        for candidate in self.verified():
            if not any(_same_outputs(existing, candidate.test_outputs) for existing in seen):
                seen.append(candidate.test_outputs)
        return len(seen)

    def best_summary_lines(self, limit: int = 4) -> list[str]:
        lines = []
        for candidate in self.ranked()[:limit]:
            lines.append(
                "cand#%d %s soft=%.2f (%d lines)"
                % (
                    candidate.distinct_id,
                    "VERIFIED" if candidate.exact else candidate.summary[:60],
                    candidate.soft_score,
                    candidate.source_lines,
                )
            )
        return lines


def _same_outputs(a: Sequence[Any], b: Sequence[Any]) -> bool:
    if len(a) != len(b):
        return False
    return all(grid_equal(x, y) for x, y in zip(a, b))


def _fallback_grid(test_input: Grid) -> Grid:
    """Always-valid placeholder: copy the input.

    Guarantees the schema is satisfiable for every task even if the agent never
    produced anything, which matters because a missing task id or a missing
    attempt invalidates the submission.
    """
    return [list(row) for row in test_input]


def select_attempts(
    pool: CandidatePool,
    task: Task,
    *,
    allow_duplicate: bool = True,
) -> list[dict[str, Any]]:
    """Build ``[{"attempt_1": grid, "attempt_2": grid}, ...]`` aligned to test inputs.

    The second-attempt policy depends on whether anything verified, because the
    two cases have different dominant risks:

    * **Something verified on every train pair.**  Land's held-out experiment put
      precision at ~97%, so the leader is very likely right and the residual risk
      is small.  ``attempt_2`` is therefore the next-best candidate by
      ``soft_score`` -- maximising the probability that one of the two is correct
      beats hedging against a family-level mistake we have good reason to doubt.
    * **Nothing verified.**  Now the dominant risk *is* having the whole
      transformation family wrong (usually visible as a shape error), so
      ``attempt_2`` prefers a candidate with a **different output shape**: two
      attempts that disagree about the shape cover two families instead of one.
    """
    ranked = pool.ranked()
    n_test = task.n_test

    primary = next((c for c in ranked if _usable(c, n_test)), None)
    primary_outputs = _pad(primary.test_outputs, n_test, task) if primary else None
    have_verified = bool(pool.verified())

    secondary: list[Grid] | None = None
    if primary_outputs is not None:
        if have_verified:
            for candidate in ranked:
                if candidate is primary or not _usable(candidate, n_test):
                    continue
                outputs = _pad(candidate.test_outputs, n_test, task)
                if any(not grid_equal(a, b) for a, b in zip(outputs, primary_outputs)):
                    secondary = outputs
                    break
        else:
            secondary = _shape_diverse_outputs(ranked, primary, primary_outputs, n_test, task)

    out: list[dict[str, Any]] = []
    for index in range(n_test):
        first = primary_outputs[index] if primary_outputs else _fallback_grid(task.test_inputs[index])
        second = secondary[index] if secondary is not None else None
        if second is None or grid_equal(first, second):
            second = _second_choice_for_index(ranked, index, first, task, prefer_shape=not have_verified)
        if second is None:
            second = [list(row) for row in first] if allow_duplicate else _fallback_grid(task.test_inputs[index])
        out.append({"attempt_1": first, "attempt_2": second})
    return out


def _shape_diverse_outputs(
    ranked: Sequence[Candidate],
    primary: Candidate,
    primary_outputs: Sequence[Grid],
    n_test: int,
    task: Task,
) -> list[Grid] | None:
    """Highest-ranked candidate whose output shapes differ from the leader's."""
    primary_shapes = [shape_of(g) for g in primary_outputs]
    best_same = None
    for candidate in ranked:
        if candidate is primary or not _usable(candidate, n_test):
            continue
        outputs = _pad(candidate.test_outputs, n_test, task)
        if all(grid_equal(a, b) for a, b in zip(outputs, primary_outputs)):
            continue
        if [shape_of(g) for g in outputs] != primary_shapes:
            return outputs
        if best_same is None:
            best_same = outputs
    return best_same


def _usable(candidate: Candidate, n_test: int) -> bool:
    return bool(candidate.test_outputs) and all(
        normalize_grid(g) is not None for g in candidate.test_outputs[:n_test]
    )


def _pad(outputs: Sequence[Any], n_test: int, task: Task) -> list[Grid]:
    result: list[Grid] = []
    for index in range(n_test):
        grid = normalize_grid(outputs[index]) if index < len(outputs) else None
        result.append(grid if grid is not None else _fallback_grid(task.test_inputs[index]))
    return result


def _second_choice_for_index(
    ranked: Sequence[Candidate], index: int, first: Grid, task: Task, *, prefer_shape: bool = False
) -> Grid | None:
    """Best grid for one test index that differs from ``first``.

    With ``prefer_shape`` (used when nothing verified) a different *shape* wins
    over a higher partial score, because under exact-match scoring the dominant
    failure is having the transformation family -- and therefore the output
    dimensions -- wrong.
    """
    first_shape = shape_of(first)
    different_shape = None
    different_cells = None
    for candidate in ranked:
        if index >= len(candidate.test_outputs):
            continue
        grid = normalize_grid(candidate.test_outputs[index])
        if grid is None or grid_equal(grid, first):
            continue
        if shape_of(grid) != first_shape:
            if different_shape is None:
                different_shape = grid
            if prefer_shape:
                break
        elif different_cells is None:
            different_cells = grid
        if different_shape is not None and different_cells is not None:
            break
    if prefer_shape:
        return different_shape or different_cells
    return different_cells or different_shape


def build_submission(
    predictions: dict[str, list[dict[str, Any]]],
    task_shapes: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Normalise into the competition schema, filling every missing task/attempt."""
    submission: dict[str, list[dict[str, Any]]] = {}
    for task_id, n_test in task_shapes.items():
        entries = predictions.get(task_id) or []
        packed: list[dict[str, Any]] = []
        for index in range(max(0, int(n_test))):
            entry = entries[index] if index < len(entries) else {}
            if not isinstance(entry, dict):
                entry = {}
            packed.append(
                {
                    "attempt_1": entry.get("attempt_1") or [],
                    "attempt_2": entry.get("attempt_2") or [],
                }
            )
        submission[task_id] = packed
    return submission


def validate_submission(submission: Any, task_shapes: dict[str, int]) -> list[str]:
    """Return a list of schema problems; empty means valid."""
    problems: list[str] = []
    if not isinstance(submission, dict):
        return ["submission is not a JSON object"]
    for task_id, n_test in task_shapes.items():
        if task_id not in submission:
            problems.append(f"missing task_id {task_id}")
            continue
        entries = submission[task_id]
        if not isinstance(entries, list):
            problems.append(f"{task_id}: value is not a list")
            continue
        if len(entries) != n_test:
            problems.append(f"{task_id}: expected {n_test} entries, got {len(entries)}")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                problems.append(f"{task_id}[{index}]: not an object")
                continue
            for key in ATTEMPT_KEYS:
                if key not in entry:
                    problems.append(f"{task_id}[{index}]: missing {key}")
                    continue
                grid = entry[key]
                if not isinstance(grid, list) or not grid:
                    problems.append(f"{task_id}[{index}].{key}: empty or not a list")
                    continue
                width = None
                for row in grid:
                    if not isinstance(row, list) or not row:
                        problems.append(f"{task_id}[{index}].{key}: malformed row")
                        width = None
                        break
                    if width is None:
                        width = len(row)
                    elif len(row) != width:
                        problems.append(f"{task_id}[{index}].{key}: ragged rows")
                        break
                    for cell in row:
                        if not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 9:
                            problems.append(f"{task_id}[{index}].{key}: cell {cell!r} not an int 0-9")
                            break
    return problems


def write_submission(path: str | os.PathLike, submission: dict[str, Any]) -> None:
    """Atomic cumulative write -- safe to call after every task and from a watchdog.

    ``os.replace`` is atomic on POSIX, which is what the Kaggle rerun uses.  On
    Windows it fails with a sharing violation if another process happens to have
    the destination open, so the rename is retried briefly; the temp file name is
    pid-scoped so concurrent writers never collide on it either.
    """
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(submission, handle)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.01 * (attempt + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    if last_error is not None:
        raise last_error


def score_submission(
    submission: dict[str, Any], solutions: dict[str, Sequence[Grid]]
) -> dict[str, Any]:
    """Competition scoring: mean over all test outputs, point iff either attempt matches."""
    total = 0
    correct = 0
    per_task: dict[str, float] = {}
    for task_id, ground_truth in solutions.items():
        entries = submission.get(task_id) or []
        hits = 0
        for index, expected in enumerate(ground_truth):
            total += 1
            entry = entries[index] if index < len(entries) else {}
            if not isinstance(entry, dict):
                continue
            for key in ATTEMPT_KEYS:
                if grid_equal(normalize_grid(entry.get(key)), normalize_grid(expected)):
                    hits += 1
                    correct += 1
                    break
        per_task[task_id] = hits / max(1, len(ground_truth))
    return {
        "n_outputs": total,
        "n_correct": correct,
        "score": correct / total if total else 0.0,
        "per_task": per_task,
    }


def score_task_entries(
    entries: Sequence[dict[str, Any]], ground_truth: Sequence[Grid]
) -> float:
    if not ground_truth:
        return 0.0
    correct = 0
    for index, expected in enumerate(ground_truth):
        entry = entries[index] if index < len(entries) else {}
        if not isinstance(entry, dict):
            continue
        for key in ATTEMPT_KEYS:
            if grid_equal(normalize_grid(entry.get(key)), normalize_grid(expected)):
                correct += 1
                break
    return correct / len(ground_truth)
