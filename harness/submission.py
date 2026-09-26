"""Cumulative submission writer + schema validation.

Competition schema (ARC Prize 2026):
    {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...one per test input...]}
- every task_id must appear; both attempts always present; order matches the
  test-input order; grids are rectangular int matrices.

The file is rewritten atomically (tmp + os.replace) after EVERY task so a
12h hard-kill always leaves a complete, valid submission behind. Unfinished
tasks keep heuristic fallbacks: attempt_1 = the test input itself (identity
guess — correct for a nonzero slice of ARC tasks), attempt_2 = background fill.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .grid import background_color
from .types import Grid, Task


def fallback_attempts(test_input: Grid) -> list[dict]:
    x = np.asarray(test_input)
    a1 = x.tolist()
    bg = background_color(x)
    a2 = np.full_like(x, bg).tolist()
    return [{"attempt_1": a1, "attempt_2": a2}]


def _grid_to_list(g: Optional[Grid], like: Grid) -> list:
    if g is None:
        return np.zeros_like(np.asarray(like)).tolist()
    a = np.asarray(g)
    return [[int(v) for v in row] for row in a]


class SubmissionWriter:
    def __init__(self, path: str | Path, tasks: Sequence[Task]):
        self.path = Path(path)
        self.tasks = {t.task_id: t for t in tasks}
        self._lock = threading.RLock()
        self._data: dict[str, list[dict]] = {}
        for t in tasks:
            self._data[t.task_id] = [fallback_attempts(x)[0] for x in t.test]
        self.flush()

    def update(
        self,
        task_id: str,
        attempts: Sequence[tuple[Optional[Grid], Optional[Grid]]],
    ) -> None:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task_id {task_id}")
        packs = []
        for t_idx, x in enumerate(task.test):
            if t_idx < len(attempts):
                a1, a2 = attempts[t_idx]
            else:
                a1, a2 = None, None
            g1 = _grid_to_list(a1, x)
            g2 = _grid_to_list(a2, x)
            packs.append({"attempt_1": g1, "attempt_2": g2})
        with self._lock:
            self._data[task_id] = packs
            self.flush()

    def flush(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)

    def data(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))


def validate_submission(data: dict, tasks: Sequence[Task]) -> list[str]:
    """Return a list of schema problems (empty = valid)."""
    problems: list[str] = []
    for t in tasks:
        if t.task_id not in data:
            problems.append(f"{t.task_id}: missing")
            continue
        packs = data[t.task_id]
        if not isinstance(packs, list) or len(packs) != t.n_test:
            problems.append(f"{t.task_id}: expected {t.n_test} packs, got {len(packs) if isinstance(packs, list) else '?'}")
            continue
        for i, pack in enumerate(packs):
            for key in ("attempt_1", "attempt_2"):
                g = pack.get(key)
                if not isinstance(g, list) or not g:
                    problems.append(f"{t.task_id}[{i}].{key}: missing/empty")
                    continue
                widths = {len(r) for r in g if isinstance(r, list)}
                if len(widths) != 1 or not all(
                    isinstance(v, int) and 0 <= v <= 9 for r in g for v in r
                ):
                    problems.append(f"{t.task_id}[{i}].{key}: non-rectangular or bad values")
    return problems


def score_submission(data: dict, tasks: Sequence[Task], solutions: dict[str, list]) -> dict:
    """pass@2 ground-truth scoring: per test output and mean over all outputs."""
    total = 0
    hits = 0
    per_task: dict[str, float] = {}
    for t in tasks:
        gt = solutions.get(t.task_id)
        if gt is None:
            continue
        packs = data.get(t.task_id, [])
        task_hits = 0
        for i, truth in enumerate(gt):
            total += 1
            if i >= len(packs):
                continue
            pack = packs[i]
            if pack.get("attempt_1") == truth or pack.get("attempt_2") == truth:
                hits += 1
                task_hits += 1
        per_task[t.task_id] = task_hits / max(1, len(gt))
    return {
        "outputs_total": total,
        "outputs_hit": hits,
        "score": hits / total if total else 0.0,
        "per_task": per_task,
    }
