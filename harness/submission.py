"""Cumulative submission writer with schema validation.

The writer is seeded with the competition's own placeholder grid for every task and
every test index before generation starts, then re-written after every wave. A hard
timeout at any point therefore still leaves a complete, schema-valid
`submission.json` on disk. Writes are atomic (temp file + `os.replace`).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Optional, Sequence

from .grids import Grid
from .tasks import ARCTask

PLACEHOLDER: Grid = [[0, 0], [0, 0]]
ATTEMPT_KEYS = ("attempt_1", "attempt_2")


def _coerce_grid(obj: object) -> Grid:
    if isinstance(obj, list) and obj and all(isinstance(r, list) for r in obj):
        out: Grid = []
        for row in obj:
            cells: List[int] = []
            for cell in row:
                if isinstance(cell, bool):
                    cells.append(int(cell))
                elif isinstance(cell, int):
                    cells.append(int(cell))
                elif isinstance(cell, float) and float(cell).is_integer():
                    cells.append(int(cell))
                else:
                    raise ValueError(f"non-integer cell in grid: {cell!r}")
            out.append(cells)
        return out
    raise ValueError(f"not a grid: {type(obj).__name__}")


def build_entry(grids: Sequence[Grid]) -> Dict[str, Grid]:
    attempt_1 = grids[0] if len(grids) > 0 else PLACEHOLDER
    attempt_2 = grids[1] if len(grids) > 1 else attempt_1
    return {"attempt_1": [list(r) for r in attempt_1], "attempt_2": [list(r) for r in attempt_2]}


def empty_submission(tasks: Sequence[ARCTask]) -> Dict[str, List[Dict[str, Grid]]]:
    return {
        task.task_id: [build_entry([PLACEHOLDER]) for _ in range(task.n_test)] for task in tasks
    }


class SubmissionWriter:
    def __init__(self, path: str, tasks: Sequence[ARCTask]) -> None:
        self.path = path
        self.tasks = {t.task_id: t for t in tasks}
        self.order = [t.task_id for t in tasks]
        self.data: Dict[str, List[Dict[str, Grid]]] = empty_submission(tasks)
        self.solved: Dict[str, int] = {}
        self.writes = 0
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def update(self, task_id: str, predictions: Sequence[Dict[str, Grid]]) -> None:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task_id {task_id!r}")
        if len(predictions) != task.n_test:
            raise ValueError(
                f"{task_id}: expected {task.n_test} predictions, got {len(predictions)}"
            )
        entry: List[Dict[str, Grid]] = []
        for pred in predictions:
            present = [k for k in ATTEMPT_KEYS if pred.get(k) is not None]
            grids = [_coerce_grid(pred[k]) for k in present]
            entry.append(build_entry(grids))
        self.data[task_id] = entry

    def write(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".submission-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh)
            os.replace(tmp, self.path)
            self.writes += 1
        except BaseException:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            raise

    def validate(self) -> List[str]:
        errors: List[str] = []
        for task_id in self.order:
            task = self.tasks[task_id]
            entries = self.data.get(task_id)
            if entries is None:
                errors.append(f"{task_id}: missing")
                continue
            if len(entries) != task.n_test:
                errors.append(f"{task_id}: {len(entries)} entries, expected {task.n_test}")
                continue
            for i, entry in enumerate(entries):
                if set(entry.keys()) != set(ATTEMPT_KEYS):
                    errors.append(f"{task_id}[{i}]: keys {sorted(entry.keys())}")
                    continue
                for key in ATTEMPT_KEYS:
                    grid = entry[key]
                    if not isinstance(grid, list) or not grid:
                        errors.append(f"{task_id}[{i}].{key}: empty")
                        continue
                    if not all(isinstance(r, list) and r for r in grid):
                        errors.append(f"{task_id}[{i}].{key}: ragged or empty rows")
                        continue
                    width = len(grid[0])
                    if any(len(r) != width for r in grid):
                        errors.append(f"{task_id}[{i}].{key}: non-rectangular")
                        continue
                    if any(
                        not isinstance(c, int) or isinstance(c, bool) or not 0 <= c <= 9
                        for r in grid
                        for c in r
                    ):
                        errors.append(f"{task_id}[{i}].{key}: cell outside 0..9")
        extra = set(self.data) - set(self.order)
        if extra:
            errors.append(f"unexpected task_ids: {sorted(extra)[:5]}")
        return errors

    def is_complete(self) -> bool:
        return not self.validate()

    @staticmethod
    def load(path: str) -> Dict[str, List[Dict[str, Grid]]]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def validate_submission_obj(
    submission: Dict[str, List[Dict[str, Grid]]], tasks: Sequence[ARCTask]
) -> List[str]:
    writer = SubmissionWriter.__new__(SubmissionWriter)
    writer.order = [t.task_id for t in tasks]
    writer.tasks = {t.task_id: t for t in tasks}
    writer.data = dict(submission)
    return writer.validate()
