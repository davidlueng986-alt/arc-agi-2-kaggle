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
import threading
import time
from typing import Dict, List, Optional, Sequence

from .grids import Grid
from .tasks import ARCTask

PLACEHOLDER: Grid = [[0, 0], [0, 0]]
ATTEMPT_KEYS = ("attempt_1", "attempt_2")
WRITE_ATTEMPTS = 4
WRITE_RETRY_DELAY = 0.05


def _unlink_quietly(path: str) -> None:
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


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
        self._lock = threading.RLock()
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
        with self._lock:
            self.data[task_id] = entry

    def write(self) -> None:
        """Atomically snapshot the submission to disk.

        Guarded by a re-entrant lock because the watchdog thread and the scheduler
        thread both write. Without it, concurrent `os.replace` raises PermissionError
        on Windows, and `json.dump` iterating `self.data` while `update` mutates it
        raises "dictionary changed size during iteration" on Linux. Losing the write
        at the generation deadline would cost the entire run, so the replace is also
        retried briefly to ride out transient filesystem locks.
        """
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            snapshot = json.dumps(self.data)
            last_error: Optional[BaseException] = None
            for attempt in range(WRITE_ATTEMPTS):
                fd, tmp = tempfile.mkstemp(dir=directory, prefix=".submission-", suffix=".tmp")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        fh.write(snapshot)
                    os.replace(tmp, self.path)
                    self.writes += 1
                    return
                except PermissionError as exc:
                    last_error = exc
                    _unlink_quietly(tmp)
                    time.sleep(WRITE_RETRY_DELAY * (attempt + 1))
                except BaseException:
                    _unlink_quietly(tmp)
                    raise
            raise last_error if last_error else RuntimeError("submission write failed")

    def validate(self) -> List[str]:
        with self._lock:
            return _validate_data(self.data, self.tasks, self.order)

    def is_complete(self) -> bool:
        return not self.validate()

    @staticmethod
    def load(path: str) -> Dict[str, List[Dict[str, Grid]]]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def _validate_data(
    data: Dict[str, List[Dict[str, Grid]]],
    tasks: Dict[str, ARCTask],
    order: Sequence[str],
) -> List[str]:
    errors: List[str] = []
    for task_id in order:
        task = tasks[task_id]
        entries = data.get(task_id)
        if entries is None:
            errors.append(f"{task_id}: missing")
            continue
        if len(entries) != task.n_test:
            errors.append(f"{task_id}: {len(entries)} entries, expected {task.n_test}")
            continue
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry.keys()) != set(ATTEMPT_KEYS):
                errors.append(f"{task_id}[{i}]: keys {sorted(entry) if isinstance(entry, dict) else type(entry).__name__}")
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
    extra = set(data) - set(order)
    if extra:
        errors.append(f"unexpected task_ids: {sorted(extra)[:5]}")
    return errors


def validate_submission_obj(
    submission: Dict[str, List[Dict[str, Grid]]], tasks: Sequence[ARCTask]
) -> List[str]:
    return _validate_data(
        submission, {t.task_id: t for t in tasks}, [t.task_id for t in tasks]
    )
