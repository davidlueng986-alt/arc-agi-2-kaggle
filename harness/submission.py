"""Cumulative submission writer + schema validation + watchdog finalization.

Kaggle contract (competition-brief.md):
  - every task_id in the test challenges MUST appear;
  - value = list with one {"attempt_1", "attempt_2"} dict per test input, in
    test-input order;
  - grids are rectangular int matrices; both attempts always present.
Written cumulatively after every task (Poetiq main.py pattern) and atomically
(tmp + os.replace) so a hard kill can never leave a truncated file.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Callable, Optional

from harness.grids import modal_color
from harness.types import Grid


def fallback_attempts(test_inputs: list[Grid], train_out: list[Grid]) -> list[dict]:
    """Schema-valid placeholder attempts for an unsolved task: identity, then
    the modal train-output colour fill. Never empty."""
    color = modal_color(train_out) if train_out else 0
    out = []
    for tin in test_inputs:
        grid = tin if tin and tin[0] else [[0]]
        identity = [list(row) for row in grid]
        alt = [[(color + 1) % 10] * len(row) for row in grid]
        if alt == identity:
            alt = [[color] * len(row) for row in grid]
        out.append({"attempt_1": identity, "attempt_2": alt})
    return out


class SubmissionWriter:
    def __init__(self, path: str, challenges: dict):
        self.path = path
        self.challenges = challenges  # task_id -> {"train": [...], "test": [...]}
        self.submission: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def update(self, task_id: str, attempts: list[dict]) -> None:
        with self._lock:
            self.submission[task_id] = attempts
            self._write_locked()

    def write(self) -> None:
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        """Atomic write: tmp file in the same directory + os.replace.
        Safe against concurrent watchdog writes (lock) and hard kills (atomic)."""
        d = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.submission, f)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def finalize(self, solver_fn: Optional[Callable[[str], Optional[list[dict]]]] = None) -> dict:
        """Fill every missing task (optionally via solver_fn) with schema-valid
        fallbacks, write once more, and return validation errors (should be [])."""
        with self._lock:
            for task_id, task in self.challenges.items():
                if task_id in self.submission and self.submission[task_id]:
                    continue
                attempts = None
                if solver_fn is not None:
                    try:
                        attempts = solver_fn(task_id)
                    except Exception:  # noqa: BLE001
                        attempts = None
                if not attempts:
                    test_inputs = [ex["input"] for ex in task.get("test", [])]
                    train_out = [ex["output"] for ex in task.get("train", [])]
                    attempts = fallback_attempts(test_inputs, train_out)
                self.submission[task_id] = attempts
            self._write_locked()
            return {"errors": validate_submission(self.submission, self.challenges)}


def validate_submission(submission: dict, challenges: dict) -> list[str]:
    """Full schema check. Returns a list of human-readable errors ([] = valid)."""
    errors: list[str] = []
    if not isinstance(submission, dict):
        return ["submission is not a dict"]
    for task_id, task in challenges.items():
        if task_id not in submission:
            errors.append(f"{task_id}: missing from submission")
            continue
        entries = submission[task_id]
        n_test = len(task.get("test", []))
        if not isinstance(entries, list):
            errors.append(f"{task_id}: value is not a list")
            continue
        if len(entries) != n_test:
            errors.append(f"{task_id}: expected {n_test} entries, got {len(entries)}")
            continue
        for j, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"{task_id}[{j}]: entry is not a dict")
                continue
            for key in ("attempt_1", "attempt_2"):
                if key not in entry:
                    errors.append(f"{task_id}[{j}]: missing {key}")
                    continue
                if not _valid_grid(entry[key]):
                    errors.append(f"{task_id}[{j}]: {key} is not a rectangular int grid")
    for task_id in submission:
        if task_id not in challenges:
            errors.append(f"{task_id}: present in submission but not in challenges")
    return errors


def _valid_grid(g) -> bool:
    if not isinstance(g, list) or not g:
        return False
    width = None
    for row in g:
        if not isinstance(row, list) or not row:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        for c in row:
            if not isinstance(c, int) or isinstance(c, bool):
                return False
    return True
