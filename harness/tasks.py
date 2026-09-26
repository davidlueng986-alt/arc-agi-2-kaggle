"""ARC task loading, container types, and Kaggle path discovery."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .grids import Grid, is_grid

DEV_DATA_DIR = r"C:\Users\daive\Downloads\arc-agi-2\data"
KAGGLE_COMP_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
KAGGLE_WORKING = "/kaggle/working"

SPLITS = ("training", "evaluation", "test")


@dataclass(frozen=True)
class ARCTask:
    task_id: str
    train: Tuple[Tuple[Grid, Grid], ...]
    test: Tuple[Grid, ...]
    solutions: Optional[Tuple[Grid, ...]] = field(default=None, compare=False)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def prompt_chars(self) -> int:
        total = 0
        for a, b in self.train:
            total += _csv_chars(a) + _csv_chars(b)
        for g in self.test:
            total += _csv_chars(g)
        return total

    def approx_prompt_tokens(self, chars_per_token: float = 3.2, overhead: int = 320) -> int:
        return int(self.prompt_chars() / chars_per_token) + overhead


def _csv_chars(grid: Grid) -> int:
    return sum(len(",".join(map(str, row))) for row in grid) + max(0, len(grid) - 1)


def _validate_raw(task_id: str, raw: dict) -> Tuple[Tuple[Tuple[Grid, Grid], ...], Tuple[Grid, ...]]:
    train_raw = raw.get("train") or []
    test_raw = raw.get("test") or []
    if not isinstance(train_raw, list) or not isinstance(test_raw, list):
        raise ValueError(f"{task_id}: train/test must be lists")
    train: List[Tuple[Grid, Grid]] = []
    for i, ex in enumerate(train_raw):
        gi, go = ex.get("input"), ex.get("output")
        if not is_grid(gi) or not is_grid(go):
            raise ValueError(f"{task_id}: bad train pair {i}")
        train.append((gi, go))
    test: List[Grid] = []
    for i, ex in enumerate(test_raw):
        gi = ex.get("input")
        if not is_grid(gi):
            raise ValueError(f"{task_id}: bad test input {i}")
        test.append(gi)
    if not test:
        raise ValueError(f"{task_id}: no test inputs")
    return tuple(train), tuple(test)


def load_challenges(path: str, solutions_path: Optional[str] = None) -> List[ARCTask]:
    with open(path, "r", encoding="utf-8") as fh:
        blob: Dict[str, dict] = json.load(fh)
    sols: Dict[str, List[Grid]] = {}
    if solutions_path and os.path.exists(solutions_path):
        with open(solutions_path, "r", encoding="utf-8") as fh:
            sols = json.load(fh)

    tasks: List[ARCTask] = []
    for task_id in sorted(blob.keys()):
        train, test = _validate_raw(task_id, blob[task_id])
        gt: Optional[Tuple[Grid, ...]] = None
        if task_id in sols:
            raw_sols = sols[task_id]
            if isinstance(raw_sols, list) and all(is_grid(g) for g in raw_sols) and len(raw_sols) == len(test):
                gt = tuple(raw_sols)
        tasks.append(ARCTask(task_id=task_id, train=train, test=test, solutions=gt))
    return tasks


def task_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return sorted(json.load(fh).keys())


def resolve_data_dir(explicit: Optional[str] = None) -> str:
    for cand in (
        explicit,
        os.environ.get("ARC_DATA_DIR"),
        KAGGLE_COMP_DIR,
        os.path.join(os.getcwd(), "data"),
        DEV_DATA_DIR,
    ):
        if cand and os.path.isdir(cand):
            return cand
    raise FileNotFoundError("could not locate ARC data directory")


def resolve_challenges_path(data_dir: str, split: str = "test") -> str:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return os.path.join(data_dir, f"arc-agi_{split}_challenges.json")


def resolve_solutions_path(data_dir: str, split: str) -> Optional[str]:
    path = os.path.join(data_dir, f"arc-agi_{split}_solutions.json")
    return path if os.path.exists(path) else None


def resolve_output_dir(explicit: Optional[str] = None) -> str:
    for cand in (explicit, os.environ.get("ARC_OUTPUT_DIR")):
        if cand:
            return cand
    if os.path.isdir(KAGGLE_WORKING):
        return KAGGLE_WORKING
    return os.path.join(os.getcwd(), "output")


def is_kaggle_rerun() -> bool:
    return bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))


def total_test_outputs(tasks: Sequence[ARCTask]) -> int:
    return sum(t.n_test for t in tasks)
