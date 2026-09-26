"""Loading ARC challenge/solution JSON files into Task objects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .grid import to_grid
from .types import Task


def load_challenges(path: str | Path) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_solutions(path: str | Path) -> dict[str, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tasks_from_challenges(challenges: dict[str, dict]) -> list[Task]:
    tasks = []
    for tid in sorted(challenges.keys()):
        ch = challenges[tid]
        train = [(to_grid(p["input"]), to_grid(p["output"])) for p in ch["train"]]
        test = [to_grid(p["input"]) for p in ch["test"]]
        tasks.append(Task(task_id=tid, train=train, test=test))
    return tasks


def load_tasks(
    data_dir: str | Path,
    split: str = "evaluation",
    limit: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
) -> list[Task]:
    data_dir = Path(data_dir)
    challenges = load_challenges(data_dir / f"arc-agi_{split}_challenges.json")
    if task_ids:
        challenges = {k: v for k, v in challenges.items() if k in task_ids}
    tasks = tasks_from_challenges(challenges)
    if limit:
        tasks = tasks[:limit]
    return tasks


def load_solutions_for(data_dir: str | Path, split: str = "evaluation") -> Optional[dict]:
    path = Path(data_dir) / f"arc-agi_{split}_solutions.json"
    if not path.exists():
        return None
    return load_solutions(path)
