"""ARC-AGI-2 core utilities: task loading, grid helpers, submission I/O."""
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(split="training"):
    ch = load_tasks(DATA_DIR / f"arc-agi_{split}_challenges.json")
    sol_path = DATA_DIR / f"arc-agi_{split}_solutions.json"
    sol = load_tasks(sol_path) if sol_path.exists() else {}
    return ch, sol


def grid_size(grid):
    return len(grid), len(grid[0]) if grid else (0, 0)


def grids_equal(a, b):
    return a == b


def is_valid_grid(g):
    if not isinstance(g, list) or not g:
        return False
    w = None
    for row in g:
        if not isinstance(row, list):
            return False
        if w is None:
            w = len(row)
        elif len(row) != w:
            return False
        if not all(isinstance(c, int) and 0 <= c <= 9 for c in row):
            return False
    return True


def make_submission(solver, challenges_path, out_path, default_grid=None):
    """Run solver over all tasks and write a well-formed submission.json.

    solver: callable(task_id, task) -> list of attempts per test input,
            each attempt is a grid (list of lists). Exactly 2 attempts
            are written per test input.
    """
    if default_grid is None:
        default_grid = [[0, 0], [0, 0]]
    tasks = load_tasks(challenges_path)
    submission = {}
    for task_id, task in tasks.items():
        try:
            preds = solver(task_id, task)
        except Exception:
            preds = []
        if not isinstance(preds, list):
            preds = [preds]
        # pad/trim to number of test inputs
        n_test = len(task.get("test", []))
        while len(preds) < n_test:
            preds.append(default_grid)
        preds = preds[:n_test]
        entry = []
        for p in preds:
            for i in range(2):
                if not is_valid_grid(p):
                    p = default_grid
            entry.append({"attempt_1": p, "attempt_2": p})
        submission[task_id] = entry
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f)
    return submission


def score_submission(submission, solutions_path):
    """Official scoring: per test-output exact match with 2 attempts, averaged."""
    sols = load_tasks(solutions_path)
    total, correct = 0, 0
    for task_id, outs in sols.items():
        preds = submission.get(task_id, [])
        for i, gt in enumerate(outs):
            total += 1
            if i < len(preds):
                p = preds[i]
                a1 = p.get("attempt_1")
                a2 = p.get("attempt_2")
                if a1 == gt or a2 == gt:
                    correct += 1
    return correct / total if total else 0.0, correct, total
