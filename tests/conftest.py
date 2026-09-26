import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.environ.get("ARC_DATA_DIR", r"C:\Users\daive\Downloads\arc-agi-2\data")


@pytest.fixture(scope="session")
def eval_challenges() -> dict:
    path = os.path.join(DATA_DIR, "arc-agi_evaluation_challenges.json")
    if not os.path.exists(path):
        pytest.skip(f"eval challenges not available at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def eval_solutions() -> dict:
    path = os.path.join(DATA_DIR, "arc-agi_evaluation_solutions.json")
    if not os.path.exists(path):
        pytest.skip(f"eval solutions not available at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ten_tasks(eval_challenges, eval_solutions):
    ids = sorted(k for k in eval_challenges if k in eval_solutions)[:10]
    return (
        {k: eval_challenges[k] for k in ids},
        {k: eval_solutions[k] for k in ids},
    )


@pytest.fixture
def tiny_task():
    """A trivial synthetic task: output = input with every cell +1 mod 10."""
    train_in = [[[0, 1], [2, 3]], [[9, 8], [7, 6]]]
    train_out = [[[1, 2], [3, 4]], [[0, 9], [8, 7]]]
    test_in = [[[5, 5], [0, 1]]]
    return {
        "train": [{"input": i, "output": o} for i, o in zip(train_in, train_out)],
        "test": [{"input": i} for i in test_in],
    }
