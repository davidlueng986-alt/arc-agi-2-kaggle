from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness.tasks import (  # noqa: E402
    DEV_DATA_DIR,
    load_challenges,
    resolve_solutions_path,
)

DATA_DIR = os.environ.get("ARC_DATA_DIR", DEV_DATA_DIR)


@pytest.fixture(scope="session")
def data_dir() -> str:
    if not os.path.isdir(DATA_DIR):
        pytest.skip(f"ARC data dir not available: {DATA_DIR}")
    return DATA_DIR


@pytest.fixture(scope="session")
def eval_tasks(data_dir):
    path = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")
    sols = resolve_solutions_path(data_dir, "evaluation")
    return load_challenges(path, sols)


@pytest.fixture(scope="session")
def test_tasks(data_dir):
    path = os.path.join(data_dir, "arc-agi_test_challenges.json")
    return load_challenges(path, resolve_solutions_path(data_dir, "test"))


@pytest.fixture
def tiny_task():
    from harness.tasks import ARCTask

    train = (
        ([[0, 1], [1, 0]], [[1, 0], [0, 1]]),
        ([[2, 2], [0, 0]], [[0, 0], [2, 2]]),
    )
    return ARCTask(
        task_id="tiny0001",
        train=train,
        test=([[3, 3], [0, 0]],),
        solutions=([[0, 0], [3, 3]],),
    )


@pytest.fixture
def two_test_task():
    from harness.tasks import ARCTask

    train = (([[1]], [[2]]),)
    return ARCTask(
        task_id="tiny0002",
        train=train,
        test=([[3]], [[4]]),
        solutions=([[4]], [[5]]),
    )
