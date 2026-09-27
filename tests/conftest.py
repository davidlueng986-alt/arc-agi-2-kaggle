"""Shared fixtures.

Everything here runs without a GPU, without torch and without network, which is
the local constraint (RTX 5060 Laptop 8GB, no torch, cannot host Qwen3.8-27B).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.grids import Pair, Task  # noqa: E402
from harness.oracle import normalize_grid  # noqa: E402

REAL_DATA_CANDIDATES = [
    os.environ.get("ARC_DATA_DIR", ""),
    r"C:\Users\daive\Downloads\arc-agi-2\data",
    os.path.expanduser("~/Downloads/arc-agi-2/data"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
]


def _find_real_data() -> str | None:
    for directory in REAL_DATA_CANDIDATES:
        if not directory:
            continue
        path = os.path.join(directory, "arc-agi_evaluation_challenges.json")
        if os.path.isfile(path):
            return directory
    return None


REAL_DATA_DIR = _find_real_data()


@pytest.fixture(scope="session")
def real_data_dir() -> str:
    if REAL_DATA_DIR is None:  # pragma: no cover
        pytest.skip("ARC-AGI-2 evaluation data not found locally")
    return REAL_DATA_DIR


@pytest.fixture(scope="session")
def eval_challenges(real_data_dir) -> dict:
    with open(
        os.path.join(real_data_dir, "arc-agi_evaluation_challenges.json"), encoding="utf-8"
    ) as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def eval_solutions(real_data_dir) -> dict:
    path = os.path.join(real_data_dir, "arc-agi_evaluation_solutions.json")
    if not os.path.isfile(path):  # pragma: no cover
        pytest.skip("no evaluation solutions file")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def eval_tasks(eval_challenges) -> list[Task]:
    from harness.grids import task_from_dict

    return [task_from_dict(tid, blob) for tid, blob in eval_challenges.items()]


# --------------------------------------------------------------------- synthetic


def _pair(grid_in, grid_out) -> Pair:
    return Pair(input=normalize_grid(grid_in), output=normalize_grid(grid_out))


@pytest.fixture
def flip_task() -> Task:
    """A task whose true rule is vertical flip -- solvable and verifiable."""
    pairs = [
        ([[1, 2], [3, 4]], [[3, 4], [1, 2]]),
        ([[0, 0, 5], [1, 1, 1]], [[1, 1, 1], [0, 0, 5]]),
        ([[2, 2], [2, 9]], [[2, 9], [2, 2]]),
    ]
    return Task(
        task_id="flipud-synthetic",
        train=tuple(_pair(a, b) for a, b in pairs),
        test_inputs=(normalize_grid([[7, 7], [8, 8]]),),
    )


@pytest.fixture
def crop_task() -> Task:
    """Output shape differs from input shape -- exercises shape diagnostics."""
    pairs = [
        ([[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]], [[2, 2], [2, 2]]),
        ([[1, 1, 1], [1, 5, 1], [1, 1, 1]], [[5]]),
    ]
    return Task(
        task_id="crop-synthetic",
        train=tuple(_pair(a, b) for a, b in pairs),
        test_inputs=(
            normalize_grid([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]]),
            normalize_grid([[4, 4], [4, 4]]),
        ),
    )


@pytest.fixture
def ambiguous_task() -> Task:
    """One train pair, two rules that both verify but disagree on the test input.

    This is the case that makes the second attempt worth anything: exact train
    verification is a strong but not conclusive signal (Land measured ~97%
    precision), and with a single train pair several distinct rules survive it.
    """
    return Task(
        task_id="ambiguous-synthetic",
        train=(_pair([[0, 0], [0, 0]], [[1, 1], [1, 1]]),),
        test_inputs=(normalize_grid([[7, 7, 7], [8, 8, 8]]),),
    )


AMBIGUOUS_RULE_A = "def transform(g):\n    return [[1] * len(row) for row in g]\nverify(transform)"
AMBIGUOUS_RULE_B = "def transform(g):\n    return [[1, 1], [1, 1]]\nverify(transform)"


@pytest.fixture
def two_test_task(flip_task) -> Task:
    return Task(
        task_id=flip_task.task_id,
        train=flip_task.train,
        test_inputs=(
            normalize_grid([[1, 1], [2, 2]]),
            normalize_grid([[3, 3, 3], [4, 4, 4]]),
        ),
    )
