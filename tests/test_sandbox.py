import os
import time

import numpy as np
import pytest

from harness.sandbox import LocalExecutor, Sandbox
from harness.types import ExecResult

OK_CODE = """
import numpy as np

def transform(grid):
    return grid + 1
"""

RAISE_CODE = """
def transform(grid):
    raise RuntimeError("boom")
"""

HANG_CODE = """
def transform(grid):
    while True:
        pass
"""

BIG_OUTPUT_CODE = """
import numpy as np

def transform(grid):
    return np.zeros((100, 100), dtype=int)
"""

NON_2D_CODE = """
def transform(grid):
    return 5
"""

PRINT_NOISE_CODE = """
import numpy as np

def transform(grid):
    print("GARBAGE THAT WOULD BREAK NAIVE JSON PARSING" * 10)
    print('{"ok": true, "result": [[9]]}')
    return grid.copy()
"""

ENV_LEAK_CODE = """
import os
import numpy as np

def transform(grid):
    leaked = os.environ.get("H3_SECRET_CANARY")
    out = np.zeros_like(grid)
    if leaked:
        out[:, :] = 1
    return out
"""

MUTATES_INPUT_CODE = """
import numpy as np

def transform(grid):
    grid[0, 0] = 9  # in-place mutation must not affect caller's grid
    return grid
"""


@pytest.fixture(scope="module")
def sandbox():
    sb = Sandbox(per_job_timeout=1.0, startup_timeout=30.0)
    yield sb
    sb.close()


def test_basic_execution(sandbox):
    g = np.array([[0, 1], [2, 3]])
    res = sandbox.run(OK_CODE, g)
    assert res.ok
    assert np.array_equal(res.output, np.array([[1, 2], [3, 4]]))


def test_input_not_mutated(sandbox):
    g = np.array([[0, 1], [2, 3]])
    res = sandbox.run(MUTATES_INPUT_CODE, g)
    assert res.ok
    assert g[0, 0] == 0, "sandbox must pass a copy; caller grid mutated"


def test_exception_captured(sandbox):
    res = sandbox.run(RAISE_CODE, np.zeros((2, 2), dtype=int))
    assert not res.ok
    assert "boom" in (res.error or "")


def test_big_output_rejected(sandbox):
    res = sandbox.run(BIG_OUTPUT_CODE, np.zeros((2, 2), dtype=int))
    assert not res.ok
    assert "too large" in (res.error or "")


def test_non_2d_output_rejected(sandbox):
    res = sandbox.run(NON_2D_CODE, np.zeros((2, 2), dtype=int))
    assert not res.ok


def test_stdout_noise_does_not_break_protocol(sandbox):
    g = np.array([[7, 7], [7, 7]])
    res = sandbox.run(PRINT_NOISE_CODE, g)
    assert res.ok
    assert np.array_equal(res.output, g)


def test_environment_isolation(sandbox, monkeypatch):
    monkeypatch.setenv("H3_SECRET_CANARY", "leaked")
    res = sandbox.run(ENV_LEAK_CODE, np.zeros((2, 2), dtype=int))
    assert res.ok
    assert np.array_equal(res.output, np.zeros((2, 2), dtype=int)), (
        "sandboxed code saw a parent-process environment variable"
    )


def test_timeout_kills_and_batch_continues(sandbox):
    g = np.zeros((2, 2), dtype=int)
    t0 = time.time()
    results = sandbox.run_batch([(OK_CODE, g), (HANG_CODE, g), (OK_CODE, g)])
    dt = time.time() - t0
    assert len(results) == 3
    assert results[0].ok
    assert not results[1].ok and results[1].timed_out
    assert results[2].ok, "jobs after a timeout must still complete (worker restart)"
    assert dt < 30, f"timeout handling too slow: {dt:.1f}s"


def test_batch_results_in_order(sandbox):
    g = np.zeros((3, 3), dtype=int)
    jobs = [(OK_CODE, g + i) for i in range(10)]
    results = sandbox.run_batch(jobs)
    for i, r in enumerate(results):
        assert r.ok
        assert np.array_equal(r.output, g + i + 1)


def test_missing_transform_rejected(sandbox):
    res = sandbox.run("def solver(grid):\n    return grid\n", np.zeros((2, 2), dtype=int))
    assert not res.ok
    assert "transform" in (res.error or "")


def test_local_executor_parity():
    g = np.array([[0, 1], [2, 3]])
    local = LocalExecutor()
    r1 = local.run(OK_CODE, g)
    r2 = local.run(RAISE_CODE, g)
    r3 = local.run(BIG_OUTPUT_CODE, g)
    assert r1.ok and np.array_equal(r1.output, g + 1)
    assert not r2.ok and "boom" in r2.error
    assert not r3.ok
    # input not mutated
    r4 = local.run(MUTATES_INPUT_CODE, g)
    assert r4.ok and g[0, 0] == 0
