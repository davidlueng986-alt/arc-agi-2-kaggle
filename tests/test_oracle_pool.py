import numpy as np

from harness.oracle import array_diff, build_feedback, evaluate_program, run_on_inputs, score_output
from harness.pool import ProgramPool, behavior_key, source_hash
from harness.sandbox import LocalExecutor
from harness.types import ProgramEval, PairScore

EX = LocalExecutor()

CORRECT = """
import numpy as np

def transform(grid):
    out = grid.copy()
    out[out == 1] = 2
    return out
"""

PARTIAL = """
import numpy as np

def transform(grid):
    out = grid.copy()
    out[0, 0] = 2 if grid[0, 0] == 1 else grid[0, 0]
    return out
"""

CRASH = """
def transform(grid):
    return grid.shape[2]  # IndexError
"""


def _pairs():
    x1 = np.array([[1, 0], [0, 1]])
    y1 = np.array([[2, 0], [0, 2]])
    x2 = np.array([[1, 1], [0, 0]])
    y2 = np.array([[2, 2], [0, 0]])
    return [(x1, y1), (x2, y2)]


def test_score_output_exact():
    a = np.array([[1, 2], [3, 4]])
    ps = score_output(a, a.copy())
    assert ps.exact and ps.soft == 1.0 and ps.shape_ok


def test_score_output_partial():
    pred = np.array([[1, 2], [3, 0]])
    truth = np.array([[1, 2], [3, 4]])
    ps = score_output(pred, truth)
    assert not ps.exact and ps.shape_ok
    assert abs(ps.soft - 0.75) < 1e-9
    assert "0/4" in ps.diff


def test_score_output_shape_mismatch():
    ps = score_output(np.array([[1, 2]]), np.array([[1, 2], [3, 4]]))
    assert not ps.exact and not ps.shape_ok and ps.soft == 0.0


def test_score_output_none():
    ps = score_output(None, np.array([[1]]))
    assert not ps.exact and ps.error


def test_array_diff_format():
    pred = np.array([[1, 9]])
    truth = np.array([[1, 2]])
    assert array_diff(pred, truth) == "1 9/2"


def test_evaluate_program_verified():
    ev = evaluate_program(EX, CORRECT, _pairs())
    assert ev.verified
    assert ev.soft_mean == 1.0
    assert ev.n_exec_errors == 0


def test_evaluate_program_partial():
    ev = evaluate_program(EX, PARTIAL, _pairs())
    assert not ev.verified
    assert 0.0 < ev.soft_mean < 1.0


def test_evaluate_program_crash():
    ev = evaluate_program(EX, CRASH, _pairs())
    assert not ev.verified
    assert ev.n_exec_errors == 2
    assert all(ps.error for ps in ev.per_pair)


def test_run_on_inputs():
    outs = run_on_inputs(EX, CORRECT, [np.array([[1, 0]]), np.array([[0, 0]])])
    assert np.array_equal(outs[0], np.array([[2, 0]]))
    assert np.array_equal(outs[1], np.array([[0, 0]]))


def test_run_on_inputs_crash_is_none():
    outs = run_on_inputs(EX, CRASH, [np.array([[1, 0]])])
    assert outs == [None]


def test_build_feedback_mentions_wrong_and_correct():
    ev = evaluate_program(EX, PARTIAL, _pairs())
    fb, score = build_feedback(ev, _pairs())
    assert "WRONG" in fb or "CORRECT" in fb
    assert isinstance(score, float)


def test_source_hash_ignores_whitespace_and_comments():
    a = "def transform(g):\n    return g  # comment\n"
    b = "def transform(g):\n    return g\n"
    assert source_hash(a) == source_hash(b)


def test_source_hash_differs_on_logic():
    assert source_hash("def transform(g):\n return g\n") != source_hash("def transform(g):\n return g+1\n")


def test_behavior_key_same_for_identical_behavior():
    p1 = _pairs()
    ev1 = evaluate_program(EX, CORRECT, p1)
    code2 = """
import numpy as np
def transform(grid):
    return np.where(grid == 1, 2, grid)
"""
    ev2 = evaluate_program(EX, code2, p1)
    k1 = behavior_key(ev1, [np.array([[5]])])
    k2 = behavior_key(ev2, [np.array([[5]])])
    assert k1 == k2


def test_pool_dedupes_identical_source():
    pool = ProgramPool()
    ev = evaluate_program(EX, CORRECT, _pairs())
    i1, new1 = pool.add(CORRECT, "e0i0", ev, [])
    i2, new2 = pool.add(CORRECT, "e0i1", ev, [])
    assert new1 and not new2
    assert i1 == i2
    assert len(pool) == 1


def test_pool_keeps_behavior_duplicates():
    pool = ProgramPool()
    ev1 = evaluate_program(EX, CORRECT, _pairs())
    code2 = "import numpy as np\ndef transform(grid):\n    return np.where(grid == 1, 2, grid)\n"
    ev2 = evaluate_program(EX, code2, _pairs())
    pool.add(CORRECT, "a", ev1, [np.array([[2]])])
    pool.add(code2, "b", ev2, [np.array([[2]])])
    assert len(pool) == 2


def test_pool_verified_behaviors_groups_by_test_output():
    pool = ProgramPool()
    pairs = _pairs()
    ev = evaluate_program(EX, CORRECT, pairs)
    pool.add(CORRECT, "a", ev, [np.array([[2, 0], [0, 2]])])
    code2 = "import numpy as np\ndef transform(grid):\n    return np.where(grid == 1, 2, grid)\n"
    ev2 = evaluate_program(EX, code2, pairs)
    pool.add(code2, "b", ev2, [np.array([[2, 0], [0, 2]])])
    groups = pool.verified_behaviors()
    assert len(groups) == 1
    assert sorted(next(iter(groups.values()))) == [0, 1]


def test_pool_incomplete_behaviors_segregated():
    pool = ProgramPool()
    pairs = _pairs()
    ev = evaluate_program(EX, CORRECT, pairs)
    pool.add(CORRECT, "a", ev, [None])  # crashed on test
    groups = pool.verified_behaviors()
    assert ("incomplete",) in groups


def test_pool_best_partial_orders_by_soft():
    pool = ProgramPool()
    pairs = _pairs()
    pool.add(PARTIAL, "p", evaluate_program(EX, PARTIAL, pairs), [])
    pool.add(CRASH, "c", evaluate_program(EX, CRASH, pairs), [])
    best = pool.best_partial()
    assert best[0] == 0  # PARTIAL (soft>0) before CRASH (soft 0)
