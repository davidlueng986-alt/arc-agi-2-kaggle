"""Synthesis-loop tests with scripted MockLLM (no GPU, deterministic)."""
from __future__ import annotations

import numpy as np
import pytest

from harness.disambiguate import DisambiguationConfig
from harness.grid import grids_equal, to_grid
from harness.llm import MockLLM, library_responder, scripted_responder
from harness.oracle import evaluate_program
from harness.pool import ProgramPool
from harness.sandbox import LocalExecutor
from harness.solver import SolverConfig, solve_task
from harness.types import Task
from tests.fixtures.synthetic import (
    output_of,
    F1_CORRECT,
    F1_OVERFIT,
    F1_TEST,
    F1_TRAIN,
    F2_CORRECT,
    F2_OVERFIT,
)

EX = LocalExecutor()


def fenced(code: str, prose: str = "Here is my solution.") -> str:
    return f"{prose}\n\n```python\n{code}\n```\n"


def _cfg(**kw) -> SolverConfig:
    base = dict(
        num_experts=2,
        max_iterations=4,
        min_calls=3,
        patience_no_new=2,
        stop_distinct_behaviors=4,
        use_adjudication=False,
        use_loo=False,
        seed=0,
        disambiguation=DisambiguationConfig(seed=3),
    )
    base.update(kw)
    return SolverConfig(**base)


def test_synthesis_finds_verified_program_and_picks_correct_behavior():
    task = Task("f1", F1_TRAIN, F1_TEST)
    scripts = [
        fenced(F1_OVERFIT, "The 2s live in the top-left, recolor them there."),
        fenced(F1_CORRECT, "Recolor every 2 as 8."),
        fenced(F1_CORRECT),  # duplicate source -> deduped
        fenced(F1_OVERFIT),  # duplicate
        fenced(F1_CORRECT),
        fenced(F1_CORRECT),
    ]
    llm = MockLLM(scripted_responder(scripts))
    res = solve_task(task, llm, EX, None, _cfg())
    assert res.n_verified >= 2
    a1, a2 = res.attempts[0]
    expected = output_of(F1_CORRECT, F1_TEST)[0]
    assert grids_equal(a1, expected), "equivariance must beat the top-left overfit end-to-end"
    assert res.llm_calls >= 2


def test_adjudication_breaks_tie():
    """Two verified behaviors, 1v1; the adjudicator program decides."""
    task = Task("f2", F1_TRAIN, F1_TEST)
    adjudicated = F1_CORRECT  # adjudicator returns the truly correct program
    scripts = [
        fenced(F1_OVERFIT),
        fenced(F1_CORRECT),
        fenced(adjudicated, "The window-hardcoded rule is an artifact of the training grids."),
        fenced(F1_CORRECT),
        fenced(F1_CORRECT),
        fenced(F1_CORRECT),
        fenced(F1_CORRECT),
        fenced(F1_CORRECT),
    ]
    llm = MockLLM(scripted_responder(scripts))
    cfg = _cfg(use_adjudication=True, min_calls=2, patience_no_new=1)
    res = solve_task(task, llm, EX, None, cfg)
    a1, _ = res.attempts[0]
    expected = output_of(F1_CORRECT, F1_TEST)[0]
    assert grids_equal(a1, expected)


def test_loo_corroboration_path_runs():
    """LOO folds synthesize a program that passes the held-out pair and
    corroborates the correct behavior."""
    task = Task("f2loo", F1_TRAIN, F1_TEST)
    scripts = [
        fenced(F1_OVERFIT),   # e0i0
        fenced(F1_CORRECT),   # e0i1 -> verified set {overfit, correct}
        fenced(F1_CORRECT),   # e0i2 dedupe
        fenced(F1_OVERFIT),   # e0i3 dedupe
        fenced(F1_CORRECT),   # e0i4 dedupe (patience stops loop)
        fenced(F1_CORRECT),   # e1i0 dedupe
        fenced(F1_CORRECT),   # e1i1
        fenced(F1_CORRECT),   # e1i2
        fenced(F1_CORRECT),   # e1i3
        # LOO fold responses (one per fold; each a distinct correct variant):
        fenced("import numpy as np\n\ndef transform(grid):\n    return np.where(grid == 2, 8, grid)\n"),
        fenced("import numpy as np\n\ndef transform(grid):\n    out = grid.copy()\n    out[grid == 2] = 8\n    return out\n"),
        fenced("import numpy as np\n\ndef transform(grid):\n    return np.where(grid == 2, 8, grid)\n"),
    ]
    llm = MockLLM(scripted_responder(scripts))
    cfg = _cfg(use_loo=True, loo_max_folds=3, loo_min_pairs=3, min_calls=2, patience_no_new=1)
    res = solve_task(task, llm, EX, None, cfg)
    a1, _ = res.attempts[0]
    expected = output_of(F1_CORRECT, F1_TEST)[0]
    assert grids_equal(a1, expected)
    assert res.extras.get("loo_map"), "LOO corroboration should have credited a behavior"


def test_no_code_in_response_is_survivable():
    task = Task("f1", F1_TRAIN, F1_TEST)
    scripts = ["I cannot solve this.", "Sorry.", fenced(F1_CORRECT), fenced(F1_CORRECT), fenced(F1_CORRECT)]
    llm = MockLLM(scripted_responder(scripts))
    res = solve_task(task, llm, EX, None, _cfg())
    assert res.n_verified >= 1


def test_budget_gates_calls():
    import time

    from harness.budget import Budget

    task = Task("f1", F1_TRAIN, F1_TEST)
    budget = Budget(deadline_ts=time.time() + 600, per_task_tokens=200, total_tokens=10_000)
    llm = MockLLM(library_responder())
    res = solve_task(task, llm, EX, budget, _cfg())
    # per-task token cap must stop the loop well before 8 calls
    assert res.llm_calls <= 3
    assert budget.task_spent(task.task_id) <= 400


def test_deadline_stops_everything():
    import time

    from harness.budget import Budget

    task = Task("f1", F1_TRAIN, F1_TEST)
    budget = Budget(deadline_ts=time.time() - 1, reserve_seconds=0)
    llm = MockLLM(library_responder())
    res = solve_task(task, llm, EX, budget, _cfg())
    assert res.llm_calls == 0
    # still emits schema-valid attempts via fallback
    a1, a2 = res.attempts[0]
    assert a1 is not None and a2 is not None


def test_pool_records_provenance_and_feedback_loop_uses_partials():
    """When nothing verifies, the loop must keep feeding partial-solution
    feedback (Poetiq mechanism) and never crash."""
    task = Task("hard", F1_TRAIN, F1_TEST)
    seen_feedback = []

    def responder(messages, call_index):
        content = messages[0]["content"]
        seen_feedback.append("PREVIOUS ATTEMPTS" in content)
        return fenced("import numpy as np\n\ndef transform(grid):\n    return grid * 0 + 5\n")

    llm = MockLLM(responder)
    res = solve_task(task, llm, EX, None, _cfg())
    assert res.n_verified == 0
    assert any(seen_feedback[1:]), "feedback block must appear after the first attempt"
    a1, a2 = res.attempts[0]
    assert a1 is not None
