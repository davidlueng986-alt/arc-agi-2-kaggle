"""End-to-end MockLLM run over real eval-split tasks.

Asserts the pipeline produces a schema-valid submission even when template
programs almost never verify on hard ARC-AGI-2 tasks (i.e. it exercises the
no-verified-program fallback path end to end, plus the full budget/watchdog/
writer plumbing). Skips when the shared data dir is absent.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from harness.budget import Budget, Watchdog
from harness.data import load_solutions_for, load_tasks
from harness.disambiguate import DisambiguationConfig
from harness.llm import MockLLM, library_responder
from harness.sandbox import LocalExecutor
from harness.solver import SolverConfig, solve_all
from harness.submission import SubmissionWriter, score_submission, validate_submission

DATA_DIR = os.environ.get("ARC_DATA_DIR", r"C:\Users\daive\Downloads\arc-agi-2\data")
HAS_DATA = Path(DATA_DIR).exists() and (Path(DATA_DIR) / "arc-agi_evaluation_challenges.json").exists()

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="shared ARC data dir not available")


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(DATA_DIR, split="evaluation", limit=10)


def test_ten_eval_tasks_produce_valid_submission(tmp_path, tasks):
    assert len(tasks) == 10
    llm = MockLLM(library_responder())
    ex = LocalExecutor()
    sub_path = tmp_path / "submission.json"
    writer = SubmissionWriter(sub_path, tasks)
    budget = Budget(deadline_ts=time.time() + 600, per_task_tokens=100_000, total_tokens=5_000_000)
    cfg = SolverConfig(
        num_experts=2,
        max_iterations=4,
        min_calls=3,
        patience_no_new=2,
        use_adjudication=False,
        use_loo=False,
        disambiguation=DisambiguationConfig(seed=1),
    )
    summary = solve_all(tasks, llm, budget, cfg, submission_writer=writer, workers=1, executor_factory=lambda: ex)

    data = json.loads(sub_path.read_text())
    problems = validate_submission(data, tasks)
    assert problems == [], f"invalid submission schema: {problems[:5]}"
    assert set(data.keys()) == {t.task_id for t in tasks}
    for t in tasks:
        packs = data[t.task_id]
        assert len(packs) == t.n_test
        for pack in packs:
            assert "attempt_1" in pack and "attempt_2" in pack
            assert pack["attempt_1"] and pack["attempt_2"]
    assert summary["tasks_run"] == 10


def test_scoring_against_ground_truth_runs(tmp_path, tasks):
    llm = MockLLM(library_responder())
    ex = LocalExecutor()
    writer = SubmissionWriter(tmp_path / "s.json", tasks)
    cfg = SolverConfig(num_experts=1, max_iterations=3, min_calls=2, patience_no_new=2,
                       use_adjudication=False, use_loo=False)
    solve_all(tasks, llm, None, cfg, submission_writer=writer, workers=1, executor_factory=lambda: ex)
    sols = load_solutions_for(DATA_DIR, "evaluation")
    report = score_submission(writer.data(), tasks, sols)
    assert report["outputs_total"] == sum(t.n_test for t in tasks)
    assert 0.0 <= report["score"] <= 1.0


def test_watchdog_finalizes_before_deadline(tmp_path, tasks):
    sub_path = tmp_path / "sub.json"
    writer = SubmissionWriter(sub_path, tasks)
    fired = []
    wd = Watchdog(time.time() + 0.3, lambda: (writer.flush(), fired.append(1)), hard_kill_grace_s=60)
    wd.start()
    time.sleep(0.7)
    wd.cancel()
    assert fired == [1]
    # a valid submission exists even though we never solved anything
    assert validate_submission(json.loads(sub_path.read_text()), tasks) == []


def test_parallel_workers_produce_valid_submission(tmp_path, tasks):
    """Thread-parallel path (each task gets its own sandbox) stays schema-valid."""
    from harness.sandbox import Sandbox

    llm = MockLLM(library_responder())
    writer = SubmissionWriter(tmp_path / "par.json", tasks[:6])
    cfg = SolverConfig(num_experts=1, max_iterations=3, min_calls=2, patience_no_new=2,
                       use_adjudication=False, use_loo=False, sandbox_timeout=2.0)
    solve_all(
        tasks[:6], llm, None, cfg, submission_writer=writer, workers=3,
        executor_factory=lambda: Sandbox(per_job_timeout=1.5),
    )
    assert validate_submission(json.loads((tmp_path / "par.json").read_text()), tasks[:6]) == []
