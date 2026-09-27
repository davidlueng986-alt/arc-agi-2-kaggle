"""End-to-end: real eval-split tasks through the whole stack with MockLLM.

This validates plumbing, never accuracy.  The mock agent tries identity, a
bounding-box crop and a constant fill, so its score on the real eval split is
near zero by construction -- what is being asserted here is that a full run
produces a *schema-valid* submission covering every task with both attempts,
that scoring works, and that the cumulative file on disk is valid at all times.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from harness.budget import Deadline
from harness.config import HarnessConfig
from harness.dataio import task_shapes
from harness.llm import MockLLM
from harness.mock_agent import build_mock_llm
from harness.runner import Runner, write_run_report
from harness.submission import score_submission, validate_submission, write_submission

N_TASKS = 10


def make_config(workers: int = 4, max_turns: int = 6) -> HarnessConfig:
    config = HarnessConfig(workers=workers)
    config.budget.max_turns = max_turns
    config.budget.hard_max_turns = max_turns + 2
    config.budget.max_task_seconds = 240
    config.budget.tool_timeout_s = 20
    config.budget.keep_recent_turns = 3
    config.budget.use_image = True
    config.budget.image_max_vision_tokens = 120
    # Plumbing validation only: keep the oracle's augmentation/probe budget small
    # so 10 real 30x30 tasks stay quick. The real run uses the config defaults.
    config.budget.verify_call_seconds = 0.8
    config.budget.verify_total_seconds = 4.0
    config.min_turns = 1
    config.diversity_wait_turns = 3
    config.stop_on_verified = True
    return config


@pytest.fixture
def subset(eval_tasks):
    return eval_tasks[:N_TASKS]


@pytest.fixture
def subset_solutions(eval_solutions, subset):
    return {task.task_id: eval_solutions[task.task_id] for task in subset if task.task_id in eval_solutions}


def test_end_to_end_mock_run_produces_a_valid_submission(tmp_path, subset, subset_solutions):
    path = str(tmp_path / "submission.json")
    shapes = {task.task_id: task.n_test for task in subset}
    started = time.perf_counter()

    result = Runner(
        build_mock_llm(),
        make_config(),
        subset,
        submission_path=path,
        task_shapes=shapes,
        solutions=subset_solutions,
        workers=4,
    ).run()

    assert os.path.isfile(path)
    on_disk = json.loads(open(path, encoding="utf-8").read())

    # schema: every task id, one entry per test input, both attempts present
    assert set(on_disk) == set(shapes)
    for task_id, n_test in shapes.items():
        assert len(on_disk[task_id]) == n_test
        for entry in on_disk[task_id]:
            assert set(entry) == {"attempt_1", "attempt_2"}
            for grid in entry.values():
                assert isinstance(grid, list) and grid
                width = len(grid[0])
                for row in grid:
                    assert isinstance(row, list) and len(row) == width
                    for cell in row:
                        assert isinstance(cell, int) and not isinstance(cell, bool)
                        assert 0 <= cell <= 9

    assert validate_submission(on_disk, shapes) == []
    assert result.schema_problems == []

    # scoring ran and is consistent with the file
    assert result.score is not None
    assert result.score["n_outputs"] == sum(shapes.values())
    assert 0.0 <= result.score["score"] <= 1.0
    assert score_submission(on_disk, subset_solutions)["n_outputs"] == result.score["n_outputs"]

    # accounting
    assert result.tasks_attempted == N_TASKS
    assert result.ledger["total"]["calls"] > 0
    assert result.ledger["total"]["python_calls"] > 0
    assert result.ledger["mean_turn_tokens"] > 0
    assert result.seconds < 600

    # per-task records
    assert len(result.per_task) == N_TASKS
    for record in result.per_task:
        assert record["task_id"] in shapes
        assert record["turns"] >= 1
        assert record["tokens"] >= 0
        assert "score" in record

    write_run_report(str(tmp_path / "report.json"), result)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["tasks_attempted"] == N_TASKS


def test_submission_file_is_valid_before_any_task_runs(tmp_path, subset, subset_solutions):
    """The pre-fill matters: a kill in the first second must still yield a valid file."""
    path = str(tmp_path / "submission.json")
    shapes = {task.task_id: task.n_test for task in subset}
    observed: list[bool] = []

    class Watcher:
        def complete(self, request):
            if not observed:
                observed.append(os.path.isfile(path))
                if observed[0]:
                    problems = validate_submission(json.loads(open(path, encoding="utf-8").read()), shapes)
                    observed.append(not problems)
            from harness.llm import text_response

            return text_response("done")

    Runner(
        Watcher(),
        make_config(workers=1, max_turns=1),
        subset,
        submission_path=path,
        task_shapes=shapes,
        solutions=subset_solutions,
        workers=1,
    ).run()
    assert observed[0] is True, "submission.json did not exist before the first LLM call"
    assert observed[1] is True, "the pre-filled submission.json was not schema-valid"


def test_deadline_stops_the_run_and_still_finalises(tmp_path, subset, subset_solutions):
    path = str(tmp_path / "submission.json")
    shapes = {task.task_id: task.n_test for task in subset}
    deadline = Deadline(0.5)
    time.sleep(0.6)

    result = Runner(
        build_mock_llm(),
        make_config(workers=1),
        subset,
        submission_path=path,
        task_shapes=shapes,
        solutions=subset_solutions,
        deadline=deadline,
        workers=1,
    ).run()

    on_disk = json.loads(open(path, encoding="utf-8").read())
    assert validate_submission(on_disk, shapes) == []
    assert result.tasks_attempted == 0


def test_watchdog_hard_finalise_writes_a_complete_file(tmp_path, subset):
    from harness.budget import Watchdog

    path = str(tmp_path / "submission.json")
    shapes = {task.task_id: task.n_test for task in subset}
    runner = Runner(
        build_mock_llm(),
        make_config(workers=1),
        subset[:1],
        submission_path=path,
        task_shapes=shapes,
        workers=1,
    )
    runner._prefill_fallbacks()
    runner._on_hard_deadline()
    on_disk = json.loads(open(path, encoding="utf-8").read())
    assert set(on_disk) == set(shapes)
    assert validate_submission(on_disk, shapes) == []


def test_adaptive_config_shrinks_when_behind_schedule(tmp_path, subset):
    runner = Runner(
        build_mock_llm(),
        make_config(workers=2),
        subset,
        submission_path=str(tmp_path / "s.json"),
        workers=2,
    )
    runner.deadline = Deadline(30)
    base = runner._effective_config(subset[0])
    assert base.budget.max_task_seconds <= runner.config.budget.max_task_seconds

    # simulate being deep into the run with almost no clock left
    runner._done = len(subset) - 1
    runner.deadline = Deadline(60)
    squeezed = runner._effective_config(subset[-1])
    assert squeezed.budget.max_task_seconds < base.budget.max_task_seconds or squeezed.budget.max_turns <= base.budget.max_turns


def test_concurrent_writers_do_not_corrupt_the_file(tmp_path, subset, subset_solutions):
    """A reader polling the file must never see a partial or invalid submission.

    On POSIX (Kaggle) ``os.replace`` is ``rename(2)`` and atomic, so this holds
    unconditionally.  On Windows a concurrent reader can get a transient sharing
    violation at the instant of the rename, which is an OS artifact rather than
    corruption -- so transient ``OSError`` is retried, while a JSON parse error or
    a schema violation fails the test immediately.
    """
    path = str(tmp_path / "submission.json")
    shapes = {task.task_id: task.n_test for task in subset}
    errors: list[str] = []
    stop = threading.Event()

    def read_once():
        for _ in range(20):
            try:
                with open(path, encoding="utf-8") as handle:
                    return json.load(handle)
            except (PermissionError, OSError):
                time.sleep(0.002)  # Windows rename sharing violation; retry
        raise OSError("could not read submission.json after retries")

    def reader():
        while not stop.is_set():
            try:
                payload = read_once()
            except json.JSONDecodeError as exc:
                errors.append(f"torn read: {exc}")
                return
            except OSError as exc:
                errors.append(f"unreadable: {exc}")
                return
            problems = validate_submission(payload, shapes)
            if problems:
                errors.extend(problems[:3])
                return
            time.sleep(0.002)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    Runner(
        build_mock_llm(),
        make_config(workers=6),
        subset,
        submission_path=path,
        task_shapes=shapes,
        solutions=subset_solutions,
        workers=6,
        log=lambda *_: None,
    ).run()
    stop.set()
    thread.join(timeout=10)
    assert errors == []
    assert validate_submission(json.loads(open(path, encoding="utf-8").read()), shapes) == []


def test_full_120_task_shapes_are_coverable(tmp_path, eval_tasks, eval_challenges):
    """The rerun must cover every task id; check the machinery scales to 120."""
    shapes = task_shapes(eval_challenges)
    assert len(shapes) == 120
    submission = {}
    for task in eval_tasks:
        submission[task.task_id] = [
            {
                "attempt_1": [list(row) for row in grid],
                "attempt_2": [list(row) for row in grid],
            }
            for grid in task.test_inputs
        ]
    assert validate_submission(submission, shapes) == []
    path = str(tmp_path / "full.json")
    write_submission(path, submission)
    assert os.path.getsize(path) > 1000
