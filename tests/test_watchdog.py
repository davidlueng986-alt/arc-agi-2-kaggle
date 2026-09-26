from __future__ import annotations

import json
import os

import pytest

from harness.budget import BudgetConfig, BudgetGovernor, ManualClock
from harness.submission import SubmissionWriter
from harness.tasks import ARCTask
from harness.watchdog import Watchdog


TRAIN = (([[0]], [[1]]),)


def make_tasks():
    return [
        ARCTask("aaa", TRAIN, ([[2]],)),
        ARCTask("bbb", TRAIN, ([[3]], [[4]])),
    ]


@pytest.fixture
def env(tmp_path):
    tasks = make_tasks()
    clock = ManualClock(0.0)
    budget = BudgetConfig(total_seconds=1000.0, submission_buffer_seconds=100.0, startup_seconds=10.0, finalize_reserve_seconds=20.0)
    governor = BudgetGovernor(budget, clock=clock)
    path = str(tmp_path / "submission.json")
    writer = SubmissionWriter(path, tasks)
    logs = []
    watchdog = Watchdog(
        governor,
        writer,
        logger=logs.append,
        poll_seconds=0.01,
        clock=clock.now,
        sleeper=clock.sleep,
    )
    return tasks, clock, governor, writer, path, logs, watchdog


def test_watchdog_is_idle_before_deadline(env):
    _tasks, clock, governor, writer, path, _logs, wd = env
    clock.set(governor.t_gen - 1)
    wd.tick()
    assert not wd.finalized
    assert not os.path.exists(path)


def test_watchdog_finalizes_at_generation_deadline(env):
    _tasks, clock, governor, writer, path, logs, wd = env
    writer.update("aaa", [{"attempt_1": [[9]], "attempt_2": [[8]]}])
    clock.set(governor.t_gen)
    wd.tick()
    assert wd.finalized
    assert wd.triggered_at == governor.t_gen
    assert os.path.exists(path)
    saved = json.load(open(path, encoding="utf-8"))
    assert saved["aaa"][0]["attempt_1"] == [[9]]
    assert saved["bbb"][0]["attempt_1"] == [[0, 0], [0, 0]]
    assert any("finalizing" in m for m in logs)


def test_watchdog_writes_once(env):
    _tasks, clock, governor, writer, path, _logs, wd = env
    clock.set(governor.t_gen + 5)
    wd.tick()
    wd.tick()
    assert writer.writes == 1


def test_watchdog_records_schema_errors_but_still_writes(env):
    _tasks, clock, governor, writer, path, _logs, wd = env
    writer.data["aaa"][0]["attempt_1"] = [[99]]
    clock.set(governor.t_gen)
    wd.tick()
    assert wd.errors
    assert os.path.exists(path)


def test_watchdog_thread_runs_and_stops(env):
    _tasks, clock, governor, writer, path, _logs, wd = env
    wd.start()
    wd.start()
    clock.set(governor.t_gen + 1)
    import time as _time

    for _ in range(200):
        if os.path.exists(path):
            break
        _time.sleep(0.01)
    wd.stop()
    assert os.path.exists(path)
    assert wd.finalized


def test_watchdog_hard_exit_disabled_by_default(env):
    _tasks, clock, governor, writer, path, _logs, wd = env
    assert wd.hard_exit is False


def test_watchdog_deadline_ordering(env):
    _tasks, _clock, governor, _writer, _path, _logs, _wd = env
    assert governor.t_gen < governor.t_submit < governor.t_hard


def test_watchdog_loop_terminates_without_hard_exit(tmp_path):
    """Past the final deadline the thread must exit, not busy-loop for the kernel's life."""
    tasks = make_tasks()
    clock = ManualClock(0.0)
    budget = BudgetConfig(
        total_seconds=1000.0, submission_buffer_seconds=100.0,
        startup_seconds=10.0, finalize_reserve_seconds=20.0,
    )
    governor = BudgetGovernor(budget, clock=clock)
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    wd = Watchdog(governor, writer, poll_seconds=0.01, clock=clock.now, sleeper=lambda _s: None)
    clock.set(governor.t_submit)
    wd.start()
    wd._thread.join(timeout=5.0)
    assert not wd._thread.is_alive(), "watchdog thread did not terminate"
    assert os.path.exists(str(tmp_path / "s.json"))
    assert wd.finalized


def test_watchdog_polls_until_deadline_then_terminates(tmp_path):
    tasks = make_tasks()
    clock = ManualClock(0.0)
    budget = BudgetConfig(
        total_seconds=100.0, submission_buffer_seconds=10.0,
        startup_seconds=0.0, finalize_reserve_seconds=5.0,
    )
    governor = BudgetGovernor(budget, clock=clock)
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)

    def sleep_and_advance(seconds: float) -> None:
        clock.advance(30.0)

    wd = Watchdog(governor, writer, poll_seconds=0.01, clock=clock.now, sleeper=sleep_and_advance)
    wd.start()
    wd._thread.join(timeout=5.0)
    assert not wd._thread.is_alive()
    assert wd.finalized
    assert os.path.exists(str(tmp_path / "s.json"))


def test_watchdog_thread_stops_early_on_request(tmp_path):
    tasks = make_tasks()
    clock = ManualClock(0.0)
    governor = BudgetGovernor(BudgetConfig(total_seconds=100000.0), clock=clock)
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    wd = Watchdog(governor, writer, poll_seconds=0.01, clock=clock.now, sleeper=lambda _s: None)
    wd.start()
    wd.stop()
    assert not wd._thread.is_alive()
    assert not wd.finalized


def test_run_harness_with_watchdog_armed(tmp_path):
    """The exact wiring the Kaggle kernel uses, minus hard_exit."""
    from harness.config import HarnessConfig, SamplingConfig
    from harness.llm import MockLLM
    from harness.runner import mock_answers, run_harness
    from harness.tasks import ARCTask

    tasks = [
        ARCTask("w1", TRAIN, ([[2]],), solutions=([[1]],)),
        ARCTask("w2", TRAIN, ([[3]], [[4]]), solutions=([[1]], [[1]])),
    ]
    config = HarnessConfig(split="evaluation", output_dir=str(tmp_path))
    config.sampling = SamplingConfig(phase_a_samples=1, wave_samples=(1,))
    config.budget = BudgetConfig(
        total_seconds=43200.0, startup_seconds=0.0, planned_tok_per_s=1e9, pessimistic_tok_per_s=1e9
    )
    client = MockLLM(answers=mock_answers(tasks), correct_prob=1.0, tok_per_sec=0)
    outcome = run_harness(
        config,
        client=client,
        clock=ManualClock(0.0),
        logger=lambda _m: None,
        tasks=tasks,
        use_watchdog=True,
        hard_exit=False,
        watchdog_poll_seconds=0.01,
        watchdog_sleeper=lambda _s: None,
    )
    assert outcome.report.submission_errors == []
    assert os.path.exists(outcome.submission_path)
    assert outcome.score.score == pytest.approx(1.0)


def test_run_harness_watchdog_finalizes_when_deadline_already_passed(tmp_path):
    from harness.config import HarnessConfig, SamplingConfig
    from harness.llm import MockLLM
    from harness.runner import mock_answers, run_harness
    from harness.tasks import ARCTask

    tasks = [ARCTask("w1", TRAIN, ([[2]],), solutions=([[1]],))]
    config = HarnessConfig(split="evaluation", output_dir=str(tmp_path))
    config.sampling = SamplingConfig(phase_a_samples=1, wave_samples=(1,))
    clock = ManualClock(0.0)
    config.budget = BudgetConfig(total_seconds=10.0, startup_seconds=0.0, submission_buffer_seconds=1.0)
    client = MockLLM(answers=mock_answers(tasks), correct_prob=1.0, tok_per_sec=0)
    outcome = run_harness(
        config, client=client, clock=clock, logger=lambda _m: None, tasks=tasks,
        use_watchdog=True, hard_exit=False, watchdog_poll_seconds=0.01,
        watchdog_sleeper=lambda _s: None,
    )
    assert os.path.exists(outcome.submission_path)
    assert outcome.report.submission_errors == []
