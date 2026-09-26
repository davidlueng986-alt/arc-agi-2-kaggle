from __future__ import annotations

import json

import pytest

from harness.budget import BudgetConfig, BudgetGovernor, ManualClock
from harness.config import HarnessConfig, SamplingConfig
from harness.grids import to_csv
from harness.llm import ChatRequest, ChatResponse, LLMClient, MockLLM, Usage
from harness.submission import SubmissionWriter
from harness.tasks import ARCTask
from harness.waves import Scheduler


def synth_tasks(n=6, n_test=1):
    tasks = []
    for i in range(n):
        grids = [[[(i + j) % 10, (i + j + 1) % 10] for _ in range(2)] for j in range(n_test)]
        tasks.append(
            ARCTask(
                task_id=f"synth{i:03d}",
                train=(([[0, 0], [0, 0]], [[1, 1], [1, 1]]),),
                test=tuple(grids),
                solutions=tuple(grids),
            )
        )
    return tasks


def make_config(**sampling):
    cfg = HarnessConfig(output_dir=None)
    cfg.sampling = SamplingConfig(**{**cfg.sampling.__dict__, **sampling})
    cfg.budget = BudgetConfig(startup_seconds=0.0, planned_tok_per_s=1e9, pessimistic_tok_per_s=1e9)
    return cfg


def make_env(tasks, client, config, tmp_path, budget=None):
    clock = ManualClock(0.0)
    governor = BudgetGovernor(budget or config.budget, clock=clock)
    writer = SubmissionWriter(str(tmp_path / "submission.json"), tasks)
    scheduler = Scheduler(client, tasks, config, governor, writer, logger=lambda _m: None)
    return scheduler, governor, writer, clock


class AlwaysDisagree(LLMClient):
    name = "disagree"

    def __init__(self):
        self.n = 0

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.n += 1
        grid = [[self.n % 10, (self.n + 1) % 10]]
        return ChatResponse(
            content=f"ANSWER 1:\n{to_csv(grid)}",
            usage=Usage(10, 100, 90),
            finish_reason="stop",
            meta=dict(request.meta),
        )


class ClockAdvancing(LLMClient):
    name = "clock-advancing"

    def __init__(self, inner: LLMClient, clock: ManualClock, seconds_per_request: float) -> None:
        self.inner = inner
        self.clock = clock
        self.seconds_per_request = seconds_per_request
        self.n = 0

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.n += 1
        self.clock.advance(self.seconds_per_request)
        return self.inner.complete(request)


def perfect_client(tasks):
    return MockLLM(answers={t.task_id: t.solutions for t in tasks}, correct_prob=1.0, tok_per_sec=0)


def test_scheduler_seeds_every_task_before_generating(tmp_path):
    tasks = synth_tasks(4)
    config = make_config()
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    assert set(scheduler.states) == {t.task_id for t in tasks}
    for state in scheduler.states.values():
        assert len(state.results) == state.task.n_test
        assert state.pool.total() == 0
        assert not state.fallbacks[0] == []


def test_full_run_produces_schema_valid_submission(tmp_path):
    tasks = synth_tasks(6)
    config = make_config(phase_a_samples=2, wave_samples=(3, 3))
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    report = scheduler.run()
    assert writer.validate() == []
    assert report.n_tasks == 6
    assert report.n_test_outputs == 6
    assert report.request_failures == 0
    assert writer.data["synth000"][0]["attempt_1"] == [[0, 1], [0, 1]]


def test_phase_a_runs_before_phase_b(tmp_path):
    tasks = synth_tasks(3)
    config = make_config(phase_a_samples=2, wave_samples=(2,))
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    report = scheduler.run()
    assert report.waves[0].phase == "A"
    assert report.waves[0].effort == "low"
    assert report.waves[1].phase == "B1"
    assert report.waves[1].effort == "xhigh"


def test_consensus_stops_later_waves(tmp_path):
    tasks = synth_tasks(4)
    config = make_config(
        phase_a_samples=0,
        wave_samples=(4, 4, 4),
        consensus_threshold=0.7,
        consensus_min_candidates=4,
    )
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    report = scheduler.run()
    assert report.consensus_stopped_tasks == 4
    assert len(report.waves) == 1
    assert sum(w.requests for w in report.waves) == 16


def test_no_consensus_runs_all_waves(tmp_path):
    tasks = synth_tasks(3)
    config = make_config(phase_a_samples=0, wave_samples=(2, 2, 2))
    client = AlwaysDisagree()
    scheduler, governor, writer, _ = make_env(tasks, client, config, tmp_path)
    report = scheduler.run()
    assert len(report.waves) == 3
    assert report.consensus_stopped_tasks == 0
    assert client.n == 3 * 6


def test_exhausted_budget_blocks_phase_b_but_keeps_phase_a(tmp_path):
    tasks = synth_tasks(3)
    config = make_config(phase_a_samples=1, wave_samples=(3, 3))
    budget = BudgetConfig(
        total_seconds=43200.0,
        submission_buffer_seconds=600.0,
        startup_seconds=0.0,
        finalize_reserve_seconds=120.0,
        safety_reserve_fraction=0.10,
        planned_tok_per_s=450.0,
        pessimistic_tok_per_s=450.0,
    )
    inner = perfect_client(tasks)
    clock = ManualClock(0.0)
    client = ClockAdvancing(inner, clock, 13000.0)
    governor = BudgetGovernor(budget, clock=clock)
    writer = SubmissionWriter(str(tmp_path / "submission.json"), tasks)
    scheduler = Scheduler(client, tasks, config, governor, writer, logger=lambda _m: None)
    report = scheduler.run()

    phase_a = report.waves[0]
    assert phase_a.phase == "A" and phase_a.requests == 3
    later = [w for w in report.waves if w.phase.startswith("B")]
    assert all(w.requests == 0 for w in later)
    assert writer.validate() == []


def test_hard_deadline_stops_everything(tmp_path):
    tasks = synth_tasks(3)
    config = make_config(phase_a_samples=2, wave_samples=(3,))
    budget = BudgetConfig(startup_seconds=0.0, planned_tok_per_s=450.0, pessimistic_tok_per_s=450.0)
    scheduler, governor, writer, clock = make_env(
        tasks, perfect_client(tasks), config, tmp_path, budget=budget
    )
    clock.set(governor.t_gen + 1.0)
    report = scheduler.run()
    assert all(w.requests == 0 for w in report.waves)
    assert writer.validate() == []


def test_failures_are_counted_but_do_not_break_the_run(tmp_path):
    tasks = synth_tasks(3)
    config = make_config(phase_a_samples=1, wave_samples=(1,))

    class Failing(LLMClient):
        name = "failing"

        def complete(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(
                error="HTTP 500: reasoning_effort", usage=Usage(0, 0, 0), meta=dict(request.meta)
            )

    scheduler, governor, writer, _ = make_env(tasks, Failing(), config, tmp_path)
    report = scheduler.run()
    assert report.request_failures == 6
    assert report.tasks_with_no_parse == 3
    assert writer.validate() == []
    assert writer.data["synth000"][0]["attempt_1"] == [[1, 1], [1, 1]]


def test_unparseable_output_falls_back_to_heuristic_grid(tmp_path):
    tasks = synth_tasks(2)
    config = make_config(phase_a_samples=2, wave_samples=())

    class Gibberish(LLMClient):
        name = "gibberish"

        def complete(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(
                content="I am unsure about this transformation.",
                usage=Usage(10, 50, 40),
                meta=dict(request.meta),
            )

    scheduler, governor, writer, _ = make_env(tasks, Gibberish(), config, tmp_path)
    report = scheduler.run()
    assert report.tasks_with_no_parse == 2
    assert writer.validate() == []
    assert writer.data["synth000"][0]["attempt_1"] == [[1, 1], [1, 1]]


def test_report_token_accounting(tmp_path):
    tasks = synth_tasks(3)
    config = make_config(phase_a_samples=2, wave_samples=(2,))
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    report = scheduler.run()
    assert governor.ledger.requests == 12
    assert report.tokens["requests"] == 12
    assert report.samples_per_task == pytest.approx(4.0)
    assert report.xhigh_samples_per_task == pytest.approx(2.0)
    assert "low" in governor.ledger.by_effort


def test_per_task_rows(tmp_path):
    tasks = synth_tasks(2)
    config = make_config(phase_a_samples=1, wave_samples=(1,))
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    scheduler.run()
    rows = scheduler.per_task_rows()
    assert len(rows) == 2
    assert {r["task_id"] for r in rows} == {t.task_id for t in tasks}
    assert all(r["samples"] == 2 for r in rows)


def test_max_xhigh_samples_cap(tmp_path):
    tasks = synth_tasks(2)
    config = make_config(phase_a_samples=0, wave_samples=(4, 4, 4, 4), max_xhigh_samples=6)
    client = AlwaysDisagree()
    scheduler, governor, writer, _ = make_env(tasks, client, config, tmp_path)
    report = scheduler.run()
    assert client.n == 2 * 6
    assert report.xhigh_samples_per_task == pytest.approx(6.0)


def test_wave_records_include_timing_and_tokens(tmp_path):
    tasks = synth_tasks(2)
    config = make_config(phase_a_samples=1, wave_samples=(1,))
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    report = scheduler.run()
    for wave in report.waves:
        assert wave.requests > 0
        assert wave.seconds >= 0.0
    assert json.loads(json.dumps(report.as_dict(), default=str))["n_tasks"] == 2


def test_multi_test_task_gets_one_entry_per_test_index(tmp_path):
    tasks = [
        ARCTask("multi", train=(([[0]], [[1]]),), test=([[2]], [[3]]), solutions=([[4]], [[5]]))
    ]
    config = make_config(phase_a_samples=2, wave_samples=(2,))
    client = MockLLM(answers={"multi": [[[4]], [[5]]]}, correct_prob=1.0, tok_per_sec=0)
    scheduler, governor, writer, _ = make_env(tasks, client, config, tmp_path)
    scheduler.run()
    assert len(writer.data["multi"]) == 2
    assert writer.data["multi"][0]["attempt_1"] == [[4]]
    assert writer.data["multi"][1]["attempt_1"] == [[5]]
    assert writer.validate() == []


def test_scheduler_shuffles_task_order_but_covers_all(tmp_path):
    tasks = synth_tasks(10)
    config = make_config(phase_a_samples=1, wave_samples=())
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    assert set(scheduler.order) == {t.task_id for t in tasks}
    assert scheduler.order != [t.task_id for t in tasks]


def test_task_state_counters(tmp_path):
    tasks = synth_tasks(1)
    config = make_config(phase_a_samples=1, wave_samples=(2,))
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    scheduler.run()
    state = scheduler.states["synth000"]
    assert state.n_samples == 3
    assert state.xhigh_samples() == 2
    assert state.samples_by_effort == {"low": 1, "xhigh": 2}


def test_concurrency_is_respected(tmp_path):
    tasks = synth_tasks(4)
    config = make_config(phase_a_samples=1, wave_samples=(2,), concurrency=2)
    scheduler, governor, writer, _ = make_env(tasks, perfect_client(tasks), config, tmp_path)
    report = scheduler.run()
    assert sum(w.requests for w in report.waves) == 4 + 8
    assert writer.validate() == []
