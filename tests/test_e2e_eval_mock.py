"""End-to-end pipeline test on real ARC-AGI-2 evaluation tasks with MockLLM.

Proves that every non-LLM component composes: real task loading -> real prompt
construction -> sampling -> tolerant parsing -> candidate pooling -> shape-diverse
two-attempt selection -> cumulative submission write -> schema validation ->
exact-match scoring.
"""

from __future__ import annotations

import json
import os

import pytest

from harness.budget import BudgetConfig, BudgetGovernor, ManualClock
from harness.config import HarnessConfig, SamplingConfig
from harness.llm import ChatRequest, ChatResponse, LLMClient, MockLLM, Usage
from harness.prompts import build_messages
from harness.runner import budget_table, mock_answers, run_harness
from harness.scoring import score_submission
from harness.submission import SubmissionWriter
from harness.tasks import ARCTask, total_test_outputs

N_TASKS = 10


@pytest.fixture(scope="module")
def eval_subset(eval_tasks):
    return eval_tasks[:N_TASKS]


def make_config(tmp_path, **sampling):
    cfg = HarnessConfig(split="evaluation", output_dir=str(tmp_path), seed=7)
    cfg.sampling = SamplingConfig(**{**cfg.sampling.__dict__, **sampling})
    cfg.budget = BudgetConfig(startup_seconds=0.0, planned_tok_per_s=1e9, pessimistic_tok_per_s=1e9)
    cfg.log_every = 1000
    return cfg


def test_prompts_are_constructible_for_real_eval_tasks(eval_subset):
    for task in eval_subset:
        messages = build_messages(task)
        assert messages[0]["role"] == "system"
        assert "Training pair 1 input:" in messages[1]["content"]
        assert f"ANSWER {task.n_test}:" in messages[1]["content"] or task.n_test == 1
        assert len(messages[1]["content"]) > 50


def test_e2e_mock_perfect_solves_everything(tmp_path, eval_subset):
    config = make_config(tmp_path, phase_a_samples=2, wave_samples=(3,))
    client = MockLLM(answers=mock_answers(eval_subset), correct_prob=1.0, tok_per_sec=0)
    outcome = run_harness(config, client=client, clock=ManualClock(0.0), logger=lambda _m: None, tasks=eval_subset)

    assert outcome.report.submission_errors == []
    assert outcome.score is not None
    assert outcome.score.score == pytest.approx(1.0)
    assert outcome.score.total_outputs == total_test_outputs(eval_subset)
    assert os.path.exists(outcome.submission_path)


def test_e2e_mock_submission_schema_is_valid(tmp_path, eval_subset):
    config = make_config(tmp_path, phase_a_samples=2, wave_samples=(2, 2))
    client = MockLLM(answers=mock_answers(eval_subset), correct_prob=0.45, seed=11, tok_per_sec=0)
    outcome = run_harness(config, client=client, clock=ManualClock(0.0), logger=lambda _m: None, tasks=eval_subset)

    submission = json.load(open(outcome.submission_path, encoding="utf-8"))
    assert set(submission) == {t.task_id for t in eval_subset}
    for task in eval_subset:
        entries = submission[task.task_id]
        assert len(entries) == task.n_test
        for entry in entries:
            assert set(entry) == {"attempt_1", "attempt_2"}
            for key in ("attempt_1", "attempt_2"):
                grid = entry[key]
                assert isinstance(grid, list) and grid
                width = len(grid[0])
                assert width > 0
                assert all(len(row) == width for row in grid)
                assert all(isinstance(c, int) and 0 <= c <= 9 for row in grid for c in row)


def test_e2e_mock_writes_artifacts(tmp_path, eval_subset):
    config = make_config(tmp_path, phase_a_samples=1, wave_samples=(1,))
    client = MockLLM(answers=mock_answers(eval_subset), correct_prob=0.5, tok_per_sec=0)
    outcome = run_harness(config, client=client, clock=ManualClock(0.0), logger=lambda _m: None, tasks=eval_subset)
    paths = outcome.write_artifacts(str(tmp_path))
    for key in ("report", "config"):
        assert os.path.exists(paths[key])
    assert os.path.exists(paths["score"])
    assert os.path.exists(str(tmp_path / "per_task_detail.json"))
    blob = json.load(open(paths["report"], encoding="utf-8"))
    assert blob["n_tasks"] == N_TASKS


def test_e2e_mock_total_failure_still_yields_valid_submission(tmp_path, eval_subset):
    config = make_config(tmp_path, phase_a_samples=2, wave_samples=(2,))

    class Broken(LLMClient):
        name = "broken"

        def complete(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(error="connection refused", usage=Usage(0, 0, 0), meta=dict(request.meta))

    outcome = run_harness(config, client=Broken(), clock=ManualClock(0.0), logger=lambda _m: None, tasks=eval_subset)
    assert outcome.report.submission_errors == []
    assert outcome.report.request_failures > 0
    assert outcome.score.score == 0.0
    submission = json.load(open(outcome.submission_path, encoding="utf-8"))
    assert len(submission) == N_TASKS


def test_e2e_mock_gibberish_uses_fallback_grids(tmp_path, eval_subset):
    config = make_config(tmp_path, phase_a_samples=1, wave_samples=(1,))

    class Gibberish(LLMClient):
        name = "gibberish"

        def complete(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(content="no idea at all", usage=Usage(5, 20, 10), meta=dict(request.meta))

    outcome = run_harness(config, client=Gibberish(), clock=ManualClock(0.0), logger=lambda _m: None, tasks=eval_subset)
    assert outcome.report.submission_errors == []
    assert outcome.report.tasks_with_no_parse == N_TASKS
    submission = json.load(open(outcome.submission_path, encoding="utf-8"))
    for task in eval_subset:
        for entry in submission[task.task_id]:
            assert entry["attempt_1"]


def test_e2e_mock_noisy_markdown_output_parses(tmp_path, eval_subset):
    """Realistic messy model output: prose, fences, bold markers, trailing chatter."""
    answers = mock_answers(eval_subset)
    config = make_config(tmp_path, phase_a_samples=2, wave_samples=(2,))

    class Noisy(LLMClient):
        name = "noisy"

        def complete(self, request: ChatRequest) -> ChatResponse:
            task_id = request.meta["task_id"]
            grids = answers[task_id]
            rows = []
            for i, g in enumerate(grids, start=1):
                body = "\n".join(",".join(str(c) for c in row) for row in g)
                rows.append(f"**ANSWER {i}:**\n```csv\n{body}\n```\n")
            text = (
                "Let me reason about the transformation.\n\n"
                "- The background is 0.\n- Objects translate right by one.\n\n"
                "So the answers are:\n\n" + "\n".join(rows) + "\nI hope that helps!\n"
            )
            return ChatResponse(
                content=text,
                reasoning="long thinking " * 20,
                usage=Usage(1200, 10000, 9500),
                finish_reason="stop",
                meta=dict(request.meta),
            )

    outcome = run_harness(config, client=Noisy(), clock=ManualClock(0.0), logger=lambda _m: None, tasks=eval_subset)
    assert outcome.report.submission_errors == []
    assert outcome.score.score == pytest.approx(1.0)
    assert outcome.report.parse_method_counts.get("marker", 0) > 0


def test_e2e_mock_deadline_midrun_keeps_valid_submission(tmp_path, eval_subset):
    config = make_config(tmp_path, phase_a_samples=2, wave_samples=(3, 3))
    config.budget = BudgetConfig(
        total_seconds=43200.0,
        submission_buffer_seconds=600.0,
        startup_seconds=0.0,
        finalize_reserve_seconds=120.0,
        planned_tok_per_s=450.0,
        pessimistic_tok_per_s=450.0,
    )
    clock = ManualClock(0.0)

    class SlowAndDraining(LLMClient):
        name = "slow"

        def __init__(self):
            self.inner = MockLLM(answers=answers, correct_prob=0.6, tok_per_sec=0)
            self.n = 0

        def complete(self, request: ChatRequest) -> ChatResponse:
            self.n += 1
            clock.advance(900.0)
            return self.inner.complete(request)

    answers = mock_answers(eval_subset)
    outcome = run_harness(
        config, client=SlowAndDraining(), clock=clock, logger=lambda _m: None, tasks=eval_subset
    )
    assert outcome.report.submission_errors == []
    submission = json.load(open(outcome.submission_path, encoding="utf-8"))
    assert len(submission) == N_TASKS


def test_e2e_second_attempt_can_add_score(tmp_path):
    """Shape-diverse attempt_2 must recover outputs attempt_1 missed."""
    tasks = [
        ARCTask(
            "s1",
            train=(([[0]], [[1]]),),
            test=([[2]], [[3]]),
            solutions=([[2, 2, 2]], [[5, 5]]),
        ),
    ]
    config = make_config(tmp_path, phase_a_samples=0, wave_samples=(4,))

    class SplitVote(LLMClient):
        name = "split"

        def __init__(self):
            self.n = 0

        def complete(self, request: ChatRequest) -> ChatResponse:
            self.n += 1
            first = "1,1" if self.n <= 3 else "2,2,2"
            return ChatResponse(
                content=f"ANSWER 1:\n{first}\n\nANSWER 2:\n5,5",
                usage=Usage(10, 100, 90),
                meta=dict(request.meta),
            )

    outcome = run_harness(config, client=SplitVote(), clock=ManualClock(0.0), logger=lambda _m: None, tasks=tasks)
    assert outcome.report.submission_errors == []
    blob = json.load(open(outcome.submission_path, encoding="utf-8"))["s1"]
    assert blob[0]["attempt_1"] == [[1, 1]]
    assert blob[0]["attempt_2"] == [[2, 2, 2]]
    assert blob[1]["attempt_1"] == [[5, 5]]
    assert outcome.score.score == pytest.approx(1.0)
    assert outcome.score.attempt_1_score == pytest.approx(0.5)
    assert outcome.score.attempt_2_only_correct == 1


def test_budget_table_matches_hypothesis_arithmetic():
    rows = {r["scenario"]: r for r in budget_table(BudgetConfig(), n_tasks=240, phase_a_samples=2)}
    assert rows["central"]["usable_seconds"] == 43200 - 600 - 1020 - 120
    assert rows["central"]["planned_seconds"] == pytest.approx(0.9 * (43200 - 600 - 1020 - 120))
    central_samples = rows["central"]["xhigh_samples_per_task_uniform"]
    assert 5.0 <= central_samples <= 9.0, central_samples
    pessimistic_samples = rows["pessimistic"]["xhigh_samples_per_task_uniform"]
    assert pessimistic_samples >= 3.0, pessimistic_samples
    assert rows["optimistic"]["xhigh_samples_per_task_uniform"] > central_samples


def test_submission_writer_seeded_from_real_test_split(tmp_path, test_tasks):
    path = str(tmp_path / "submission.json")
    writer = SubmissionWriter(path, test_tasks)
    writer.write()
    blob = json.load(open(path, encoding="utf-8"))
    assert len(blob) == 240
    assert sum(len(v) for v in blob.values()) == total_test_outputs(test_tasks)
    assert writer.validate() == []
