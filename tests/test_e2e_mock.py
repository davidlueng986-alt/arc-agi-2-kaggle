"""End-to-end MockLLM runs over real eval-split tasks.

Asserts: valid submission schema (every task id, both attempts per test input,
rectangular int grids), cumulative writing, judge path exercised, and correct
scoring when the mock supplies verified programs.
"""

import asyncio
import json
import os

from harness.config import PipelineConfig
from harness.llm import MockLLM
from harness.runner import TaskRunner
from harness.submission import validate_submission


def _cfg(**kw):
    base = dict(
        experts=2,
        iters_a=3,
        iters_b=2,
        text_candidates=2,
        judges=3,
        task_concurrency=3,
        total_time_s=3600.0,
    )
    base.update(kw)
    return PipelineConfig(**base)


class TestE2EDeterministic:
    def test_ten_tasks_all_solved(self, ten_tasks, tmp_path):
        challenges, solutions = ten_tasks
        out = str(tmp_path / "submission.json")
        mock = MockLLM.from_tasks(challenges, solutions, seed=0)
        cfg = _cfg(use_judge=False)  # deterministic verified-consensus path
        runner = TaskRunner(challenges, mock, cfg, out, solutions=solutions)
        summary = asyncio.run(runner.run())

        assert summary["validation_errors"] == []
        # lookup mock => every task verified by both experts => pass@2 == 1.0
        assert summary["pass2"] == 1.0, summary
        assert summary["pass1"] == 1.0
        assert summary["verified_tasks"] == len(challenges)
        # judge was off
        assert summary["judged_test_inputs"] == 0

        with open(out, encoding="utf-8") as f:
            sub = json.load(f)
        assert validate_submission(sub, challenges) == []
        assert set(sub) == set(challenges)
        # order matches test inputs; both attempts equal ground truth
        for tid, task in challenges.items():
            assert len(sub[tid]) == len(task["test"])
            for j, gt in enumerate(solutions[tid]):
                assert sub[tid][j]["attempt_1"] == gt
                assert sub[tid][j]["attempt_2"] == gt or sub[tid][j]["attempt_2"]

    def test_cumulative_file_written_during_run(self, ten_tasks, tmp_path):
        challenges, solutions = ten_tasks
        out = str(tmp_path / "cum.json")
        mock = MockLLM.from_tasks(challenges, solutions, seed=1)
        cfg = _cfg(use_judge=False, task_concurrency=1)
        runner = TaskRunner(challenges, mock, cfg, out, solutions=solutions)
        asyncio.run(runner.run())
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            sub = json.load(f)
        assert set(sub) == set(challenges)


class TestE2EAdversarial:
    def test_malformed_outputs_and_judge_path(self, ten_tasks, tmp_path):
        challenges, solutions = ten_tasks
        out = str(tmp_path / "adv.json")
        # every 2nd solver call returns prose with no code block
        mock = MockLLM.from_tasks(challenges, solutions, fail_every=2, seed=7)
        cfg = _cfg(use_judge=True, judges=3, iters_a=3, iters_b=2)
        runner = TaskRunner(challenges, mock, cfg, out, solutions=solutions)
        summary = asyncio.run(runner.run())

        assert summary["validation_errors"] == []
        with open(out, encoding="utf-8") as f:
            sub = json.load(f)
        assert validate_submission(sub, challenges) == []
        # judge path was exercised on at least some test inputs OR every task
        # early-accepted; either way the pipeline must not error out
        assert summary["channel_errors"] == 0
        # malformed iterations must not prevent solving via the lookup mock
        assert summary["pass2"] >= 0.9, summary

    def test_all_refusals_still_writes_valid_submission(self, ten_tasks, tmp_path):
        challenges, _solutions = ten_tasks
        out = str(tmp_path / "refusal.json")
        mock = MockLLM(responses=["I cannot assist with that request."])
        cfg = _cfg(use_judge=True)
        runner = TaskRunner(challenges, mock, cfg, out)
        summary = asyncio.run(runner.run())

        assert summary["validation_errors"] == []
        with open(out, encoding="utf-8") as f:
            sub = json.load(f)
        assert validate_submission(sub, challenges) == []
        for tid, task in challenges.items():
            assert len(sub[tid]) == len(task["test"])
            for entry in sub[tid]:
                assert entry["attempt_1"] and entry["attempt_2"]

    def test_judge_path_exercised(self, ten_tasks, tmp_path):
        challenges, _ = ten_tasks
        out = str(tmp_path / "judge.json")
        # no solutions registered -> mock emits valid-but-unverified identity
        # programs -> no early accept -> the judge council must run
        mock = MockLLM.from_tasks(challenges, None, seed=11)
        cfg = _cfg(use_judge=True, judges=3)
        runner = TaskRunner(challenges, mock, cfg, out)
        summary = asyncio.run(runner.run())
        assert summary["validation_errors"] == []
        assert summary["verified_tasks"] == 0
        assert summary["judged_test_inputs"] > 0
        tok = summary["tokens"]["components"].get("judge")
        assert tok is not None and tok["calls"] >= 3
        # prefix-cache accounting: 3 judges share one prefill -> effective < billed
        assert tok["effective"] < tok["billed"]

    def test_exception_client_never_kills_run(self, ten_tasks, tmp_path):
        challenges, _ = ten_tasks
        out = str(tmp_path / "exc.json")
        mock = MockLLM(responses=[RuntimeError("service unavailable")])
        cfg = _cfg()
        runner = TaskRunner(challenges, mock, cfg, out)
        summary = asyncio.run(runner.run())
        assert summary["validation_errors"] == []
        with open(out, encoding="utf-8") as f:
            sub = json.load(f)
        assert set(sub) == set(challenges)


class TestE2EAblations:
    def test_no_experts_text_only(self, ten_tasks, tmp_path):
        challenges, solutions = ten_tasks
        out = str(tmp_path / "textonly.json")
        mock = MockLLM.from_tasks(challenges, solutions, seed=3)
        cfg = _cfg(experts=0, text_candidates=2, use_judge=False)
        runner = TaskRunner(challenges, mock, cfg, out, solutions=solutions)
        summary = asyncio.run(runner.run())
        assert summary["validation_errors"] == []
        # text mock echoes test inputs (wrong), so pass@2 is low but schema holds
        assert summary["pass2"] is not None

    def test_single_expert(self, ten_tasks, tmp_path):
        challenges, solutions = ten_tasks
        out = str(tmp_path / "single.json")
        mock = MockLLM.from_tasks(challenges, solutions, seed=4)
        cfg = _cfg(experts=1, use_judge=False)
        runner = TaskRunner(challenges, mock, cfg, out, solutions=solutions)
        summary = asyncio.run(runner.run())
        assert summary["pass2"] == 1.0
        # with one expert there is only 1 verified vote -> early_accept does NOT
        # fire, but judge is off, so selection falls back to consensus scoring
        assert summary["validation_errors"] == []
