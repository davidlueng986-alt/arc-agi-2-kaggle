"""Top-level orchestration shared by `scripts/eval_local.py` and the Kaggle notebook."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .budget import BudgetConfig, BudgetGovernor, Clock, SystemClock
from .config import HarnessConfig
from .diagnostics import CoverageStats, CoverageTracker
from .grids import Grid
from .llm import HTTPChatClient, LLMClient, MockLLM, validate_effort
from .scoring import ScoreReport, score_submission
from .submission import SubmissionWriter
from .tasks import (
    ARCTask,
    load_challenges,
    resolve_challenges_path,
    resolve_data_dir,
    resolve_output_dir,
    resolve_solutions_path,
    total_test_outputs,
)
from .watchdog import Watchdog
from .waves import RunReport, Scheduler

Logger = Callable[[str], None]


def default_logger(prefix: str = "h1") -> Logger:
    def _log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {prefix}: {message}", flush=True)

    return _log


def load_tasks(config: HarnessConfig) -> List[ARCTask]:
    data_dir = resolve_data_dir(config.data_dir)
    challenges = resolve_challenges_path(data_dir, config.split)
    solutions = resolve_solutions_path(data_dir, config.split)
    tasks = load_challenges(challenges, solutions)
    if config.task_ids:
        wanted = set(config.task_ids)
        tasks = [t for t in tasks if t.task_id in wanted]
    if config.limit:
        tasks = tasks[: config.limit]
    return tasks


def build_client(config: HarnessConfig, answers: Optional[Dict[str, Sequence[Grid]]] = None) -> LLMClient:
    if answers is not None:
        return MockLLM(answers=answers)
    return HTTPChatClient(
        base_url=config.serving.base_url,
        model=config.serving.model,
        api_key=config.serving.api_key,
        timeout=config.sampling.request_timeout,
        max_retries=config.serving.max_retries,
        effort_mechanism=config.serving.effort_mechanism,
        extra_payload=config.serving.extra_payload,
    )


def mock_answers(tasks: Sequence[ARCTask]) -> Dict[str, List[Grid]]:
    return {t.task_id: [list(map(list, g)) for g in (t.solutions or [])] for t in tasks}


@dataclass
class RunOutcome:
    report: RunReport
    score: Optional[ScoreReport]
    submission_path: str
    tasks: List[ARCTask]
    config: HarnessConfig

    def write_artifacts(self, output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        report_path = os.path.join(output_dir, "run_report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(self.report.as_dict(), fh, indent=2, default=str)
        paths["report"] = report_path

        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(self.config.to_dict(), fh, indent=2, default=str)
        paths["config"] = config_path

        if self.score is not None:
            score_path = os.path.join(output_dir, "score.json")
            with open(score_path, "w", encoding="utf-8") as fh:
                json.dump(self.score.as_dict(), fh, indent=2)
            paths["score"] = score_path
        return paths


def run_harness(
    config: HarnessConfig,
    client: Optional[LLMClient] = None,
    clock: Optional[Clock] = None,
    logger: Optional[Logger] = None,
    tasks: Optional[Sequence[ARCTask]] = None,
    governor: Optional[BudgetGovernor] = None,
    submission_path: Optional[str] = None,
    use_watchdog: bool = False,
    hard_exit: bool = False,
    watchdog_poll_seconds: float = 5.0,
    watchdog_sleeper: Optional[Callable[[float], None]] = None,
) -> RunOutcome:
    log = logger or default_logger()
    clock = clock or SystemClock()
    tasks = list(tasks) if tasks is not None else load_tasks(config)
    if not tasks:
        raise ValueError("no tasks loaded")

    validate_effort(config.sampling.phase_a_effort)
    validate_effort(config.sampling.phase_b_effort)

    output_dir = resolve_output_dir(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    path = submission_path or os.path.join(output_dir, config.submission_name)

    governor = governor or BudgetGovernor(config.budget, clock=clock)
    writer = SubmissionWriter(path, tasks)
    if client is None:
        client = build_client(config, answers=mock_answers(tasks) if config.split == "mock" else None)

    log(
        f"run_harness: {len(tasks)} tasks, {total_test_outputs(tasks)} test outputs, "
        f"split={config.split}, submission={path}"
    )
    tracker = CoverageTracker(tasks)
    watchdog = None
    if use_watchdog:
        watchdog = Watchdog(
            governor,
            writer,
            logger=log,
            hard_exit=hard_exit,
            poll_seconds=watchdog_poll_seconds,
            clock=clock.now,
            sleeper=watchdog_sleeper or clock.sleep,
        )
        watchdog.start()
        log(
            f"watchdog armed: finalize at t+{governor.t_gen - governor.t0:.0f}s, "
            f"hard exit at t+{governor.t_submit - governor.config.finalize_reserve_seconds - governor.t0:.0f}s"
        )
    try:
        scheduler = Scheduler(client, tasks, config, governor, writer, logger=log, tracker=tracker)
        report = scheduler.run()
    finally:
        if watchdog is not None:
            watchdog.stop()
            writer.write()

    score: Optional[ScoreReport] = None
    if tracker.enabled:
        score = score_submission(writer.data, tasks)
        report.score = score.as_dict()
        coverage = tracker.finalize(score.score)
        report.coverage = coverage.as_dict()
        log(
            f"score={score.score * 100:.2f}% "
            f"(attempt_1 only={score.attempt_1_score * 100:.2f}%, "
            f"2nd-attempt gain={score.second_attempt_gain * 100:.2f}%, "
            f"{score.total_correct}/{score.total_outputs} outputs)"
        )
        log(
            f"coverage(pass@N)={coverage.coverage * 100:.2f}% "
            f"pass@1={coverage.pass_at_1 * 100:.2f}% "
            f"selection_efficiency={coverage.selection_efficiency * 100:.1f}% "
            f"-> {coverage.verdict()}"
        )

    errors = report.submission_errors
    if errors:
        log(f"SUBMISSION SCHEMA ERRORS ({len(errors)}): {errors[:5]}")
    else:
        log("submission schema valid")

    outcome = RunOutcome(report=report, score=score, submission_path=path, tasks=tasks, config=config)
    detail_path = os.path.join(output_dir, "per_task_detail.json")
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump(scheduler.per_task_rows(), fh, indent=2)
    outcome.write_artifacts(output_dir)
    return outcome


def write_config_snapshot(config: HarnessConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2, default=str)


def budget_table(
    config: Optional[BudgetConfig] = None,
    n_tasks: int = 240,
    phase_a_samples: int = 2,
) -> List[Dict[str, object]]:
    """Reproduce the HYPOTHESIS.md budget arithmetic from live config values."""
    c = config or BudgetConfig()
    usable = (
        c.total_seconds
        - c.submission_buffer_seconds
        - c.startup_seconds
        - c.finalize_reserve_seconds
    )
    planned = usable * (1.0 - c.safety_reserve_fraction)
    phase_a_tokens = n_tasks * phase_a_samples * c.low_tokens_per_sample
    rows = []
    for name, tps in (
        ("pessimistic", c.pessimistic_tok_per_s),
        ("central", c.planned_tok_per_s),
        ("optimistic", c.optimistic_tok_per_s),
    ):
        tokens = planned * tps
        phase_b = max(0.0, tokens - phase_a_tokens)
        rows.append(
            {
                "scenario": name,
                "tok_per_s": tps,
                "usable_seconds": round(usable, 0),
                "planned_seconds": round(planned, 0),
                "completion_token_budget": round(tokens, 0),
                "phase_a_tokens": round(phase_a_tokens, 0),
                "tokens_per_task": round(tokens / max(1, n_tasks), 0),
                "xhigh_samples_per_task_uniform": round(
                    phase_b / max(1, n_tasks) / c.xhigh_tokens_per_sample, 2
                ),
                "xhigh_wave_hours_uniform": round(
                    phase_b / max(1e-9, tps) / 3600.0, 2
                ),
            }
        )
    return rows
