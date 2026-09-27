"""Multi-task orchestration: workers, cumulative submission, watchdog.

Design points that matter on a Kaggle rerun:

* **The submission file is valid at every instant.**  It is pre-filled with
  fallback entries for every task id in the challenge file before the first task
  starts, rewritten atomically after every completed task, and finalised by a
  watchdog thread before the 600s buffer.  A kill at any second still yields a
  schema-valid file covering all 240 task ids with both attempts present.
* **Adaptive scheduling.**  The per-task time and turn allowance are recomputed
  from ``remaining_time * workers / remaining_tasks`` before each dispatch, so a
  slow start automatically shortens later tasks instead of running out of clock.
* **Threads, not asyncio.**  Each worker's hot path is a blocking HTTP call plus
  a blocking subprocess; threads keep both simple and let the sandbox stay a
  plain ``subprocess.Popen`` with a reader thread (which the interactive
  ``verify`` round-trip needs anyway).
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .agent import AgentResult, DuckREPLAgent
from .budget import BudgetLedger, Deadline, Usage, Watchdog
from .config import HarnessConfig
from .grids import Grid, Task
from .llm import LLMClient
from .submission import (
    build_submission,
    score_submission,
    validate_submission,
    write_submission,
)


@dataclass
class RunResult:
    submission: dict[str, Any] = field(default_factory=dict)
    score: dict[str, Any] | None = None
    ledger: dict[str, Any] = field(default_factory=dict)
    per_task: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    tasks_attempted: int = 0
    tasks_verified: int = 0
    submission_path: str = ""
    schema_problems: list[str] = field(default_factory=list)


class Runner:
    def __init__(
        self,
        llm: LLMClient,
        config: HarnessConfig,
        tasks: Sequence[Task],
        *,
        submission_path: str,
        task_shapes: dict[str, int] | None = None,
        solutions: dict[str, Sequence[Grid]] | None = None,
        total_tasks: int | None = None,
        deadline: Deadline | None = None,
        workers: int | None = None,
        keep_transcript: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        self.llm = llm
        self.config = config
        self.tasks = list(tasks)
        self.submission_path = submission_path
        self.solutions = solutions
        self.deadline = deadline or Deadline(config.budget.usable_seconds)
        self.workers = max(1, int(workers or config.workers))
        self.keep_transcript = keep_transcript
        self.log = log
        self.ledger = BudgetLedger()
        self.total_tasks = int(total_tasks or len(self.tasks))

        shapes = dict(task_shapes or {})
        for task in self.tasks:
            shapes.setdefault(task.task_id, task.n_test)
        self.task_shapes = shapes

        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._predictions: dict[str, list[dict[str, Any]]] = {}
        self._per_task: list[dict[str, Any]] = []
        self._done = 0
        self._fallback: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ public

    def run(self) -> RunResult:
        started = time.perf_counter()
        self._prefill_fallbacks()
        self._write()

        watchdog = Watchdog(
            self.deadline,
            on_soft=self._on_soft_deadline,
            on_hard=self._on_hard_deadline,
        )
        watchdog.start()

        try:
            if self.workers == 1:
                for task in self.tasks:
                    if self.deadline.expired():
                        break
                    self._run_one(task)
            else:
                with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="arc26") as pool:
                    futures = [pool.submit(self._guarded_run_one, task) for task in self.tasks]
                    for future in futures:
                        try:
                            future.result()
                        except Exception as exc:
                            self.log(f"! worker crashed: {type(exc).__name__}: {exc}")
                        if self.deadline.expired():
                            for pending in futures:
                                pending.cancel()
                            break
        finally:
            self.finalize()

        elapsed = time.perf_counter() - started
        with self._lock:
            submission = copy.deepcopy(self._current_submission())
            per_task = list(self._per_task)
        problems = validate_submission(submission, self.task_shapes)
        score = None
        if self.solutions:
            score = score_submission(submission, self.solutions)
        result = RunResult(
            submission=submission,
            score=score,
            ledger=self.ledger.snapshot(),
            per_task=per_task,
            seconds=elapsed,
            tasks_attempted=len(per_task),
            tasks_verified=self.ledger.tasks_solved_verified,
            submission_path=self.submission_path,
            schema_problems=problems,
        )
        return result

    def finalize(self) -> None:
        """Blocking, idempotent: guarantee a complete schema-valid file on disk."""
        with self._lock:
            submission = self._current_submission()
        problems = validate_submission(submission, self.task_shapes)
        if problems:
            submission = build_submission(self._predictions_with_fallbacks(), self.task_shapes)
        try:
            with self._write_lock:
                write_submission(self.submission_path, submission)
        except Exception as exc:
            self.log(f"!! could not write {self.submission_path}: {exc}")

    # ----------------------------------------------------------------- internals

    def _guarded_run_one(self, task: Task) -> None:
        if self.deadline.expired():
            return
        try:
            self._run_one(task)
        except Exception as exc:
            self.log(f"! {task.task_id} crashed: {type(exc).__name__}: {exc}")
            with self._lock:
                self._per_task.append({"task_id": task.task_id, "error": f"{type(exc).__name__}: {exc}"[:300]})
            self._write()

    def _run_one(self, task: Task) -> None:
        effective = self._effective_config(task)
        agent = DuckREPLAgent(
            self.llm,
            effective,
            deadline=self.deadline,
            keep_transcript=self.keep_transcript,
        )
        result = agent.solve(task)
        self.ledger.record(task.task_id, result.usage)
        self.ledger.finish_task(task.task_id, verified=result.verified)
        with self._lock:
            if result.entries:
                self._predictions[task.task_id] = result.entries
            self._done += 1
            done = self._done
            record = {
                "task_id": task.task_id,
                "turns": result.turns,
                "python_calls": result.python_calls,
                "verify_calls": result.verify_calls,
                "verified": result.verified,
                "n_verified": result.n_verified,
                "pool_size": result.pool_size,
                "best_soft_score": round(result.best_soft_score, 4),
                "stop_reason": result.stop_reason,
                "error": result.error,
                "seconds": round(result.elapsed_s, 1),
                "tokens": result.usage.total_tokens,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            }
            if self.solutions and task.task_id in self.solutions:
                from .submission import score_task_entries

                record["score"] = score_task_entries(result.entries, self.solutions[task.task_id])
            self._per_task.append(record)
        self._write()
        mark = "x" if record.get("score") == 1.0 else ("+" if result.verified else "-")
        extra = f" score={record['score']:.2f}" if "score" in record else ""
        self.log(
            f"{mark} {task.task_id} [{done}/{self.total_tasks}] turns={result.turns} "
            f"tok={result.usage.total_tokens} {result.elapsed_s:.0f}s "
            f"verified={result.n_verified}{extra} ({result.stop_reason})"
        )

    def _effective_config(self, task: Task) -> HarnessConfig:
        """Shrink per-task allowances when the run is behind schedule."""
        effective = copy.deepcopy(self.config)
        with self._lock:
            remaining_tasks = max(1, self.total_tasks - self._done)
        remaining_seconds = self.deadline.remaining()
        if remaining_seconds > 0:
            allowance = remaining_seconds * self.workers / remaining_tasks
            effective.budget.max_task_seconds = float(
                min(max(30.0, allowance * 0.9), self.config.budget.max_task_seconds)
            )
            fraction = effective.budget.max_task_seconds / max(1.0, self.config.budget.max_task_seconds)
            if fraction < 0.75:
                scaled = max(3, int(round(self.config.budget.max_turns * fraction)))
                effective.budget.max_turns = min(effective.budget.max_turns, scaled)
                effective.budget.hard_max_turns = min(
                    effective.budget.hard_max_turns, max(scaled + 2, effective.budget.max_turns)
                )
        if remaining_seconds < 120:
            effective.budget.max_turns = min(effective.budget.max_turns, 2)
            effective.budget.hard_max_turns = min(effective.budget.hard_max_turns, 3)
        return effective

    def _prefill_fallbacks(self) -> None:
        by_id = {t.task_id: t for t in self.tasks}
        for task_id, n_test in self.task_shapes.items():
            task = by_id.get(task_id)
            entries = []
            for index in range(max(0, int(n_test))):
                grid: Grid = (
                    [list(row) for row in task.test_inputs[index]]
                    if task is not None and index < task.n_test
                    else [[0, 0], [0, 0]]
                )
                entries.append({"attempt_1": grid, "attempt_2": [list(r) for r in grid]})
            self._fallback[task_id] = entries

    def _predictions_with_fallbacks(self) -> dict[str, list[dict[str, Any]]]:
        merged = dict(self._fallback)
        merged.update(self._predictions)
        return merged

    def _current_submission(self) -> dict[str, Any]:
        return build_submission(self._predictions_with_fallbacks(), self.task_shapes)

    def _write(self) -> None:
        """Cumulative write after every task.

        The rename is serialised on its own lock: two workers finishing at the
        same instant must not race on the destination path.  On POSIX (Kaggle)
        ``os.replace`` is ``rename(2)`` and atomic, so a reader always sees either
        the old or the new complete file; the lock just keeps the ordering
        deterministic.
        """
        try:
            with self._lock:
                submission = self._current_submission()
            with self._write_lock:
                write_submission(self.submission_path, submission)
        except Exception as exc:
            self.log(f"! cumulative submission write failed: {exc}")

    def _on_soft_deadline(self) -> None:
        self.log("watchdog: soft deadline reached -- stopping after in-flight tasks")
        self.deadline.request_stop()

    def _on_hard_deadline(self) -> None:
        self.log("watchdog: hard deadline -- finalising submission now")
        self.deadline.request_stop()
        self.finalize()


def write_run_report(path: str, result: RunResult) -> None:
    payload = {
        "seconds": round(result.seconds, 1),
        "tasks_attempted": result.tasks_attempted,
        "tasks_verified": result.tasks_verified,
        "score": result.score,
        "ledger": {k: v for k, v in result.ledger.items() if k != "per_task"},
        "schema_problems": result.schema_problems,
        "per_task": result.per_task,
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
