"""Wave scheduler: cheap triage pass over every task, then budget-routed xhigh waves.

Design consequences of the budget model in HYPOTHESIS.md:

* Phase A runs `low`-effort samples over **all** tasks first. It costs ~4% of the
  token budget and ~25 minutes, and after it completes every task already has a
  schema-valid two-attempt answer on disk. That converts a 12-hour all-or-nothing
  run into a run whose worst case is "phase A quality", not zero.
* Phase B runs in *waves* rather than task-by-task. A single task only needs ~7
  concurrent xhigh samples, and a batch of 7 decodes at ~220 tok/s against a
  ~760 tok/s ceiling at batch 32. Waves over all tasks keep the in-flight pool at
  `concurrency` requests, which is what actually buys the throughput the budget
  assumes.
* Before every wave the governor re-plans against *measured* throughput, so a 2x
  throughput miss degrades samples-per-task instead of blowing the deadline.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .budget import BudgetGovernor
from .config import HarnessConfig
from .diagnostics import CoverageTracker
from .grids import Grid, fallback_grid
from .llm import ChatRequest, LLMClient
from .parse import merge_reasoning_and_content, parse_answers
from .prompts import build_messages
from .select import CandidatePool, IndexResult, task_consensus
from .submission import SubmissionWriter
from .tasks import ARCTask, total_test_outputs

Logger = Callable[[str], None]


@dataclass
class _Job:
    request: ChatRequest
    state: "TaskState"
    phase: str
    effort: str


@dataclass
class TaskState:
    task: ARCTask
    pool: CandidatePool
    base_seed: int
    results: List[IndexResult] = field(default_factory=list)
    fallbacks: List[Grid] = field(default_factory=list)
    stopped: bool = False
    samples_by_effort: Dict[str, int] = field(default_factory=dict)
    parse_methods: Dict[str, int] = field(default_factory=dict)
    failures: int = 0
    seconds: float = 0.0

    @property
    def n_samples(self) -> int:
        return sum(self.samples_by_effort.values())

    def xhigh_samples(self) -> int:
        return self.samples_by_effort.get("xhigh", 0)


@dataclass
class WaveRecord:
    index: int
    phase: str
    effort: str
    tasks: int
    requests: int
    samples_per_task: int
    completion_tokens: int = 0
    seconds: float = 0.0
    trimmed_reason: str = ""
    stopped_early: bool = False
    consensus_after: int = 0


@dataclass
class RunReport:
    n_tasks: int = 0
    n_test_outputs: int = 0
    elapsed_s: float = 0.0
    waves: List[WaveRecord] = field(default_factory=list)
    samples_per_task: float = 0.0
    xhigh_samples_per_task: float = 0.0
    consensus_stopped_tasks: int = 0
    tasks_with_no_parse: int = 0
    request_failures: int = 0
    parse_method_counts: Dict[str, int] = field(default_factory=dict)
    measured_tok_per_s: Optional[float] = None
    governor: Dict[str, object] = field(default_factory=dict)
    tokens: Dict[str, object] = field(default_factory=dict)
    score: Optional[Dict[str, object]] = None
    coverage: Optional[Dict[str, object]] = None
    submission_errors: List[str] = field(default_factory=list)
    submission_path: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_tasks": self.n_tasks,
            "n_test_outputs": self.n_test_outputs,
            "elapsed_s": round(self.elapsed_s, 1),
            "samples_per_task": round(self.samples_per_task, 2),
            "xhigh_samples_per_task": round(self.xhigh_samples_per_task, 2),
            "consensus_stopped_tasks": self.consensus_stopped_tasks,
            "tasks_with_no_parse": self.tasks_with_no_parse,
            "request_failures": self.request_failures,
            "parse_method_counts": self.parse_method_counts,
            "measured_tok_per_s": self.measured_tok_per_s,
            "governor": self.governor,
            "tokens": self.tokens,
            "score": self.score,
            "coverage": self.coverage,
            "submission_errors": self.submission_errors,
            "submission_path": self.submission_path,
            "waves": [w.__dict__ for w in self.waves],
        }


class Scheduler:
    def __init__(
        self,
        client: LLMClient,
        tasks: Sequence[ARCTask],
        config: HarnessConfig,
        governor: BudgetGovernor,
        writer: SubmissionWriter,
        logger: Optional[Logger] = None,
        tracker: Optional["CoverageTracker"] = None,
    ) -> None:
        self.client = client
        self.tasks = list(tasks)
        self.config = config
        self.governor = governor
        self.writer = writer
        self.tracker = tracker
        self.log: Logger = logger or (lambda _m: None)
        rng = random.Random(config.seed)
        order = list(self.tasks)
        rng.shuffle(order)
        self.states: Dict[str, TaskState] = {}
        for i, task in enumerate(order):
            state = TaskState(
                task=task,
                pool=CandidatePool(task.n_test),
                base_seed=rng.randrange(1, 2**30),
                fallbacks=[fallback_grid(task.test[j], task.train) for j in range(task.n_test)],
            )
            state.results = state.pool.select(state.fallbacks)
            self.states[task.task_id] = state
        self.order: List[str] = [t.task_id for t in order]
        self.report = RunReport()
        self._parse_counts: Dict[str, int] = {}

    def _request(self, state: TaskState, effort: str, sample_idx: int, phase: str) -> ChatRequest:
        sampling = self.config.sampling
        seed = state.base_seed + 1013 * sample_idx
        return ChatRequest(
            messages=build_messages(state.task, shuffle_train=sampling.shuffle_train, seed=seed),
            max_tokens=sampling.max_tokens_for(effort),
            temperature=sampling.temperature_for(effort),
            top_p=sampling.top_p,
            seed=seed,
            reasoning_effort=effort,
            meta={
                "task_id": state.task.task_id,
                "phase": phase,
                "effort": effort,
                "sample": sample_idx,
                "n_test": state.task.n_test,
            },
        )

    def _handle(self, response, job: _Job) -> None:
        state = job.state
        usage = response.usage
        self.governor.record(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            phase=job.phase,
            effort=job.effort,
            failed=not response.ok,
        )
        state.seconds += response.elapsed
        if not response.ok:
            state.failures += 1
            self.log(f"  ! {state.task.task_id} {job.phase}/{job.effort}: {response.error}")
            return
        text = merge_reasoning_and_content(response.reasoning, response.content)
        parsed = parse_answers(text, state.task.n_test)
        self._parse_counts[parsed.method] = self._parse_counts.get(parsed.method, 0) + 1
        state.parse_methods[parsed.method] = state.parse_methods.get(parsed.method, 0) + 1
        grids = parsed.as_list(state.task.n_test)
        if self.tracker is not None and self.tracker.enabled:
            for i, grid in enumerate(grids):
                if grid is not None:
                    self.tracker.observe(state.task.task_id, i, grid, job.effort)
        added = state.pool.add_sample(grids, effort=job.effort, truncated=response.truncated)
        if added:
            state.samples_by_effort[job.effort] = state.samples_by_effort.get(job.effort, 0) + 1
        self._refresh(state)
        if self.config.write_after_each_task:
            self._update_writer(state)

    def _refresh(self, state: TaskState) -> None:
        state.results = state.pool.select(state.fallbacks)
        sampling = self.config.sampling
        state.stopped = task_consensus(
            state.results, sampling.consensus_min_candidates, sampling.consensus_threshold
        )

    def _update_writer(self, state: TaskState) -> None:
        self.writer.update(state.task.task_id, [r.as_dict() for r in state.results])

    def _flush_all(self) -> None:
        for task_id in self.order:
            self._update_writer(self.states[task_id])
        self.writer.write()

    def _eligible(self) -> List[str]:
        return [tid for tid in self.order if not self.states[tid].stopped]

    def _execute(self, jobs: Sequence[_Job], wave_timeout: float) -> Tuple[int, bool]:
        """Run a batch of requests; returns (completed, stopped_early)."""
        if not jobs:
            return 0, False
        workers = max(1, min(self.config.sampling.concurrency, len(jobs)))
        completed = 0
        stopped_early = False
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm")
        futures = {executor.submit(self.client.complete, job.request): job for job in jobs}
        try:
            for future in as_completed(futures, timeout=max(1.0, wave_timeout)):
                job = futures[future]
                try:
                    response = future.result()
                except Exception as exc:
                    response = type(
                        "R",
                        (),
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed": 0.0,
                            "usage": type("U", (), {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0})(),
                            "truncated": False,
                            "content": "",
                            "reasoning": "",
                        },
                    )()
                self._handle(response, job)
                completed += 1
                if completed % self.config.log_every == 0:
                    self.log(
                        f"    {completed}/{len(jobs)} done | "
                        f"completion_tokens={self.governor.completion_tokens} | "
                        f"gen_left={self.governor.gen_seconds_left():.0f}s"
                    )
                if self.governor.must_stop_generating():
                    stopped_early = True
                    break
        except FuturesTimeout:
            stopped_early = True
            self.log(f"    wave timeout after {wave_timeout:.0f}s")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return completed, stopped_early

    def run(self) -> RunReport:
        started = time.monotonic()
        self.report.n_tasks = len(self.tasks)
        self.report.n_test_outputs = total_test_outputs(self.tasks)
        self.log(
            f"scheduler: {self.report.n_tasks} tasks / {self.report.n_test_outputs} test outputs; "
            f"plan horizon {self.governor.plan_seconds_left():.0f}s; "
            f"projected {self.governor.projected_tok_per_s():.0f} tok/s"
        )
        self._flush_all()

        self._run_phase_a()
        self._run_phase_b()

        self._flush_all()
        self.report.elapsed_s = time.monotonic() - started
        self._finalize_report()
        return self.report

    def _run_phase_a(self) -> None:
        sampling = self.config.sampling
        n = sampling.phase_a_samples
        if n <= 0:
            return
        allowed, reason = self.governor.plan_wave(len(self.order), n, sampling.phase_a_effort)
        self.log(f"phase A ({sampling.phase_a_effort}): plan={n}/task -> {allowed} ({reason})")
        if allowed <= 0:
            return
        t0 = time.monotonic()
        before = self.governor.completion_tokens
        jobs: List[_Job] = []
        for task_id in self.order:
            state = self.states[task_id]
            for s in range(allowed):
                jobs.append(
                    _Job(
                        request=self._request(state, sampling.phase_a_effort, s, "A"),
                        state=state,
                        phase="A",
                        effort=sampling.phase_a_effort,
                    )
                )
        wave_timeout = min(
            self.governor.gen_seconds_left(),
            max(60.0, self.governor.seconds_for_tokens(len(jobs) * sampling.phase_a_max_tokens) * 2.0),
        )
        completed, stopped = self._execute(jobs, wave_timeout)
        record = WaveRecord(
            index=0,
            phase="A",
            effort=sampling.phase_a_effort,
            tasks=len(self.order),
            requests=completed,
            samples_per_task=allowed,
            completion_tokens=self.governor.completion_tokens - before,
            seconds=time.monotonic() - t0,
            trimmed_reason=reason if reason != "full" else "",
            stopped_early=stopped,
            consensus_after=sum(1 for s in self.states.values() if s.stopped),
        )
        self.report.waves.append(record)
        self._flush_all()
        self.log(
            f"phase A done: {completed} requests, {record.completion_tokens} completion tokens, "
            f"{record.seconds:.0f}s, consensus on {record.consensus_after} tasks; submission written"
        )

    def _run_phase_b(self) -> None:
        sampling = self.config.sampling
        for wave_idx, requested in enumerate(sampling.wave_samples, start=1):
            if self.governor.must_stop_generating():
                self.log(f"phase B: stopping before wave {wave_idx}, generation deadline reached")
                break
            eligible = self._eligible()
            if not eligible:
                self.log(f"phase B: all {len(self.order)} tasks at consensus after wave {wave_idx - 1}")
                break
            remaining_cap = sampling.max_xhigh_samples
            allowed, reason = self.governor.plan_wave(len(eligible), requested, sampling.phase_b_effort)
            self.log(
                f"phase B wave {wave_idx}: {len(eligible)} eligible tasks, "
                f"plan={requested} -> {allowed} samples/task ({reason}); cap={remaining_cap}"
            )
            if allowed <= 0:
                self.report.waves.append(
                    WaveRecord(
                        index=wave_idx,
                        phase="B",
                        effort=sampling.phase_b_effort,
                        tasks=len(eligible),
                        requests=0,
                        samples_per_task=0,
                        trimmed_reason=reason,
                    )
                )
                break

            t0 = time.monotonic()
            before = self.governor.completion_tokens
            jobs: List[_Job] = []
            for task_id in eligible:
                state = self.states[task_id]
                room = max(0, remaining_cap - state.xhigh_samples())
                for s in range(min(allowed, room)):
                    jobs.append(
                        _Job(
                            request=self._request(
                                state, sampling.phase_b_effort, 100 + state.xhigh_samples() + s, f"B{wave_idx}"
                            ),
                            state=state,
                            phase=f"B{wave_idx}",
                            effort=sampling.phase_b_effort,
                        )
                    )
            est_tokens = len(jobs) * self.governor.config.tokens_per_sample(sampling.phase_b_effort)
            wave_timeout = min(
                self.governor.gen_seconds_left(),
                max(120.0, self.governor.seconds_for_tokens(est_tokens) * 1.8),
            )
            completed, stopped = self._execute(jobs, wave_timeout)
            record = WaveRecord(
                index=wave_idx,
                phase=f"B{wave_idx}",
                effort=sampling.phase_b_effort,
                tasks=len(eligible),
                requests=completed,
                samples_per_task=allowed,
                completion_tokens=self.governor.completion_tokens - before,
                seconds=time.monotonic() - t0,
                trimmed_reason=reason if reason != "full" else "",
                stopped_early=stopped,
                consensus_after=sum(1 for s in self.states.values() if s.stopped),
            )
            self.report.waves.append(record)
            self._flush_all()
            self.log(
                f"phase B wave {wave_idx} done: {completed} requests, "
                f"{record.completion_tokens} completion tokens, {record.seconds:.0f}s, "
                f"consensus on {record.consensus_after}/{len(self.order)} tasks"
            )
            if stopped:
                self.log("phase B: wave stopped early by deadline, finalizing")
                break

    def _finalize_report(self) -> None:
        states = list(self.states.values())
        n = max(1, len(states))
        self.report.samples_per_task = sum(s.n_samples for s in states) / n
        self.report.xhigh_samples_per_task = sum(s.xhigh_samples() for s in states) / n
        self.report.consensus_stopped_tasks = sum(1 for s in states if s.stopped)
        self.report.tasks_with_no_parse = sum(1 for s in states if s.pool.total() == 0)
        self.report.request_failures = self.governor.ledger.failures
        self.report.parse_method_counts = dict(self._parse_counts)
        measured = self.governor.measured_tok_per_s()
        self.report.measured_tok_per_s = round(measured, 1) if measured else None
        self.report.governor = self.governor.snapshot()
        self.report.tokens = self.governor.ledger.as_dict()
        self.report.submission_path = self.writer.path
        self.report.submission_errors = self.writer.validate()

    def per_task_rows(self) -> List[Dict[str, object]]:
        rows = []
        for task_id in sorted(self.states):
            state = self.states[task_id]
            rows.append(
                {
                    "task_id": task_id,
                    "n_test": state.task.n_test,
                    "samples": state.n_samples,
                    "xhigh_samples": state.xhigh_samples(),
                    "candidates": state.pool.total(),
                    "distinct": sum(r.n_distinct for r in state.results),
                    "consensus": round(min((r.consensus for r in state.results), default=0.0), 3),
                    "stopped": state.stopped,
                    "failures": state.failures,
                    "seconds": round(state.seconds, 1),
                    "parse": state.parse_methods,
                }
            )
        return rows
