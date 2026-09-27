"""Budget accounting and the wall-clock watchdog.

Two separate currencies are tracked, because they bind differently:

* **Tokens.**  A competitor's ARC-AGI-3 A/B put it exactly: "Tokens are the real
  currency, not actions. Every run hits the wallclock cap... any change that
  makes a turn more expensive directly buys you fewer moves."  Their animation
  feature cost +17% tokens/action and mechanically cost them moves.  So every
  turn here is accounted for, prompt and completion separately, because prefill
  and decode have different throughput on the serving side.
* **Wall clock.**  A Kaggle rerun is killed at 12h.  The watchdog finalises
  ``submission.json`` before the safety buffer, and the per-task deadline lets a
  single hard task be abandoned rather than eating the run.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    python_calls: int = 0
    verify_calls: int = 0
    seconds: float = 0.0
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls
        self.python_calls += other.python_calls
        self.verify_calls += other.verify_calls
        self.seconds += other.seconds
        self.estimated_prompt_tokens += other.estimated_prompt_tokens
        self.estimated_completion_tokens += other.estimated_completion_tokens

    @property
    def tokens_per_call(self) -> float:
        return self.total_tokens / self.calls if self.calls else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "python_calls": self.python_calls,
            "verify_calls": self.verify_calls,
            "seconds": round(self.seconds, 2),
            "tokens_per_call": round(self.tokens_per_call, 1),
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "estimated_completion_tokens": self.estimated_completion_tokens,
        }


class Deadline:
    """Shared monotonic deadline with a soft-stop flag the loops poll."""

    def __init__(self, total_seconds: float, *, start: float | None = None) -> None:
        self._start = time.monotonic() if start is None else start
        # No 1s floor: tests and the runner's adaptive squeeze both need
        # sub-second deadlines to mean what they say.
        self._end = self._start + max(1e-3, float(total_seconds))
        self._stop = threading.Event()

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def expired(self) -> bool:
        return time.monotonic() >= self._end or self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()


@dataclass
class TaskBudget:
    """Per-task limits. ``exceeded`` returns a reason string or ``None``."""

    max_turns: int
    max_tokens: int
    max_seconds: float
    deadline: Deadline | None = None
    started: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def exceeded(self, usage: Usage, turns: int) -> str | None:
        if turns >= self.max_turns:
            return f"turn cap {self.max_turns}"
        if usage.total_tokens >= self.max_tokens:
            return f"token cap {self.max_tokens}"
        if self.elapsed() >= self.max_seconds:
            return f"task time cap {self.max_seconds:.0f}s"
        if self.deadline is not None and self.deadline.expired():
            return "global deadline"
        return None


class Watchdog(threading.Thread):
    """Finalises the submission before the hard deadline.

    Runs ``on_soft`` when the soft deadline is reached (the runner uses it to set
    a global stop flag and shrink per-task turn caps) and ``on_hard`` shortly
    before the real deadline (an immediate blocking write of a complete,
    schema-valid submission covering every task id).
    """

    def __init__(
        self,
        deadline: Deadline,
        *,
        on_soft: Callable[[], None],
        on_hard: Callable[[], None],
        soft_fraction: float = 0.93,
        hard_lead_seconds: float = 90.0,
        poll_seconds: float = 2.0,
    ) -> None:
        super().__init__(daemon=True, name="arc26-watchdog")
        self.deadline = deadline
        self.on_soft = on_soft
        self.on_hard = on_hard
        self.soft_fraction = soft_fraction
        self.hard_lead_seconds = hard_lead_seconds
        self.poll_seconds = poll_seconds
        self.soft_fired = False
        self.hard_fired = False

    def run(self) -> None:  # pragma: no cover - timing dependent
        total = max(1.0, self.deadline.end - self.deadline.start)
        soft_at = self.deadline.start + total * self.soft_fraction
        hard_at = self.deadline.end - self.hard_lead_seconds
        while True:
            now = time.monotonic()
            if not self.soft_fired and now >= soft_at:
                self.soft_fired = True
                try:
                    self.on_soft()
                except Exception:
                    pass
            if not self.hard_fired and now >= hard_at:
                self.hard_fired = True
                try:
                    self.on_hard()
                except Exception:
                    pass
                return
            if now >= self.deadline.end:
                self.hard_fired = True
                try:
                    self.on_hard()
                except Exception:
                    pass
                return
            time.sleep(min(self.poll_seconds, max(0.05, hard_at - now)))


class BudgetLedger:
    """Thread-safe global accounting across all worker threads.

    The lock must be re-entrant: ``snapshot()`` calls ``mean_turn_tokens()``
    while holding it.  A plain ``Lock`` there deadlocked the first time the
    runner tried to write a report -- which is exactly what the test suite is
    for.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.total = Usage()
        self.per_task: dict[str, Usage] = {}
        self.turn_token_samples: list[int] = []
        self.tasks_done = 0
        self.tasks_solved_verified = 0

    def record(self, task_id: str, usage: Usage) -> None:
        with self._lock:
            self.total.add(usage)
            existing = self.per_task.setdefault(task_id, Usage())
            existing.add(usage)
            if usage.calls:
                self.turn_token_samples.append(int(usage.tokens_per_call))

    def record_turn(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.turn_token_samples.append(prompt_tokens + completion_tokens)

    def finish_task(self, task_id: str, *, verified: bool) -> None:
        with self._lock:
            self.tasks_done += 1
            if verified:
                self.tasks_solved_verified += 1

    def mean_turn_tokens(self) -> float:
        with self._lock:
            if not self.turn_token_samples:
                return 0.0
            return sum(self.turn_token_samples) / len(self.turn_token_samples)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": self.total.as_dict(),
                "tasks_done": self.tasks_done,
                "tasks_verified": self.tasks_solved_verified,
                "mean_turn_tokens": round(self.mean_turn_tokens(), 1),
                "per_task": {k: v.as_dict() for k, v in self.per_task.items()},
            }
