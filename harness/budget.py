"""Budget accounting (tokens + wall clock) and the hard-stop watchdog."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class BudgetStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tasks_done: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Budget:
    """Thread-safe token + wall-clock accountant.

    Per-task token caps implement the quantitative plan in HYPOTHESIS.md:
    ~55-75K raw tokens/task (mostly prefix-cached prompt) and <= ~14K
    completion tokens/task keeps 240 tasks inside the 12h L4x4 envelope.
    """

    def __init__(
        self,
        deadline_ts: float,
        per_task_tokens: int = 75_000,
        total_tokens: int = 24_000_000,
        reserve_seconds: float = 420.0,
    ):
        self.deadline_ts = deadline_ts
        self.per_task_tokens = per_task_tokens
        self.total_tokens = total_tokens
        self.reserve_seconds = reserve_seconds
        self.stats = BudgetStats()
        self._lock = threading.Lock()
        self._task_spent: dict[str, int] = {}

    # -- charging ------------------------------------------------------------

    def charge(self, task_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.stats.prompt_tokens += prompt_tokens
            self.stats.completion_tokens += completion_tokens
            self.stats.llm_calls += 1
            self._task_spent[task_id] = self._task_spent.get(task_id, 0) + (
                prompt_tokens + completion_tokens
            )

    def task_spent(self, task_id: str) -> int:
        with self._lock:
            return self._task_spent.get(task_id, 0)

    def task_exhausted(self, task_id: str) -> bool:
        return self.task_spent(task_id) >= self.per_task_tokens

    def mark_task_done(self) -> None:
        with self._lock:
            self.stats.tasks_done += 1

    # -- gating ---------------------------------------------------------------

    def expired(self) -> bool:
        return time.time() >= self.deadline_ts - self.reserve_seconds

    def seconds_left(self) -> float:
        return max(0.0, self.deadline_ts - self.reserve_seconds - time.time())

    def can_call(self, task_id: str, est_tokens: int = 0) -> bool:
        with self._lock:
            if self.stats.total_tokens + est_tokens > self.total_tokens:
                return False
            if self._task_spent.get(task_id, 0) + est_tokens > self.per_task_tokens:
                return False
        return not self.expired()

    def per_task_time_cap(self, tasks_remaining: int) -> float:
        """Fair share of remaining wall clock, x1.5 slack, floored at 60s."""
        if tasks_remaining <= 0:
            return 60.0
        return max(60.0, self.seconds_left() / tasks_remaining * 1.5)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.stats.prompt_tokens,
                "completion_tokens": self.stats.completion_tokens,
                "llm_calls": self.stats.llm_calls,
                "tasks_done": self.stats.tasks_done,
                "seconds_left": round(self.seconds_left(), 1),
            }


class Watchdog:
    """Hard guarantee that the submission file is finalized before the wall.

    `on_expire` runs in a daemon thread at `deadline_ts`; after
    `hard_kill_grace_s` more the process is terminated with os._exit so a
    wedged main loop cannot eat the submission (lb3389 pattern).
    """

    def __init__(self, deadline_ts: float, on_expire, hard_kill_grace_s: float = 120.0):
        self.deadline_ts = deadline_ts
        self.on_expire = on_expire
        self.hard_kill_grace_s = hard_kill_grace_s
        self.fired = False
        self._timer: threading.Timer | None = None
        self._killer: threading.Timer | None = None

    def start(self) -> None:
        delay = max(0.0, self.deadline_ts - time.time())
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.fired = True
        try:
            self.on_expire()
        finally:
            import os

            self._killer = threading.Timer(self.hard_kill_grace_s, lambda: os._exit(0))
            self._killer.daemon = True
            self._killer.start()

    def cancel(self) -> None:
        if self._timer:
            self._timer.cancel()
        if self._killer:
            self._killer.cancel()
