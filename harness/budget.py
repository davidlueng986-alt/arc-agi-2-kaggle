"""Token + wall-clock budget accounting, and the deadline-driven degradation ladder."""

from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass
from typing import Optional

# Degradation ladder thresholds (fraction of total wall clock elapsed).
LEVEL_THRESHOLDS = (0.60, 0.75, 0.88, 0.93)

# Current task scope for token accounting; each asyncio.Task gets its own copy
# of the context, so concurrent tasks never cross-attribute tokens.
_CURRENT_TASK: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "ledger_task", default=None
)


class Deadline:
    """Global wall-clock deadline with a 4-level degradation ladder.

    level(): 0 normal, 1 @60%, 2 @75%, 3 @88%, 4 @93% (stop solving, finalize).
    """

    def __init__(self, total_seconds: float, start: Optional[float] = None):
        self.total_seconds = float(total_seconds)
        self.start = time.monotonic() if start is None else start

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed())

    def fraction_used(self) -> float:
        if self.total_seconds <= 0:
            return 1.0
        return min(1.0, self.elapsed() / self.total_seconds)

    def level(self) -> int:
        f = self.fraction_used()
        level = 0
        for i, t in enumerate(LEVEL_THRESHOLDS, start=1):
            if f >= t:
                level = i
        return level

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def task_deadline(self, tasks_remaining: int, cap_s: float = 900.0) -> float:
        """Per-task soft deadline: fair share of remaining time x1.5 front-load, capped."""
        if tasks_remaining <= 0:
            return 0.0
        return min(cap_s, self.remaining() / tasks_remaining * 1.5)


@dataclass
class ComponentUsage:
    billed_prompt: int = 0
    billed_completion: int = 0
    effective_prompt: int = 0  # prefix-cache-aware: shared prefills counted once
    calls: int = 0

    @property
    def billed(self) -> int:
        return self.billed_prompt + self.billed_completion

    @property
    def effective(self) -> int:
        return self.effective_prompt + self.billed_completion


class TokenLedger:
    """Records per-component token usage; distinguishes billed vs GPU-effective
    tokens (a repeated cache_key prefill within one task is counted once)."""

    def __init__(self, per_task_effective_cap: int = 120_000):
        self.per_task_effective_cap = per_task_effective_cap
        self._lock = threading.Lock()
        self.components: dict[str, ComponentUsage] = {}
        self._seen_by_task: dict[str, set[str]] = {}
        self.task_totals: dict[str, dict[str, int]] = {}

    # -- task scoping ---------------------------------------------------------
    def begin_task(self, task_id: str) -> None:
        _CURRENT_TASK.set(task_id)
        with self._lock:
            self.task_totals.setdefault(task_id, {"billed": 0, "effective": 0})
            self._seen_by_task.setdefault(task_id, set())

    def end_task(self) -> dict[str, int]:
        tid = _CURRENT_TASK.get()
        _CURRENT_TASK.set(None)
        with self._lock:
            totals = dict(self.task_totals.get(tid or "", {"billed": 0, "effective": 0}))
            self._seen_by_task.pop(tid or "", None)
            return totals

    # -- accounting -----------------------------------------------------------
    def add(
        self,
        component: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_key: Optional[str] = None,
    ) -> None:
        tid = _CURRENT_TASK.get()
        with self._lock:
            c = self.components.setdefault(component, ComponentUsage())
            c.billed_prompt += prompt_tokens
            c.billed_completion += completion_tokens
            c.calls += 1
            effective_prompt = prompt_tokens
            if cache_key is not None:
                seen = self._seen_by_task.setdefault(tid or "", set())
                if cache_key in seen:
                    effective_prompt = 0  # prefix-cache hit: no fresh prefill
                else:
                    seen.add(cache_key)
            c.effective_prompt += effective_prompt
            if tid is not None:
                t = self.task_totals.setdefault(tid, {"billed": 0, "effective": 0})
                t["billed"] += prompt_tokens + completion_tokens
                t["effective"] += effective_prompt + completion_tokens

    # -- queries --------------------------------------------------------------
    def task_effective(self, task_id: Optional[str] = None) -> int:
        tid = task_id if task_id is not None else _CURRENT_TASK.get()
        with self._lock:
            if tid is None:
                return 0
            return self.task_totals.get(tid, {}).get("effective", 0)

    def task_over_budget(self, task_id: Optional[str] = None) -> bool:
        return self.task_effective(task_id) >= self.per_task_effective_cap

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "components": {
                    k: {
                        "billed": v.billed,
                        "effective": v.effective,
                        "calls": v.calls,
                    }
                    for k, v in self.components.items()
                },
                "total_billed": sum(v.billed for v in self.components.values()),
                "total_effective": sum(v.effective for v in self.components.values()),
                "tasks": len(self.task_totals),
            }
