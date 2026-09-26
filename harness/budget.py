"""Wall-clock and token budget accounting.

The 12h Kaggle rerun is a hard deadline, so the scheduler never trusts its own
throughput forecast: it plans against a safety-discounted horizon, measures the
achieved aggregate decode rate, and re-plans before every wave. All time reads go
through an injectable `Clock` so the governor is unit-testable in milliseconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


class Clock:
    def now(self) -> float:
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class ManualClock(Clock):
    """Test clock: advance it explicitly instead of waiting."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += max(0.0, seconds)

    def advance(self, seconds: float) -> None:
        self._t += max(0.0, seconds)

    def set(self, value: float) -> None:
        self._t = float(value)


@dataclass
class BudgetConfig:
    total_seconds: float = 43200.0
    submission_buffer_seconds: float = 600.0
    startup_seconds: float = 1020.0
    finalize_reserve_seconds: float = 120.0
    safety_reserve_fraction: float = 0.10
    planned_tok_per_s: float = 450.0
    pessimistic_tok_per_s: float = 280.0
    optimistic_tok_per_s: float = 700.0
    min_measured_window_s: float = 30.0
    min_measured_tokens: float = 20000.0
    xhigh_tokens_per_sample: int = 10000
    medium_tokens_per_sample: int = 4000
    low_tokens_per_sample: int = 1300

    def tokens_per_sample(self, effort: str) -> int:
        return {
            "xhigh": self.xhigh_tokens_per_sample,
            "medium": self.medium_tokens_per_sample,
            "low": self.low_tokens_per_sample,
        }.get(effort, self.xhigh_tokens_per_sample)


@dataclass
class TokenLedger:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0
    failures: int = 0
    by_phase: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_effort: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
        phase: str = "?",
        effort: str = "?",
        failed: bool = False,
    ) -> None:
        self.requests += 1
        if failed:
            self.failures += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.reasoning_tokens += reasoning_tokens
        for bucket, key in ((self.by_phase, phase), (self.by_effort, effort)):
            entry = bucket.setdefault(key, {"prompt": 0, "completion": 0, "requests": 0})
            entry["prompt"] += prompt_tokens
            entry["completion"] += completion_tokens
            entry["requests"] += 1

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> Dict[str, object]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "requests": self.requests,
            "failures": self.failures,
            "by_phase": self.by_phase,
            "by_effort": self.by_effort,
        }


class BudgetGovernor:
    def __init__(self, config: Optional[BudgetConfig] = None, clock: Optional[Clock] = None) -> None:
        self.config = config or BudgetConfig()
        self.clock = clock or SystemClock()
        self.ledger = TokenLedger()
        self.t0: float = self.clock.now()
        c = self.config
        self.t_hard = self.t0 + c.total_seconds
        self.t_submit = self.t_hard - c.submission_buffer_seconds
        self.t_gen = self.t_submit - c.finalize_reserve_seconds
        self.t_gen_start = self.t0 + c.startup_seconds
        span = max(1.0, self.t_gen - self.t_gen_start)
        self.t_plan = self.t_gen - c.safety_reserve_fraction * span
        self._gen_tokens_at_start = 0

    @property
    def elapsed(self) -> float:
        return self.clock.now() - self.t0

    def seconds_to(self, horizon: float) -> float:
        return max(0.0, horizon - self.clock.now())

    def plan_seconds_left(self) -> float:
        return self.seconds_to(self.t_plan)

    def gen_seconds_left(self) -> float:
        return self.seconds_to(self.t_gen)

    def hard_seconds_left(self) -> float:
        return self.seconds_to(self.t_submit)

    def must_stop_generating(self) -> bool:
        return self.clock.now() >= self.t_gen

    def must_finalize_now(self) -> bool:
        return self.clock.now() >= self.t_submit - self.config.finalize_reserve_seconds

    def startup_budget_exceeded(self) -> bool:
        return self.clock.now() > self.t_gen_start

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int = 0,
        phase: str = "?",
        effort: str = "?",
        failed: bool = False,
    ) -> None:
        self.ledger.record(
            prompt_tokens, completion_tokens, reasoning_tokens, phase, effort, failed
        )

    @property
    def completion_tokens(self) -> int:
        return self.ledger.completion_tokens

    def generation_seconds(self) -> float:
        return max(0.0, self.clock.now() - self.t_gen_start)

    def measured_tok_per_s(self) -> Optional[float]:
        secs = self.generation_seconds()
        tokens = self.ledger.completion_tokens - self._gen_tokens_at_start
        if secs < self.config.min_measured_window_s or tokens < self.config.min_measured_tokens:
            return None
        return tokens / secs

    def projected_tok_per_s(self) -> float:
        measured = self.measured_tok_per_s()
        if measured is None:
            return self.config.planned_tok_per_s
        return max(self.config.pessimistic_tok_per_s * 0.5, measured)

    def affordable_completion_tokens(self, horizon: Optional[float] = None) -> float:
        target = self.t_plan if horizon is None else horizon
        return max(0.0, self.seconds_to(target) * self.projected_tok_per_s())

    def estimate_wave_tokens(self, n_tasks: int, samples_per_task: int, effort: str) -> float:
        return n_tasks * samples_per_task * self.config.tokens_per_sample(effort)

    def plan_wave(
        self, n_tasks: int, samples_per_task: int, effort: str
    ) -> Tuple[int, str]:
        """Clip a proposed wave to what the remaining budget can actually pay for."""
        if n_tasks <= 0 or samples_per_task <= 0:
            return 0, "empty-wave"
        if self.must_stop_generating():
            return 0, "generation-deadline-passed"
        per_sample = self.config.tokens_per_sample(effort)
        remaining = self.affordable_completion_tokens()
        affordable_samples = int(remaining // max(1, n_tasks * per_sample))
        allowed = max(0, min(samples_per_task, affordable_samples))
        if allowed == 0:
            return 0, (
                f"token-budget-exhausted (remaining={remaining:.0f}, "
                f"need={n_tasks * per_sample} for 1 sample/task)"
            )
        if allowed < samples_per_task:
            return allowed, f"trimmed {samples_per_task}->{allowed} to fit token budget"
        return allowed, "full"

    def seconds_for_tokens(self, tokens: float) -> float:
        tps = self.projected_tok_per_s()
        return tokens / tps if tps > 0 else float("inf")

    def snapshot(self) -> Dict[str, object]:
        return {
            "elapsed_s": round(self.elapsed, 1),
            "plan_seconds_left": round(self.plan_seconds_left(), 1),
            "gen_seconds_left": round(self.gen_seconds_left(), 1),
            "hard_seconds_left": round(self.hard_seconds_left(), 1),
            "completion_tokens": self.ledger.completion_tokens,
            "prompt_tokens": self.ledger.prompt_tokens,
            "requests": self.ledger.requests,
            "failures": self.ledger.failures,
            "measured_tok_per_s": (
                round(self.measured_tok_per_s(), 1)
                if self.measured_tok_per_s() is not None
                else None
            ),
            "projected_tok_per_s": round(self.projected_tok_per_s(), 1),
            "affordable_completion_tokens": round(self.affordable_completion_tokens(), 0),
        }
