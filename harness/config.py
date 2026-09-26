"""Pipeline configuration + the deadline-driven degradation ladder (HYPOTHESIS.md §4.4)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from harness.experts import ExpertConfig


@dataclass
class PipelineConfig:
    # experts
    experts: int = 2  # 0 = text-only, 1 = Expert A only, 2 = A + B
    iters_a: int = 8
    iters_b: int = 4
    pool_size: int = 5
    seed: int = 0
    solver_temperature: float = 1.0
    effort_solver: str = "low"
    max_tokens_solver: int = 2048
    sandbox_timeout: float = 5.0
    # text channel
    text_candidates: int = 3
    effort_text: str = "low"
    # judge
    use_judge: bool = True
    judges: int = 3
    effort_judge: str = "medium"
    judge_max_entries: int = 12
    judge_trace_cap_chars: int = 2500
    max_tokens_judge: int = 4096
    # selection
    diversity: bool = True
    # budget / time
    token_budget_per_task: int = 120_000  # GPU-effective
    total_time_s: float = 41_700.0
    max_task_seconds: float = 900.0
    task_concurrency: int = 2
    # client
    model: str = "qwen3.8-27b"
    base_url: str = "http://127.0.0.1:8000/v1"

    def expert_configs(self) -> list[ExpertConfig]:
        cfgs: list[ExpertConfig] = []
        if self.experts >= 1:
            cfgs.append(
                ExpertConfig(
                    name="expert_a",
                    variant="A",
                    max_iterations=self.iters_a,
                    # disjoint seed ranges (Poetiq solve_parallel_coding):
                    # each expert's iterations use seed + it, so ranges of
                    # max_iterations guarantee no seed collision.
                    seed=self.seed,
                    temperature=self.solver_temperature,
                    reasoning_effort=self.effort_solver,
                    max_tokens=self.max_tokens_solver,
                    pool_size=self.pool_size,
                    sandbox_timeout=self.sandbox_timeout,
                )
            )
        if self.experts >= 2:
            cfgs.append(
                ExpertConfig(
                    name="expert_b",
                    variant="B",
                    max_iterations=self.iters_b,
                    seed=self.seed + max(self.iters_a, 1) * 1000,
                    temperature=self.solver_temperature,
                    reasoning_effort=self.effort_solver,
                    max_tokens=self.max_tokens_solver,
                    pool_size=self.pool_size,
                    sandbox_timeout=self.sandbox_timeout,
                )
            )
        return cfgs


def degraded(cfg: PipelineConfig, level: int) -> PipelineConfig:
    """Degradation ladder driven by Deadline.level():
      L1 @60%: iters A 8->5, text 3->1
      L2 @75%: iters ->3, no text, 1 judge
      L3 @88%: single-shot experts, no judge, consensus-only
      L4 @93%: handled by the runner (stop solving, finalize).
    """
    if level <= 0:
        return cfg
    if level == 1:
        return replace(cfg, iters_a=min(cfg.iters_a, 5), iters_b=min(cfg.iters_b, 3),
                       text_candidates=min(cfg.text_candidates, 1))
    if level == 2:
        return replace(cfg, iters_a=min(cfg.iters_a, 3), iters_b=min(cfg.iters_b, 2),
                       text_candidates=0, judges=1)
    return replace(cfg, iters_a=min(cfg.iters_a, 1), iters_b=min(cfg.iters_b, 1),
                   text_candidates=0, use_judge=False, judges=1)
