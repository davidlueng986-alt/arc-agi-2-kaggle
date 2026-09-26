"""Coverage instrumentation: the instrument that actually tests hypothesis H1.

The achieved score decomposes into two independent factors:

    score = coverage x selection_efficiency

* `coverage` (pass@N) is the fraction of test outputs for which at least one
  sampled candidate equals the ground truth. It is a hard ceiling set purely by
  model capability plus sample count. No amount of harness work can exceed it.
* `selection_efficiency` is score / coverage: how much of that ceiling the
  plurality + shape-diverse two-attempt rule actually captures.
* `pass_at_1` is the per-sample success rate, the single number that predicts
  coverage and therefore the number worth measuring first on a fresh model.

If coverage on the eval split is below 0.50, H1 ("pure reasoning reaches >=50%") is
refuted no matter what the selection rule does, and the honest conclusion is that
the binding constraint is per-sample capability rather than harness design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .grids import Grid, to_key
from .tasks import ARCTask


@dataclass
class CoverageStats:
    samples: int = 0
    correct_samples: int = 0
    outputs: int = 0
    outputs_covered: int = 0
    tasks: int = 0
    tasks_fully_covered: int = 0
    score: float = 0.0
    by_effort: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def pass_at_1(self) -> float:
        return self.correct_samples / self.samples if self.samples else 0.0

    @property
    def coverage(self) -> float:
        return self.outputs_covered / self.outputs if self.outputs else 0.0

    @property
    def selection_efficiency(self) -> float:
        return self.score / self.coverage if self.coverage else 0.0

    @property
    def uncovered_outputs(self) -> int:
        return self.outputs - self.outputs_covered

    def as_dict(self) -> Dict[str, object]:
        return {
            "samples": self.samples,
            "correct_samples": self.correct_samples,
            "pass_at_1": round(self.pass_at_1, 5),
            "outputs": self.outputs,
            "outputs_covered": self.outputs_covered,
            "coverage_pass_at_n": round(self.coverage, 5),
            "uncovered_outputs": self.uncovered_outputs,
            "tasks": self.tasks,
            "tasks_fully_covered": self.tasks_fully_covered,
            "score": round(self.score, 5),
            "selection_efficiency": round(self.selection_efficiency, 5),
            "by_effort": self.by_effort,
            "verdict": self.verdict(),
        }

    def verdict(self) -> str:
        if not self.outputs:
            return "no ground truth"
        if self.coverage >= 0.5:
            return "H1 ALIVE: coverage ceiling >= 0.50, score is selection-limited"
        if self.coverage >= 0.4:
            return "H1 MARGINAL: coverage ceiling 0.40-0.50, needs near-perfect selection"
        return "H1 REFUTED: coverage ceiling < 0.40, unreachable by any selection rule"


class CoverageTracker:
    """Records per-sample correctness against ground truth as candidates arrive."""

    def __init__(self, tasks: Sequence[ARCTask]) -> None:
        self._truth: Dict[str, Tuple[Tuple[Tuple[int, ...], ...], ...]] = {}
        self._covered: Dict[str, List[bool]] = {}
        self.stats = CoverageStats()
        for task in tasks:
            if task.solutions is None:
                continue
            self._truth[task.task_id] = tuple(to_key(g) for g in task.solutions)
            self._covered[task.task_id] = [False] * task.n_test
        self.stats.tasks = len(self._truth)
        self.stats.outputs = sum(len(v) for v in self._covered.values())

    @property
    def enabled(self) -> bool:
        return bool(self._truth)

    def observe(self, task_id: str, test_index: int, grid: Grid, effort: str = "?") -> bool:
        truth = self._truth.get(task_id)
        if truth is None:
            return False
        self.stats.samples += 1
        bucket = self.stats.by_effort.setdefault(effort, {"samples": 0, "correct": 0})
        bucket["samples"] += 1
        hit = test_index < len(truth) and to_key(grid) == truth[test_index]
        if hit:
            self.stats.correct_samples += 1
            bucket["correct"] += 1
            flags = self._covered[task_id]
            if test_index < len(flags):
                flags[test_index] = True
        return hit

    def finalize(self, score: Optional[float] = None) -> CoverageStats:
        self.stats.outputs_covered = sum(
            1 for flags in self._covered.values() for f in flags if f
        )
        self.stats.tasks_fully_covered = sum(
            1 for flags in self._covered.values() if flags and all(flags)
        )
        if score is not None:
            self.stats.score = score
        return self.stats
