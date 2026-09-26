"""Candidate pooling and the 2-attempt selection rule.

Scoring is per-test-output exact match with two attempts, so votes are tallied
independently per test index (a sample that only parsed for test 1 still
contributes). attempt_1 is the weighted plurality; attempt_2 is the best candidate
with a *different output shape* whenever one exists, because if the plurality shape
is wrong then every same-shape candidate is wrong too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .grids import Grid, GridKey, from_key, shape, to_key
from .llm import EFFORT_RANK

EFFORT_WEIGHT = {name: 1.0 + 0.5 * rank for name, rank in EFFORT_RANK.items()}
TRUNCATION_PENALTY = 0.3
UNKNOWN_EFFORT_WEIGHT = 1.0


@dataclass
class Candidate:
    grid: Grid
    effort: str = "xhigh"
    order: int = 0
    truncated: bool = False

    @property
    def key(self) -> GridKey:
        return to_key(self.grid)

    @property
    def weight(self) -> float:
        w = EFFORT_WEIGHT.get(self.effort, UNKNOWN_EFFORT_WEIGHT)
        return w * (TRUNCATION_PENALTY if self.truncated else 1.0)

    @property
    def effort_rank(self) -> int:
        return EFFORT_RANK.get(self.effort, -1)


@dataclass
class IndexResult:
    attempt_1: Grid
    attempt_2: Grid
    votes: Dict[GridKey, float] = field(default_factory=dict)
    counts: Dict[GridKey, int] = field(default_factory=dict)
    consensus: float = 0.0
    n_candidates: int = 0
    n_distinct: int = 0
    shape_diverse: bool = False
    used_fallback: bool = False

    def as_dict(self) -> Dict[str, Grid]:
        return {"attempt_1": self.attempt_1, "attempt_2": self.attempt_2}


class CandidatePool:
    """Accumulates candidates for one task across all phases and samples."""

    def __init__(self, n_test: int) -> None:
        if n_test < 1:
            raise ValueError("n_test must be >= 1")
        self.n_test = n_test
        self._buckets: List[List[Candidate]] = [[] for _ in range(n_test)]
        self._counter = 0

    def add(self, index: int, grid: Grid, effort: str = "xhigh", truncated: bool = False) -> None:
        if not 0 <= index < self.n_test:
            raise IndexError(f"test index {index} out of range for n_test={self.n_test}")
        self._buckets[index].append(
            Candidate(grid=grid, effort=effort, order=self._counter, truncated=truncated)
        )
        self._counter += 1

    def add_sample(self, grids: Sequence[Optional[Grid]], effort: str, truncated: bool = False) -> int:
        added = 0
        for i, grid in enumerate(grids):
            if grid is not None:
                self.add(i, grid, effort=effort, truncated=truncated)
                added += 1
        return added

    def candidates(self, index: int) -> List[Candidate]:
        return list(self._buckets[index])

    def total(self) -> int:
        return sum(len(b) for b in self._buckets)

    def min_index_count(self) -> int:
        return min(len(b) for b in self._buckets)

    def consensus(self, index: int) -> float:
        if not self._buckets[index]:
            return 0.0
        votes = _tally(self._buckets[index])
        top = max(votes.values())
        return top / sum(votes.values())

    def select(self, fallbacks: Sequence[Grid]) -> List[IndexResult]:
        if len(fallbacks) != self.n_test:
            raise ValueError(f"expected {self.n_test} fallback grids, got {len(fallbacks)}")
        return [_select_index(self._buckets[i], fallbacks[i]) for i in range(self.n_test)]


def _tally(cands: Sequence[Candidate]) -> Dict[GridKey, float]:
    votes: Dict[GridKey, float] = {}
    for c in cands:
        votes[c.key] = votes.get(c.key, 0.0) + c.weight
    return votes


def _rank_key(item: Tuple[GridKey, float], cands: Sequence[Candidate]) -> Tuple[float, int, int]:
    key, weight = item
    best_rank = max(c.effort_rank for c in cands if c.key == key)
    first_order = min(c.order for c in cands if c.key == key)
    return (-weight, -best_rank, first_order)


def _select_index(cands: Sequence[Candidate], fallback: Grid) -> IndexResult:
    if not cands:
        return IndexResult(
            attempt_1=[list(r) for r in fallback],
            attempt_2=[list(r) for r in fallback],
            used_fallback=True,
        )

    votes = _tally(cands)
    counts: Dict[GridKey, int] = {}
    for c in cands:
        counts[c.key] = counts.get(c.key, 0) + 1
    ordered = sorted(votes.items(), key=lambda kv: _rank_key(kv, cands))

    top_key = ordered[0][0]
    top_shape = shape(from_key(top_key))

    diverse = next((k for k, _ in ordered[1:] if shape(from_key(k)) != top_shape), None)
    second_key = diverse if diverse is not None else (ordered[1][0] if len(ordered) > 1 else top_key)

    total_weight = sum(votes.values())
    return IndexResult(
        attempt_1=from_key(top_key),
        attempt_2=from_key(second_key),
        votes=dict(votes),
        counts=counts,
        consensus=(votes[top_key] / total_weight) if total_weight else 0.0,
        n_candidates=len(cands),
        n_distinct=len(votes),
        shape_diverse=diverse is not None,
    )


def consensus_reached(result: IndexResult, min_candidates: int = 4, threshold: float = 0.7) -> bool:
    if result.n_candidates < min_candidates:
        return False
    return result.consensus >= threshold


def task_consensus(results: Sequence[IndexResult], min_candidates: int, threshold: float) -> bool:
    """True when every test index of the task has a confident plurality."""
    if not results:
        return False
    return all(consensus_reached(r, min_candidates, threshold) for r in results)
