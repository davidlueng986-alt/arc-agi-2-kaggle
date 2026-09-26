"""Core data types for the H3 code-synthesis + verified-set disambiguation harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

Grid = np.ndarray  # 2-D array of ints 0..9


@dataclass(frozen=True)
class Task:
    task_id: str
    train: list[tuple[Grid, Grid]]
    test: list[Grid]

    @property
    def n_pairs(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


@dataclass
class ExecResult:
    ok: bool
    output: Optional[Grid] = None
    error: Optional[str] = None
    timed_out: bool = False
    ms: float = 0.0


@dataclass
class PairScore:
    """Outcome of running one program on one (input, expected_output) pair."""

    exact: bool
    soft: float  # mean cell accuracy when shapes match, else 0.0
    shape_ok: bool
    error: Optional[str] = None  # sandbox/execution error, if any
    diff: Optional[str] = None  # pred/truth diagnostic grid text


@dataclass
class ProgramEval:
    per_pair: list[PairScore]
    verified: bool  # all pairs exact
    soft_mean: float
    n_exec_errors: int = 0


@dataclass
class Candidate:
    """One synthesized program plus everything we know about it."""

    code: str
    source: str  # provenance tag, e.g. "e0i3" (expert 0, iteration 3)
    eval: ProgramEval
    test_outputs: list[Optional[Grid]] = field(default_factory=list)
    adj_vote: int = 0  # +1 when an adjudication/LOO program corroborates this behavior
    loo_corroboration: float = 0.0  # fraction of LOO folds whose verified program matches this behavior

    @property
    def verified(self) -> bool:
        return self.eval.verified

    @property
    def soft_mean(self) -> float:
        return self.eval.soft_mean


@dataclass
class ClusterScore:
    rep_index: int
    test_output: Optional[Grid]
    score: float
    support: float  # independence-weighted member count
    members: list[int]
    features: dict[str, float]


@dataclass
class DisambiguationResult:
    attempt_1: Optional[Grid]
    attempt_2: Optional[Grid]
    clusters: list[ClusterScore]
    features: list[dict[str, Any]]  # per-candidate feature dicts (for caching/ablation)
    agreement: Optional[np.ndarray] = None  # pairwise probe-agreement matrix
    notes: list[str] = field(default_factory=list)


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str = "mock"


@dataclass
class TaskResult:
    task_id: str
    attempts: list[tuple[Optional[Grid], Optional[Grid]]]  # per test input
    n_candidates: int
    n_verified: int
    n_distinct_verified_behaviors: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    disambiguation: Optional[DisambiguationResult] = None
    extras: dict[str, Any] = field(default_factory=dict)
